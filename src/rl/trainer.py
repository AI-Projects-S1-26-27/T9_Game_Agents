"""MAPPO training loop."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from ctf.config import CTFConfig
from rl.buffer import RolloutBuffer
from rl.checkpoint import CheckpointManager, load_checkpoint, restore_training_state
from rl.config import RLConfig
from rl.curriculum import Curriculum
from rl.league import LeagueStore, LeagueWriter, resolve_league_dir, validate_file_opponents
from rl.mappo import MAPPO, ValueNorm
from rl.metrics import MetricsLogger, summarize_episodes
from rl.networks import ActorCritic, NetworkSpec
from rl.rollout import collect
from rl.schema import ConfigError
from rl.train_eval import NoiseFreeEvaluator
from rl.vec_env import make_vec_env


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _lerp(start: float, end: float, progress: float) -> float:
    return start + (end - start) * min(max(progress, 0.0), 1.0)


class Trainer:
    def __init__(
        self,
        cfg: RLConfig,
        env_config: CTFConfig,
        run_dir: Path,
        resume_from: Path | None = None,
    ) -> None:
        self.cfg, self.env_config, self.run_dir = cfg, env_config, run_dir
        self.device = resolve_device(cfg.run.device)
        torch.set_num_threads(cfg.run.torch_threads)
        torch.manual_seed(cfg.run.seed)

        spec = NetworkSpec.from_env(env_config)
        self.model = ActorCritic(spec, cfg.model).to(self.device)
        self.value_norm = ValueNorm(cfg.ppo.value_normalization).to(self.device)
        self.algo = MAPPO(self.model, cfg.ppo, self.value_norm)
        self.curriculum = Curriculum(cfg.curriculum.phases)
        self.checkpoints = CheckpointManager(run_dir, cfg.run.keep_checkpoints)
        self.counters = {"total_steps": 0, "update": 0}
        # Fails here, before any training, if the eval YAML is invalid.
        self.evaluator = NoiseFreeEvaluator(cfg, env_config, run_dir) if cfg.noise_free_eval.enabled else None
        validate_file_opponents(cfg)  # a missing opponent file stops the run now, not hours in
        self.league_dir = resolve_league_dir(cfg, run_dir)
        if cfg.league.directory and resume_from is None and LeagueStore(self.league_dir, 1).snapshots():
            raise ConfigError(
                f"league.directory {self.league_dir} already holds snapshots, and a new run would play them as its "
                "opponents. Use an empty folder (or delete them), or --resume to continue the run they belong to."
            )
        self.league = LeagueWriter(self.league_dir, cfg.league.keep_snapshots) if cfg.league.snapshot_every_steps > 0 else None
        if resume_from is not None:
            payload = load_checkpoint(resume_from, map_location="cpu")
            self.counters = restore_training_state(
                payload,
                model=self.model,
                optimizer=self.algo.optimizer,
                value_norm=self.value_norm,
                curriculum=self.curriculum,
            )
        self.buffer = RolloutBuffer(
            cfg.rollout.rollout_length, cfg.rollout.num_envs, env_config.agents_per_team, spec
        )
        self.generator = None if self.device.type == "mps" else torch.Generator(device=self.device)
        if self.generator is not None:
            self.generator.manual_seed(cfg.run.seed + self.counters["update"])

    # ------------------------------------------------------------------ helpers

    def _state(self) -> dict:
        return {
            "model": self.model,
            "optimizer": self.algo.optimizer,
            "value_norm": self.value_norm,
            "curriculum": self.curriculum,
            "counters": self.counters,
            "config": self.cfg,
            "env_config": self.env_config,
        }

    def _scalars(self, stats, episodes, lr, entropy_coef, scout_scale, sps) -> dict[str, float]:
        roles = [r.value for r in self.cfg.roles.slots]
        scalars = {f"train/{k}": v for k, v in stats.items()}
        scalars.update({f"episode/{k}": v for k, v in summarize_episodes(episodes, roles).items()})
        scalars["curriculum/phase"] = float(self.curriculum.index)
        if self.curriculum.win_rate is not None:
            scalars["curriculum/rolling_win_rate"] = self.curriculum.win_rate
        scalars["schedule/learning_rate"] = lr
        scalars["schedule/entropy_coef"] = entropy_coef
        scalars["schedule/scout_scale"] = scout_scale
        scalars["perf/steps_per_sec"] = sps
        if self.league is not None:
            scalars["league/num_snapshots"] = float(len(LeagueStore(self.league_dir, 1).snapshots()))
            ages = [self.counters["total_steps"] - e["snapshot_step"] for e in episodes if e.get("snapshot_step") is not None]
            if episodes:
                scalars["league/games_fraction"] = len(ages) / len(episodes)
            if ages:  # how old (in training steps) the snapshots the learner just played against were
                scalars["league/opponent_age_steps"] = sum(ages) / len(ages)
        return scalars

    def _noise_free_eval(self, logger: MetricsLogger) -> None:
        """One noise-free evaluation of the current policy. A failure is reported and training goes on:
        a broken eval must not cost a multi-hour run."""
        assert self.evaluator is not None
        step = self.counters["total_steps"]
        try:
            scalars, report = self.evaluator.run(self.model, self.value_norm, step, self.device)
            path = self.evaluator.save(report, step)
        except Exception as exc:  # noqa: BLE001 - see docstring
            print(f"noise-free eval FAILED at step {step:,}: {type(exc).__name__}: {exc}", flush=True)
            logger.event(step, "noise_free_eval_failed", error=f"{type(exc).__name__}: {exc}")
            return
        logger.log_eval(step, scalars)
        print(f"--> noise-free eval @ {step / 1e6:.3f}M | {self.evaluator.summary(scalars)} | {path}", flush=True)

    # --------------------------------------------------------------------- train

    def run(self) -> dict[str, int]:
        cfg = self.cfg
        steps_per_update = cfg.rollout.rollout_length * cfg.rollout.num_envs
        # Resumed runs use different env seeds than the segment they continue.
        seed = cfg.run.seed + self.counters["update"] * cfg.rollout.num_envs
        vec = make_vec_env(cfg, self.env_config, seed, self.league_dir)
        logger = MetricsLogger(self.run_dir, cfg.run.tensorboard)
        pending: list[dict] = []
        window_start, window_steps = time.time(), 0
        next_checkpoint = (self.counters["total_steps"] // cfg.run.checkpoint_every_steps + 1) * (
            cfg.run.checkpoint_every_steps
        )
        every = cfg.noise_free_eval.every_steps
        next_eval = (self.counters["total_steps"] // every + 1) * every
        last_eval_step = -1
        snap_every = cfg.league.snapshot_every_steps
        next_snapshot = (self.counters["total_steps"] // snap_every + 1) * snap_every if snap_every > 0 else None
        try:
            vec.set_phase(self.curriculum.index)
            obs = vec.reset()
            while self.counters["total_steps"] < cfg.rollout.total_steps:
                total = self.counters["total_steps"]
                progress = total / cfg.rollout.total_steps
                lr = _lerp(cfg.ppo.learning_rate, cfg.ppo.learning_rate_final, progress)
                entropy_coef = _lerp(cfg.ppo.entropy_coef, cfg.ppo.entropy_coef_final, progress)
                scout_scale = max(0.0, 1.0 - total / cfg.rewards.scout_anneal_steps)
                vec.set_scout_scale(scout_scale)

                obs, episodes = collect(vec, obs, self.model, self.value_norm, self.buffer, self.device)
                self.buffer.compute_gae(cfg.ppo.gamma, cfg.ppo.gae_lambda)
                self.model.train()
                stats = self.algo.update(self.buffer.to_batch(self.device), lr, entropy_coef, self.generator)

                self.counters["total_steps"] += steps_per_update
                self.counters["update"] += 1
                window_steps += steps_per_update
                pending.extend(episodes)

                if self.curriculum.record([e["outcome"] for e in episodes], steps_per_update):
                    phase = self.curriculum.phase
                    vec.set_phase(self.curriculum.index)
                    logger.event(self.counters["total_steps"], "phase_advance",
                                 phase_index=self.curriculum.index, phase=phase.name)
                    self.checkpoints.save_phase_start(self.curriculum.index, phase.name, **self._state())
                    print(f"--> phase {self.curriculum.index}: {phase.name}", flush=True)

                if self.counters["update"] % cfg.run.log_every_updates == 0:
                    sps = window_steps / max(time.time() - window_start, 1e-9)
                    scalars = self._scalars(stats, pending, lr, entropy_coef, scout_scale, sps)
                    logger.log(self.counters["total_steps"], scalars)
                    self._print(scalars)
                    pending, window_start, window_steps = [], time.time(), 0

                if self.counters["total_steps"] >= next_checkpoint:
                    self.checkpoints.save_periodic(self.counters["total_steps"], **self._state())
                    next_checkpoint += cfg.run.checkpoint_every_steps

                if self.league is not None and self.counters["total_steps"] >= next_snapshot:
                    self.league.save(
                        self.counters["total_steps"], model=self.model, value_norm=self.value_norm,
                        counters=self.counters, config=cfg, env_config=self.env_config,
                    )
                    while next_snapshot <= self.counters["total_steps"]:
                        next_snapshot += snap_every

                if self.evaluator is not None and self.counters["total_steps"] >= next_eval:
                    self._noise_free_eval(logger)
                    last_eval_step = self.counters["total_steps"]
                    while next_eval <= self.counters["total_steps"]:
                        next_eval += every
            self.checkpoints.save_periodic(self.counters["total_steps"], **self._state())
            if self.evaluator is not None and cfg.noise_free_eval.at_end and self.counters["total_steps"] != last_eval_step:
                self._noise_free_eval(logger)  # after the final checkpoint is safely on disk
        except KeyboardInterrupt:
            print("\ninterrupted: saving checkpoint...", flush=True)
            self.checkpoints.save_periodic(self.counters["total_steps"], **self._state())
        finally:
            vec.close()
            logger.close()
        return dict(self.counters)

    def _print(self, s: dict[str, float]) -> None:
        phase = self.cfg.curriculum.phases[self.curriculum.index].name
        win = s.get("episode/win_rate")
        draw = s.get("episode/draw_rate")
        episode = f"win {win:.2f} draw {draw:.2f}" if win is not None else "no finished episodes"
        print(
            f"upd {self.counters['update']:6d} | step {self.counters['total_steps'] / 1e6:7.3f}M"
            f" | {s['perf/steps_per_sec']:6.0f} sps | {phase} | {episode}"
            f" | ent {s['train/entropy']:.2f} | kl {s['train/approx_kl']:.4f}"
            f" | ev {s['train/explained_variance']:.2f}",
            flush=True,
        )
