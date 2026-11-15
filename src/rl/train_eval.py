"""Noise-free evaluation hook for the training loop (``noise_free_eval`` in the RL config).

The numbers in ``metrics.jsonl`` that training logs per update come from games with sampled
actions, ``opponent_noise`` and whatever curriculum phase is active, so they are not a clean
measurement. Every ``noise_free_eval.every_steps`` environment steps the trainer calls
:class:`NoiseFreeEvaluator`, which plays the CURRENT policy through the same code as
``python -m rl.evaluate`` (argmax actions, exactly scripted opponents, fixed seeds) on the small
eval YAML named in the config, and returns plain scalars for the metrics log plus a full report.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from ctf.config import CTFConfig, config_to_dict
from rl.config import RLConfig
from rl.eval_config import EvalConfig, PolicyMode, load_eval_config
from rl.evaluate import evaluate_extras, evaluate_policy
from rl.mappo import ValueNorm
from rl.networks import ActorCritic
from rl.rl_strategy import RLStrategy
from rl.schema import ConfigError, to_dict


def load_training_eval_config(cfg: RLConfig, root: Path | None = None) -> EvalConfig:
    """The eval config named by ``cfg.noise_free_eval``; relative paths resolve against ``root``."""
    nf = cfg.noise_free_eval
    path = Path(nf.config_path)
    if not path.is_absolute() and root is not None:
        path = root / path
    eval_cfg = load_eval_config(path, nf.overrides)
    n_roles = len(cfg.roles.slots)
    for opp in eval_cfg.opponents:
        roles = opp.params.get("roles")
        if roles is not None and len(roles) != n_roles:
            raise ConfigError(
                f"noise_free_eval: opponent {opp.label!r} in {path} has {len(roles)} roles but the team has {n_roles}"
            )
    return eval_cfg


def flatten_scalars(results: dict[str, Any], extras: dict[str, Any]) -> dict[str, float]:
    """The numbers worth plotting from an evaluation, as ``{tag: float}`` (TensorBoard-style tags)."""
    out: dict[str, float] = {}
    wins, stays = [], []
    for label, block in results.items():
        s = block["all"]
        out[f"eval/win_rate/{label}"] = s["win"]["rate"]
        out[f"eval/draw_rate/{label}"] = s["draw"]["rate"]
        out[f"eval/loss_rate/{label}"] = s["loss"]["rate"]
        wins.append(s["win"]["rate"])
        stays.append(s["stay_fraction"]["all"])
    if wins:
        out["eval/mean_win_rate"] = sum(wins) / len(wins)
        out["eval/min_win_rate"] = min(wins)
        out["eval/stay_fraction"] = sum(stays) / len(stays)
    held = extras.get("held_out")
    if held:
        all_wins = []
        for map_label, per_opp in held.items():
            map_wins = [b["all"]["win"]["rate"] for b in per_opp.values()]
            out[f"eval_held_out/{map_label}/win_rate"] = sum(map_wins) / len(map_wins)
            all_wins += map_wins
        out["eval_held_out/mean_win_rate"] = sum(all_wins) / len(all_wins)
    swap = extras.get("role_swap")
    if swap and swap["all"]["rate"] is not None:
        out["eval/role_swap_changed"] = swap["all"]["rate"]
        for pair, stat in swap["pairs"].items():
            out[f"eval/role_swap/{pair}"] = stat["rate"]
    check = extras.get("policy_check")
    if check:
        diffs = [c["win_difference"]["diff"] for c in check.values() if c["win_difference"]["diff"] is not None]
        if diffs:
            out["eval/argmax_minus_sampled_win"] = sum(diffs) / len(diffs)
    return {k: float(v) for k, v in out.items() if v is not None}


class NoiseFreeEvaluator:
    def __init__(self, cfg: RLConfig, env_config: CTFConfig, run_dir: Path, root: Path | None = None) -> None:
        self.cfg, self.env_config, self.run_dir = cfg, env_config, run_dir
        self.eval_cfg = load_training_eval_config(cfg, root if root is not None else Path.cwd())

    def run(
        self, model: ActorCritic, value_norm: ValueNorm, total_steps: int, device: torch.device
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Evaluate the live model; returns (scalars for the log, full report). Training state is untouched:
        the model's train/eval mode is restored and the evaluation uses its own random generators."""
        was_training = model.training
        t0 = time.time()
        try:
            strategy = RLStrategy(
                model, self.cfg, self.env_config, value_norm, seed=self.eval_cfg.seed, device=device,
                deterministic=self.eval_cfg.policy_mode is PolicyMode.ARGMAX,
            )
            roles = list(strategy.roles)
            results = evaluate_policy(strategy, roles, self.env_config, self.eval_cfg)
            extras = evaluate_extras(strategy, roles, self.env_config, self.eval_cfg)
        finally:
            model.train(was_training)
        seconds = time.time() - t0
        scalars = flatten_scalars(results, extras)
        scalars["eval/seconds"] = seconds
        report = {
            "meta": {
                "step": int(total_steps),
                "seconds": round(seconds, 1),
                "eval_config": to_dict(self.eval_cfg),
                "env_config": config_to_dict(self.env_config),
            },
            "results": results,
            **extras,
        }
        return scalars, report

    def save(self, report: dict[str, Any], total_steps: int) -> Path:
        out = self.run_dir / "eval" / f"step_{total_steps:010d}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return out

    @staticmethod
    def summary(scalars: dict[str, float]) -> str:
        parts = [f"mean win {scalars.get('eval/mean_win_rate', float('nan')):.2f}",
                 f"min {scalars.get('eval/min_win_rate', float('nan')):.2f}"]
        if "eval_held_out/mean_win_rate" in scalars:
            parts.append(f"held-out {scalars['eval_held_out/mean_win_rate']:.2f}")
        if "eval/role_swap_changed" in scalars:
            parts.append(f"role-swap {scalars['eval/role_swap_changed']:.2f}")
        return " | ".join(parts) + f" | {scalars['eval/seconds']:.0f}s"
