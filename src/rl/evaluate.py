"""Noise-free evaluation of a trained policy against fixed opponents.

    python -m rl.evaluate --checkpoint runs/<run>/checkpoints/latest.pt
    python -m rl.evaluate --checkpoint ... --opponents random scripted_chaser --set episodes=40
    python -m rl.evaluate --checkpoint ... --env-set visibility_radius=6 --set policy_mode=sampled

What "noise-free" means here: the opponents play exactly as scripted (no ``opponent_noise``), the
policy plays argmax actions by default, and every episode's spawns / opponent randomness come from a
seed derived from ``seed``, the opponent label, the side and the episode index. Two policies
evaluated with the same config therefore face the same games (paired comparison).

For each opponent and side the policy plays, the report gives win / draw / loss rates with Wilson
intervals, and behaviour metrics: captures, time to first pickup, pickup-to-capture conversion, tags,
STAY share, scout map coverage, per-role action histograms and position heatmaps, route choice
(through which side of the centre column the attackers pass) and how far the defender stays from
its own flag. Everything tunable lives in the eval YAML (``configs/rl/eval.yaml``); see
docs/evaluation.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from random import Random
from typing import Any

import numpy as np

from ctf.actions import Action
from ctf.agent import Team
from ctf.config import CTFConfig, config_from_dict, config_to_dict
from ctf.environment import CTFEnvironment
from ctf.game_state import GameState
from ctf.observations import ObservationMode, manhattan_cells
from rl.canonical import ACTION_INDEX, NUM_ACTIONS
from rl.config import OpponentSpec
from rl.eval_config import EvalConfig, EvalOpponent, PolicyMode, Side, load_eval_config
from rl.eval_stats import mean_stat, rate
from rl.opponents import KNOWN_KINDS, build_opponent
from rl.schema import ConfigError, parse_override
from strategies import BaseStrategy, Role

ACTION_NAMES = [a.name for a in Action]

# Opponent kinds that live outside rl.opponents (the classical AI, the LLM agent) register a
# factory here: kind -> callable(spec, env_config, rng) -> BaseStrategy. Nothing is registered by
# default; the eval YAML may only name a kind that is built in or registered.
EXTERNAL_OPPONENTS: dict[str, Callable[[EvalOpponent, CTFConfig, Random], BaseStrategy]] = {}


def build_eval_opponent(spec: EvalOpponent, env_config: CTFConfig, rng: Random) -> BaseStrategy:
    if spec.kind in EXTERNAL_OPPONENTS:
        return EXTERNAL_OPPONENTS[spec.kind](spec, env_config, rng)
    if spec.kind not in KNOWN_KINDS:
        raise ConfigError(
            f"unknown eval opponent kind {spec.kind!r} (label {spec.label!r}); "
            f"built in: {KNOWN_KINDS}, registered: {sorted(EXTERNAL_OPPONENTS)}"
        )
    return build_opponent(OpponentSpec(spec.kind, spec.label, 1.0, dict(spec.params)), env_config, rng)


# ------------------------------------------------------------------------------ environments


def eval_env_config(base: CTFConfig, overrides: dict[str, Any], *, fixed_spawns: bool) -> CTFConfig:
    """``base`` with the eval overrides applied; ``fixed_spawns`` forces spawn_mode=fixed."""
    merged = {**config_to_dict(base), **overrides}
    if fixed_spawns:
        merged["spawn_mode"] = "fixed"
    try:
        return config_from_dict(merged)
    except ValueError as exc:
        raise ConfigError(f"eval env_overrides invalid: {exc}") from exc


def is_mirror_symmetric(config: CTFConfig) -> bool:
    """Is the map left-right symmetric (so BLUE's view can be flipped onto RED's)?"""
    w = config.width
    return (
        {(w - 1 - x, y) for x, y in config.obstacles} == set(config.obstacles)
        and (w - 1 - config.red_base[0], config.red_base[1]) == tuple(config.blue_base)
    )


# ---------------------------------------------------------------------------- per-episode data


@dataclass
class EpisodeRecord:
    outcome: str  # win | draw | loss, from the policy's point of view
    length: int
    side: str  # red | blue
    captures: int
    pickups: int
    first_pickup: int | None  # timestep (1-based) of the policy team's first pickup
    tags_inflicted: int
    tags_suffered: int
    action_counts: np.ndarray  # (roles, NUM_ACTIONS)
    heat: np.ndarray  # (roles, H, W) active-agent position counts, in environment coordinates
    scout_coverage: float | None  # share of traversable cells within sight of an active scout; None without a scout
    route: dict[str, int]  # centre-column entries by route_roles agents: top / middle / bottom
    route_carrying: dict[str, int]  # the same, only entries made while carrying the enemy flag
    defender_dist_base: float | None  # mean Manhattan distance defender -> own base cell
    defender_dist_flag: float | None  # mean Manhattan distance defender -> own flag's current cell
    mirrored: bool = field(default=False)  # BLUE episode on a symmetric map: flip x to get team-relative coordinates


class _Recorder:
    """Collects one episode's behaviour data from state snapshots, events and chosen actions."""

    def __init__(self, env_config: CTFConfig, team: Team, roles: list[Role], route_roles: set[Role]) -> None:
        self.cfg, self.team, self.roles, self.route_roles = env_config, team, roles, route_roles
        self.role_names = list(dict.fromkeys(r.value for r in roles))
        self.ids = [f"{'R' if team is Team.RED else 'B'}{i}" for i in range(1, len(roles) + 1)]
        self.slot_role = {aid: self.role_names.index(r.value) for aid, r in zip(self.ids, roles)}
        self.slot_kind = dict(zip(self.ids, roles))
        self.actions = np.zeros((len(self.role_names), NUM_ACTIONS), dtype=np.int64)
        self.heat = np.zeros((len(self.role_names), env_config.height, env_config.width), dtype=np.int64)
        obstacles = set(env_config.obstacles)
        self.traversable = {
            (x, y) for y in range(env_config.height) for x in range(env_config.width) if (x, y) not in obstacles
        }
        self.seen: set[tuple[int, int]] = set()
        self.has_scout = Role.SCOUT in roles
        self.center_cols = {(env_config.width - 1) // 2, env_config.width // 2}
        self.mid_row = (env_config.height - 1) / 2
        self.prev_in_center: dict[str, bool] = {}
        self.route = {"top": 0, "middle": 0, "bottom": 0}
        self.route_carrying = {"top": 0, "middle": 0, "bottom": 0}
        self.def_base: list[float] = []
        self.def_flag: list[float] = []
        self.pickups = self.captures = self.tags_inflicted = self.tags_suffered = 0
        self.first_pickup: int | None = None

    def snapshot(self, state: GameState) -> None:
        own_base = state.red_base if self.team is Team.RED else state.blue_base
        own_flag = state.flags[self.team]
        for aid in self.ids:
            agent = state.agents[aid]
            if not agent.active:
                self.prev_in_center.pop(aid, None)
                continue
            role = self.slot_kind[aid]
            self.heat[self.slot_role[aid], agent.y, agent.x] += 1
            if role is Role.SCOUT:
                self.seen |= manhattan_cells(agent.x, agent.y, self.cfg.visibility_radius, self.cfg.width, self.cfg.height)
            if role is Role.DEFENDER:
                self.def_base.append(abs(agent.x - own_base[0]) + abs(agent.y - own_base[1]))
                self.def_flag.append(abs(agent.x - own_flag.x) + abs(agent.y - own_flag.y))
            if role in self.route_roles:
                inside = agent.x in self.center_cols
                if inside and not self.prev_in_center.get(aid, False):
                    lane = "top" if agent.y < self.mid_row else "bottom" if agent.y > self.mid_row else "middle"
                    self.route[lane] += 1
                    if agent.carrying_flag:
                        self.route_carrying[lane] += 1
                self.prev_in_center[aid] = inside

    def act(self, actions: dict[str, Action]) -> None:
        for aid, action in actions.items():
            self.actions[self.slot_role[aid], ACTION_INDEX[action]] += 1

    def events(self, events: list[dict], timestep: int) -> None:
        mine = set(self.ids)
        for ev in events:
            kind = ev.get("type")
            if kind == "flag_pickup" and ev["agent_id"] in mine:
                self.pickups += 1
                if self.first_pickup is None:
                    self.first_pickup = timestep
            elif kind == "flag_capture" and ev["agent_id"] in mine:
                self.captures += 1
            elif kind == "tag":
                self.tags_suffered += ev["agent_id"] in mine  # the tagged carrier is ours
                self.tags_inflicted += ev["by_agent_id"] in mine

    def finish(self, outcome: str, length: int, side: str, mirrored: bool) -> EpisodeRecord:
        coverage = len(self.seen & self.traversable) / len(self.traversable) if self.has_scout else None
        return EpisodeRecord(
            outcome=outcome, length=length, side=side, captures=self.captures, pickups=self.pickups,
            first_pickup=self.first_pickup, tags_inflicted=self.tags_inflicted, tags_suffered=self.tags_suffered,
            action_counts=self.actions, heat=self.heat, scout_coverage=coverage, route=self.route,
            route_carrying=self.route_carrying,
            defender_dist_base=float(np.mean(self.def_base)) if self.def_base else None,
            defender_dist_flag=float(np.mean(self.def_flag)) if self.def_flag else None,
            mirrored=mirrored,
        )


def play_episode(
    env: CTFEnvironment,
    learner: BaseStrategy,
    roles: list[Role],
    opponent: BaseStrategy,
    team: Team,
    route_roles: set[Role],
    env_seed: int,
    mirrored: bool = False,
) -> EpisodeRecord:
    """One game: ``learner`` plays ``team``, ``opponent`` the other side. No noise on either."""
    state = env.reset(seed=env_seed)
    learner.reset()
    opponent.reset()
    rec = _Recorder(env.config, team, roles, route_roles)
    rec.snapshot(state)
    done = False
    while not done:
        mine = learner.act(env.observe(ObservationMode.PARTIAL, team=team), team)
        theirs = opponent.act(env.observe(ObservationMode.PARTIAL, team=team.opponent), team.opponent)
        rec.act(mine)
        state, _, done, info = env.step({**mine, **theirs})
        rec.events(info.get("events", []), state.timestep)
        rec.snapshot(state)
    outcome = "win" if state.winner is team else "draw" if state.winner is None else "loss"
    return rec.finish(outcome, state.timestep, "red" if team is Team.RED else "blue", mirrored)


# ----------------------------------------------------------------------------- aggregation


def _lane_stats(records: list[EpisodeRecord], attr: str, confidence: float) -> dict[str, Any]:
    totals = {lane: sum(getattr(r, attr)[lane] for r in records) for lane in ("top", "middle", "bottom")}
    crossings = sum(totals.values())
    return {**totals, "top_share": rate(totals["top"], crossings, confidence)}


def summarize(
    records: list[EpisodeRecord], role_names: list[str], confidence: float, *, team_relative: bool
) -> dict[str, Any]:
    """Aggregate episode records into the report numbers (plain JSON types)."""
    n = len(records)
    out: dict[str, Any] = {"n": n}
    for outcome in ("win", "draw", "loss"):
        out[outcome] = rate(sum(r.outcome == outcome for r in records), n, confidence)
    out["episode_length"] = mean_stat([r.length for r in records], confidence)
    out["captures"] = {"total": sum(r.captures for r in records), "per_episode": mean_stat([r.captures for r in records], confidence)}
    pickups = sum(r.pickups for r in records)
    with_pickup = [r for r in records if r.first_pickup is not None]
    out["pickups"] = {"total": pickups, "per_episode": mean_stat([r.pickups for r in records], confidence)}
    out["episodes_with_pickup"] = rate(len(with_pickup), n, confidence)
    out["time_to_first_pickup"] = mean_stat([r.first_pickup for r in with_pickup], confidence)
    # Each pickup either ends in a capture or is lost (tagged, timeout); pickups within one episode are
    # not independent, so read this interval as approximate.
    out["pickup_to_capture"] = rate(sum(r.captures for r in records), pickups, confidence)
    out["tags_inflicted"] = mean_stat([r.tags_inflicted for r in records], confidence)
    out["tags_suffered"] = mean_stat([r.tags_suffered for r in records], confidence)

    actions = sum((r.action_counts for r in records), np.zeros((len(role_names), NUM_ACTIONS), dtype=np.int64))
    stay = ACTION_INDEX[Action.STAY]
    out["stay_fraction"] = {
        "all": float(actions[:, stay].sum() / max(actions.sum(), 1)),
        **{role: float(actions[i, stay] / max(actions[i].sum(), 1)) for i, role in enumerate(role_names)},
    }
    out["action_histograms"] = {
        role: {name: float(actions[i, j] / max(actions[i].sum(), 1)) for j, name in enumerate(ACTION_NAMES)}
        for i, role in enumerate(role_names)
    }
    coverage = [r.scout_coverage for r in records if r.scout_coverage is not None]
    out["scout_coverage"] = mean_stat(coverage, confidence) if coverage else None

    # Heatmaps need one coordinate frame: a single side (environment coordinates), or both sides on a
    # symmetric map with BLUE flipped so every team's own base is on the left (team-relative).
    heat = None
    if records and (team_relative or len({r.side for r in records}) == 1):
        heat = np.zeros_like(records[0].heat)
        for r in records:
            heat += r.heat[:, :, ::-1] if (team_relative and r.mirrored) else r.heat
    out["heatmaps"] = (
        None
        if heat is None
        else {role: (heat[i] / max(heat[i].sum(), 1)).tolist() for i, role in enumerate(role_names)}  # rows = y
    )
    out["heatmaps_team_relative"] = bool(team_relative)
    out["route"] = _lane_stats(records, "route", confidence)
    out["route_carrying"] = _lane_stats(records, "route_carrying", confidence)
    out["defender_distance"] = {
        "to_base": mean_stat([r.defender_dist_base for r in records if r.defender_dist_base is not None], confidence),
        "to_flag": mean_stat([r.defender_dist_flag for r in records if r.defender_dist_flag is not None], confidence),
    }
    return out


# --------------------------------------------------------------------------------- driver


def evaluate_policy(
    learner: BaseStrategy,
    roles: list[Role],
    env_config: CTFConfig,
    cfg: EvalConfig,
    *,
    only: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Play every configured opponent on every configured side; returns ``{label: {side|all: summary}}``."""
    if len(roles) != env_config.agents_per_team:
        raise ConfigError(f"policy has {len(roles)} roles but the env has {env_config.agents_per_team} agents per team")
    opponents = cfg.opponents
    if only:
        unknown = sorted(set(only) - {o.label for o in opponents})
        if unknown:
            raise ConfigError(f"unknown opponent label(s) {unknown}; configured: {[o.label for o in opponents]}")
        opponents = [o for o in opponents if o.label in only]
    envs = {
        fixed: CTFEnvironment(eval_env_config(env_config, cfg.env_overrides, fixed_spawns=fixed))
        for fixed in (True, False)
    }
    role_names = list(dict.fromkeys(r.value for r in roles))
    route_roles = set(cfg.route_roles)
    symmetric = is_mirror_symmetric(env_config)
    results: dict[str, Any] = {}
    for opp in opponents:
        per_side: dict[str, list[EpisodeRecord]] = {}
        for side in cfg.sides:
            records: list[EpisodeRecord] = []
            for i in range(cfg.episodes_per_side):
                key = f"{cfg.seed}:{opp.label}:{side.value}:{i}"
                draw = Random(key)
                fixed = draw.random() < cfg.fixed_spawn_fraction
                env_seed, opp_seed = draw.randrange(2**31), draw.randrange(2**31)
                opponent = build_eval_opponent(opp, envs[fixed].config, Random(opp_seed))
                records.append(
                    play_episode(
                        envs[fixed], learner, roles, opponent, side.team, route_roles, env_seed,
                        mirrored=symmetric and side is Side.BLUE,
                    )
                )
            per_side[side.value] = records
            if progress:
                wins = sum(r.outcome == "win" for r in records)
                progress(f"  vs {opp.label:<18} as {side.value:<4} {wins}/{len(records)} wins")
        everything = [r for recs in per_side.values() for r in recs]
        block = {"all": summarize(everything, role_names, cfg.confidence, team_relative=symmetric)}
        for side_name, recs in per_side.items():
            block[side_name] = summarize(recs, role_names, cfg.confidence, team_relative=False)
        results[opp.label] = block
    return results


# ---------------------------------------------------------------------------------- report


def _pct(stat: dict) -> str:
    return "  n/a " if stat["rate"] is None else f"{100 * stat['rate']:5.1f}"


def _pct_text(stat: dict) -> str:
    return "n/a" if stat["rate"] is None else f"{100 * stat['rate']:.0f}%"


def _ci(stat: dict) -> str:
    return f"[{100 * stat['lo']:4.0f},{100 * stat['hi']:4.0f}]"


def _num(stat: dict | None, fmt: str = "{:5.1f}") -> str:
    return "  n/a" if stat is None or stat["mean"] is None else fmt.format(stat["mean"])


def format_report(results: dict[str, Any], meta: dict[str, Any]) -> str:
    """Plain-text table: one row per opponent (both sides pooled), then one row per side."""
    lines = [
        f"checkpoint: {meta.get('checkpoint', '-')}  (step {meta.get('checkpoint_step', '-')})",
        f"env: {meta['env_summary']}  policy: {meta['policy_mode']}  "
        f"episodes/opponent: {meta['episodes']}  intervals: {int(round(100 * meta['confidence']))}% "
        "(Wilson for rates)",
        "",
        f"{'opponent':<18}{'side':<6}{'n':>5} {'win%':>6} {'win CI':>11} {'draw%':>6} {'loss%':>6} "
        f"{'len':>5} {'pick%':>6} {'1st':>5} {'conv%':>6} {'tag+':>5} {'tag-':>5} {'stay%':>6} {'scout%':>7}",
    ]
    for label, block in results.items():
        for side in ["all", *[k for k in block if k != "all"]]:
            s = block[side]
            scout = "    n/a" if s["scout_coverage"] is None else "{:7.1f}".format(100 * s["scout_coverage"]["mean"])
            lines.append(
                f"{label if side == 'all' else '':<18}{side:<6}{s['n']:>5} {_pct(s['win'])} {_ci(s['win']):>11} "
                f"{_pct(s['draw'])} {_pct(s['loss'])} {_num(s['episode_length'])} {_pct(s['episodes_with_pickup'])} "
                f"{_num(s['time_to_first_pickup'])} {_pct(s['pickup_to_capture'])} {_num(s['tags_inflicted'], '{:5.2f}')} "
                f"{_num(s['tags_suffered'], '{:5.2f}')} {100 * s['stay_fraction']['all']:6.1f} "
                f"{scout}"
            )
    lines += ["", "route: top share of centre-column entries by route_roles (all sides) | defender distance base / flag:"]
    for label, block in results.items():
        s = block["all"]
        d = s["defender_distance"]
        lanes = s["route"]
        lines.append(
            f"  {label:<18} top {_pct_text(lanes['top_share'])} of {lanes['top'] + lanes['middle'] + lanes['bottom']:>4}"
            f" entries | while carrying: top {_pct_text(s['route_carrying']['top_share'])}"
            f" | defender {_num(d['to_base'])} / {_num(d['to_flag'])}"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------------------------ CLI


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Noise-free evaluation of a trained CTF policy.")
    p.add_argument("--checkpoint", required=True, help="checkpoint .pt file")
    p.add_argument("--eval-config", default="configs/rl/eval.yaml", help="eval YAML (opponents, episodes, ...)")
    p.add_argument("--env-config", default=None, help="env YAML to play on (default: the one stored in the checkpoint)")
    p.add_argument("--env-set", action="append", default=[], dest="env_overrides", metavar="KEY=VALUE",
                   help="override a CTFConfig key on top of the eval env_overrides, e.g. visibility_radius=6")
    p.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE",
                   help="override an eval-config value by dotted path, e.g. episodes=40")
    p.add_argument("--opponents", nargs="+", default=None, metavar="LABEL", help="only these opponent labels")
    p.add_argument("--out", default=None, help="JSON output (default: outputs/eval/<run>_<checkpoint>.json)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    import torch
    import yaml

    from rl.rl_strategy import RLStrategy

    try:
        overrides = dict(parse_override(t) for t in args.overrides)
        cfg = load_eval_config(args.eval_config, overrides)
        env_over = dict(parse_override(t) for t in args.env_overrides)
        if env_over:
            cfg = replace(cfg, env_overrides={**cfg.env_overrides, **env_over})
        env_config = None
        if args.env_config:
            with open(args.env_config, encoding="utf-8") as f:
                env_config = config_from_dict(yaml.safe_load(f))
        mode = cfg.policy_mode
        strategy = RLStrategy.from_checkpoint(
            args.checkpoint, env_config, deterministic=mode is PolicyMode.ARGMAX, seed=cfg.seed
        )
        torch.set_num_threads(strategy.config.run.torch_threads)
        base_env = strategy.env_config
        # Build every env once now so a bad override fails before any games are played.
        eval_env_config(base_env, cfg.env_overrides, fixed_spawns=False)
        t0 = time.time()
        results = evaluate_policy(
            strategy, list(strategy.roles), base_env, cfg, only=args.opponents,
            progress=lambda msg: print(msg, flush=True),
        )
    except (ConfigError, ValueError, FileNotFoundError, KeyError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2

    shown_env = eval_env_config(base_env, cfg.env_overrides, fixed_spawns=False)
    meta = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": strategy.checkpoint_step,
        "env_summary": f"{shown_env.width}x{shown_env.height}, {shown_env.agents_per_team} agents/team, "
        f"radius {shown_env.visibility_radius}, spawns {shown_env.spawn_mode.value}",
        "policy_mode": mode.value,
        "episodes": cfg.episodes,
        "confidence": cfg.confidence,
        "seconds": round(time.time() - t0, 1),
        "roles": [r.value for r in strategy.roles],
        "eval_config": {**cfg.__dict__, "sides": [s.value for s in cfg.sides], "policy_mode": mode.value,
                        "route_roles": [r.value for r in cfg.route_roles],
                        "opponents": [o.__dict__ for o in cfg.opponents]},
        "env_config": config_to_dict(shown_env),
        "rl_config_name": strategy.config.run.name,
    }
    print()
    print(format_report(results, meta))
    ckpt = Path(args.checkpoint)
    out = Path(args.out) if args.out else Path("outputs/eval") / f"{ckpt.parent.parent.name}_{ckpt.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
