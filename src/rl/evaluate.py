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
its own flag. Three further checks are configured in the same YAML: held-out maps (other 11x7
layouts from ``configs/maps``), a role-swap sensitivity check (does the argmax action change when
the role inputs are swapped?) and an argmax-vs-sampled check. Everything tunable lives in the eval
YAML (``configs/rl/eval.yaml``); see docs/evaluation.md.
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
from rl.eval_stats import mean_stat, rate, rate_difference
from rl.features import ROLE_ORDER
from rl.opponents import KNOWN_KINDS, build_opponent
from rl.schema import ConfigError, parse_override, read_yaml, to_dict
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


# Keys a held-out map file contributes. Everything else (rules, visibility, spawn mode, episode
# length, carrier slots, ...) stays what the policy's own environment config says, so a held-out
# result differs from the main result in the walls and base positions only.
MAP_KEYS = ("obstacles", "red_base", "blue_base", "red_spawns", "blue_spawns")


def held_out_env_config(base: CTFConfig, map_path: str | Path) -> CTFConfig:
    """``base`` with the geometry (walls, bases, spawn cells) of the map file at ``map_path``."""
    data = read_yaml(map_path)
    for key in ("width", "height", "agents_per_team"):
        if data.get(key) != getattr(base, key):
            raise ConfigError(
                f"held-out map {map_path}: {key} is {data.get(key)!r} but the policy's environment has "
                f"{getattr(base, key)!r}; only layouts of the same size and team size can be used"
            )
    missing = [k for k in MAP_KEYS if k not in data]
    if missing:
        raise ConfigError(f"held-out map {map_path}: missing key(s) {missing}")
    try:
        return config_from_dict({**config_to_dict(base), **{k: data[k] for k in MAP_KEYS}})
    except ValueError as exc:
        raise ConfigError(f"held-out map {map_path} is invalid: {exc}") from exc


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


def _play_side(
    learner: BaseStrategy,
    roles: list[Role],
    envs: dict[bool, CTFEnvironment],
    cfg: EvalConfig,
    opp: EvalOpponent,
    side: Side,
    episodes: int,
    key_prefix: str,
    symmetric: bool,
) -> list[EpisodeRecord]:
    """``episodes`` seeded games of ``learner`` (on ``side``) against ``opp``. Same key = same games."""
    route_roles = set(cfg.route_roles)
    records: list[EpisodeRecord] = []
    for i in range(episodes):
        draw = Random(f"{cfg.seed}:{key_prefix}{opp.label}:{side.value}:{i}")
        fixed = draw.random() < cfg.fixed_spawn_fraction
        env_seed, opp_seed = draw.randrange(2**31), draw.randrange(2**31)
        opponent = build_eval_opponent(opp, envs[fixed].config, Random(opp_seed))
        records.append(
            play_episode(
                envs[fixed], learner, roles, opponent, side.team, route_roles, env_seed,
                mirrored=symmetric and side is Side.BLUE,
            )
        )
    return records


def _eval_envs(env_config: CTFConfig, cfg: EvalConfig) -> dict[bool, CTFEnvironment]:
    return {
        fixed: CTFEnvironment(eval_env_config(env_config, cfg.env_overrides, fixed_spawns=fixed))
        for fixed in (True, False)
    }


def _select_opponents(cfg: EvalConfig, labels: list[str]) -> list[EvalOpponent]:
    unknown = sorted(set(labels) - {o.label for o in cfg.opponents})
    if unknown:
        raise ConfigError(f"unknown opponent label(s) {unknown}; configured: {[o.label for o in cfg.opponents]}")
    return [o for o in cfg.opponents if o.label in labels]


def _run_matchups(
    learner: BaseStrategy,
    roles: list[Role],
    env_config: CTFConfig,
    cfg: EvalConfig,
    opponents: list[EvalOpponent],
    episodes_per_side: int,
    key_prefix: str,
    progress: Callable[[str], None] | None,
    tag: str = "",
) -> dict[str, Any]:
    envs = _eval_envs(env_config, cfg)
    role_names = list(dict.fromkeys(r.value for r in roles))
    symmetric = is_mirror_symmetric(env_config)
    results: dict[str, Any] = {}
    for opp in opponents:
        per_side: dict[str, list[EpisodeRecord]] = {}
        for side in cfg.sides:
            records = _play_side(learner, roles, envs, cfg, opp, side, episodes_per_side, key_prefix, symmetric)
            per_side[side.value] = records
            if progress:
                wins = sum(r.outcome == "win" for r in records)
                progress(f"  {tag}vs {opp.label:<18} as {side.value:<4} {wins}/{len(records)} wins")
        everything = [r for recs in per_side.values() for r in recs]
        block = {"all": summarize(everything, role_names, cfg.confidence, team_relative=symmetric)}
        for side_name, recs in per_side.items():
            block[side_name] = summarize(recs, role_names, cfg.confidence, team_relative=False)
        results[opp.label] = block
    return results


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
    opponents = _select_opponents(cfg, only) if only else cfg.opponents
    return _run_matchups(learner, roles, env_config, cfg, opponents, cfg.episodes_per_side, "", progress)


# ----------------------------------------------------------------- extra checks (held-out maps etc.)


def evaluate_held_out(
    learner: BaseStrategy,
    roles: list[Role],
    env_config: CTFConfig,
    cfg: EvalConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """The configured opponents on each held-out map: ``{map: {opponent: {side|all: summary}}}``.

    ``None`` when the config lists no held-out maps.
    """
    ho = cfg.held_out
    if not ho.maps:
        return None
    opponents = _select_opponents(cfg, ho.opponents)
    out: dict[str, Any] = {}
    for held in ho.maps:
        map_config = held_out_env_config(env_config, held.path)  # also fails early on a wrong-sized map
        out[held.label] = _run_matchups(
            learner, roles, map_config, cfg, opponents, ho.episodes // len(cfg.sides),
            f"held_out:{held.label}:", progress, tag=f"[{held.label}] ",
        )
    return out


class _RoleSwapProbe(BaseStrategy):
    """Plays like ``inner`` while counting, per role pair, how often swapping the roles changes the argmax action."""

    def __init__(self, inner: Any, roles: list[Role], pairs: list[tuple[Role, Role]]) -> None:
        self.inner, self.roles, self.pairs = inner, roles, pairs
        self.counts = {pair: [0, 0] for pair in pairs}  # pair -> [changed decisions, decisions]

    def reset(self) -> None:
        self.inner.reset()

    def act(self, observation, team):
        actions = self.inner.act(observation, team)
        decision = self.inner.last_decision[team]
        actual = self.inner.greedy_actions(team)
        for a, b in self.pairs:
            swapped = np.array(
                [ROLE_ORDER.index(b if r is a else a if r is b else r) for r in self.roles], dtype=np.int64
            )
            alt = self.inner.greedy_actions(team, swapped)
            for slot, role in enumerate(self.roles):
                if decision.alive[slot] and role in (a, b):
                    self.counts[(a, b)][1] += 1
                    self.counts[(a, b)][0] += int(alt[slot] != actual[slot])
        return actions


def role_swap_check(
    learner: Any,
    roles: list[Role],
    env_config: CTFConfig,
    cfg: EvalConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Does the policy use its role input? Swap roles in its inputs and count changed argmax actions.

    The games are played by the policy itself (in the configured policy mode) against the
    ``role_swap.opponents``; at every decision each configured pair of roles (A, B) is swapped for
    the whole team (agents playing A are shown B, agents playing B are shown A; both the role
    one-hot and the role-specific output head switch) and the argmax action is compared with the
    unswapped argmax. A policy that ignores roles scores 0. Only the role inputs change: what an
    agent sees and its ``self_can_carry`` flag stay as they were.
    """
    rs = cfg.role_swap
    if not hasattr(learner, "greedy_actions"):
        raise ConfigError("the role-swap check needs a neural policy (RLStrategy)")
    pairs = [(a, b) for a, b in rs.swaps]
    absent = sorted({r.value for pair in pairs for r in pair} - {r.value for r in roles})
    if absent:
        raise ConfigError(f"role_swap.swaps names role(s) {absent} that this policy's team does not have")
    probe = _RoleSwapProbe(learner, roles, pairs)
    envs = _eval_envs(env_config, cfg)
    symmetric = is_mirror_symmetric(env_config)
    for opp in _select_opponents(cfg, rs.opponents):
        for side in cfg.sides:
            _play_side(probe, roles, envs, cfg, opp, side, rs.episodes // len(cfg.sides), "role_swap:", symmetric)
            if progress:
                progress(f"  role-swap games vs {opp.label:<18} as {side.value}")
    per_pair: dict[str, Any] = {}
    for (a, b), (changed, total) in probe.counts.items():
        stat = rate(changed, total, cfg.confidence)
        per_pair[f"{a.value}<->{b.value}"] = {
            **stat, "passed": stat["rate"] is not None and stat["rate"] >= rs.min_change_rate,
        }
    all_changed = sum(c for c, _ in probe.counts.values())
    all_total = sum(t for _, t in probe.counts.values())
    return {
        "episodes": rs.episodes,
        "opponents": list(rs.opponents),
        "min_change_rate": rs.min_change_rate,
        "pairs": per_pair,
        "all": rate(all_changed, all_total, cfg.confidence),
        "passed": bool(per_pair) and all(p["passed"] for p in per_pair.values()),
    }


class _TopProbability(BaseStrategy):
    """Wraps a policy and averages the probability of its most likely action over every decision."""

    def __init__(self, inner: Any) -> None:
        self.inner, self.total, self.count = inner, 0.0, 0

    def reset(self) -> None:
        self.inner.reset()

    def act(self, observation, team):
        actions = self.inner.act(observation, team)
        decision = self.inner.last_decision[team]
        top = decision.probs.max(axis=1)
        self.total += float(top[decision.alive].sum())
        self.count += int(decision.alive.sum())
        return actions

    @property
    def mean(self) -> float | None:
        return self.total / self.count if self.count else None


def policy_mode_check(
    learner: Any,
    roles: list[Role],
    env_config: CTFConfig,
    cfg: EvalConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """The same policy on the same seeded games, once with argmax actions and once sampled.

    Training sampled its actions, the demo plays argmax; a large gap means the demo behaviour is
    not what was trained (or the policy is still very random). ``top_action_prob`` is the mean
    probability of the most likely action (1.0 = fully deterministic policy). The policy's own
    mode is restored afterwards.
    """
    pc = cfg.policy_check
    if not hasattr(learner, "set_policy_mode"):
        raise ConfigError("the policy check needs a neural policy (RLStrategy)")
    original = bool(learner.deterministic)
    envs = _eval_envs(env_config, cfg)
    symmetric = is_mirror_symmetric(env_config)
    out: dict[str, Any] = {}
    try:
        for opp in _select_opponents(cfg, pc.opponents):
            modes: dict[str, Any] = {}
            counts: dict[str, tuple[int, int]] = {}
            for mode in (PolicyMode.ARGMAX, PolicyMode.SAMPLED):
                learner.set_policy_mode(mode is PolicyMode.ARGMAX, seed=cfg.seed)
                tracker = _TopProbability(learner)
                records: list[EpisodeRecord] = []
                for side in cfg.sides:
                    records += _play_side(
                        tracker, roles, envs, cfg, opp, side, pc.episodes // len(cfg.sides), "policy_check:", symmetric
                    )
                n = len(records)
                wins = sum(r.outcome == "win" for r in records)
                counts[mode.value] = (wins, n)
                modes[mode.value] = {
                    "win": rate(wins, n, cfg.confidence),
                    "draw": rate(sum(r.outcome == "draw" for r in records), n, cfg.confidence),
                    "loss": rate(sum(r.outcome == "loss" for r in records), n, cfg.confidence),
                    "top_action_prob": tracker.mean,
                }
                if progress:
                    progress(f"  policy check vs {opp.label:<18} {mode.value:<8} {wins}/{n} wins")
            out[opp.label] = {
                **modes,
                "win_difference": rate_difference(*counts["argmax"], *counts["sampled"], cfg.confidence),
            }
    finally:
        learner.set_policy_mode(original)
    return out


EXTRA_CHECKS = ("held_out", "role_swap", "policy_check")


def evaluate_extras(
    learner: Any,
    roles: list[Role],
    env_config: CTFConfig,
    cfg: EvalConfig,
    *,
    skip: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Held-out maps, role-swap check and policy-mode check; a disabled or skipped one is ``None``."""
    skip = skip or []
    unknown = sorted(set(skip) - set(EXTRA_CHECKS))
    if unknown:
        raise ConfigError(f"unknown check(s) {unknown} to skip; valid: {list(EXTRA_CHECKS)}")
    if len(roles) != env_config.agents_per_team:
        raise ConfigError(f"policy has {len(roles)} roles but the env has {env_config.agents_per_team} agents per team")
    return {
        "held_out": None if "held_out" in skip else evaluate_held_out(learner, roles, env_config, cfg, progress=progress),
        "role_swap": (
            role_swap_check(learner, roles, env_config, cfg, progress=progress)
            if cfg.role_swap.enabled and "role_swap" not in skip else None
        ),
        "policy_check": (
            policy_mode_check(learner, roles, env_config, cfg, progress=progress)
            if cfg.policy_check.enabled and "policy_check" not in skip else None
        ),
    }


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


def format_extras(extras: dict[str, Any], meta: dict[str, Any]) -> str:
    """Text for the held-out maps, role-swap check and policy-mode check (only those that were run)."""
    lines: list[str] = []
    held = extras.get("held_out")
    if held:
        lines += ["", f"held-out maps (maps the policy was not trained on; sides pooled; intervals {int(round(100 * meta['confidence']))}% Wilson)",
                  f"  {'map':<18}{'opponent':<18}{'n':>5} {'win%':>6} {'win CI':>11} {'draw%':>6} {'loss%':>6}"]
        for map_label, per_opp in held.items():
            wins = []
            for opp_label, block in per_opp.items():
                s = block["all"]
                wins.append(s["win"]["rate"])
                lines.append(
                    f"  {map_label:<18}{opp_label:<18}{s['n']:>5} {_pct(s['win'])} {_ci(s['win']):>11} "
                    f"{_pct(s['draw'])} {_pct(s['loss'])}"
                )
            if wins:
                lines.append(f"  {map_label:<18}{'(mean over opponents)':<18}{'':>5} {100 * sum(wins) / len(wins):6.1f}")
    swap = extras.get("role_swap")
    if swap:
        lines += ["", f"role-swap check: share of decisions whose argmax action changes when roles are swapped "
                      f"(a pair passes at >= {100 * swap['min_change_rate']:.0f}%; a policy that ignores roles scores 0)"]
        for pair, stat in swap["pairs"].items():
            lines.append(
                f"  {pair:<22} {_pct(stat)}% {_ci(stat)} of {stat['n']:>6} decisions  {'PASS' if stat['passed'] else 'FAIL'}"
            )
        lines.append(f"  overall {'PASS' if swap['passed'] else 'FAIL'}: {_pct(swap['all'])}% of {swap['all']['n']} decisions changed")
    check = extras.get("policy_check")
    if check:
        lines += ["", "argmax vs sampled actions (same seeded games; difference = argmax win% - sampled win%, approximate interval):",
                  f"  {'opponent':<18}{'argmax win%':>12}{'sampled win%':>13}{'diff':>7} {'diff CI':>14}{'top-prob a/s':>14}"]
        for label, c in check.items():
            d = c["win_difference"]
            diff = "  n/a" if d["diff"] is None else f"{100 * d['diff']:+6.1f}"
            ci = "" if d["lo"] is None else f"[{100 * d['lo']:+5.0f},{100 * d['hi']:+5.0f}]"
            top = f"{c['argmax']['top_action_prob']:.2f}/{c['sampled']['top_action_prob']:.2f}" if c["argmax"]["top_action_prob"] is not None else "n/a"
            lines.append(f"  {label:<18}{_pct(c['argmax']['win']):>12}{_pct(c['sampled']['win']):>13}{diff:>7} {ci:>14}{top:>14}")
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
    p.add_argument("--opponents", nargs="+", default=None, metavar="LABEL", help="only these opponent labels (main table only)")
    p.add_argument("--skip", nargs="+", default=[], choices=list(EXTRA_CHECKS), metavar="CHECK",
                   help=f"do not run these extra checks: {', '.join(EXTRA_CHECKS)}")
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
        say = lambda msg: print(msg, flush=True)  # noqa: E731
        results = evaluate_policy(strategy, list(strategy.roles), base_env, cfg, only=args.opponents, progress=say)
        extras = evaluate_extras(strategy, list(strategy.roles), base_env, cfg, skip=args.skip, progress=say)
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
        "eval_config": to_dict(cfg),
        "env_config": config_to_dict(shown_env),
        "rl_config_name": strategy.config.run.name,
    }
    print()
    print(format_report(results, meta))
    extra_text = format_extras(extras, meta)
    if extra_text:
        print(extra_text)
    ckpt = Path(args.checkpoint)
    out = Path(args.out) if args.out else Path("outputs/eval") / f"{ckpt.parent.parent.name}_{ckpt.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "results": results, **extras}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
