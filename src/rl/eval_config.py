"""Evaluation configuration (``configs/rl/eval*.yaml``), strict like every other config.

Every key is required. There is deliberately NO opponent-noise key: evaluation always plays the
opponents exactly as scripted (training's ``opponent_noise`` replaces a share of their moves with
random ones, which is useful for learning but would blur a measurement).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ctf.agent import Team
from rl.schema import ConfigError, from_dict, read_yaml
from strategies import Role


class Side(str, Enum):
    RED = "red"
    BLUE = "blue"

    @property
    def team(self) -> Team:
        return Team.RED if self is Side.RED else Team.BLUE


class PolicyMode(str, Enum):
    ARGMAX = "argmax"  # every agent plays its most likely action (what the demo uses)
    SAMPLED = "sampled"  # every agent samples from its action distribution (what training used)


@dataclass(frozen=True)
class EvalOpponent:
    kind: str  # random | stay | scripted | minimax (see rl.opponents) or a registered external kind
    label: str  # name in reports and --opponents filters; unique, [a-z0-9_]+
    params: dict[str, Any]


@dataclass(frozen=True)
class HeldOutMap:
    label: str  # name in reports; unique, [a-z0-9_]+
    path: str  # map YAML (configs/maps/*.yaml); only its geometry is used, and it must match the policy's grid size


@dataclass(frozen=True)
class HeldOutConfig:
    """Play on maps the policy was never trained on (same opponents, same rules, other walls / bases)."""

    episodes: int  # per (map, opponent), split evenly over `sides`
    opponents: list[str]  # labels of entries in the top-level `opponents` list (minimax is slow: leave it out)
    maps: list[HeldOutMap]  # empty = no held-out evaluation


@dataclass(frozen=True)
class RoleSwapConfig:
    """Does the policy actually use its role input? Re-ask it with swapped roles and count changed argmax actions."""

    enabled: bool
    episodes: int  # games played (per opponent) to collect decisions, split evenly over `sides`
    opponents: list[str]  # labels from the top-level `opponents` list
    swaps: list[list[Role]]  # pairs [A, B]: every agent playing A is shown B and vice versa
    min_change_rate: float  # a pair passes when at least this share of decisions changes


@dataclass(frozen=True)
class PolicyCheckConfig:
    """Argmax (demo) vs sampled (training) behaviour of the same policy on the same seeded games."""

    enabled: bool
    episodes: int  # per opponent and mode, split evenly over `sides`
    opponents: list[str]  # labels from the top-level `opponents` list


@dataclass(frozen=True)
class EvalConfig:
    episodes: int  # per opponent, split evenly over `sides`
    seed: int  # every episode's spawns and opponent randomness derive from it (same seed = same games)
    confidence: float  # level of every confidence interval (Wilson for rates, normal for means)
    sides: list[Side]  # sides the policy plays; a policy trained mirrored should be checked on both
    policy_mode: PolicyMode
    fixed_spawn_fraction: float  # share of episodes that use the env's fixed spawns (0 = all randomized)
    env_overrides: dict[str, Any]  # CTFConfig keys applied on top of the env config (e.g. spawn_mode)
    route_roles: list[Role]  # roles whose entries into the centre column count for route choice
    opponents: list[EvalOpponent]
    held_out: HeldOutConfig
    role_swap: RoleSwapConfig
    policy_check: PolicyCheckConfig

    def __post_init__(self) -> None:
        _validate(self)

    @property
    def episodes_per_side(self) -> int:
        return self.episodes // len(self.sides)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _validate(cfg: EvalConfig) -> None:
    _check(cfg.episodes >= 1, "eval episodes must be >= 1")
    _check(len(cfg.sides) >= 1, "eval sides must not be empty")
    _check(len(set(cfg.sides)) == len(cfg.sides), f"eval sides must be unique, got {[s.value for s in cfg.sides]}")
    _check(
        cfg.episodes % len(cfg.sides) == 0,
        f"eval episodes ({cfg.episodes}) must be divisible by the number of sides ({len(cfg.sides)})",
    )
    _check(0.0 < cfg.confidence < 1.0, "eval confidence must be strictly between 0 and 1")
    _check(0.0 <= cfg.fixed_spawn_fraction <= 1.0, "eval fixed_spawn_fraction must be in [0, 1]")
    _check(len(cfg.opponents) >= 1, "eval opponents must not be empty")
    labels = [o.label for o in cfg.opponents]
    _check(all(re.fullmatch(r"[a-z0-9_]+", lab) for lab in labels), f"eval opponent labels must match [a-z0-9_]+, got {labels}")
    _check(len(set(labels)) == len(labels), f"eval opponent labels must be unique, got {labels}")

    def known(section: str, names: list[str]) -> None:
        unknown = sorted(set(names) - set(labels))
        _check(not unknown, f"eval {section}.opponents names unknown opponent label(s) {unknown}; configured: {labels}")

    def split(section: str, episodes: int) -> None:
        _check(episodes >= 1, f"eval {section}.episodes must be >= 1")
        _check(
            episodes % len(cfg.sides) == 0,
            f"eval {section}.episodes ({episodes}) must be divisible by the number of sides ({len(cfg.sides)})",
        )

    ho = cfg.held_out
    split("held_out", ho.episodes)
    known("held_out", ho.opponents)
    map_labels = [m.label for m in ho.maps]
    _check(all(re.fullmatch(r"[a-z0-9_]+", lab) for lab in map_labels), f"eval held-out map labels must match [a-z0-9_]+, got {map_labels}")
    _check(len(set(map_labels)) == len(map_labels), f"eval held-out map labels must be unique, got {map_labels}")
    _check(not ho.maps or len(ho.opponents) >= 1, "eval held_out.opponents must not be empty when held_out.maps is set")

    rs = cfg.role_swap
    split("role_swap", rs.episodes)
    known("role_swap", rs.opponents)
    _check(0.0 <= rs.min_change_rate <= 1.0, "eval role_swap.min_change_rate must be in [0, 1]")
    for pair in rs.swaps:
        _check(len(pair) == 2 and pair[0] != pair[1], f"eval role_swap.swaps entries must be two different roles, got {[r.value for r in pair]}")
    if rs.enabled:
        _check(len(rs.opponents) >= 1 and len(rs.swaps) >= 1, "eval role_swap needs at least one opponent and one swap when enabled")

    pc = cfg.policy_check
    split("policy_check", pc.episodes)
    known("policy_check", pc.opponents)
    if pc.enabled:
        _check(len(pc.opponents) >= 1, "eval policy_check.opponents must not be empty when enabled")


def load_eval_config(path: str | Path, overrides: dict[str, Any] | None = None) -> EvalConfig:
    from rl.schema import apply_overrides

    data = read_yaml(path)
    if overrides:
        data = apply_overrides(data, overrides)
    return from_dict(EvalConfig, data)
