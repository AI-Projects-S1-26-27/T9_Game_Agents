"""Controllers (RL checkpoint, scripted team, random, stay) the viewer can put on either side."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

from ctf.config import CTFConfig
from rl.config import OpponentSpec
from rl.opponents import KNOWN_KINDS, ScriptedParams, build_opponent
from rl.schema import ConfigError, from_dict, read_yaml
from strategies import BaseStrategy

KINDS = ("rl", *KNOWN_KINDS)
ROLE_NAMES = ("attacker", "defender", "scout")


class ControllerError(RuntimeError):
    """A controller could not be built (missing checkpoint, wrong map size, ...)."""


@dataclass(frozen=True)
class ControllerSpec:
    label: str
    kind: str
    checkpoint: str  # path to a .pt file (kind=rl); "" otherwise
    deterministic: bool  # rl: argmax instead of sampling
    params: dict[str, Any]
    # Role label shown on each team slot for kinds that do not use roles themselves (random, stay).
    # Display only: it changes how the players are labelled, never how they move. Scripted teams take
    # their roles from params and RL teams from the checkpoint. Empty = show "unknown".
    roles: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        bad = [r for r in self.roles if r not in ROLE_NAMES]
        if bad:
            raise ConfigError(f"controller {self.label!r}: unknown role(s) {bad}; valid roles: {list(ROLE_NAMES)}")
        if self.kind not in KINDS:
            raise ConfigError(f"controller {self.label!r}: kind must be one of {KINDS}, got {self.kind!r}")
        if self.kind == "rl" and not self.checkpoint:
            raise ConfigError(f"controller {self.label!r}: kind rl needs a checkpoint path")


@dataclass(frozen=True)
class ControllersFile:
    controllers: list[ControllerSpec]

    def __post_init__(self) -> None:
        if not self.controllers:
            raise ConfigError("controllers: list must not be empty")


@dataclass
class Controller:
    label: str
    strategy: BaseStrategy
    roles: list[str]  # role name of each team slot ("unknown" when the controller has none)
    rl: Any  # RLStrategy | None (typed Any so torch is only needed when an RL controller is used)
    spec: ControllerSpec


def load_controller_specs(path: str | Path) -> list[ControllerSpec]:
    return from_dict(ControllersFile, read_yaml(path)).controllers


class ControllerFactory:
    """Builds controllers; keeps loaded checkpoints so switching sides or episodes is instant."""

    def __init__(self) -> None:
        self._loaded: dict[tuple[str, int], tuple] = {}

    def build(self, spec: ControllerSpec, env_config: CTFConfig, seed: int) -> Controller:
        n = env_config.agents_per_team
        if spec.kind == "rl":
            return self._build_rl(spec, env_config, seed)
        rng = Random(seed)
        strategy = build_opponent(OpponentSpec(spec.kind, spec.kind, 1.0, spec.params), env_config, rng)
        if spec.kind == "scripted":
            params = from_dict(ScriptedParams, spec.params)
            roles = [r.value for r in params.roles]
        elif spec.roles:
            if len(spec.roles) != n:
                raise ControllerError(
                    f"controller {spec.label!r}: roles lists {len(spec.roles)} slots but the map has {n} agents per team"
                )
            roles = list(spec.roles)
        else:
            roles = ["unknown"] * n
        return Controller(spec.label, strategy, roles, None, spec)

    def _build_rl(self, spec: ControllerSpec, env_config: CTFConfig, seed: int) -> Controller:
        try:
            from rl.checkpoint import load_policy
            from rl.mappo import ValueNorm
            from rl.rl_strategy import RLStrategy
        except ImportError as exc:  # torch not installed
            raise ControllerError("RL controllers need PyTorch (install the rl extra)") from exc
        path = Path(spec.checkpoint)
        if not path.exists():
            raise ControllerError(f"checkpoint not found: {path}")
        key = (str(path.resolve()), path.stat().st_mtime_ns)
        if key not in self._loaded:
            self._loaded = {k: v for k, v in self._loaded.items() if k[0] != key[0]}
            self._loaded[key] = load_policy(path, "cpu")
        model, meta = self._loaded[key]
        value_norm = ValueNorm(meta["config"].ppo.value_normalization)
        value_norm.load_state_dict(meta["value_norm"])
        try:
            strategy = RLStrategy(
                model, meta["config"], env_config, value_norm,
                deterministic=spec.deterministic, checkpoint_step=int(meta["counters"]["total_steps"]), seed=seed,
                layout=meta["layout"],
            )
        except ValueError as exc:
            raise ControllerError(str(exc)) from exc
        step = strategy.checkpoint_step
        label = f"{spec.label} @ {step / 1e6:.2f}M steps" if step else spec.label
        return Controller(label, strategy, [r.value for r in meta["config"].roles.slots], strategy, spec)
