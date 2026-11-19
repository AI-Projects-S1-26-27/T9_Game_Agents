"""The classical AI (``src/classical_ai``: A* + influence map + behaviour tree) as an opponent team.

``src/classical_ai`` is a verbatim copy of the ``classical_ai`` branch (written by a teammate, not edited
here). This module is only the adapter that lets it play in our evaluation and in the viewer:

* **Explicit roles.** The classical coordinator's default role assignment is positional over the agents
  that are still ACTIVE (first = attacker, second = defender, the rest scouts). That is wrong for the v2
  rules (slots 1-2 attackers, slot 3 defender, slot 4 scout) and it shifts when an agent is tagged. The
  adapter always passes a fixed ``{agent id: role}`` table built from ``params.roles``.
* **A real ``reset()``.** ``ClassicalStrategy`` keeps its coordinator (shared belief map) for as long as the
  object lives, so a reused instance would carry what it learned in one episode into the next. ``reset()``
  drops it; the next ``act`` builds a fresh one.
* **Rule checks.** Only the carrier slots of the environment may hold an attacker (the classical code does
  not know the ``carrier_slots`` rule).

The classical AI itself is deterministic and ignores the ``rng`` the opponent factory passes.
"""

from __future__ import annotations

from dataclasses import dataclass

from classical_ai.classical_strategy import ClassicalStrategy
from ctf.actions import Action
from ctf.agent import Team
from ctf.config import CTFConfig
from ctf.observations import TeamObservation
from rl.classical_fixes import FixedClassicalStrategy, FixSettings
from rl.schema import ConfigError
from strategies import BaseStrategy, Role

_PREFIX = {Team.RED: "R", Team.BLUE: "B"}


@dataclass(frozen=True)
class ClassicalParams:
    """``roles`` is required. Every other key is optional and its default is the ORIGINAL behaviour (all fixes off,
    the old literals), so entries and saved configs written before the fixes existed mean exactly what they meant.
    The ``classical_fixed`` entries in configs/rl/eval.yaml and configs/viz/controllers.yaml spell every key out."""

    roles: list[Role]  # role of each team slot, slot 1 first (same meaning as scripted opponents)
    remember_dropped_flag: bool = False
    defend_dropped_flag: bool = False
    buddy_system: bool = False
    intercept_in_pairs: bool = False
    pair_guard_radius: int = 2
    pair_wait_steps: int = 20
    carrier_danger_radius: int = 2
    influence_radius: int = 3
    enemy_penalty: float = 5.0
    probability_weight: float = 3.0

    def __post_init__(self) -> None:
        for name in ("pair_guard_radius", "pair_wait_steps", "carrier_danger_radius", "influence_radius"):
            if getattr(self, name) < 0:
                raise ConfigError(f"classical params.{name} must be >= 0, got {getattr(self, name)}")
        for name in ("enemy_penalty", "probability_weight"):
            if getattr(self, name) < 0:
                raise ConfigError(f"classical params.{name} must be >= 0, got {getattr(self, name)}")

    def fix_settings(self) -> FixSettings:
        return FixSettings(
            remember_dropped_flag=self.remember_dropped_flag,
            defend_dropped_flag=self.defend_dropped_flag,
            buddy_system=self.buddy_system,
            intercept_in_pairs=self.intercept_in_pairs,
            pair_guard_radius=self.pair_guard_radius,
            pair_wait_steps=self.pair_wait_steps,
            carrier_danger_radius=self.carrier_danger_radius,
            influence_radius=self.influence_radius,
            enemy_penalty=self.enemy_penalty,
            probability_weight=self.probability_weight,
        )

    def is_original(self) -> bool:
        """True when nothing differs from the verbatim classical code (then that code runs untouched)."""
        return self == ClassicalParams(roles=self.roles)


class ClassicalOpponent(BaseStrategy):
    def __init__(self, env_config: CTFConfig, params: ClassicalParams) -> None:
        if len(params.roles) != env_config.agents_per_team:
            raise ConfigError(
                f"classical opponent has {len(params.roles)} roles for {env_config.agents_per_team} agents per team"
            )
        for slot, role in enumerate(params.roles, start=1):
            if role is Role.ATTACKER and not env_config.can_carry(slot):
                raise ConfigError(
                    f"classical opponent has an attacker in slot {slot}, but the environment only lets "
                    f"slots {env_config.carrier_slots} carry the flag"
                )
        self._roles = {
            team: {f"{prefix}{slot}": role.value for slot, role in enumerate(params.roles, start=1)}
            for team, prefix in _PREFIX.items()
        }
        self._params = params
        self._inner = None

    def reset(self) -> None:
        self._inner = None

    def act(self, observation: TeamObservation, team: Team) -> dict[str, Action]:
        if self._inner is None:
            table = {t: dict(r) for t, r in self._roles.items()}
            if self._params.is_original():
                self._inner = ClassicalStrategy(role_assignments=table)
            else:
                self._inner = FixedClassicalStrategy(table, self._params.fix_settings())
        return self._inner.act(observation, team)
