"""Opponent teams for experiments, built from the other branches' code.

Names: aggressive | balanced | defensive | camper | chaser | escort | random | stay   (RL branch scripted teams)
       classical                                                  (Team 1's classical AI)
       rl                                                         (Team 2's trained RL policy, needs a checkpoint)
       baseline                                                   (this package's role-default team)
"""
from __future__ import annotations

from random import Random

from strategies.base import BaseStrategy

def _roles(kind: str, n: int, can_carry=lambda slot: True) -> list[str]:
    """Role of each team slot for n agents per team (3 agents: A,D,S; 4 agents: A,A,D,S).
    Only slots that may carry the flag (can_carry) are given the attacker role when everyone attacks."""
    if kind == "aggressive":
        return ["attacker" if can_carry(slot) else "scout" for slot in range(1, n + 1)]
    if kind == "defensive":
        return ["attacker"] + ["defender"] * min(2, n - 1) + ["scout"] * max(0, n - 3)
    return ["attacker"] * max(1, n - 2) + ["defender", "scout"][: max(0, n - 1)]       # balanced, camper, chaser, escort


# name -> (guard_mode, guard_leash, scout_behavior, chase_radius, escort)
PRESETS = {
    "aggressive": ("leash", 2, "random", 0, False),
    "balanced":   ("leash", 2, "random", 0, False),
    "defensive":  ("leash", 2, "stay", 0, False),
    "camper":     ("on_flag", 0, "stay", 0, False),
    "chaser":     ("chaser", 0, "random", 4, False),
    "escort":     ("leash", 2, "random", 0, True),
}
SIMPLE_KINDS = ("random", "stay")
ALL_OPPONENTS = tuple(PRESETS) + SIMPLE_KINDS + ("classical", "rl", "baseline")


def make_opponent(name: str, env_config, seed: int = 0, rl_checkpoint: str | None = None) -> BaseStrategy:
    if name not in ALL_OPPONENTS:
        raise ValueError(f"unknown opponent '{name}'; choose from {', '.join(ALL_OPPONENTS)}")
    if name == "classical":
        from classical_ai.classical_strategy import ClassicalStrategy

        return ClassicalStrategy()
    if name == "rl":
        from .rl_adapter import make_rl_strategy

        return make_rl_strategy(rl_checkpoint, env_config)
    if name == "baseline":
        from .baseline import ScriptedStrategy

        return ScriptedStrategy("baseline")
    from rl.opponents import GuardMode, RandomStrategy, ScoutBehavior, ScriptedParams, ScriptedTeamStrategy
    from strategies import Role, StayStrategy

    rng = Random(seed)
    if name == "random":
        return RandomStrategy(rng)
    if name == "stay":
        return StayStrategy()
    guard, leash, scout, chase, escort = PRESETS[name]
    params = ScriptedParams(roles=[Role(r) for r in _roles(name, env_config.agents_per_team, env_config.can_carry)],
                            guard_mode=GuardMode(guard), guard_leash=leash, scout_behavior=ScoutBehavior(scout),
                            chase_radius=chase, escort=escort)
    return ScriptedTeamStrategy(env_config, params, rng)
