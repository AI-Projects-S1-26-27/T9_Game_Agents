"""The LLM commander as an ordinary team (opponent kind ``llm``), and the classical baseline with v2 roles.

Both exist so that the commander can play in the viewer and in ``rl.evaluate`` like every other approach: any team
is built from ``{kind, params}`` through ``rl.opponents.build_opponent``; this module is the ``llm`` entry.

``LLMTeamParams`` (strict YAML, every key optional):

    condition       any condition of ``conditions.py`` ("llm", "llm_classical", "llm_rl", the ablations, "random_commander"),
                    with an optional "@K" trigger period ("llm@3")
    provider        "mock" (offline, no key; a few fixed rules, a plumbing test, NOT a result), a provider name
                    (openai, anthropic, gemini, groq, ollama, custom, ...) or "auto" (see docs/llm_commander.md)
    model           model name for a provider ("" = the provider's default)
    rpm             requests per minute the backend may use
    cache           file that stores every answer so a game can be replayed without the service ("" = none)
    every_k_turns   the commander is asked at least every K turns (the "@K" of the condition wins)
    rl_checkpoint   .pt file for "llm_rl" ("" = none)
    log_path        JSON-lines log of the commander's decisions ("" = none)

The viewer calls ``act`` once per step and ``reset`` once per game, which ``LLMCommanderStrategy`` supports.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from ctf.config import CTFConfig
from rl.schema import ConfigError
from strategies import BaseStrategy, Role
from strategies.llm_commander.commands import assign_roles, rules_from_config
from strategies.llm_commander.conditions import needs_rl, parse_condition


@dataclass(frozen=True)
class LLMTeamParams:
    condition: str = "llm"
    provider: str = "mock"
    model: str = ""
    rpm: float = 10.0
    cache: str = ""
    every_k_turns: int = 5
    rl_checkpoint: str = ""
    log_path: str = ""

    def __post_init__(self) -> None:
        try:
            parse_condition(self.condition)
        except ValueError as exc:
            raise ConfigError(f"llm params.condition: {exc}") from exc
        if self.every_k_turns < 1:
            raise ConfigError(f"llm params.every_k_turns must be >= 1, got {self.every_k_turns}")
        if self.rpm <= 0:
            raise ConfigError(f"llm params.rpm must be > 0, got {self.rpm}")
        if needs_rl(self.condition) and not self.rl_checkpoint:
            raise ConfigError(f"llm params: condition {self.condition!r} needs rl_checkpoint")


def slot_roles(env_config: CTFConfig | None) -> list[str]:
    """Role of each team slot (slot 1 first) under the rules of ``env_config``: attackers in the carrier slots, then one
    defender, the rest scouts (4 agents: attacker, attacker, defender, scout). Original rules: attacker, defender, scout."""
    n = env_config.agents_per_team if env_config is not None else 3
    roles = assign_roles([f"R{i}" for i in range(1, n + 1)], rules_from_config(env_config))
    return [roles[f"R{i}"] for i in range(1, n + 1)]


def build_llm_team(env_config: CTFConfig, params: LLMTeamParams, rng: Random) -> BaseStrategy:
    from strategies.llm_commander.conditions import build_condition
    from strategies.llm_commander.llm import make_llm

    try:
        llm = make_llm(params.provider, model=params.model or None, rpm=params.rpm, cache=params.cache or None)
    except RuntimeError as exc:  # no key, unknown provider, ...: surface it as a config problem (the viewer shows it)
        raise ConfigError(f"llm params.provider {params.provider!r}: {exc}") from exc
    return build_condition(
        params.condition, llm=llm, max_steps=env_config.max_episode_steps, seed=rng.randrange(2**31),
        default_k=params.every_k_turns, log_path=params.log_path or None,
        rl_checkpoint=params.rl_checkpoint or None, config=env_config,
    )


def make_classical(config: CTFConfig | None) -> BaseStrategy:
    """Team 1's classical AI as a baseline.

    With carrier slots (the v2 rules) the roles are the fixed table of the game (``slot_roles``) and the team gets a real
    ``reset()``. The coordinator's own default assigns roles by position among the ACTIVE agents (first = attacker, second
    = defender, the rest scouts), which is wrong for 4 agents and shifts when an agent is tagged. Without a config or with
    the original rules it is the plain ``ClassicalStrategy`` exactly as before, so results on those rules stay comparable."""
    if config is None or rules_from_config(config).carrier_slots is None:
        from classical_ai.classical_strategy import ClassicalStrategy

        return ClassicalStrategy()
    from rl.classical_opponent import ClassicalOpponent, ClassicalParams

    return ClassicalOpponent(config, ClassicalParams(roles=[Role(r) for r in slot_roles(config)]))
