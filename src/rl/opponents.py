"""Opponent teams for training and evaluation.

Kinds (selected by ``OpponentSpec.kind`` in the curriculum config):

``random``    uniformly random actions.
``stay``      never moves (note: an agent standing on its own flag cell makes that
              flag unpickable, because pickup and tag happen in the same step).
``scripted``  a simple rule-based team: attackers run to the enemy flag (wherever it lies) and
              home, defenders hold a post, chase flag carriers and touch a dropped own flag to
              send it home, scouts wander or wait. Options: ``guard_mode: chaser`` leaves the post
              to chase any enemy that comes close; ``escort: true`` makes attackers move as a pair
              so the carrier is never alone (the headcount rule punishes a lone carrier).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from random import Random

from ctf.actions import Action
from ctf.agent import Team
from ctf.config import CTFConfig
from ctf.game_state import FlagStatus
from ctf.observations import TeamObservation
from rl.config import OpponentSpec
from rl.navigation import MOVES, GridNav
from rl.schema import ConfigError, from_dict
from strategies import BaseStrategy, Role, StayStrategy

KNOWN_KINDS = ("random", "stay", "scripted")


class GuardMode(str, Enum):
    ON_FLAG = "on_flag"  # defender parks on its own flag cell (unbeatable while it stays)
    LEASH = "leash"  # defender holds a post `guard_leash` cells in front of the flag
    CHASER = "chaser"  # defender leaves its flag to chase enemies within `chase_radius`, then returns


class ScoutBehavior(str, Enum):
    RANDOM = "random"
    STAY = "stay"


@dataclass(frozen=True)
class ScriptedParams:
    roles: list[Role]  # role of each team slot, in sorted agent-id order
    guard_mode: GuardMode
    guard_leash: int
    scout_behavior: ScoutBehavior
    chase_radius: int  # chaser only: chase enemies within this Manhattan distance (>= 1)
    escort: bool  # attackers move together: the lead waits for the others, then all move as one


def _active_team_agents(observation: TeamObservation, team: Team) -> list:
    return [a for a in observation.agents.values() if a.team is team and a.active]


class RandomStrategy(BaseStrategy):
    def __init__(self, rng: Random) -> None:
        self._rng = rng

    def act(self, observation: TeamObservation, team: Team) -> dict[str, Action]:
        actions = list(Action)
        return {a.agent_id: self._rng.choice(actions) for a in _active_team_agents(observation, team)}


class ScriptedTeamStrategy(BaseStrategy):
    def __init__(self, env_config: CTFConfig, params: ScriptedParams, rng: Random) -> None:
        if len(params.roles) != env_config.agents_per_team:
            raise ConfigError(
                f"scripted opponent has {len(params.roles)} roles for "
                f"{env_config.agents_per_team} agents per team"
            )
        if params.guard_leash < 0:
            raise ConfigError("scripted opponent guard_leash must be >= 0")
        if params.guard_mode is GuardMode.CHASER and params.chase_radius < 1:
            raise ConfigError("scripted opponent with guard_mode chaser needs chase_radius >= 1")
        if params.chase_radius < 0:
            raise ConfigError("scripted opponent chase_radius must be >= 0")
        for slot, role in enumerate(params.roles, start=1):
            if role is Role.ATTACKER and not env_config.can_carry(slot):
                raise ConfigError(
                    f"scripted opponent has an attacker in slot {slot}, but the environment only lets "
                    f"slots {env_config.carrier_slots} carry the flag"
                )
        self._cfg = env_config
        self._params = params
        self._rng = rng
        traversable = [
            [(x, y) not in set(env_config.obstacles) for x in range(env_config.width)]
            for y in range(env_config.height)
        ]
        self._nav = GridNav(traversable)
        self._post = {team: self._leash_post(team) for team in Team}

    def _bases(self, team: Team) -> tuple[tuple[int, int], tuple[int, int]]:
        red, blue = self._cfg.red_base, self._cfg.blue_base
        return (red, blue) if team is Team.RED else (blue, red)

    def _leash_post(self, team: Team) -> tuple[int, int]:
        own, other = self._bases(team)
        if self._params.guard_mode in (GuardMode.ON_FLAG, GuardMode.CHASER) or self._params.guard_leash == 0:
            return own
        ring = [
            (x, y)
            for y in range(self._cfg.height)
            for x in range(self._cfg.width)
            if self._nav.distance(own, (x, y)) == self._params.guard_leash
        ]
        if not ring:
            return own
        return min(ring, key=lambda c: (self._nav.distance(c, other), c[1], c[0]))

    def act(self, observation: TeamObservation, team: Team) -> dict[str, Action]:
        own_base, enemy_base = self._bases(team)
        own_flag = observation.flags[team]
        enemy_flag = observation.flags.get(team.opponent)
        enemies = [a for a in observation.agents.values() if a.team is not team and a.active]
        enemy_carriers = [a for a in enemies if a.carrying_flag]
        mine = sorted(
            (a for a in observation.agents.values() if a.team is team), key=lambda a: a.agent_id
        )
        attackers = [
            a for slot, a in enumerate(mine) if a.active and self._params.roles[slot] is Role.ATTACKER
        ]
        lead = attackers[0] if self._params.escort and len(attackers) > 1 else None
        actions: dict[str, Action] = {}
        lead_action: Action | None = None
        for slot, agent in enumerate(mine):
            if not agent.active:
                continue
            here = (agent.x, agent.y)
            role = self._params.roles[slot]
            if role is Role.ATTACKER:
                if agent.carrying_flag:
                    target = own_base
                elif enemy_flag is not None:
                    target = (enemy_flag.x, enemy_flag.y)  # at its base, dropped, or on our carrier
                else:
                    target = enemy_base
                if lead is not None and agent is not lead:
                    # Follower: copy the lead's move when sharing its cell, otherwise catch up.
                    if (lead.x, lead.y) == here and lead_action is not None:
                        actions[agent.agent_id] = lead_action
                    else:
                        actions[agent.agent_id] = self._nav.step_toward(here, (lead.x, lead.y), self._rng)
                    continue
                move = self._nav.step_toward(here, target, self._rng)
                if lead is not None and not agent.carrying_flag:
                    others = [a for a in attackers if a is not agent]
                    if any((o.x, o.y) != here for o in others):
                        move = Action.STAY  # wait until the escorts have caught up
                if lead is not None:
                    lead_action = move
                actions[agent.agent_id] = move
            elif role is Role.DEFENDER:
                actions[agent.agent_id] = self._nav.step_toward(
                    here, self._defender_target(here, team, own_flag, enemies, enemy_carriers), self._rng
                )
            else:
                actions[agent.agent_id] = self._scout_action(here)
        return actions

    def _defender_target(self, here, team, own_flag, enemies, enemy_carriers) -> tuple[int, int]:
        if own_flag.status in (FlagStatus.CARRIED, FlagStatus.DROPPED):
            return (own_flag.x, own_flag.y)  # chase the carrier, or touch the dropped flag to send it home
        if enemy_carriers:
            nearest = min(enemy_carriers, key=lambda c: self._nav.distance(here, (c.x, c.y)))
            return (nearest.x, nearest.y)
        if self._params.guard_mode is GuardMode.CHASER:
            close = [e for e in enemies if abs(e.x - here[0]) + abs(e.y - here[1]) <= self._params.chase_radius]
            if close:
                nearest = min(close, key=lambda c: self._nav.distance(here, (c.x, c.y)))
                return (nearest.x, nearest.y)
        return self._post[team]

    def _scout_action(self, here: tuple[int, int]) -> Action:
        if self._params.scout_behavior is ScoutBehavior.STAY:
            return Action.STAY
        options = [m for m in MOVES if self._nav.is_free(here[0] + m.delta()[0], here[1] + m.delta()[1])]
        return self._rng.choice(options) if options else Action.STAY


def build_opponent(spec: OpponentSpec, env_config: CTFConfig, rng: Random) -> BaseStrategy:
    """Create a fresh opponent from its config entry (raises ConfigError if invalid)."""
    if spec.kind not in KNOWN_KINDS:
        raise ConfigError(f"unknown opponent kind {spec.kind!r}; known kinds: {KNOWN_KINDS}")
    if spec.kind == "scripted":
        params = from_dict(ScriptedParams, spec.params, f"opponent[{spec.kind}].params")
        return ScriptedTeamStrategy(env_config, params, rng)
    if spec.params:
        raise ConfigError(f"opponent kind {spec.kind!r} takes no params, got {sorted(spec.params)}")
    return RandomStrategy(rng) if spec.kind == "random" else StayStrategy()
