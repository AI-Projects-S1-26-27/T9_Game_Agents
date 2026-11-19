"""The classical AI for the v2 game (4 agents per team, carrier slots, headcount tagging, flags that stay where they drop).

``classical_ai`` was written for the original game: 3 agents, any enemy tags, a tagged carrier's flag goes home. The
original modules in this package (behaviour tree, role controllers, coordinator, influence map, path finding) are
untouched and still describe what each role WANTS. This module is the v2 layer around them. It subclasses the
original roles and coordinator, so changing the original files and this module never conflict.

What it adds, each part with its own switch in ``ClassicalParams`` (defaults = the ORIGINAL behaviour, so
configs and results written before this module existed mean exactly what they meant; with every switch at its
default the unmodified ``ClassicalStrategy`` runs):

* **Fixed roles per slot** (``ClassicalOpponent``). The original coordinator assigns roles by position among the
  agents that are still ACTIVE (first = attacker, second = defender, the rest scouts). That is wrong for v2 (slots
  1-2 attackers, slot 3 defender, slot 4 scout) and shifts when an agent is tagged. ``params.roles`` gives a fixed
  ``{agent id: role}`` table; only slots that may carry the flag (``carrier_slots``) may be attackers.
* **A real ``reset()``.** The original strategy keeps its belief map for as long as the object lives.
* ``remember_dropped_flag``: the enemy flag is only in the observation while one of our agents is within the
  visibility radius, so a flag dropped out of sight made the attackers walk back to the enemy base. The strategy
  remembers the last seen DROPPED enemy flag and puts it back into the observation until the cell is in view again.
* ``defend_dropped_flag``: the defender ignored a DROPPED own flag. It now walks onto it (an active agent of the
  flag's own team standing on a dropped flag sends it home).
* ``buddy_system``: the headcount rule tags a carrier only when enemies on its cell outnumber its friendly
  non-carriers there. The attackers now (a) enter a guarded enemy flag cell together (the first one waits next to
  it until its partner is next to it too; non-carriers cannot be tagged, so waiting is safe), (b) the carrier
  waits one step for an adjacent escort when an enemy is close, (c) an escort on the carrier's cell repeats the
  carrier's move so the pair stays on one cell.
* ``intercept_in_pairs``: with the headcount rule one interceptor can never tag a carrier that has an escort. When an
  enemy carrier is visible, the defender and the scout together pick the first cell on the carrier's shortest way
  home that both can reach in time and wait there; the carrier then walks into two agents.
* Always on in the fixed strategy: it notices a new episode itself (the step counter went back) and starts with
  an empty belief map, so callers that never call ``reset()`` no longer leak knowledge between games; the scout
  view radius for global observations is the observation's own radius, not a literal 3.
* The influence-map literals (radius, penalty, probability weight) are parameters.

``rl.opponents`` (kind ``classical``), the viewer (``configs/viz/controllers.yaml``) and ``rl.evaluate`` build this
class through ``rl/classical_opponent.py``, which only re-exports it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from classical_ai.behavior_tree import decide_action
from classical_ai.classical_strategy import ClassicalStrategy
from classical_ai.coordinator import ClassicalCoordinator
from classical_ai.influence_map import build_influence_map
from classical_ai.pathfinding import astar_path, next_action
from classical_ai.roles.attacker import AttackerRole
from classical_ai.roles.defender import DefenderRole
from classical_ai.roles.scout import ScoutRole
from coordination.shared_map import build_scout_view
from ctf.actions import Action
from ctf.agent import Agent, Team
from ctf.config import CTFConfig
from ctf.game_state import Flag, FlagStatus
from ctf.observations import GlobalObservation, PartialTeamObservation, TeamObservation, manhattan_distance
from rl.schema import ConfigError  # the project's strict-YAML error type
from strategies import BaseStrategy, Role

Position = tuple[int, int]


@dataclass(frozen=True)
class FixSettings:
    """The tunables of the fixes (built from ``ClassicalParams`` by the adapter; values come from YAML)."""

    remember_dropped_flag: bool
    defend_dropped_flag: bool
    buddy_system: bool
    intercept_in_pairs: bool
    pair_guard_radius: int  # enemy within this many cells of the enemy flag counts as "the flag is guarded"
    pair_wait_steps: int  # longest a buddy rule may hold an agent still before it goes on alone
    carrier_danger_radius: int  # carrier waits for an adjacent escort when an enemy is this close (Manhattan)
    influence_radius: int
    enemy_penalty: float
    probability_weight: float


def _path_len(start: Position, goal: Position, observation: TeamObservation) -> int | None:
    path = astar_path(start, goal, observation, None)
    return None if path is None else len(path) - 1


def _enemy_carriers(observation: TeamObservation, team: Team) -> list[Agent]:
    return [a for a in observation.agents.values() if a.team is team.opponent and a.active and a.carrying_flag]


def ambush_cell(observation: TeamObservation, team: Team, carrier: Agent, interceptors: list[Agent]) -> Position:
    """First cell on the carrier's shortest way home that every interceptor reaches in at most as many steps as the
    carrier needs; the carrier's own cell (plain chase) if there is none."""
    home = observation.red_base if team.opponent is Team.RED else observation.blue_base
    path = astar_path((carrier.x, carrier.y), home, observation, None)
    if path:
        for k, cell in enumerate(path[1:], start=1):
            lengths = [_path_len((i.x, i.y), cell, observation) for i in interceptors]
            if all(n is not None and n <= k for n in lengths):
                return cell
    return (carrier.x, carrier.y)


def _ambush_target(settings: "FixSettings", observation, team, agent: Agent, interceptor_ids) -> Position | None:
    if not settings.intercept_in_pairs:
        return None
    carriers = _enemy_carriers(observation, team)
    if not carriers:
        return None
    carrier = min(carriers, key=lambda c: manhattan_distance(c.x, c.y, agent.x, agent.y))
    team_mates = [
        observation.agents[i]
        for i in interceptor_ids(team)
        if i in observation.agents and observation.agents[i].active and not observation.agents[i].carrying_flag
    ]
    if agent.agent_id not in {a.agent_id for a in team_mates}:
        team_mates.append(agent)
    return ambush_cell(observation, team, carrier, team_mates)


class _Influence:
    """Builds the influence map with the configured literals."""

    def __init__(self, settings: FixSettings) -> None:
        self.settings = settings

    def build(self, observation: TeamObservation, team: Team, shared_map) -> list[list[float]]:
        s = self.settings
        return build_influence_map(
            observation=observation,
            team=team,
            shared_map=shared_map,
            enemy_radius=s.influence_radius,
            enemy_penalty=s.enemy_penalty,
            probability_weight=s.probability_weight,
        )


class FixedDefenderRole(DefenderRole):
    def __init__(self, shared_map, settings: FixSettings, interceptor_ids) -> None:
        super().__init__(shared_map)
        self.settings = settings
        self._influence = _Influence(settings)
        self._interceptor_ids = interceptor_ids  # callable: team -> ids of the defender and scout slots

    def find_threat(self, observation, team, start):
        threat = super().find_threat(observation, team, start)
        if threat is not None or not self.settings.defend_dropped_flag:
            return threat
        own_flag = observation.flags.get(team)
        if own_flag is not None and own_flag.status == FlagStatus.DROPPED:
            return (own_flag.x, own_flag.y)
        return None

    def act(self, agent_id, observation, team):
        agent = observation.agents.get(agent_id)
        if agent is None or not agent.active or agent.team is not team:
            return Action.STAY
        start = (agent.x, agent.y)
        influence = self._influence.build(observation, team, self.shared_map)
        target = _ambush_target(self.settings, observation, team, agent, self._interceptor_ids)
        if target is None:
            target = self.find_threat(observation, team, start)
        if target is None:
            target = self.get_defensive_position(observation, team)
        return next_action(start=start, goal=target, observation=observation, algorithm="astar", influence_map=influence)


class FixedScoutRole(ScoutRole):
    """Same as the original scout except for the influence literals (the original ``act`` calls the module function)."""

    def __init__(self, shared_map, settings: FixSettings, interceptor_ids) -> None:
        super().__init__(shared_map)
        self.settings = settings
        self._influence = _Influence(settings)
        self._interceptor_ids = interceptor_ids

    def act(self, agent_id, observation, team):
        agent = observation.agents.get(agent_id)
        if agent is None or not agent.active or agent.team is not team:
            return Action.STAY
        self.update_knowledge(agent_id, observation, team)
        influence = self._influence.build(observation, team, self.shared_map)
        if agent.carrying_flag:
            return decide_action(agent_id=agent_id, observation=observation, team=team, influence_map=influence)
        start = (agent.x, agent.y)
        ambush = _ambush_target(self.settings, observation, team, agent, self._interceptor_ids)
        if ambush is not None:
            return next_action(start=start, goal=ambush, observation=observation, algorithm="astar", influence_map=influence)
        target = self.find_exploration_target(observation, start, influence)
        if target is None:
            return Action.STAY
        return next_action(start=start, goal=target, observation=observation, algorithm="astar", influence_map=influence)

    def update_knowledge(self, agent_id, observation, team):
        if isinstance(observation, GlobalObservation):
            # the original used a literal 3 here; the environment's own radius is the right one
            view = build_scout_view(observation, team, agent_id, visibility_radius=observation.visibility_radius)
            self.shared_map.update_from_scout(view)
            return
        super().update_knowledge(agent_id, observation, team)


class FixedAttackerRole(AttackerRole):
    """Original attacker plus the buddy rules (headcount awareness) when ``buddy_system`` is on."""

    def __init__(self, shared_map, settings: FixSettings, attacker_ids) -> None:
        super().__init__(shared_map)
        self.settings = settings
        self._influence = _Influence(settings)
        self._attacker_ids = attacker_ids  # callable: team -> ids of the attackers in the role table
        self._waited: dict[str, int] = {}
        self._budget_spent = False  # set by _wait during the current decision when the budget ran out

    # -- original behaviour with the configured influence literals
    def _default_action(self, agent_id, observation, team, influence) -> Action:
        return decide_action(
            agent_id=agent_id, observation=observation, team=team, influence_map=influence, algorithm="astar"
        )

    def act(self, agent_id, observation, team):
        agent = observation.agents.get(agent_id)
        if agent is None or not agent.active or agent.team is not team:
            return Action.STAY
        influence = self._influence.build(observation, team, self.shared_map)
        if self.settings.buddy_system:
            self._budget_spent = False
            held = self._buddy_action(agent, observation, team, influence)
            if held is not None:
                return held
            if not self._budget_spent:  # a spent budget stays spent while the same situation lasts
                self._waited[agent_id] = 0
        return self._default_action(agent_id, observation, team, influence)

    # -- buddy rules
    def _wait(self, agent_id: str) -> Action | None:
        """STAY while the wait budget lasts; None (go on alone) once it is spent."""
        used = self._waited.get(agent_id, 0)
        if used >= self.settings.pair_wait_steps:
            self._budget_spent = True
            return None
        self._waited[agent_id] = used + 1
        return Action.STAY

    def _partners(self, agent: Agent, observation, team) -> list[Agent]:
        out = []
        for other_id in self._attacker_ids(team):
            other = observation.agents.get(other_id)
            if other is not None and other.active and other.team is team and other_id != agent.agent_id:
                out.append(other)
        return out

    def _buddy_action(self, agent: Agent, observation, team, influence) -> Action | None:
        enemy_flag = observation.flags.get(team.opponent)
        enemies = [a for a in observation.agents.values() if a.team is team.opponent and a.active]
        here = (agent.x, agent.y)
        if agent.carrying_flag:
            return self._carrier_rule(agent, observation, team, enemies)
        if enemy_flag is None:
            return None
        flag_cell = (enemy_flag.x, enemy_flag.y)
        if enemy_flag.status is FlagStatus.CARRIED:
            # escort: a friendly carrier standing on my cell -> repeat its move so we stay on one cell
            carrier = observation.agents.get(enemy_flag.carrier_id) if enemy_flag.carrier_id else None
            if carrier is not None and carrier.team is team and carrier.active and (carrier.x, carrier.y) == here:
                return self._default_action(carrier.agent_id, observation, team, influence)
            return None
        # flag at base or dropped: enter a guarded flag cell together
        partners = [p for p in self._partners(agent, observation, team) if not p.carrying_flag]
        guarded = any(manhattan_distance(e.x, e.y, *flag_cell) <= self.settings.pair_guard_radius for e in enemies)
        if not guarded or not partners:
            return None
        mine = _path_len(here, flag_cell, observation)
        if mine != 1:
            return None  # far away: walk up; standing on the cell: nothing to wait for
        theirs = [_path_len((p.x, p.y), flag_cell, observation) for p in partners]
        if any(d is not None and d <= 1 for d in theirs):
            return None  # both are next to the flag: step in together
        return self._wait(agent.agent_id)

    def _carrier_rule(self, agent: Agent, observation, team, enemies: list[Agent]) -> Action | None:
        here = (agent.x, agent.y)
        danger = any(
            manhattan_distance(e.x, e.y, *here) <= self.settings.carrier_danger_radius for e in enemies
        )
        if not danger:
            return None
        friends = [
            a for a in observation.agents.values() if a.team is team and a.active and not a.carrying_flag
        ]
        if any((a.x, a.y) == here for a in friends):
            return None  # an escort is already on my cell: keep moving (it repeats my move)
        if any(manhattan_distance(a.x, a.y, *here) == 1 for a in friends):
            return self._wait(agent.agent_id)  # one step, so the adjacent escort can join me
        return None


class FixedClassicalCoordinator(ClassicalCoordinator):
    def __init__(self, width, height, role_assignments, settings: FixSettings) -> None:
        super().__init__(width, height, role_assignments)
        self.settings = settings
        self._attackers: dict[Team, list[str]] = {}
        self._interceptors: dict[Team, list[str]] = {}

    def _attacker_ids(self, team: Team) -> list[str]:
        return list(self._attackers.get(team, []))

    def _interceptor_ids(self, team: Team) -> list[str]:
        return list(self._interceptors.get(team, []))

    def initialize_roles(self, team: Team) -> None:
        if team in self.role_controllers:
            return
        shared_map = self.team_maps.for_team(team)
        self.role_controllers[team] = {
            "attacker": FixedAttackerRole(shared_map, self.settings, self._attacker_ids),
            "defender": FixedDefenderRole(shared_map, self.settings, self._interceptor_ids),
            "scout": FixedScoutRole(shared_map, self.settings, self._interceptor_ids),
        }

    def get_role_assignments(self, observation, team):
        assignments = super().get_role_assignments(observation, team)
        self._attackers[team] = sorted(a for a, role in assignments.items() if role == "attacker")
        self._interceptors[team] = sorted(a for a, role in assignments.items() if role in ("defender", "scout"))
        return assignments

    def update_team_knowledge(self, observation, team) -> None:
        if isinstance(observation, GlobalObservation):
            shared_map = self.team_maps.for_team(team)
            for agent_id, agent in observation.agents.items():
                if agent.team is not team or not agent.active:
                    continue
                view = build_scout_view(
                    observation=observation,
                    team=team,
                    scout_agent_id=agent_id,
                    visibility_radius=observation.visibility_radius,
                )
                shared_map.update_from_scout(view)
            return
        super().update_team_knowledge(observation, team)


class FixedClassicalStrategy(BaseStrategy):
    """Drop-in replacement for ``ClassicalStrategy`` with the fixes; see the module docstring."""

    def __init__(self, role_assignments, settings: FixSettings) -> None:
        self.role_assignments = role_assignments
        self.settings = settings
        self.coordinator: FixedClassicalCoordinator | None = None
        self._last_step: int | None = None
        self._remembered: dict[Team, Flag] = {}

    def reset(self) -> None:
        self.coordinator = None
        self._last_step = None
        self._remembered = {}

    def act(self, observation: TeamObservation, team: Team) -> dict[str, Action]:
        if self._last_step is not None and observation.timestep < self._last_step:
            self.reset()  # a new episode started and nobody called reset()
        self._last_step = observation.timestep
        if self.coordinator is None:
            self.coordinator = FixedClassicalCoordinator(
                observation.width, observation.height, self.role_assignments, self.settings
            )
        if self.settings.remember_dropped_flag:
            observation = self._with_remembered_flag(observation, team)
        return self.coordinator.act(observation=observation, team=team)

    def _with_remembered_flag(self, observation: TeamObservation, team: Team) -> TeamObservation:
        enemy = team.opponent
        seen = observation.flags.get(enemy)
        if seen is not None:
            if seen.status is FlagStatus.DROPPED:
                self._remembered[enemy] = seen
            else:
                self._remembered.pop(enemy, None)
            return observation
        memory = self._remembered.get(enemy)
        if memory is None or not isinstance(observation, PartialTeamObservation):
            return observation
        in_view = any(
            a.team is team and a.active and manhattan_distance(a.x, a.y, memory.x, memory.y) <= observation.visibility_radius
            for a in observation.agents.values()
        )
        if in_view:  # we can see the cell and the flag is not reported there any more: the memory is stale
            self._remembered.pop(enemy, None)
            return observation
        return replace(observation, flags={**observation.flags, enemy: memory})


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
        """True when nothing differs from the original classical code (then that code runs untouched)."""
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
