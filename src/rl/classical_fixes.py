"""Fixes for the known problems of the classical AI, WITHOUT editing ``src/classical_ai`` (a verbatim copy).

Everything here subclasses or wraps the teammate's classes, so re-copying ``classical_ai`` keeps working.
Each fix has its own switch in ``ClassicalParams`` (rl/classical_opponent.py); with every switch off and the
numbers at their old literals the original code runs unchanged (``ClassicalOpponent`` then does not use this
module at all), so earlier measurements stay comparable.

Fixes:

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
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from classical_ai.behavior_tree import decide_action
from classical_ai.coordinator import ClassicalCoordinator
from classical_ai.influence_map import build_influence_map
from classical_ai.pathfinding import astar_path, next_action
from classical_ai.roles.attacker import AttackerRole
from classical_ai.roles.defender import DefenderRole
from classical_ai.roles.scout import ScoutRole
from coordination.shared_map import build_scout_view
from ctf.actions import Action
from ctf.agent import Agent, Team
from ctf.game_state import Flag, FlagStatus
from ctf.observations import GlobalObservation, PartialTeamObservation, TeamObservation, manhattan_distance
from strategies import BaseStrategy

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
