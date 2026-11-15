"""CTF environment step/reset logic."""

from __future__ import annotations

import random
from typing import Any

from ctf.actions import Action
from ctf.agent import Agent, Team
from ctf.config import CTFConfig, TaggingRule
from ctf.game_state import FlagStatus, GameState
from ctf.observations import ObservationMode, TeamObservation, build_observation


class CTFEnvironment:
    """Discrete multi-agent CTF simulator (Prototype 1)."""

    def __init__(self, config: CTFConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self._state: GameState | None = None

    @property
    def state(self) -> GameState:
        if self._state is None:
            raise RuntimeError("Environment not reset; call reset() first")
        return self._state.copy()

    @property
    def done(self) -> bool:
        return self._state is not None and self._state.terminated

    def reset(self, *, seed: int | None = None) -> GameState:
        """Start a new episode.

        ``seed`` (re)seeds the environment's private RNG. Without it the RNG stream
        simply continues, so consecutive ``reset()`` calls give different episodes
        when ``spawn_mode`` is random, yet the whole sequence is reproducible from
        ``config.seed`` (or from the last explicit ``seed``). ``step()`` never uses
        the RNG: given the same spawns and actions, transitions are deterministic.
        """
        if seed is not None:
            self._rng = random.Random(seed)
        self._state = GameState.initial(
            self.config,
            red_spawns=self.config.sample_spawns(Team.RED, self._rng),
            blue_spawns=self.config.sample_spawns(Team.BLUE, self._rng),
        )
        return self._state.copy()

    def observe(
        self,
        mode: ObservationMode = ObservationMode.GLOBAL,
        *,
        team: Team | None = None,
        visibility_radius: int | None = None,
    ) -> TeamObservation:
        """Observation of the current state. Radius defaults to ``config.visibility_radius``."""
        if self._state is None:
            raise RuntimeError("Environment not reset; call reset() first")
        return build_observation(
            self._state, mode=mode, team=team, visibility_radius=visibility_radius
        )

    def step(
        self,
        actions: dict[str, Action | str],
    ) -> tuple[GameState, dict[str, float], bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Environment not reset; call reset() first")
        if self._state.terminated:
            raise RuntimeError("Cannot step after episode termination")

        parsed = self._parse_actions(actions)
        state = self._state.copy()
        events: list[dict[str, Any]] = []
        info: dict[str, Any] = {"rejected_moves": {}, "events": events}

        proposals = self._propose_moves(state, parsed, info)
        state.agents = {
            aid: agent.with_position(*proposals[aid])
            for aid, agent in state.agents.items()
        }

        self._sync_carried_flag_positions(state)
        self._handle_flag_returns(state, events)
        self._handle_flag_pickups(state, events)
        self._handle_tagging(state, events)
        self._handle_scoring(state, events)

        state.timestep += 1
        if not state.terminated and state.timestep >= self.config.max_episode_steps:
            state.truncation = True
            state.terminated = True
            events.append(
                {
                    "type": "episode_end",
                    "winner": None,
                    "reason": "max_steps",
                }
            )

        self._state = state
        rewards = {aid: 0.0 for aid in state.agents}
        return state.copy(), rewards, state.terminated, info

    def _parse_actions(self, actions: dict[str, Action | str]) -> dict[str, Action]:
        parsed: dict[str, Action] = {}
        for raw_id, raw_action in actions.items():
            if raw_id not in self._state.agents:
                raise KeyError(f"Unknown agent id: {raw_id}")
            action = raw_action if isinstance(raw_action, Action) else Action(raw_action)
            parsed[raw_id] = action

        for aid, agent in self._state.agents.items():
            if not agent.active:
                continue
            if aid not in parsed:
                raise KeyError(f"Missing action for active agent: {aid}")
        return parsed

    def _propose_moves(
        self,
        state: GameState,
        actions: dict[str, Action],
        info: dict[str, Any],
    ) -> dict[str, tuple[int, int]]:
        proposals: dict[str, tuple[int, int]] = {}
        for aid in sorted(state.agents.keys()):
            agent = state.agents[aid]
            if not agent.active:
                proposals[aid] = (agent.x, agent.y)
                continue
            action = actions.get(aid, Action.STAY)
            dx, dy = action.delta()
            nx, ny = agent.x + dx, agent.y + dy
            if action is Action.STAY:
                proposals[aid] = (agent.x, agent.y)
                continue
            if not state.grid.is_traversable(nx, ny):
                info["rejected_moves"][aid] = action.value
                proposals[aid] = (agent.x, agent.y)
            else:
                proposals[aid] = (nx, ny)

        return proposals

    def _sync_carried_flag_positions(self, state: GameState) -> None:
        for team, flag in list(state.flags.items()):
            if flag.status is not FlagStatus.CARRIED or flag.carrier_id is None:
                continue
            carrier = state.agents.get(flag.carrier_id)
            if carrier is None or not carrier.active:
                continue
            state.flags[team] = flag.with_position(carrier.x, carrier.y)

    def _handle_flag_returns(self, state: GameState, events: list[dict[str, Any]]) -> None:
        """A dropped flag goes home when an active agent of its own team is on its cell.

        Runs before pickups, so if a defender and an enemy attacker reach a dropped flag in the
        same step, the defender wins the race and the flag is sent home.
        """
        if not self.config.dropped_flag_return_on_touch:
            return
        for flag_team, flag in list(state.flags.items()):
            if flag.status is not FlagStatus.DROPPED:
                continue
            for aid in sorted(state.agents):
                agent = state.agents[aid]
                if agent.active and agent.team is flag_team and (agent.x, agent.y) == (flag.x, flag.y):
                    state.flags[flag_team] = flag.with_at_base(self.config)
                    events.append({"type": "flag_return", "agent_id": aid, "flag_team": flag_team.value})
                    break

    def _handle_flag_pickups(self, state: GameState, events: list[dict[str, Any]]) -> None:
        for aid, agent in state.agents.items():
            if not agent.active or agent.carrying_flag:
                continue
            if not self.config.can_carry(agent.slot):
                continue
            enemy = agent.team.opponent
            enemy_flag = state.flags[enemy]
            from_dropped = enemy_flag.status is FlagStatus.DROPPED
            if enemy_flag.status is FlagStatus.CARRIED:
                continue
            if from_dropped and not self.config.dropped_flag_enemy_pickup:
                continue
            if agent.x == enemy_flag.x and agent.y == enemy_flag.y:
                state.agents[aid] = agent.with_carrying_flag(True)
                state.flags[enemy] = enemy_flag.with_carried(aid, agent.x, agent.y)
                event = {
                    "type": "flag_pickup",
                    "agent_id": aid,
                    "flag_team": enemy.value,
                }
                if from_dropped:  # key only present for re-pickups, so original events are unchanged
                    event["from_dropped"] = True
                events.append(event)

    def _handle_tagging(self, state: GameState, events: list[dict[str, Any]]) -> None:
        if not state.tagging_enabled:
            return
        carriers = [
            aid
            for aid, ag in state.agents.items()
            if ag.active and ag.carrying_flag
        ]
        if self.config.tagging_rule is TaggingRule.HEADCOUNT:
            self._tag_by_headcount(state, carriers, events)
            return
        for carrier_id in carriers:
            carrier = state.agents[carrier_id]
            for aid, other in state.agents.items():
                if aid == carrier_id or not other.active:
                    continue
                if other.team is carrier.team:
                    continue
                if other.x == carrier.x and other.y == carrier.y:
                    self._tag(state, carrier_id, aid, events)
                    break

    def _tag_by_headcount(
        self, state: GameState, carriers: list[str], events: list[dict[str, Any]]
    ) -> None:
        """Tag every carrier outnumbered on its cell. All carriers are judged on the same snapshot."""
        tagged: list[tuple[str, str]] = []
        for carrier_id in carriers:
            carrier = state.agents[carrier_id]
            here = [
                (aid, ag)
                for aid, ag in sorted(state.agents.items())
                if ag.active and aid != carrier_id and (ag.x, ag.y) == (carrier.x, carrier.y)
            ]
            enemies = [aid for aid, ag in here if ag.team is not carrier.team]
            escorts = [aid for aid, ag in here if ag.team is carrier.team and not ag.carrying_flag]
            if len(enemies) > len(escorts):
                tagged.append((carrier_id, enemies[0]))
        for carrier_id, by_id in tagged:
            self._tag(state, carrier_id, by_id, events)

    def _tag(
        self, state: GameState, carrier_id: str, by_id: str, events: list[dict[str, Any]]
    ) -> None:
        carrier = state.agents[carrier_id]
        flag_team = carrier.team.opponent
        x, y = carrier.x, carrier.y
        self._eliminate_carrier(state, carrier_id)
        events.append(
            {
                "type": "tag",
                "agent_id": carrier_id,
                "by_agent_id": by_id,
                "flag_team": flag_team.value,
            }
        )
        if self.config.drop_flag_on_tag:
            events.append(
                {
                    "type": "flag_drop",
                    "agent_id": carrier_id,
                    "flag_team": flag_team.value,
                    "x": x,
                    "y": y,
                }
            )

    def _eliminate_carrier(self, state: GameState, carrier_id: str) -> None:
        carrier = state.agents[carrier_id]
        enemy_flag_team = carrier.team.opponent
        state.agents[carrier_id] = carrier.with_active(False).with_carrying_flag(False)
        flag = state.flags[enemy_flag_team]
        if self.config.drop_flag_on_tag:
            state.flags[enemy_flag_team] = flag.with_dropped(carrier.x, carrier.y)
        else:
            state.flags[enemy_flag_team] = flag.with_at_base(self.config)

    def _handle_scoring(self, state: GameState, events: list[dict[str, Any]]) -> None:
        for aid, agent in state.agents.items():
            if not agent.active or not agent.carrying_flag:
                continue
            base = state.red_base if agent.team is Team.RED else state.blue_base
            if (agent.x, agent.y) != base:
                continue
            flag_team = agent.team.opponent
            state.winner = agent.team
            state.terminated = True
            events.append(
                {
                    "type": "flag_capture",
                    "agent_id": aid,
                    "flag_team": flag_team.value,
                }
            )
            events.append(
                {
                    "type": "episode_end",
                    "winner": agent.team.value,
                    "reason": "flag_capture",
                }
            )
            return
