"""A live episode: two controllers play on a map and every step is recorded for the viewer."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

import numpy as np

from coordination.shared_map import SharedTeamMap, update_shared_map_from_partial_team
from ctf.actions import Action
from ctf.agent import Team
from ctf.config import CTFConfig
from ctf.environment import CTFEnvironment
from ctf.game_state import FlagStatus
from ctf.observations import ObservationMode
from viz.controllers import Controller
from viz.frames import FORMAT_VERSION, AgentFrame, FlagFrame, Frame, Recording, RecordingMeta

ACTION_NAMES = [a.name for a in Action]


def _plain(event: dict) -> dict:
    return {k: (v.value if isinstance(v, Enum) else v) for k, v in event.items()}


def _slot(agent_id: str) -> int:
    return int(agent_id[1:]) - 1


class Session:
    def __init__(
        self,
        env_config: CTFConfig,
        red: Controller,
        blue: Controller,
        *,
        seed: int = 0,
        note: str = "",
    ) -> None:
        self.env_config = env_config
        self.controllers = {Team.RED: red, Team.BLUE: blue}
        self.note = note
        self.seed = seed
        self.frames: list[Frame] = []
        self.reset(seed)

    # -------------------------------------------------------------------- lifecycle

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        cfg = self.env_config
        self.env = CTFEnvironment(replace(cfg, seed=self.seed))
        self.env.reset(seed=self.seed)
        self.belief = {t: SharedTeamMap(cfg.width, cfg.height, t) for t in Team}
        self.calcs = {}
        for team, ctrl in self.controllers.items():
            ctrl.strategy.reset()
            self.calcs[team] = ctrl.rl.make_reward_calculator(team) if ctrl.rl is not None else None
        glob = self.env.observe(ObservationMode.GLOBAL)
        for calc in self.calcs.values():
            if calc is not None:
                calc.reset(glob)
        self._cumulative = {aid: 0.0 for aid in glob.agents}
        self._prev_agents = dict(glob.agents)
        self.frames = []
        self._pending: dict[str, Action] = {}
        self._append_frame(events=[], rewards={})

    @property
    def done(self) -> bool:
        return self.frames[-1].done

    def step(self) -> Frame | None:
        """Advance one environment step and return the new frame (None if the episode is over)."""
        if self.done:
            return None
        _, _, _, info = self.env.step(self._pending)
        events = [_plain(e) for e in info.get("events", [])]
        glob = self.env.observe(ObservationMode.GLOBAL)
        rewards: dict[str, tuple[float, dict[str, float]]] = {}
        for team, calc in self.calcs.items():
            if calc is None:
                continue
            scale = self.controllers[team].rl.scout_scale
            result = calc.step(glob, self._prev_agents, events, scale)
            prefix = "R" if team is Team.RED else "B"
            for slot in range(self.env_config.agents_per_team):
                comps = {k: round(float(v[slot]), 4) for k, v in result.components.items() if abs(v[slot]) > 1e-9}
                rewards[f"{prefix}{slot + 1}"] = (float(result.total[slot]), comps)
        self._prev_agents = dict(glob.agents)
        return self._append_frame(events, rewards)

    def run_to_end(self, limit: int | None = None) -> None:
        cap = limit if limit is not None else self.env_config.max_episode_steps + 5
        while not self.done and len(self.frames) <= cap:
            self.step()

    # ------------------------------------------------------------------------ frames

    def _append_frame(self, events: list[dict], rewards: dict) -> Frame:
        cfg = self.env_config
        glob = self.env.observe(ObservationMode.GLOBAL)
        done = bool(glob.terminated or glob.truncation)
        probs: dict[str, list[float]] = {}
        values: dict[str, float] = {}
        actions: dict[str, Action] = {}
        belief: dict[str, dict] = {}
        for team in Team:
            partial = self.env.observe(ObservationMode.PARTIAL, team=team)
            update_shared_map_from_partial_team(self.belief[team], partial)
            belief[team.value] = self._belief_frame(team)
            if done:
                continue
            ctrl = self.controllers[team]
            chosen = ctrl.strategy.act(partial, team)
            prefix = "R" if team is Team.RED else "B"
            for slot in range(cfg.agents_per_team):
                aid = f"{prefix}{slot + 1}"
                if glob.agents[aid].active:
                    actions[aid] = chosen.get(aid, Action.STAY)
            if ctrl.rl is not None:
                decision = ctrl.rl.last_decision[team]
                vals = ctrl.rl.value(glob, team)
                for slot, aid in enumerate(decision.agent_ids):
                    probs[aid] = [round(float(p), 4) for p in decision.probs[slot]]
                    values[aid] = float(vals[slot])
        self._pending = actions

        agent_frames = []
        for aid in sorted(glob.agents):
            agent = glob.agents[aid]
            team = agent.team
            slot = _slot(aid)
            reward, comps = rewards.get(aid, (None, {}))
            if reward is not None:
                self._cumulative[aid] += reward
            agent_frames.append(AgentFrame(
                id=aid, team=team.value, slot=slot, role=self.controllers[team].roles[slot],
                x=agent.x, y=agent.y, active=agent.active, carrying=agent.carrying_flag,
                action=actions[aid].name if aid in actions else "",
                probs=probs.get(aid, []), value=values.get(aid), reward=reward, components=comps,
            ))
        flags = [
            FlagFrame(team.value, flag.status.value if isinstance(flag.status, FlagStatus) else str(flag.status),
                      flag.x, flag.y, flag.carrier_id or "")
            for team, flag in sorted(glob.flags.items(), key=lambda kv: kv[0].value)
        ]
        frame = Frame(
            t=glob.timestep, agents=agent_frames, flags=flags, events=events, belief=belief,
            winner=glob.winner.value if glob.winner else "", done=done,
        )
        self.frames.append(frame)
        return frame

    def _belief_frame(self, team: Team) -> dict:
        b = self.belief[team]
        last = [[-1 if v is None else int(v) for v in row] for row in b.last_observed_timestep]
        return {
            "enemy_prob": np.round(np.asarray(b.enemy_probability, dtype=float), 2).tolist(),
            "observed": [[bool(v) for v in row] for row in b.observed],
            "last_seen": last,
        }

    def cumulative(self, agent_id: str, up_to: int) -> float:
        """Total reward received by ``agent_id`` over frames 1..up_to."""
        return float(sum((f.agents[_idx(f, agent_id)].reward or 0.0) for f in self.frames[1 : up_to + 1]))

    # --------------------------------------------------------------------- recording

    def recording(self) -> Recording:
        cfg = self.env_config
        meta = RecordingMeta(
            width=cfg.width, height=cfg.height,
            obstacles=[list(c) for c in sorted(cfg.obstacles)],
            red_base=list(cfg.red_base), blue_base=list(cfg.blue_base),
            red_spawns=[list(c) for c in cfg.default_spawns(Team.RED)],
            blue_spawns=[list(c) for c in cfg.default_spawns(Team.BLUE)],
            visibility_radius=cfg.visibility_radius, max_steps=cfg.max_episode_steps,
            teams={t.value: c.label for t, c in self.controllers.items()},
            opponent=self.controllers[Team.BLUE].label, seed=self.seed, note=self.note,
            carrier_slots=list(cfg.carrier_slots or []), tagging_rule=cfg.tagging_rule.value,
            drop_flag_on_tag=cfg.drop_flag_on_tag,
        )
        return Recording(FORMAT_VERSION, meta, list(self.frames))


def _idx(frame: Frame, agent_id: str) -> int:
    return next(i for i, a in enumerate(frame.agents) if a.id == agent_id)


def record_episode(
    env_config: CTFConfig, red: Controller, blue: Controller, *, seed: int, note: str = ""
) -> Recording:
    session = Session(env_config, red, blue, seed=seed, note=note)
    session.run_to_end()
    return session.recording()
