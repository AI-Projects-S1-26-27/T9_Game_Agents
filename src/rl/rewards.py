"""Reward computation (the environment itself always returns 0).

Per-agent reward = team outcome (scaled per role) + individual events + potential-based shaping
(+ scout exploration bonus, escort bonus, step penalty). Shaping uses true BFS path distances, so
it respects the wall, and is potential-based (F = gamma*phi(s') - phi(s)), which does
not change which policy is optimal. All weights come from ``RewardsConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ctf.agent import Agent, Team
from ctf.config import CTFConfig
from ctf.game_state import FlagStatus
from ctf.observations import GlobalObservation, manhattan_cells
from rl.config import RewardsConfig
from rl.navigation import GridNav
from strategies.roles import Role

COMPONENTS = (
    "outcome",
    "pickup",
    "tagged",
    "tag",
    "team_share",
    "potential",
    "scout",
    "step",
    "carry",
    "escort",
)


@dataclass(frozen=True)
class RewardResult:
    total: np.ndarray  # (A,) float64
    components: dict[str, np.ndarray]  # each (A,)


class RewardCalculator:
    def __init__(
        self,
        env_config: CTFConfig,
        team: Team,
        roles: list[Role],
        cfg: RewardsConfig,
        gamma: float,
        nav: GridNav,
    ) -> None:
        self.cfg = cfg
        self.gamma = gamma
        self.team = team
        self.roles = roles
        self.nav = nav
        self.width, self.height = env_config.width, env_config.height
        prefix = "R" if team is Team.RED else "B"
        self.agent_ids = [f"{prefix}{i}" for i in range(1, env_config.agents_per_team + 1)]
        self._slot_of = {aid: i for i, aid in enumerate(self.agent_ids)}
        own = env_config.red_base if team is Team.RED else env_config.blue_base
        other = env_config.blue_base if team is Team.RED else env_config.red_base
        self.own_base, self.enemy_base = own, other
        # Distance a flag must be carried from its base to ours; makes the attacker's
        # potential continuous at the moment of pickup.
        self._carry_offset = nav.distance(other, own)
        share = {
            Role.ATTACKER: cfg.outcome_share_attacker,
            Role.DEFENDER: cfg.outcome_share_defender,
            Role.SCOUT: cfg.outcome_share_scout,
        }
        self._outcome_share = np.array([share[r] for r in roles])
        self._escort_roles = set(cfg.escort_roles)
        self._seen = np.zeros((self.height, self.width), dtype=bool)
        self._prev_potential = np.zeros(len(self.agent_ids))

    # ------------------------------------------------------------------ lifecycle

    def reset(self, obs: GlobalObservation) -> None:
        self._seen[:] = False
        self._mark_seen(obs, credit=False)
        self._prev_potential = self._potentials(obs)

    # ------------------------------------------------------------------ potentials

    def _potentials(self, obs: GlobalObservation) -> np.ndarray:
        phi = np.zeros(len(self.agent_ids))
        own_flag = obs.flags[self.team]
        enemy_flag = obs.flags[self.team.opponent]
        for slot, aid in enumerate(self.agent_ids):
            agent = obs.agents[aid]
            if not agent.active or obs.terminated:
                continue  # absorbing states have potential 0
            here = (agent.x, agent.y)
            role = self.roles[slot]
            if role is Role.ATTACKER:
                if agent.carrying_flag:
                    d_home = self.nav.distance(here, self.own_base)
                    phi[slot] = self.cfg.attacker_carry_potential * (self._carry_offset - d_home)
                else:
                    target = (enemy_flag.x, enemy_flag.y)
                    phi[slot] = -self.cfg.attacker_potential * self.nav.distance(here, target)
            elif role is Role.DEFENDER:
                carriers = self._enemy_carrier_cells(obs)
                if carriers and self.cfg.defender_intercept_potential > 0:
                    nearest = min(self.nav.distance(here, c) for c in carriers)
                    phi[slot] = -self.cfg.defender_intercept_potential * nearest
                else:  # guard the flag wherever it lies (home, carried away, or dropped)
                    target = (own_flag.x, own_flag.y)
                    phi[slot] = -self.cfg.defender_potential * self.nav.distance(here, target)
        return phi

    def _enemy_carrier_cells(self, obs: GlobalObservation) -> list[tuple[int, int]]:
        return [(a.x, a.y) for a in obs.agents.values() if a.team is not self.team and a.active and a.carrying_flag]

    # ----------------------------------------------------------------- exploration

    def _mark_seen(self, obs: GlobalObservation, *, credit: bool) -> np.ndarray:
        """Update the seen-set; return newly seen cell counts credited per agent."""
        new_counts = np.zeros(len(self.agent_ids))
        for slot, aid in enumerate(self.agent_ids):
            agent = obs.agents[aid]
            if not agent.active:
                continue
            cells = manhattan_cells(agent.x, agent.y, obs.visibility_radius, self.width, self.height)
            fresh = 0
            for x, y in cells:
                if not self._seen[y, x]:
                    self._seen[y, x] = True
                    fresh += 1
            if credit:
                new_counts[slot] = fresh
        return new_counts

    # ---------------------------------------------------------------------- step

    def step(
        self,
        obs: GlobalObservation,
        prev_agents: dict[str, Agent],
        events: list[dict],
        scout_scale: float,
    ) -> RewardResult:
        """Rewards for the transition that produced ``obs`` (events as emitted by the env)."""
        n = len(self.agent_ids)
        comp = {name: np.zeros(n) for name in COMPONENTS}
        cfg = self.cfg

        # Team outcome: everyone shares it, including eliminated agents.
        if obs.terminated:
            if obs.winner is self.team:
                comp["outcome"][:] = cfg.win * self._outcome_share
            elif obs.winner is not None:
                comp["outcome"][:] = cfg.loss * self._outcome_share
            else:
                comp["outcome"][:] = cfg.timeout * self._outcome_share

        for event in events:
            kind = event.get("type")
            if kind == "flag_pickup":
                slot = self._slot_of.get(event["agent_id"])
                if slot is not None:
                    comp["pickup"][slot] += cfg.pickup
            elif kind == "tag":
                carrier = self._slot_of.get(event["agent_id"])
                tagger = self._slot_of.get(event["by_agent_id"])
                if carrier is not None:
                    comp["tagged"][carrier] += cfg.tagged_while_carrying
                if tagger is not None:
                    comp["tag"][tagger] += cfg.tag_enemy_carrier
                    for slot in range(n):
                        if slot != tagger:
                            comp["team_share"][slot] += cfg.tag_team_share

        # Potential-based shaping.
        phi = self._potentials(obs)
        comp["potential"] = self.gamma * phi - self._prev_potential
        self._prev_potential = phi

        # Scout exploration bonus and per-step cost (only for agents that acted).
        fresh = self._mark_seen(obs, credit=True)
        scale = cfg.scout_floor_fraction + (1.0 - cfg.scout_floor_fraction) * scout_scale
        carriers = [(a.x, a.y) for a in obs.agents.values() if a.team is self.team and a.active and a.carrying_flag]
        for slot in range(n):
            if self.roles[slot] is Role.SCOUT:
                comp["scout"][slot] = cfg.scout_new_cell * scale * fresh[slot]
            agent = obs.agents[self.agent_ids[slot]]
            if (
                carriers
                and agent.active
                and not agent.carrying_flag
                and self.roles[slot] in self._escort_roles
                and any(abs(agent.x - cx) + abs(agent.y - cy) <= cfg.escort_radius for cx, cy in carriers)
            ):
                comp["escort"][slot] = cfg.escort_bonus
            before = prev_agents[self.agent_ids[slot]]
            if before.active:
                comp["step"][slot] = cfg.step_penalty
                if before.carrying_flag:
                    comp["carry"][slot] = cfg.carry_step_penalty

        total = sum(comp.values())
        return RewardResult(total=np.asarray(total, dtype=np.float64), components=comp)
