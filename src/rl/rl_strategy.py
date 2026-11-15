"""A trained checkpoint as a ``BaseStrategy`` (plugs into the game like any other controller).

The strategy only ever uses what the team can see: a full observation is first reduced to
the team's fog-of-war view. Besides acting, it records its last decision (action
probabilities in environment order) and can estimate the critic's value, which the
visualisation shows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ctf.actions import Action
from ctf.agent import Team
from ctf.config import CTFConfig
from ctf.observations import GlobalObservation, PartialTeamObservation, TeamObservation, restrict_to_team_view
from rl.canonical import ACTION_INDEX, NUM_ACTIONS
from rl.checkpoint import load_policy
from rl.config import RLConfig
from rl.features import FeatureBuilder
from rl.mappo import ValueNorm
from rl.navigation import GridNav
from rl.networks import ActorCritic
from rl.rewards import RewardCalculator
from strategies import BaseStrategy, Role

_LEFT, _RIGHT = ACTION_INDEX[Action.LEFT], ACTION_INDEX[Action.RIGHT]


@dataclass(frozen=True)
class Decision:
    agent_ids: list[str]  # team agents in slot order
    probs: np.ndarray  # (A, NUM_ACTIONS), environment action order (UP, DOWN, LEFT, RIGHT, STAY)
    alive: np.ndarray  # (A,)
    actions: dict[str, Action]  # chosen action for each active agent


class RLStrategy(BaseStrategy):
    def __init__(
        self,
        model: ActorCritic,
        config: RLConfig,
        env_config: CTFConfig,
        value_norm: ValueNorm,
        *,
        deterministic: bool = True,
        controlled_roles: set[Role] | None = None,
        checkpoint_step: int = 0,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        spec = model.spec
        if (spec.height, spec.width) != (env_config.height, env_config.width):
            raise ValueError(
                f"this policy was trained on a {spec.width}x{spec.height} grid and cannot play on a "
                f"{env_config.width}x{env_config.height} map"
            )
        self.model = model.to(device).eval()
        self.config = config
        self.env_config = env_config
        self.value_norm = value_norm.to(device)
        self.deterministic = deterministic
        self.controlled_roles = controlled_roles
        self.checkpoint_step = checkpoint_step
        self.device = torch.device(device)
        self.roles: list[Role] = list(config.roles.slots)
        self._generator = torch.Generator().manual_seed(seed)
        self._builders: dict[Team, FeatureBuilder] = {}
        self._last_t: dict[Team, int] = {}
        self.last_decision: dict[Team, Decision] = {}

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        env_config: CTFConfig | None = None,
        **kwargs,
    ) -> "RLStrategy":
        model, meta = load_policy(path, "cpu")
        config: RLConfig = meta["config"]
        value_norm = ValueNorm(config.ppo.value_normalization)
        value_norm.load_state_dict(meta["value_norm"])
        return cls(
            model,
            config,
            env_config or meta["env_config"],
            value_norm,
            checkpoint_step=int(meta["counters"]["total_steps"]),
            **kwargs,
        )

    # ---------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        self._builders.clear()
        self._last_t.clear()
        self.last_decision.clear()

    def _builder(self, team: Team) -> FeatureBuilder:
        if team not in self._builders:
            self._builders[team] = FeatureBuilder(self.env_config, team, self.roles, self.config.features)
        return self._builders[team]

    # --------------------------------------------------------------------- acting

    def act(self, observation: TeamObservation, team: Team) -> dict[str, Action]:
        partial = (
            observation
            if isinstance(observation, PartialTeamObservation)
            else restrict_to_team_view(observation, team)
        )
        builder = self._builder(team)
        if self._last_t.get(team, -1) > partial.timestep:  # a new episode started
            builder.reset()
        self._last_t[team] = partial.timestep

        feats = builder.actor_features(partial)
        grid = torch.from_numpy(feats.grid).to(self.device)
        vec = torch.from_numpy(feats.vec).to(self.device)
        mask = torch.from_numpy(feats.mask).to(self.device)
        roles = torch.from_numpy(feats.role_index).to(self.device)
        with torch.no_grad():
            probs = self.model.policy(grid, vec, mask, roles).probs.cpu()
        if self.deterministic:
            chosen = probs.argmax(dim=-1)
        else:
            chosen = torch.multinomial(probs, 1, generator=self._generator).squeeze(-1)

        env_probs = probs.numpy().copy()
        if builder.canon.flip:  # undo the left-right mirroring of BLUE's view
            env_probs[:, [_LEFT, _RIGHT]] = env_probs[:, [_RIGHT, _LEFT]]
        actions: dict[str, Action] = {}
        for slot, agent_id in enumerate(builder.agent_ids):
            if not feats.alive[slot]:
                continue
            if self.controlled_roles is not None and self.roles[slot] not in self.controlled_roles:
                continue
            actions[agent_id] = builder.canon.to_env_action(int(chosen[slot]))
        self.last_decision[team] = Decision(list(builder.agent_ids), env_probs, feats.alive.copy(), actions)
        return actions

    # ------------------------------------------------------------------ insight

    def value(self, observation: GlobalObservation, team: Team) -> np.ndarray:
        """The critic's estimate (in reward units) for each of the team's agents."""
        feats = self._builder(team).critic_features(observation)
        with torch.no_grad():
            raw = self.model.value(
                torch.from_numpy(feats.grid).to(self.device), torch.from_numpy(feats.vec).to(self.device)
            )
            return self.value_norm.denormalize(raw).cpu().numpy()

    def make_reward_calculator(self, team: Team) -> RewardCalculator:
        cfg = self.env_config
        obstacles = set(cfg.obstacles)
        nav = GridNav([[(x, y) not in obstacles for x in range(cfg.width)] for y in range(cfg.height)])
        return RewardCalculator(cfg, team, self.roles, self.config.rewards, self.config.ppo.gamma, nav)

    @property
    def scout_scale(self) -> float:
        """The scout exploration-bonus scale in effect at this checkpoint."""
        return max(0.0, 1.0 - self.checkpoint_step / self.config.rewards.scout_anneal_steps)
