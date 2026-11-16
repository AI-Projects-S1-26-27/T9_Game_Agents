"""Integration with Team 2's reinforcement-learning agent (branch rl_agent).

make_rl_strategy     loads a trained checkpoint as the RL team's own RLStrategy (rl/rl_strategy.py).
RLControllerAdapter  mixed team: agents with an LLM order follow it (shortest path); all other agents
                     are played by the RL policy. The policy has no input for orders, so the LLM can only
                     take over individual agents, not steer the policy's own decisions.

The RL agent is trained on the v2 rules with 4 agents per team (configs/ctf_v2.yaml), so games that
include it need `--config configs/ctf_v2.yaml`. Requires the rl_agent branch and PyTorch
(uv sync --extra dev --extra rl).
"""
from __future__ import annotations

import os
from pathlib import Path

from ctf.actions import Action

from .controller import Controller

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = "runs/mappo_default/checkpoints/latest.pt"   # where `python -m rl.train` writes
UNTRAINED = "untrained"        # random weights: only for testing the plumbing, plays badly


def resolve_checkpoint(path: str | None) -> str:
    """Checkpoint path from the argument, the RL_CHECKPOINT environment variable or the default location.
    Raises RuntimeError with instructions if the file does not exist ('untrained' is always accepted)."""
    path = path or os.environ.get("RL_CHECKPOINT") or DEFAULT_CHECKPOINT
    if path == UNTRAINED:
        return path
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    if not p.exists():
        raise RuntimeError(
            f"RL checkpoint not found: {path}\nTrain the RL agent first (python -m rl.train --config "
            f"configs/rl/default.yaml), or pass an existing file with --rl-checkpoint. "
            f"'--rl-checkpoint untrained' uses random weights and only tests that the plumbing works.")
    return str(p)


def make_rl_strategy(checkpoint: str | None = None, env_config=None, *, deterministic: bool = True, seed: int = 0):
    """The RL team's RLStrategy for a checkpoint file ('untrained' = random weights).
    env_config: the game's CTFConfig, used to check that the checkpoint fits the game."""
    import torch

    from rl.checkpoint import CheckpointError
    from rl.rl_strategy import RLStrategy

    checkpoint = resolve_checkpoint(checkpoint)
    try:
        if checkpoint == UNTRAINED:
            from rl.config import load_base_env_config, load_rl_config
            from rl.mappo import ValueNorm
            from rl.networks import ActorCritic, NetworkSpec

            cfg = load_rl_config(str(REPO_ROOT / "configs/rl/default.yaml"), {})
            env = env_config or load_base_env_config(cfg, root=REPO_ROOT)
            torch.manual_seed(seed)
            model = ActorCritic(NetworkSpec.from_env(env), cfg.model)
            strategy = RLStrategy(model, cfg, env, ValueNorm(cfg.ppo.value_normalization),
                                  deterministic=deterministic, seed=seed)
        else:
            strategy = RLStrategy.from_checkpoint(checkpoint, env_config, deterministic=deterministic, seed=seed)
    except (CheckpointError, ValueError) as e:
        raise RuntimeError(f"The RL checkpoint does not fit this game: {e}\n"
                           f"The RL agent is trained for configs/ctf_v2.yaml: use --config configs/ctf_v2.yaml.") from e
    if env_config is not None and len(strategy.roles) != env_config.agents_per_team:
        raise RuntimeError(f"The RL checkpoint was trained for {len(strategy.roles)} agents per team but this game "
                           f"has {env_config.agents_per_team}. Use --config configs/ctf_v2.yaml (the RL agent's rules).")
    return strategy


class RLControllerAdapter(Controller):
    """Mixed team: LLM orders for the agents that have one, the RL policy for everybody else."""

    def __init__(self, checkpoint: str | None = None, policy=None, env_config=None) -> None:
        super().__init__()
        self.policy = policy or make_rl_strategy(checkpoint, env_config)
        self.observation = None
        self.team = None

    def reset(self) -> None:
        super().reset()
        self.policy.reset()

    def roles_for(self, agent_ids: list[str]):
        """Roles of the agents as the RL policy sees them (its slot order), so both agree on who is who."""
        slots = getattr(self.policy, "roles", None)
        ordered = sorted(agent_ids, key=lambda a: int("".join(c for c in a if c.isdigit()) or 0))
        if not slots or len(slots) != len(ordered):
            return None
        return {aid: getattr(role, "value", str(role)) for aid, role in zip(ordered, slots)}

    def attach(self, observation, team, roles) -> None:
        self.observation, self.team = observation, team

    def actions(self, view) -> dict[str, Action]:
        walking = super().actions(view)                          # shortest-path moves for ordered agents
        learned = self.policy.act(self.observation, self.team)   # the RL policy's move for every agent
        return {a.id: (walking[a.id] if a.id in self.orders else learned.get(a.id, Action.STAY))
                for a in view.agents if a.active}
