"""Shared helpers for the league tests (not collected as tests)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ctf.config import CTFConfig
from rl.config import LeagueConfig, OpponentSpec, RLConfig, load_base_env_config
from rl.league import LeagueWriter, SNAPSHOT_NAME
from rl.mappo import ValueNorm
from rl.networks import ActorCritic, NetworkSpec
from rl_helpers import ROOT, tiny_config

LEAGUE_PARAMS = {"source": "league", "window": 0, "path": "", "deterministic": False}


def league_spec(label: str = "league", weight: float = 3.0, **params) -> OpponentSpec:
    return OpponentSpec("checkpoint", label, weight, {**LEAGUE_PARAMS, **params})


def file_spec(path: Path | str, label: str = "frozen", weight: float = 1.0, deterministic: bool = True) -> OpponentSpec:
    return OpponentSpec(
        "checkpoint", label, weight, {"source": "file", "window": 0, "path": str(path), "deterministic": deterministic}
    )


def league_config(
    tmp_path,
    *,
    opponents: list[OpponentSpec] | None = None,
    snapshot_every: int = 16,
    keep: int = 0,
    cache_size: int = 2,
    **tiny_kwargs,
) -> RLConfig:
    """Tiny 3-agent config whose single phase mixes `stay` with league opponents."""
    cfg = tiny_config(str(tmp_path), **tiny_kwargs)
    chosen = opponents or [OpponentSpec("stay", "stay", 1.0, {}), league_spec()]
    phase = replace(cfg.curriculum.phases[0], opponents=chosen, opponent_noise=0.0)
    return replace(
        cfg,
        league=LeagueConfig(snapshot_every_steps=snapshot_every, keep_snapshots=keep, cache_size=cache_size),
        curriculum=replace(cfg.curriculum, phases=[phase]),
    )


def env_config_for(cfg: RLConfig) -> CTFConfig:
    return load_base_env_config(cfg, ROOT)


def write_snapshot(directory: Path, step: int, cfg: RLConfig, *, keep: int = 0) -> Path:
    """Write a snapshot of a freshly initialised network, the way the trainer does."""
    env_config = env_config_for(cfg)
    model = ActorCritic(NetworkSpec.from_env(env_config), cfg.model)
    value_norm = ValueNorm(cfg.ppo.value_normalization)
    return LeagueWriter(directory, keep).save(
        step, model=model, value_norm=value_norm, counters={"total_steps": step, "update": step // 16},
        config=cfg, env_config=env_config,
    )


__all__ = ["league_spec", "file_spec", "league_config", "env_config_for", "write_snapshot", "SNAPSHOT_NAME", "ROOT"]
