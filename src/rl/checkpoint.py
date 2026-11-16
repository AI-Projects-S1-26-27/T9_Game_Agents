"""Checkpoint saving/loading.

A checkpoint is self-contained: weights, optimizer, value normalizer, curriculum
state, counters, the resolved configs, the git commit, and a record of the
observation layout. Loading refuses a checkpoint whose observation layout does not
match the current code, instead of silently feeding the network scrambled inputs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from ctf.config import CTFConfig, config_from_dict, config_to_dict
from rl.canonical import ACTIONS
from rl.config import RLConfig, config_to_plain
from rl.curriculum import Curriculum
from rl.features import (
    ACTOR_CHANNELS,
    ACTOR_VECTOR,
    CRITIC_CHANNELS,
    CRITIC_VECTOR,
    ROLE_ORDER,
    ObservationLayout,
)
from rl.mappo import ValueNorm
from rl.networks import ActorCritic, NetworkSpec
from rl.schema import from_dict

FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    pass


def observation_spec() -> dict[str, list[str]]:
    return {
        "actor_channels": list(ACTOR_CHANNELS),
        "actor_vector": list(ACTOR_VECTOR),
        "critic_channels": list(CRITIC_CHANNELS),
        "critic_vector": list(CRITIC_VECTOR),
        "roles": [r.value for r in ROLE_ORDER],
        "actions": [a.name for a in ACTIONS],
    }


def git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def save_checkpoint(
    path: Path,
    *,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    value_norm: ValueNorm,
    curriculum: Curriculum,
    counters: dict[str, int],
    config: RLConfig,
    env_config: CTFConfig,
) -> None:
    payload = {
        "format_version": FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "value_norm": value_norm.state_dict(),
        "curriculum": curriculum.state_dict(),
        "counters": dict(counters),
        "torch_rng": torch.get_rng_state(),
        "config": config_to_plain(config),
        "env_config": config_to_dict(env_config),
        "obs_spec": observation_spec(),
        "git_commit": git_commit(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written checkpoint


# Settings added after the first checkpoints were written, with the value that reproduces the old
# behaviour. Applied to a checkpoint's stored config when the key is missing, so old checkpoints
# keep loading while the YAML schema still requires every key. Entries are (section.key, value).
# Opponent labels (= the opponent's kind) and scripted-opponent params are handled in migrate_config.
CONFIG_COMPAT: tuple[tuple[str, Any], ...] = (
    ("rewards.outcome_share_attacker", 1.0),
    ("rewards.outcome_share_defender", 1.0),
    ("rewards.outcome_share_scout", 1.0),
    ("rewards.escort_bonus", 0.0),
    ("rewards.escort_radius", 0),
    ("rewards.escort_roles", []),
    ("rewards.defender_intercept_potential", 0.0),
    ("rewards.scout_floor_fraction", 0.0),
)
# Whole sections added later: checkpoints written before them get the "off" value.
SECTION_COMPAT: dict[str, dict[str, Any]] = {
    "noise_free_eval": {
        "enabled": False, "every_steps": 1, "at_end": False, "config_path": "configs/rl/eval_train.yaml", "overrides": {},
    },
    "league": {"snapshot_every_steps": 0, "keep_snapshots": 0, "cache_size": 4},
}
SCRIPTED_COMPAT: tuple[tuple[str, Any], ...] = (("chase_radius", 0), ("escort", False))


def migrate_config(plain: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a stored RL config dict with settings added since it was written filled in."""
    import copy

    cfg = copy.deepcopy(plain)
    for section, default in SECTION_COMPAT.items():
        cfg.setdefault(section, copy.deepcopy(default))
    for path, value in CONFIG_COMPAT:
        section, key = path.split(".")
        if isinstance(cfg.get(section), dict):
            cfg[section].setdefault(key, value)
    for phase in cfg.get("curriculum", {}).get("phases", []):
        used: dict[str, int] = {}
        for opp in phase.get("opponents", []):
            if "label" not in opp:  # old runs lumped same-kind opponents: name them scripted, scripted_2, ...
                used[opp["kind"]] = used.get(opp["kind"], 0) + 1
                n = used[opp["kind"]]
                opp["label"] = opp["kind"] if n == 1 else f"{opp['kind']}_{n}"
            if opp["kind"] == "scripted":
                for key, value in SCRIPTED_COMPAT:
                    opp["params"].setdefault(key, value)
    return cfg


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu", *, exact_layout: bool = True
) -> dict[str, Any]:
    """Read a checkpoint. ``exact_layout=False`` also accepts one trained on fewer features.

    Resuming training needs the exact layout (the network shapes must match). Playing a policy
    does not: features added later are simply not shown to an older network.
    """
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(
            f"{path}: unsupported checkpoint format {payload.get('format_version')!r} "
            f"(this code reads format {FORMAT_VERSION})"
        )
    saved, current = payload["obs_spec"], observation_spec()
    if saved != current:
        diff = sorted(k for k in current if saved.get(k) != current[k])
        if exact_layout or any(k in diff for k in ("roles", "actions")):
            raise CheckpointError(
                f"{path}: observation layout differs from the current code in {diff}; "
                "this checkpoint was trained with a different feature set"
            )
        try:  # an older, smaller feature set is fine for inference
            ObservationLayout.from_spec(saved)
        except ValueError as exc:
            raise CheckpointError(f"{path}: {exc}; this checkpoint was trained with a different feature set") from exc
    return payload


def restore_training_state(
    payload: dict[str, Any],
    *,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    value_norm: ValueNorm,
    curriculum: Curriculum,
) -> dict[str, int]:
    if payload.get("weights_only"):
        raise CheckpointError(
            "this file is a league snapshot (weights only: no optimizer or curriculum state), so training cannot "
            "resume from it; resume from a file in <run>/checkpoints/ instead"
        )
    try:
        model.load_state_dict(payload["model"])
    except RuntimeError as exc:
        raise CheckpointError(
            "checkpoint weights do not fit the configured model (did model.* change?)"
        ) from exc
    optimizer.load_state_dict(payload["optimizer"])
    value_norm.load_state_dict(payload["value_norm"])
    curriculum.load_state_dict(payload["curriculum"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    return {k: int(v) for k, v in payload["counters"].items()}


def load_policy(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[ActorCritic, dict[str, Any]]:
    """Rebuild the trained network for inference (no training state needed)."""
    payload = load_checkpoint(path, map_location="cpu", exact_layout=False)
    config = from_dict(RLConfig, migrate_config(payload["config"]))
    env_config = config_from_dict(payload["env_config"])
    layout = ObservationLayout.from_spec(payload["obs_spec"])
    model = ActorCritic(NetworkSpec.from_env(env_config, layout), config.model)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    meta = {
        "layout": layout,
        "config": config,
        "env_config": env_config,
        "counters": payload["counters"],
        "git_commit": payload["git_commit"],
        "value_norm": payload["value_norm"],
    }
    return model, meta


class CheckpointManager:
    """Writes periodic checkpoints (pruned), a ``latest.pt`` copy, and permanent phase snapshots."""

    def __init__(self, run_dir: Path, keep: int) -> None:
        self.dir = run_dir / "checkpoints"
        self.keep = keep

    @property
    def latest(self) -> Path:
        return self.dir / "latest.pt"

    def save_periodic(self, step: int, **state: Any) -> Path:
        path = self.dir / f"step_{step:010d}.pt"
        save_checkpoint(path, **state)
        tmp = self.latest.with_suffix(".pt.tmp")
        shutil.copyfile(path, tmp)
        os.replace(tmp, self.latest)
        self._prune()
        return path

    def save_phase_start(self, index: int, name: str, **state: Any) -> Path:
        path = self.dir / f"phase_{index}_{name}_start.pt"
        save_checkpoint(path, **state)
        return path

    def _prune(self) -> None:
        if self.keep <= 0:
            return
        periodic = sorted(self.dir.glob("step_*.pt"))
        for old in periodic[: max(len(periodic) - self.keep, 0)]:
            old.unlink()
