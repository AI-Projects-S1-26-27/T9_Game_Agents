"""League training: frozen past policies as opponents.

Two kinds of source for the ``checkpoint`` opponent (``CheckpointParams``):

``source: league``  the trainer saves a small weights-only *snapshot* every
                    ``league.snapshot_every_steps`` environment steps into ``<run>/league/``. Each
                    episode draws one snapshot from the newest ``window`` ones (``0`` = all).
                    While the league is still empty the entry is skipped and the phase's other
                    opponents are used (so every phase needs at least one non-league opponent).
``source: file``    always plays one fixed checkpoint or snapshot (``path``): "new model against the
                    old model", or a fixed strong sparring partner.

Snapshots have the same layout as normal checkpoints minus optimizer and curriculum state, so they
play in the viewer and in ``rl.evaluate`` like any checkpoint. They cannot be resumed from.

Worker processes load policies on the CPU and keep the newest ones in a small cache
(``league.cache_size``). Torch is imported only when a checkpoint opponent is actually built.
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any

from ctf.config import CTFConfig
from rl.schema import ConfigError, from_dict

if TYPE_CHECKING:  # pragma: no cover
    from rl.config import OpponentSpec, RLConfig
    from strategies import BaseStrategy

SNAPSHOT_NAME = "snapshot_{step:010d}.pt"
_SNAPSHOT_RE = re.compile(r"snapshot_(\d+)\.pt")
_DRAW_ATTEMPTS = 5  # a snapshot can be pruned between listing the folder and loading it


class LeagueSource(str, Enum):
    LEAGUE = "league"
    FILE = "file"


@dataclass(frozen=True)
class CheckpointParams:
    source: LeagueSource
    window: int  # league: draw from the newest `window` snapshots (0 = all); file: must be 0
    path: str  # file: the checkpoint / snapshot to play; league: must be ""
    deterministic: bool  # true = the frozen policy plays argmax actions, false = it samples


def parse_params(params: dict[str, Any], where: str = "opponent[checkpoint].params") -> CheckpointParams:
    p = from_dict(CheckpointParams, params, where)
    if p.window < 0:
        raise ConfigError(f"{where}: window must be >= 0 (0 = every snapshot)")
    if p.source is LeagueSource.FILE:
        if not p.path:
            raise ConfigError(f"{where}: source file needs a path")
        if p.window != 0:
            raise ConfigError(f"{where}: window only applies to source league, use 0 with source file")
    elif p.path:
        raise ConfigError(f"{where}: source league takes no path (snapshots come from the run's league folder)")
    return p


def is_league_spec(spec: "OpponentSpec") -> bool:
    return spec.kind == "checkpoint" and spec.params.get("source") == LeagueSource.LEAGUE.value


def uses_league_source(cfg: "RLConfig") -> bool:
    return any(is_league_spec(o) for ph in cfg.curriculum.phases for o in ph.opponents)


def validate_league_settings(cfg: "RLConfig") -> None:
    """Config-load checks for ``checkpoint`` opponents (called from ``rl.config``)."""
    league_used = False
    for i, phase in enumerate(cfg.curriculum.phases):
        loc = f"curriculum.phases[{i}] ({phase.name})"
        for opp in phase.opponents:
            if opp.kind == "checkpoint":
                parse_params(opp.params, f"{loc}: opponent {opp.label!r} params")
        if any(is_league_spec(o) for o in phase.opponents):
            league_used = True
            if all(is_league_spec(o) for o in phase.opponents):
                raise ConfigError(
                    f"{loc}: needs at least one opponent that is not a league opponent "
                    "(the league is empty at the start, and a snapshot can vanish)"
                )
    lg = cfg.league
    if lg.snapshot_every_steps < 0 or lg.keep_snapshots < 0:
        raise ConfigError("league.snapshot_every_steps and league.keep_snapshots must be >= 0")
    if lg.cache_size < 1:
        raise ConfigError("league.cache_size must be >= 1")
    if league_used and lg.snapshot_every_steps < 1:
        raise ConfigError("a phase has a league opponent but league.snapshot_every_steps is 0: the league would stay empty")


def validate_file_opponents(cfg: "RLConfig") -> None:
    """Fail before training starts if a ``source: file`` opponent names a file that does not exist."""
    for phase in cfg.curriculum.phases:
        for opp in phase.opponents:
            if opp.kind != "checkpoint":
                continue
            p = parse_params(opp.params)
            if p.source is LeagueSource.FILE and not Path(p.path).exists():
                raise ConfigError(f"phase {phase.name!r} opponent {opp.label!r}: checkpoint file not found: {p.path}")


# ----------------------------------------------------------------------------- loading


class PolicyCache:
    """Small LRU cache of loaded policies (CPU). The key includes the file's modification time."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._items: OrderedDict[tuple[str, int], tuple[Any, dict[str, Any]]] = OrderedDict()

    def get(self, path: Path) -> tuple[Any, dict[str, Any]]:
        from rl.checkpoint import load_policy

        stat = os.stat(path)  # FileNotFoundError propagates: callers decide what a vanished file means
        key = (str(path), stat.st_mtime_ns)
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        policy = load_policy(path, "cpu")
        for old in [k for k in self._items if k[0] == key[0]]:  # a rewritten file replaces its old entry
            del self._items[old]
        self._items[key] = policy
        while len(self._items) > self.size:
            self._items.popitem(last=False)
        return policy


class LeagueStore:
    """The snapshots in one folder, with a policy cache. Read-only: only the trainer writes."""

    def __init__(self, directory: Path, cache_size: int) -> None:
        self.directory = Path(directory)
        self.cache = PolicyCache(cache_size)

    def snapshots(self) -> list[tuple[int, Path]]:
        """``(step, path)`` of every snapshot, oldest first (half-written ``.tmp`` files are ignored)."""
        try:
            names = os.listdir(self.directory)
        except FileNotFoundError:
            return []
        found = []
        for name in names:
            match = _SNAPSHOT_RE.fullmatch(name)
            if match:
                found.append((int(match.group(1)), self.directory / name))
        return sorted(found)

    def draw(self, window: int, rng: Random) -> tuple[int, Any, dict[str, Any]] | None:
        """``(step, model, meta)`` of a random snapshot among the newest ``window`` (0 = all), or None if
        the league is empty. A snapshot pruned while being drawn is simply drawn again."""
        for _ in range(_DRAW_ATTEMPTS):
            snaps = self.snapshots()
            if not snaps:
                return None
            if window > 0:
                snaps = snaps[-window:]
            step, path = rng.choice(snaps)
            try:
                model, meta = self.cache.get(path)
            except FileNotFoundError:
                continue
            return step, model, meta
        return None


_STORES: dict[str, LeagueStore] = {}
_FILE_CACHE = PolicyCache(2)


def get_store(directory: Path, cache_size: int) -> LeagueStore:
    """One store (and so one policy cache) per folder and process, shared by all envs of a worker."""
    key = str(Path(directory).resolve())
    store = _STORES.get(key)
    if store is None or store.cache.size != cache_size:
        store = _STORES[key] = LeagueStore(Path(directory), cache_size)
    return store


def make_policy_opponent(
    model: Any, meta: dict[str, Any], env_config: CTFConfig, *, deterministic: bool, seed: int
) -> "BaseStrategy":
    """A loaded policy as a team controller for ``env_config`` (a clear ConfigError if it does not fit)."""
    from rl.mappo import ValueNorm
    from rl.rl_strategy import RLStrategy

    config = meta["config"]
    if len(config.roles.slots) != env_config.agents_per_team:
        raise ConfigError(
            f"checkpoint opponent was trained with {len(config.roles.slots)} agents per team "
            f"but this environment has {env_config.agents_per_team}"
        )
    value_norm = ValueNorm(config.ppo.value_normalization)
    value_norm.load_state_dict(meta["value_norm"])
    try:
        return RLStrategy(
            model, config, env_config, value_norm,
            deterministic=deterministic, checkpoint_step=int(meta["counters"]["total_steps"]),
            seed=seed, layout=meta["layout"],
        )
    except ValueError as exc:
        raise ConfigError(f"checkpoint opponent does not fit this environment: {exc}") from exc


def build_checkpoint_opponent(
    params: dict[str, Any], env_config: CTFConfig, rng: Random, league: LeagueStore | None
) -> tuple["BaseStrategy", int | None]:
    """Build a ``checkpoint`` opponent; returns it with the snapshot's step (None for ``source: file``)."""
    p = parse_params(params)
    seed = rng.randrange(2**31)
    if p.source is LeagueSource.FILE:
        try:
            model, meta = _FILE_CACHE.get(Path(p.path))
        except FileNotFoundError as exc:
            raise ConfigError(f"checkpoint opponent: file not found: {p.path}") from exc
        return make_policy_opponent(model, meta, env_config, deterministic=p.deterministic, seed=seed), None
    if league is None:
        raise ConfigError("source league needs a training run (it plays the run's own snapshots)")
    drawn = league.draw(p.window, rng)
    if drawn is None:
        raise ConfigError("source league: the league is empty (no snapshots yet)")
    step, model, meta = drawn
    return make_policy_opponent(model, meta, env_config, deterministic=p.deterministic, seed=seed), step


# ----------------------------------------------------------------------------- writing


class LeagueWriter:
    """Writes snapshots into ``directory`` and keeps only the newest ``keep`` (0 = keep all)."""

    def __init__(self, directory: Path, keep: int) -> None:
        self.directory = Path(directory)
        self.keep = keep

    def save(self, step: int, **state: Any) -> Path:
        """``state`` as for ``save_checkpoint``: model, value_norm, counters, config, env_config (the optimizer and
        curriculum are not needed and not stored)."""
        import torch

        from ctf.config import config_to_dict
        from rl.checkpoint import FORMAT_VERSION, git_commit, observation_spec
        from rl.config import config_to_plain

        payload = {
            "format_version": FORMAT_VERSION,
            "weights_only": True,  # a snapshot: cannot be resumed from
            "model": {k: v.detach().cpu() for k, v in state["model"].state_dict().items()},
            "value_norm": {k: v.detach().cpu() for k, v in state["value_norm"].state_dict().items()},
            "counters": dict(state["counters"]),
            "config": config_to_plain(state["config"]),
            "env_config": config_to_dict(state["env_config"]),
            "obs_spec": observation_spec(),
            "git_commit": git_commit(),
        }
        path = self.directory / SNAPSHOT_NAME.format(step=step)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomic: a worker never sees a half-written snapshot
        self._prune()
        return path

    def _prune(self) -> None:
        if self.keep <= 0:
            return
        snaps = LeagueStore(self.directory, 1).snapshots()
        for _, old in snaps[: max(len(snaps) - self.keep, 0)]:
            try:
                old.unlink()
            except FileNotFoundError:
                pass
