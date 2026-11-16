"""Map files (configs/maps/*.yaml) used as training pools and held-out evaluation maps.

A map file contributes the geometry only: walls, bases and spawn cells (``MAP_KEYS``). Rules, visibility,
spawn mode, episode length and everything else stay what the base environment config says, so two maps
differ in the layout and nothing else.

Training maps must be

* the same size (width, height, agents per team) as the base environment, because the network shape
  contains the board size;
* left-right mirror symmetric (walls and bases), because BLUE's view is mirrored so one policy can play
  both sides;
* connected (a path from the RED base to the BLUE base), otherwise no flag could ever be captured.
"""

from __future__ import annotations

import os
from pathlib import Path

from ctf.config import CTFConfig, config_from_dict, config_to_dict
from rl.navigation import GridNav
from rl.schema import ConfigError, read_yaml

# Keys a map file contributes. Everything else (rules, visibility, spawn mode, episode length, carrier
# slots, ...) stays what the policy's own environment config says.
MAP_KEYS = ("obstacles", "red_base", "blue_base", "red_spawns", "blue_spawns")


def map_env_config(base: CTFConfig, map_path: str | Path, *, what: str = "map") -> CTFConfig:
    """``base`` with the geometry (walls, bases, spawn cells) of the map file at ``map_path``.

    ``what`` only changes the wording of error messages (\"held-out map\", \"training map\", ...).
    """
    data = read_yaml(map_path)
    for key in ("width", "height", "agents_per_team"):
        if data.get(key) != getattr(base, key):
            raise ConfigError(
                f"{what} {map_path}: {key} is {data.get(key)!r} but the policy's environment has "
                f"{getattr(base, key)!r}; only layouts of the same size and team size can be used"
            )
    missing = [k for k in MAP_KEYS if k not in data]
    if missing:
        raise ConfigError(f"{what} {map_path}: missing key(s) {missing}")
    try:
        return config_from_dict({**config_to_dict(base), **{k: data[k] for k in MAP_KEYS}})
    except ValueError as exc:
        raise ConfigError(f"{what} {map_path} is invalid: {exc}") from exc


def layout_key(config: CTFConfig) -> tuple:
    """Walls and bases of a map: two maps with the same key are the same layout, whatever their files are called."""
    return (frozenset(tuple(c) for c in config.obstacles), tuple(config.red_base), tuple(config.blue_base))


def mirror_cell(config: CTFConfig, cell: tuple[int, int]) -> tuple[int, int]:
    """The cell ``cell`` flipped left-right on the board of ``config``."""
    return (config.width - 1 - cell[0], cell[1])


def symmetry_problem(config: CTFConfig) -> str | None:
    """Why the map is not left-right mirror symmetric (so BLUE's view cannot be flipped onto RED's), or None."""
    walls = {tuple(c) for c in config.obstacles}
    mirrored = {mirror_cell(config, c) for c in walls}
    if mirrored != walls:
        diff = sorted(mirrored ^ walls)
        return f"walls are not mirror symmetric (cells without a mirrored partner: {diff})"
    if mirror_cell(config, tuple(config.red_base)) != tuple(config.blue_base):
        return f"blue_base {tuple(config.blue_base)} is not the mirror of red_base {tuple(config.red_base)}"
    return None


def is_mirror_symmetric(config: CTFConfig) -> bool:
    """Is the map left-right symmetric (walls and bases)?"""
    return symmetry_problem(config) is None


def bases_connected(config: CTFConfig) -> bool:
    """Is there a walkable path between the two bases (static walls only)?"""
    walls = {tuple(c) for c in config.obstacles}
    free = [[(x, y) not in walls for x in range(config.width)] for y in range(config.height)]
    nav = GridNav(free)
    return nav.distance(tuple(config.red_base), tuple(config.blue_base)) < nav.unreachable


def training_map_env_config(base: CTFConfig, map_path: str | Path) -> CTFConfig:
    """Like :func:`map_env_config`, plus the checks a TRAINING map must pass (symmetry, connected bases)."""
    config = map_env_config(base, map_path, what="training map")
    problem = symmetry_problem(config)
    if problem is not None:
        raise ConfigError(f"training map {map_path}: {problem}; the policy plays both sides through a mirrored view")
    if not bases_connected(config):
        raise ConfigError(f"training map {map_path}: there is no path between the two bases")
    return config


def same_file(a: str | Path, b: str | Path) -> bool:
    """Do two paths name the same file (existing files by identity, otherwise by resolved path)?"""
    pa, pb = Path(a), Path(b)
    if pa.exists() and pb.exists():
        return os.path.samefile(pa, pb)
    return pa.resolve() == pb.resolve()
