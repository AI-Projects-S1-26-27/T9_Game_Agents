"""Visual theme, layout, playback and key bindings (configs/viz/default.yaml). Strict and fully required."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from rl.schema import ConfigError, apply_overrides, from_dict, read_yaml
from viz.commands import COMMANDS

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def rgb(color: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (r, g, b)."""
    if not _HEX.match(color):
        raise ConfigError(f"not a #rrggbb colour: {color!r}")
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def _check_colors(owner: str, **colors: str) -> None:
    for name, value in colors.items():
        if not _HEX.match(value):
            raise ConfigError(f"{owner}.{name}: {value!r} is not a #rrggbb colour")


class ViewTeam(str, Enum):
    RED = "red"
    BLUE = "blue"
    NONE = "none"


@dataclass(frozen=True)
class WindowStyle:
    width: int
    height: int
    fps: int
    title: str
    resizable: bool


@dataclass(frozen=True)
class LayoutStyle:
    margin: int
    panel_width: int
    header_height: int
    timeline_height: int


@dataclass(frozen=True)
class FontStyle:
    family: str
    title_size: int
    label_size: int
    small_size: int
    tiny_size: int


@dataclass(frozen=True)
class Palette:
    background: str
    panel: str
    panel_alt: str
    border: str
    text: str
    text_dim: str
    accent: str
    grid_line: str
    floor: str
    floor_alt: str
    wall: str
    wall_edge: str
    win: str
    loss: str
    draw: str
    warn: str
    fog: str

    def __post_init__(self) -> None:
        _check_colors("palette", **{k: getattr(self, k) for k in self.__dataclass_fields__})


@dataclass(frozen=True)
class TeamStyle:
    name: str
    color: str
    dark: str
    light: str

    def __post_init__(self) -> None:
        _check_colors("team", color=self.color, dark=self.dark, light=self.light)


@dataclass(frozen=True)
class TeamsStyle:
    red: TeamStyle
    blue: TeamStyle


@dataclass(frozen=True)
class RoleStyle:
    label: str
    letter: str
    color: str

    def __post_init__(self) -> None:
        _check_colors("role", color=self.color)
        if len(self.letter) != 1:
            raise ConfigError(f"role letter must be one character, got {self.letter!r}")


@dataclass(frozen=True)
class RolesStyle:
    attacker: RoleStyle
    defender: RoleStyle
    scout: RoleStyle
    unknown: RoleStyle  # agents whose controller has no roles (random, stay, ...)


@dataclass(frozen=True)
class BoardStyle:
    agent_radius: float  # fraction of a cell
    ring_width: int
    flag_scale: float
    trail_length: int
    trail_alpha: int
    fog_unseen_alpha: int
    fog_memory_alpha: int
    belief_max_alpha: int
    vision_alpha: int
    arrow_scale: float
    event_flash_seconds: float
    min_label_cell: int  # smallest cell size (px) at which full role labels are drawn


@dataclass(frozen=True)
class OverlayDefaults:
    fog: bool
    belief: bool
    vision: bool
    trails: bool
    probabilities: bool
    arrows: bool
    labels: bool
    coordinates: bool
    spawns: bool
    view_team: ViewTeam


@dataclass(frozen=True)
class PlaybackStyle:
    speeds: list[float]  # environment steps per second
    default_speed_index: int
    autoloop: bool
    live_start_playing: bool


@dataclass(frozen=True)
class PathsStyle:
    output_dir: str
    maps_dir: str
    controllers_file: str
    default_map: str


@dataclass(frozen=True)
class ReportStyle:
    smoothing_window: int
    figure_width: float
    figure_height: float
    dpi: int
    tail_updates: int


@dataclass(frozen=True)
class TimelapseStyle:
    panel_width: int
    panel_height: int
    fps: int
    hold_final_frames: int
    columns: int
    title_height: int


@dataclass(frozen=True)
class ExportStyle:
    fps: int
    width: int
    height: int


@dataclass(frozen=True)
class EditorStyle:
    min_size: int  # smallest map side the map maker allows
    max_size: int  # largest map side the map maker allows


@dataclass(frozen=True)
class VizConfig:
    window: WindowStyle
    layout: LayoutStyle
    fonts: FontStyle
    palette: Palette
    teams: TeamsStyle
    roles: RolesStyle
    board: BoardStyle
    overlays: OverlayDefaults
    playback: PlaybackStyle
    paths: PathsStyle
    report: ReportStyle
    timelapse: TimelapseStyle
    export: ExportStyle
    editor: EditorStyle
    keys: dict[str, str]  # command -> pygame key name

    def __post_init__(self) -> None:
        missing = [c for c in COMMANDS if c not in self.keys]
        unknown = [k for k in self.keys if k not in COMMANDS]
        if missing or unknown:
            raise ConfigError(f"keys: missing commands {missing}, unknown commands {unknown}")
        names = list(self.keys.values())
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ConfigError(f"keys: the same key is bound to several commands: {dupes}")
        speeds = self.playback.speeds
        if not speeds or any(s <= 0 for s in speeds):
            raise ConfigError("playback.speeds must be a non-empty list of positive numbers")
        if not 0 <= self.playback.default_speed_index < len(speeds):
            raise ConfigError("playback.default_speed_index is out of range")
        if not 3 <= self.editor.min_size <= self.editor.max_size:
            raise ConfigError("editor.min_size must be >= 3 and <= editor.max_size")
        if self.report.smoothing_window < 1:
            raise ConfigError("report.smoothing_window must be >= 1")

    def team(self, name: str) -> TeamStyle:
        return self.teams.red if name.upper() == "RED" else self.teams.blue

    def role(self, name: str) -> RoleStyle:
        return getattr(self.roles, name if name in ("attacker", "defender", "scout") else "unknown")


def load_viz_config(path: str | Path, overrides: dict[str, Any] | None = None) -> VizConfig:
    data = read_yaml(path)
    if overrides:
        data = apply_overrides(data, overrides)
    return from_dict(VizConfig, data)
