"""Episode recordings: a self-contained, viewer-independent description of one episode.

Frame ``t`` holds the state at step ``t``. ``events`` / ``reward`` describe what happened on the
way to this frame; ``action`` / ``probs`` / ``value`` describe the decision made *from* this frame.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl.schema import ConfigError, from_dict, to_dict

FORMAT_VERSION = 1
MEANINGFUL_EVENTS = ("flag_pickup", "tag", "flag_capture")  # shown on timelines and in event logs


@dataclass(frozen=True)
class AgentFrame:
    id: str
    team: str  # "RED" | "BLUE"
    slot: int
    role: str  # "attacker" | "defender" | "scout" | "unknown"
    x: int
    y: int
    active: bool
    carrying: bool
    action: str  # chosen action name ("" when none)
    probs: list[float]  # UP, DOWN, LEFT, RIGHT, STAY (board orientation); [] if unknown
    value: float | None
    reward: float | None  # reward received on the way to this frame
    components: dict[str, float]  # that reward, split by component


@dataclass(frozen=True)
class FlagFrame:
    team: str
    status: str  # "AT_BASE" | "CARRIED"
    x: int
    y: int
    carrier: str  # agent id or ""


@dataclass(frozen=True)
class Frame:
    t: int
    agents: list[AgentFrame]
    flags: list[FlagFrame]
    events: list[dict]
    belief: dict[str, Any]  # team -> {"enemy_prob": HxW, "observed": HxW, "last_seen": HxW}
    winner: str  # "" if none
    done: bool


@dataclass(frozen=True)
class RecordingMeta:
    width: int
    height: int
    obstacles: list[list[int]]
    red_base: list[int]
    blue_base: list[int]
    red_spawns: list[list[int]]
    blue_spawns: list[list[int]]
    visibility_radius: int
    max_steps: int
    teams: dict[str, str]  # team -> controller label
    opponent: str
    seed: int
    note: str


@dataclass(frozen=True)
class Recording:
    format_version: int
    meta: RecordingMeta
    frames: list[Frame]

    @property
    def outcome(self) -> str:
        last = self.frames[-1]
        if not last.done:
            return "unfinished"
        return f"{last.winner} wins" if last.winner else "draw"

    def event_steps(self) -> list[tuple[int, str]]:
        """(frame index, event type) for every event, used for timeline markers."""
        return [(i, e["type"]) for i, f in enumerate(self.frames) for e in f.events if e["type"] in MEANINGFUL_EVENTS]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(to_dict(self), separators=(",", ":"))
        if path.suffix == ".gz":
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(text)
        else:
            path.write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Recording":
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
            data = json.load(f)
        if data.get("format_version") != FORMAT_VERSION:
            raise ConfigError(f"{path}: unsupported recording format {data.get('format_version')!r}")
        return from_dict(cls, data)
