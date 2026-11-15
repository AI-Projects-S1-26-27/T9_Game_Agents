"""Map editing model (no GUI): paint walls, bases and spawns; validate; save/load as env-config YAML.

A saved map is an ordinary environment config, so it works everywhere:
``--set env.config_path=configs/maps/my_map.yaml`` for training and evaluation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

from ctf.agent import Team
from ctf.config import CTFConfig, config_from_dict, config_to_dict, load_config

Cell = tuple[int, int]


class MapError(ValueError):
    pass


class Tool(str, Enum):
    WALL = "wall"
    ERASE = "erase"
    RED_BASE = "red_base"
    BLUE_BASE = "blue_base"
    RED_SPAWN = "red_spawn"
    BLUE_SPAWN = "blue_spawn"


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning" | "info"
    message: str


class MapEditor:
    def __init__(self, config: CTFConfig, *, min_size: int, max_size: int) -> None:
        self.template = config
        self.min_size, self.max_size = min_size, max_size
        self.width, self.height = config.width, config.height
        self.obstacles: set[Cell] = set(config.obstacles)
        self.red_base: Cell = tuple(config.red_base)  # type: ignore[assignment]
        self.blue_base: Cell = tuple(config.blue_base)  # type: ignore[assignment]
        self.spawns: dict[Team, list[Cell]] = {t: list(config.default_spawns(t)) for t in Team}
        self.mirror = False
        self.tool = Tool.WALL

    @classmethod
    def from_file(cls, path: str | Path, *, min_size: int, max_size: int) -> "MapEditor":
        return cls(load_config(path), min_size=min_size, max_size=max_size)

    # ----------------------------------------------------------------------- helpers

    @property
    def agents_per_team(self) -> int:
        return self.template.agents_per_team

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def mirrored(self, cell: Cell) -> Cell:
        return (self.width - 1 - cell[0], cell[1])

    def _is_base(self, cell: Cell) -> bool:
        return cell in (self.red_base, self.blue_base)

    def _is_spawn(self, cell: Cell) -> bool:
        return any(cell in cells for cells in self.spawns.values())

    # ---------------------------------------------------------------------- editing

    def paint(self, cell: Cell, tool: Tool | None = None) -> str:
        """Apply ``tool`` (default: the current tool) at ``cell``. Returns a message ('' = ok)."""
        tool = tool or self.tool
        if not self.in_bounds(cell):
            return ""
        message = self._apply(cell, tool, mirror=False)
        if self.mirror and not message:
            twin = self.mirrored(cell)
            if twin != cell:
                self._apply(twin, self._twin_tool(tool), mirror=True)
        return message

    @staticmethod
    def _twin_tool(tool: Tool) -> Tool:
        return {
            Tool.RED_BASE: Tool.BLUE_BASE, Tool.BLUE_BASE: Tool.RED_BASE,
            Tool.RED_SPAWN: Tool.BLUE_SPAWN, Tool.BLUE_SPAWN: Tool.RED_SPAWN,
        }.get(tool, tool)

    def _apply(self, cell: Cell, tool: Tool, *, mirror: bool) -> str:
        if tool is Tool.WALL:
            if self._is_base(cell) or self._is_spawn(cell):
                return "can't put a wall on a base or a spawn"
            self.obstacles.add(cell)
        elif tool is Tool.ERASE:
            if cell in self.obstacles:
                self.obstacles.discard(cell)
            else:
                for cells in self.spawns.values():
                    if cell in cells:
                        cells.remove(cell)
        elif tool in (Tool.RED_BASE, Tool.BLUE_BASE):
            self.obstacles.discard(cell)
            if tool is Tool.RED_BASE:
                self.red_base = cell
            else:
                self.blue_base = cell
        else:
            team = Team.RED if tool is Tool.RED_SPAWN else Team.BLUE
            cells = self.spawns[team]
            if cell in cells:
                if not mirror:
                    cells.remove(cell)
                return ""
            if cell in self.obstacles:
                return "can't spawn inside a wall"
            if len(cells) < self.agents_per_team:
                cells.append(cell)
            else:
                cells[-1] = cell
        return ""

    def clear_walls(self) -> None:
        self.obstacles.clear()

    def resize(self, dw: int, dh: int) -> str:
        width, height = self.width + dw, self.height + dh
        if not (self.min_size <= width <= self.max_size and self.min_size <= height <= self.max_size):
            return f"map sides must stay between {self.min_size} and {self.max_size}"
        self.width, self.height = width, height
        inside = lambda c: 0 <= c[0] < width and 0 <= c[1] < height  # noqa: E731
        clamp = lambda c: (min(max(c[0], 0), width - 1), min(max(c[1], 0), height - 1))  # noqa: E731
        self.obstacles = {c for c in self.obstacles if inside(c)}
        self.red_base, self.blue_base = clamp(self.red_base), clamp(self.blue_base)
        self.spawns = {t: [c for c in cells if inside(c)] for t, cells in self.spawns.items()}
        if self.mirror:
            self.symmetrize()
        return ""

    def symmetrize(self) -> None:
        """Rebuild the right half as the mirror image of the left half (keeps trained policies valid)."""
        half = (self.width - 1) // 2
        left = {c for c in self.obstacles if c[0] <= half}
        self.obstacles = left | {self.mirrored(c) for c in left}
        self.blue_base = self.mirrored(self.red_base)
        self.spawns[Team.BLUE] = [self.mirrored(c) for c in self.spawns[Team.RED]]

    def is_symmetric(self) -> bool:
        return (
            {self.mirrored(c) for c in self.obstacles} == self.obstacles
            and self.mirrored(self.red_base) == self.blue_base
            and [self.mirrored(c) for c in self.spawns[Team.RED]] == self.spawns[Team.BLUE]
        )

    # -------------------------------------------------------------------- validation

    def path_length(self) -> int | None:
        """Shortest path (in steps) from the RED to the BLUE base, or None if walled off."""
        start, goal = self.red_base, self.blue_base
        if start in self.obstacles or goal in self.obstacles:
            return None
        seen = {start: 0}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return seen[(x, y)]
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                n = (nx, ny)
                if self.in_bounds(n) and n not in self.obstacles and n not in seen:
                    seen[n] = seen[(x, y)] + 1
                    queue.append(n)
        return None

    def issues(self) -> list[Issue]:
        out: list[Issue] = []
        err = lambda m: out.append(Issue("error", m))  # noqa: E731
        warn = lambda m: out.append(Issue("warning", m))  # noqa: E731
        if self.red_base == self.blue_base:
            err("the two bases are on the same cell")
        for name, base in (("RED", self.red_base), ("BLUE", self.blue_base)):
            if base in self.obstacles:
                err(f"{name} base is inside a wall")
        for team in Team:
            cells = self.spawns[team]
            if len(cells) != self.agents_per_team:
                err(f"{team.value} needs {self.agents_per_team} spawns, has {len(cells)}")
            if len(set(cells)) != len(cells):
                err(f"{team.value} has duplicate spawns")
            for c in cells:
                if c in self.obstacles:
                    err(f"a {team.value} spawn is inside a wall")
            enemy_base = self.blue_base if team is Team.RED else self.red_base
            if enemy_base in cells:
                warn(f"a {team.value} agent spawns on the enemy flag cell")
        length = self.path_length()
        if length is None:
            err("the bases are walled off from each other")
        else:
            out.append(Issue("info", f"shortest path between the bases: {length} steps"))
        if not self.is_symmetric():
            warn("not left-right symmetric: trained policies assume a symmetric map (BLUE is mirrored)")
        return out

    @property
    def is_valid(self) -> bool:
        return not any(i.level == "error" for i in self.issues())

    # --------------------------------------------------------------------- save/load

    def to_config(self) -> CTFConfig:
        errors = [i.message for i in self.issues() if i.level == "error"]
        if errors:
            raise MapError("; ".join(errors))
        data = config_to_dict(self.template)
        data.update(
            width=self.width, height=self.height,
            obstacles=[list(c) for c in sorted(self.obstacles)],
            red_base=list(self.red_base), blue_base=list(self.blue_base),
            red_spawns=[list(c) for c in self.spawns[Team.RED]],
            blue_spawns=[list(c) for c in self.spawns[Team.BLUE]],
        )
        return config_from_dict(data)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        config = self.to_config()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False), encoding="utf-8")
        return path
