"""Drawing code (pygame). Pure functions of (theme, recording frame, view state): the viewer and the
exporters share it, so a screenshot, a GIF and the live window always look the same."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pygame

from viz.commands import DESCRIPTIONS, GROUPS
from viz.frames import AgentFrame, Frame, RecordingMeta
from viz.theme import ViewTeam, VizConfig, rgb

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0), "STAY": (0, 0)}
EVENT_LABEL = {
    "flag_pickup": "picked up the flag", "tag": "was tagged", "flag_capture": "captured the flag",
    "flag_drop": "dropped the flag", "flag_return": "sent the flag home",
}
Cell = tuple[int, int]


@dataclass
class ViewState:
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
    mode: str = "replay"  # replay | live | edit
    selected: str | None = None
    playing: bool = False
    speed_index: int = 0
    autoloop: bool = False
    show_help: bool = False
    message: str = ""
    message_until: float = 0.0
    flash_t: float = 0.0  # seconds since the current frame appeared (drives event animations)
    tool: str = "wall"
    mirror: bool = False
    issues: list = field(default_factory=list)  # map-maker validation results
    map_name: str = ""
    live_labels: dict = field(default_factory=dict)  # team -> controller label (live mode)

    def belief_team(self) -> str:
        return "BLUE" if self.view_team is ViewTeam.BLUE else "RED"


@dataclass
class Geometry:
    width: int
    height: int
    obstacles: set
    red_base: Cell
    blue_base: Cell
    red_spawns: list
    blue_spawns: list
    visibility_radius: int
    max_steps: int
    # Flag rules, shown in the side panel. Defaults = the original rules.
    carrier_slots: tuple = ()  # empty = every agent may carry
    tagging_rule: str = "any_enemy"
    drop_flag_on_tag: bool = False

    @classmethod
    def from_meta(cls, m: RecordingMeta) -> "Geometry":
        return cls(m.width, m.height, {tuple(c) for c in m.obstacles}, tuple(m.red_base), tuple(m.blue_base),
                   [tuple(c) for c in m.red_spawns], [tuple(c) for c in m.blue_spawns], m.visibility_radius, m.max_steps,
                   tuple(m.carrier_slots), m.tagging_rule, m.drop_flag_on_tag)

    def rules_text(self) -> str:
        carry = "any agent carries" if not self.carrier_slots else "only slot " + "/".join(str(s) for s in self.carrier_slots) + " carries"
        tag = "headcount tagging" if self.tagging_rule == "headcount" else "one enemy tags"
        drop = "flag drops where tagged" if self.drop_flag_on_tag else "flag returns to base"
        return f"{carry}  ·  {tag}  ·  {drop}"

    @classmethod
    def from_editor(cls, ed, radius: int, max_steps: int) -> "Geometry":
        from ctf.agent import Team
        return cls(ed.width, ed.height, set(ed.obstacles), ed.red_base, ed.blue_base,
                   list(ed.spawns[Team.RED]), list(ed.spawns[Team.BLUE]), radius, max_steps)


@dataclass
class Hotspot:
    rect: pygame.Rect
    kind: str  # "command" | "agent" | "timeline"
    payload: str


@dataclass
class Layout:
    header: pygame.Rect
    board: pygame.Rect
    panel: pygame.Rect
    timeline: pygame.Rect
    cell: int


def compute_layout(size: tuple[int, int], grid: tuple[int, int], cfg: VizConfig) -> Layout:
    w, h = size
    m = cfg.layout.margin
    header = pygame.Rect(0, 0, w, cfg.layout.header_height)
    timeline = pygame.Rect(0, h - cfg.layout.timeline_height, w, cfg.layout.timeline_height)
    panel = pygame.Rect(w - cfg.layout.panel_width, header.bottom, cfg.layout.panel_width, timeline.top - header.bottom)
    area = pygame.Rect(m, header.bottom + m, max(panel.left - 2 * m, 40), max(timeline.top - header.bottom - 2 * m, 40))
    cell = max(8, min(area.w // grid[0], area.h // grid[1]))
    board = pygame.Rect(0, 0, cell * grid[0], cell * grid[1])
    board.center = area.center
    return Layout(header, board, panel, timeline, cell)


def cell_rect(lay: Layout, x: int, y: int) -> pygame.Rect:
    return pygame.Rect(lay.board.x + x * lay.cell, lay.board.y + y * lay.cell, lay.cell, lay.cell)


def pixel_to_cell(lay: Layout, pos: tuple[int, int], grid: tuple[int, int]) -> Cell | None:
    if not lay.board.collidepoint(pos):
        return None
    x, y = (pos[0] - lay.board.x) // lay.cell, (pos[1] - lay.board.y) // lay.cell
    return (x, y) if 0 <= x < grid[0] and 0 <= y < grid[1] else None


class Fonts:
    def __init__(self, cfg: VizConfig) -> None:
        pygame.font.init()
        f = cfg.fonts
        make = lambda size, bold=False: pygame.font.SysFont(f.family, size, bold=bold)  # noqa: E731
        self.title, self.label, self.small, self.tiny = make(f.title_size, True), make(f.label_size), make(f.small_size), make(f.tiny_size)
        self.label_bold, self.small_bold, self.tiny_bold = make(f.label_size, True), make(f.small_size, True), make(f.tiny_size, True)


def _alpha(size: tuple[int, int]) -> pygame.Surface:
    return pygame.Surface(size, pygame.SRCALPHA)


def text(surface, font, s: str, pos, color, anchor: str = "topleft") -> pygame.Rect:
    img = font.render(s, True, color)
    rect = img.get_rect(**{anchor: pos})
    surface.blit(img, rect)
    return rect


def with_alpha(color: tuple[int, int, int], a: int) -> tuple[int, int, int, int]:
    return (*color, max(0, min(255, a)))


def draw_flag(surface, x: float, y: float, h: float, color, outline: bool = False) -> None:
    top = (x, y - h)
    tri = [top, (x + h * 0.75, y - h * 0.78), (x, y - h * 0.56)]
    if outline:
        pygame.draw.polygon(surface, color, tri, 2)
        pygame.draw.line(surface, color, (x, y), top, 2)
    else:
        pygame.draw.line(surface, (235, 235, 235), (x, y), top, 3)
        pygame.draw.polygon(surface, color, tri)
        pygame.draw.polygon(surface, (20, 20, 20), tri, 1)


# ------------------------------------------------------------------------------ board

def _visible_cells(frame: Frame, team: str, radius: int, w: int, h: int) -> set[Cell]:
    out: set[Cell] = set()
    for a in frame.agents:
        if a.team == team and a.active:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) <= radius and 0 <= a.x + dx < w and 0 <= a.y + dy < h:
                        out.add((a.x + dx, a.y + dy))
    return out


def _agent_centers(frame: Frame, lay: Layout, cfg: VizConfig) -> dict[str, tuple[float, float, float]]:
    groups: dict[Cell, list[AgentFrame]] = {}
    for a in frame.agents:
        groups.setdefault((a.x, a.y), []).append(a)
    out: dict[str, tuple[float, float, float]] = {}
    base_r = lay.cell * cfg.board.agent_radius
    for (x, y), members in groups.items():
        r = cell_rect(lay, x, y)
        n = len(members)
        for i, a in enumerate(sorted(members, key=lambda m: m.id)):
            if n == 1:
                out[a.id] = (r.centerx, r.centery, base_r)
            else:
                ang = 2 * math.pi * i / n - math.pi / 2
                d = lay.cell * (0.24 if n > 2 else 0.2)
                out[a.id] = (r.centerx + d * math.cos(ang), r.centery + d * math.sin(ang), base_r * (0.82 if n < 4 else 0.7))
    return out


def draw_board(surface, cfg: VizConfig, fonts: Fonts, lay: Layout, geo: Geometry, frames: list[Frame], index: int,
               view: ViewState, show_spawns: bool = False) -> dict[str, pygame.Rect]:
    """Draw the arena. Returns clickable rectangles for each visible agent."""
    pad = lay.cell // 2 + 6  # room for labels just outside the arena
    surface.set_clip(lay.board.inflate(2 * pad, 2 * pad))
    try:
        return _draw_board(surface, cfg, fonts, lay, geo, frames, index, view, show_spawns)
    finally:
        surface.set_clip(None)


def _draw_board(surface, cfg: VizConfig, fonts: Fonts, lay: Layout, geo: Geometry, frames: list[Frame], index: int,
                view: ViewState, show_spawns: bool) -> dict[str, pygame.Rect]:
    pal, cell = cfg.palette, lay.cell
    for y in range(geo.height):
        for x in range(geo.width):
            pygame.draw.rect(surface, rgb(pal.floor if (x + y) % 2 == 0 else pal.floor_alt), cell_rect(lay, x, y))
    for x in range(geo.width + 1):
        pygame.draw.line(surface, rgb(pal.grid_line), (lay.board.x + x * cell, lay.board.y), (lay.board.x + x * cell, lay.board.bottom))
    for y in range(geo.height + 1):
        pygame.draw.line(surface, rgb(pal.grid_line), (lay.board.x, lay.board.y + y * cell), (lay.board.right, lay.board.y + y * cell))

    # bases
    for team, base in (("RED", geo.red_base), ("BLUE", geo.blue_base)):
        r, st = cell_rect(lay, *base), cfg.team(team)
        pygame.draw.rect(surface, rgb(st.dark), r)
        pygame.draw.rect(surface, rgb(st.color), r, 3)
        if cell >= 44:
            text(surface, fonts.tiny_bold, f"{st.name} BASE", (r.centerx, r.bottom - 3), rgb(st.light), "midbottom")
    # walls
    for (x, y) in geo.obstacles:
        r = cell_rect(lay, x, y)
        pygame.draw.rect(surface, rgb(pal.wall), r)
        pygame.draw.rect(surface, rgb(pal.wall_edge), r, 2)
        step = max(8, cell // 5)
        for k in range(-r.h, r.w, step):
            a, b = (r.x + max(k, 0), r.y + max(-k, 0)), (r.x + min(k + r.h, r.w), r.y + min(r.h, r.w - k))
            pygame.draw.line(surface, rgb(pal.wall_edge), a, b, 1)
    if show_spawns or view.spawns:
        for team, spawns in (("RED", geo.red_spawns), ("BLUE", geo.blue_spawns)):
            st = cfg.team(team)
            for i, (x, y) in enumerate(spawns):
                r = cell_rect(lay, x, y)
                pygame.draw.circle(surface, rgb(st.light), r.center, int(cell * 0.38), 2)
                text(surface, fonts.small_bold, f"{st.name[0]}{i + 1}", (r.x + 4, r.y + 2), rgb(st.light))
    if view.coordinates:
        for y in range(geo.height):
            for x in range(geo.width):
                text(surface, fonts.tiny, f"{x},{y}", (cell_rect(lay, x, y).x + 3, cell_rect(lay, x, y).y + 2), rgb(pal.text_dim))

    clicks: dict[str, pygame.Rect] = {}
    if not frames:
        return clicks
    frame = frames[index]
    w, h = geo.width, geo.height
    teams = ["RED", "BLUE"] if view.view_team is ViewTeam.NONE else [view.view_team.value.upper()]

    # fog of war (what the viewed team can see / remembers)
    visible_all: set[Cell] = set()
    if view.fog and view.view_team is not ViewTeam.NONE:
        team = teams[0]
        vis = _visible_cells(frame, team, geo.visibility_radius, w, h)
        observed = frame.belief[team]["observed"]
        fog = _alpha(lay.board.size)
        for y in range(h):
            for x in range(w):
                if (x, y) in vis:
                    continue
                a = cfg.board.fog_memory_alpha if observed[y][x] else cfg.board.fog_unseen_alpha
                fog.fill(with_alpha(rgb(pal.fog), a), pygame.Rect(x * cell, y * cell, cell, cell))
        surface.blit(fog, lay.board.topleft)
        visible_all = vis
    # belief map
    if view.belief:
        team = view.belief_team()
        enemy = "BLUE" if team == "RED" else "RED"
        probs, last, obs_mask = frame.belief[team]["enemy_prob"], frame.belief[team]["last_seen"], frame.belief[team]["observed"]
        layer = _alpha(lay.board.size)
        ec = rgb(cfg.team(enemy).color)
        for y in range(h):
            for x in range(w):
                p = probs[y][x]
                if p > 0.05:
                    layer.fill(with_alpha(ec, int(p * cfg.board.belief_max_alpha)), pygame.Rect(x * cell, y * cell, cell, cell))
        surface.blit(layer, lay.board.topleft)
        for y in range(h):
            for x in range(w):
                if probs[y][x] >= 0.5 and last[y][x] >= 0 and cell >= 44:
                    text(surface, fonts.tiny, f"seen t={last[y][x]}", (cell_rect(lay, x, y).centerx, cell_rect(lay, x, y).y + 3), rgb(cfg.team(enemy).light), "midtop")
    # vision ranges
    if view.vision:
        layer = _alpha(lay.board.size)
        for team in teams:
            vis = _visible_cells(frame, team, geo.visibility_radius, w, h)
            for (x, y) in vis:
                layer.fill(with_alpha(rgb(cfg.team(team).color), cfg.board.vision_alpha), pygame.Rect(x * cell, y * cell, cell, cell))
        surface.blit(layer, lay.board.topleft)
        outer_clip = surface.get_clip()
        surface.set_clip(lay.board)
        for a in frame.agents:
            if a.active and a.team in teams:
                c = cell_rect(lay, a.x, a.y).center
                r = (geo.visibility_radius + 0.5) * cell
                pts = [(c[0], c[1] - r), (c[0] + r, c[1]), (c[0], c[1] + r), (c[0] - r, c[1])]
                pygame.draw.polygon(surface, rgb(cfg.team(a.team).light), pts, 1)
        surface.set_clip(outer_clip)

    def shown(a: AgentFrame) -> bool:
        if view.view_team is ViewTeam.NONE or not view.fog:
            return True
        if a.team.lower() == view.view_team.value:
            return True
        return a.active and (a.x, a.y) in visible_all

    centers = _agent_centers(frame, lay, cfg)
    # trails
    if view.trails and index > 0:
        layer = _alpha(lay.board.size)
        start = max(0, index - cfg.board.trail_length)
        for a in frame.agents:
            if not shown(a):
                continue
            pts = []
            for f in frames[start : index + 1]:
                g = next((q for q in f.agents if q.id == a.id), None)
                if g is not None:
                    r = cell_rect(lay, g.x, g.y)
                    pts.append((r.centerx - lay.board.x, r.centery - lay.board.y))
            col = rgb(cfg.team(a.team).color)
            for i in range(1, len(pts)):
                if pts[i] != pts[i - 1]:
                    al = int(cfg.board.trail_alpha * i / len(pts))
                    pygame.draw.line(layer, with_alpha(col, al), pts[i - 1], pts[i], max(2, cell // 14))
        surface.blit(layer, lay.board.topleft)

    # flags at their bases (outlined "ghost" when taken), plus a solid flag where one lies dropped
    for fl in frame.flags:
        st = cfg.team(fl.team)
        away = fl.status in ("CARRIED", "DROPPED")
        base = geo.red_base if fl.team == "RED" else geo.blue_base
        r = cell_rect(lay, *base)
        size = cell * 0.36 * cfg.board.flag_scale
        pos = (r.x + cell * 0.2, r.y + cell * 0.55)
        draw_flag(surface, *pos, size, rgb(st.light) if away else rgb(st.color), outline=away)
        if fl.status == "DROPPED" and (view.view_team is ViewTeam.NONE or not view.fog or fl.team.lower() == view.view_team.value
                                       or (fl.x, fl.y) in visible_all):
            dr = cell_rect(lay, fl.x, fl.y)
            pygame.draw.circle(surface, rgb(cfg.palette.warn), dr.center, int(cell * 0.42), 2)
            draw_flag(surface, dr.x + cell * 0.2, dr.y + cell * 0.55, size, rgb(st.color))

    # agents (selected on top)
    order = sorted(frame.agents, key=lambda a: (a.id == view.selected, a.active))
    for a in order:
        if not shown(a):
            continue
        cx, cy, rad = centers[a.id]
        _draw_agent(surface, cfg, fonts, a, cx, cy, rad, lay.cell, view)
        clicks[a.id] = pygame.Rect(cx - rad, cy - rad, rad * 2, rad * 2)

    # decision arrows / probability compass
    if view.arrows or view.probabilities:
        layer = _alpha(lay.board.size)
        for a in frame.agents:
            if not a.active or not a.action or not shown(a):
                continue
            cx, cy, rad = centers[a.id]
            cx, cy = cx - lay.board.x, cy - lay.board.y
            if view.arrows and a.action != "STAY":
                dx, dy = DELTA[a.action]
                L = cell * cfg.board.arrow_scale
                tip = (cx + dx * (rad + L), cy + dy * (rad + L))
                pygame.draw.line(layer, (255, 255, 255, 220), (cx + dx * rad, cy + dy * rad), tip, 3)
                px, py = -dy, dx
                pygame.draw.polygon(layer, (255, 255, 255, 235), [tip, (tip[0] - dx * 9 + px * 6, tip[1] - dy * 9 + py * 6), (tip[0] - dx * 9 - px * 6, tip[1] - dy * 9 - py * 6)])
            elif view.arrows:
                pygame.draw.circle(layer, (255, 255, 255, 200), (cx, cy), int(rad + 5), 2)
            if view.probabilities and a.id == view.selected and a.probs:
                for name, p in zip(ACTIONS[:4], a.probs[:4]):
                    dx, dy = DELTA[name]
                    base = (cx + dx * cell * 0.46, cy + dy * cell * 0.46)
                    size = 4 + 16 * p
                    px, py = -dy, dx
                    tri = [(base[0] + dx * size, base[1] + dy * size), (base[0] + px * size * 0.7, base[1] + py * size * 0.7), (base[0] - px * size * 0.7, base[1] - py * size * 0.7)]
                    pygame.draw.polygon(layer, (*rgb(cfg.palette.accent), int(60 + 195 * p)), tri)
        surface.blit(layer, lay.board.topleft)

    # event animations (one-step bursts)
    t = view.flash_t / max(cfg.board.event_flash_seconds, 1e-3)
    if 0 <= t <= 1:
        for ev in frame.events:
            kind = ev["type"]
            who = ev.get("agent_id")
            a = next((q for q in frame.agents if q.id == who), None)
            if a is None:
                continue
            cx, cy, rad = centers[a.id]
            col = {"flag_pickup": rgb(cfg.palette.accent), "tag": rgb(cfg.palette.loss), "flag_capture": rgb(cfg.palette.win),
                   "flag_drop": rgb(cfg.palette.warn), "flag_return": rgb(cfg.palette.draw)}.get(kind, (255, 255, 255))
            layer = _alpha(lay.board.size)
            pygame.draw.circle(layer, (*col, int(230 * (1 - t))), (cx - lay.board.x, cy - lay.board.y), int(rad + t * cell * 0.9), 4)
            surface.blit(layer, lay.board.topleft)
    return clicks


def _draw_agent(surface, cfg: VizConfig, fonts: Fonts, a: AgentFrame, cx: float, cy: float, rad: float, cell: int, view: ViewState) -> None:
    team, role = cfg.team(a.team), cfg.role(a.role)
    c = (int(cx), int(cy))
    rr = int(rad)
    if not a.active:
        pygame.draw.circle(surface, (58, 63, 71), c, rr)
        pygame.draw.circle(surface, rgb(team.dark), c, rr, 3)
        d = int(rad * 0.55)
        pygame.draw.line(surface, (190, 195, 202), (c[0] - d, c[1] - d), (c[0] + d, c[1] + d), 3)
        pygame.draw.line(surface, (190, 195, 202), (c[0] - d, c[1] + d), (c[0] + d, c[1] - d), 3)
    else:
        if a.carrying:
            pygame.draw.circle(surface, rgb(cfg.palette.accent), c, rr + 6, 3)
        pygame.draw.circle(surface, rgb(team.color), c, rr)
        pygame.draw.circle(surface, rgb(role.color), c, rr, cfg.board.ring_width)
        text(surface, fonts.label_bold if rr < 22 else fonts.title, role.letter, c, (255, 255, 255), "center")
        if a.carrying:
            other = cfg.team("BLUE" if a.team == "RED" else "RED")
            draw_flag(surface, cx + rad * 0.55, cy - rad * 0.2, rad * 1.5, rgb(other.color))
    if a.id == view.selected:
        pulse = 5 + 2.5 * math.sin(view.flash_t * 7 + 1)
        pygame.draw.circle(surface, (255, 255, 255), c, int(rr + pulse), 2)
    if view.labels:
        full = cell >= cfg.board.min_label_cell
        label = f"{a.id} · {role.label}" if full else a.id
        if not a.active:
            label += " (OUT)" if full else " x"
        font = fonts.tiny_bold if cell < 90 else fonts.small_bold
        img = font.render(label, True, (255, 255, 255))
        box = img.get_rect(midtop=(c[0], c[1] + rr + (7 if a.carrying else 3)))
        pill = _alpha((box.w + 8, box.h + 2))
        pygame.draw.rect(pill, (8, 10, 13, 185), pill.get_rect(), border_radius=5)
        pygame.draw.rect(pill, (*rgb(role.color), 230), pill.get_rect(), 1, border_radius=5)
        surface.blit(pill, (box.x - 4, box.y - 1))
        surface.blit(img, box)


# ------------------------------------------------------------------------------ chrome

def _button(surface, fonts, rect, label, active=False, cfg=None, hint="") -> None:
    pal = cfg.palette
    pygame.draw.rect(surface, rgb(pal.accent) if active else rgb(pal.panel_alt), rect, border_radius=6)
    pygame.draw.rect(surface, rgb(pal.border), rect, 1, border_radius=6)
    col = (20, 20, 20) if active else rgb(pal.text)
    text(surface, fonts.small_bold, label, (rect.x + 8, rect.centery), col, "midleft")
    if hint:
        text(surface, fonts.tiny, hint, (rect.right - 8, rect.centery), (60, 60, 60) if active else rgb(pal.text_dim), "midright")


def draw_header(surface, cfg, fonts, lay, meta: RecordingMeta | None, view: ViewState) -> None:
    pal = cfg.palette
    pygame.draw.rect(surface, rgb(pal.panel), lay.header)
    pygame.draw.line(surface, rgb(pal.border), lay.header.bottomleft, lay.header.bottomright)
    text(surface, fonts.title, "CTF Viewer", (14, lay.header.centery), rgb(pal.text), "midleft")
    x = 190
    labels = (meta.teams if meta else view.live_labels) or {}
    for team in ("RED", "BLUE"):
        st = cfg.team(team)
        if team in labels:
            pygame.draw.circle(surface, rgb(st.color), (x, lay.header.centery), 9)
            r = text(surface, fonts.label_bold, f"{st.name}: {labels[team]}", (x + 16, lay.header.centery), rgb(st.light), "midleft")
            x = r.right + 26
    mode = {"replay": "REPLAY", "live": "LIVE", "edit": "MAP MAKER"}[view.mode]
    col = {"replay": pal.text_dim, "live": pal.win, "edit": pal.warn}[view.mode]
    r = text(surface, fonts.label_bold, mode, (lay.header.right - 14, lay.header.centery), rgb(col), "midright")
    text(surface, fonts.small, "H = help", (r.left - 16, lay.header.centery), rgb(pal.text_dim), "midright")


def draw_timeline(surface, cfg, fonts, lay, frames: list[Frame], index: int, view: ViewState, total: int) -> Hotspot | None:
    pal = cfg.palette
    pygame.draw.rect(surface, rgb(pal.panel), lay.timeline)
    pygame.draw.line(surface, rgb(pal.border), lay.timeline.topleft, lay.timeline.topright)
    if not frames:
        text(surface, fonts.label, "No episode yet: press K to play on this map", lay.timeline.center, rgb(pal.text_dim), "center")
        return None
    track = pygame.Rect(120, lay.timeline.y + 26, lay.timeline.w - 120 - 150, 14)
    span = max(len(frames) - 1, 1) if frames[-1].done else max(total, len(frames) - 1, 1)
    pygame.draw.rect(surface, rgb(pal.panel_alt), track, border_radius=7)
    fill = track.copy(); fill.w = int(track.w * index / span)
    pygame.draw.rect(surface, rgb(pal.accent), fill, border_radius=7)
    colors = {"flag_pickup": pal.accent, "tag": pal.loss, "flag_capture": pal.win, "flag_return": pal.draw}
    for i, f in enumerate(frames):
        for ev in (e for e in f.events if e["type"] in colors):
            x = track.x + int(track.w * i / span)
            pygame.draw.rect(surface, rgb(colors.get(ev["type"], pal.text)), pygame.Rect(x - 2, track.y - 7, 4, track.h + 14), border_radius=2)
    kx = track.x + int(track.w * index / span)
    pygame.draw.circle(surface, (255, 255, 255), (kx, track.centery), 9)
    f = frames[index]
    text(surface, fonts.label_bold, f"t = {f.t}", (14, track.centery), rgb(pal.text), "midleft")
    text(surface, fonts.small, f"/ {span}", (78, track.centery), rgb(pal.text_dim), "midleft")
    speed = cfg.playback.speeds[view.speed_index]
    state = "PLAYING" if view.playing else "PAUSED"
    text(surface, fonts.label_bold, f"{state}   x{speed:g}", (track.right + 16, track.centery), rgb(pal.text), "midleft")
    legend_y = lay.timeline.bottom - 20
    x = 14
    for name, col in (("pickup", pal.accent), ("tag", pal.loss), ("capture", pal.win), ("return", pal.draw)):
        pygame.draw.rect(surface, rgb(col), pygame.Rect(x, legend_y - 6, 4, 14), border_radius=2)
        r = text(surface, fonts.tiny, name, (x + 9, legend_y), rgb(pal.text_dim), "midleft")
        x = r.right + 16
    text(surface, fonts.tiny, "click the bar to seek   ·   , .  jump between events", (x + 10, legend_y), rgb(pal.text_dim), "midleft")
    return Hotspot(track.inflate(0, 28), "timeline", f"{track.x},{track.w},{span}")


def _status(cfg, frame: Frame) -> tuple[str, str]:
    if not frame.done:
        return "RUNNING", cfg.palette.text_dim
    if frame.winner:
        return f"{frame.winner} WINS", cfg.team(frame.winner).light
    return "DRAW (timeout)", cfg.palette.draw


def draw_panel(surface, cfg, fonts, lay, geo: Geometry, meta: RecordingMeta | None, frames: list[Frame], index: int,
               view: ViewState, cumulative) -> list[Hotspot]:
    pal = cfg.palette
    pygame.draw.rect(surface, rgb(pal.panel), lay.panel)
    pygame.draw.line(surface, rgb(pal.border), lay.panel.topleft, lay.panel.bottomleft)
    spots: list[Hotspot] = []
    x0, x1 = lay.panel.x + 14, lay.panel.right - 14
    y = lay.panel.y + 10
    heading = lambda s: text(surface, fonts.small_bold, s.upper(), (x0, y), rgb(pal.text_dim))  # noqa: E731

    if view.mode == "edit":
        return _draw_editor_panel(surface, cfg, fonts, lay, geo, view)

    frame = frames[index] if frames else None
    if frame is None:
        text(surface, fonts.label, "No episode loaded.", (x0, y), rgb(pal.text))
        return spots
    status, scol = _status(cfg, frame)
    heading("Episode"); y += 20
    text(surface, fonts.title, f"Step {frame.t} / {geo.max_steps}", (x0, y), rgb(pal.text))
    text(surface, fonts.label_bold, status, (x1, y + 4), rgb(scol), "topright"); y += 30
    seed = f"seed {meta.seed}" if meta else ""
    vt = view.view_team.value.upper() if view.view_team is not ViewTeam.NONE else "EVERYONE"
    text(surface, fonts.small, f"{seed}   ·   vision radius {geo.visibility_radius}   ·   viewing as {vt}", (x0, y), rgb(pal.text_dim)); y += 24

    heading("Teams"); y += 18
    for team in ("RED", "BLUE"):
        st = cfg.team(team)
        mine = [a for a in frame.agents if a.team == team]
        fl = next((f for f in frame.flags if f.team == ("BLUE" if team == "RED" else "RED")), None)
        flag = "enemy flag: at its base"
        if fl and fl.status == "CARRIED":
            flag = f"enemy flag: CARRIED by {fl.carrier}"
        elif fl and fl.status == "DROPPED":
            flag = f"enemy flag: DROPPED at ({fl.x},{fl.y})"
        pygame.draw.circle(surface, rgb(st.color), (x0 + 6, y + 8), 6)
        label = (meta.teams.get(team, "") if meta else view.live_labels.get(team, ""))
        text(surface, fonts.small_bold, f"{st.name}  {label}", (x0 + 18, y), rgb(st.light)); y += 17
        text(surface, fonts.tiny, f"{sum(a.active for a in mine)}/{len(mine)} active  ·  {flag}", (x0 + 18, y), rgb(pal.text_dim)); y += 18
    text(surface, fonts.tiny, "Rules: " + geo.rules_text(), (x0, y), rgb(pal.text_dim)); y += 18
    y += 4

    heading("Players (click to select)"); y += 18
    for a in frame.agents:
        row = pygame.Rect(x0 - 4, y - 1, x1 - x0 + 8, 20)
        if a.id == view.selected:
            pygame.draw.rect(surface, rgb(pal.panel_alt), row, border_radius=4)
            pygame.draw.rect(surface, rgb(pal.accent), row, 1, border_radius=4)
        st, role = cfg.team(a.team), cfg.role(a.role)
        pygame.draw.circle(surface, rgb(st.color), (x0 + 6, y + 9), 6)
        text(surface, fonts.small_bold, a.id, (x0 + 18, y), rgb(pal.text))
        pill = pygame.Rect(x0 + 50, y + 1, 92, 17)
        pygame.draw.rect(surface, rgb(role.color), pill, border_radius=8)
        text(surface, fonts.tiny_bold, f"{role.letter} {role.label}", pill.center, (15, 15, 15), "center")
        state = "OUT" if not a.active else ("HAS FLAG" if a.carrying else "")
        text(surface, fonts.tiny_bold, state, (x0 + 150, y + 2), rgb(pal.loss if not a.active else pal.accent))
        text(surface, fonts.tiny, f"({a.x},{a.y})", (x0 + 214, y + 2), rgb(pal.text_dim))
        text(surface, fonts.tiny, a.action or "-", (x1, y + 2), rgb(pal.text), "topright")
        spots.append(Hotspot(row, "agent", a.id))
        y += 21
    y += 6

    sel = next((a for a in frame.agents if a.id == view.selected), None)
    heading("Selected player"); y += 18
    if sel is None:
        text(surface, fonts.small, "Click a player (or press N) to see its decision.", (x0, y), rgb(pal.text_dim)); y += 22
    else:
        role = cfg.role(sel.role)
        text(surface, fonts.label_bold, f"{sel.id} · {role.label}", (x0, y), rgb(cfg.team(sel.team).light))
        if sel.value is not None:
            text(surface, fonts.small, f"critic value {sel.value:+.2f}", (x1, y + 2), rgb(pal.text), "topright")
        y += 22
        if sel.probs:
            best = max(range(5), key=lambda i: sel.probs[i])
            for i, (name, p) in enumerate(zip(ACTIONS, sel.probs)):
                text(surface, fonts.tiny_bold if name == sel.action else fonts.tiny, name, (x0, y), rgb(pal.accent if name == sel.action else pal.text_dim))
                bar = pygame.Rect(x0 + 52, y + 2, 190, 9)
                pygame.draw.rect(surface, rgb(pal.panel_alt), bar, border_radius=4)
                fill = bar.copy(); fill.w = max(2, int(bar.w * p))
                pygame.draw.rect(surface, rgb(pal.accent if i == best else pal.text_dim), fill, border_radius=4)
                text(surface, fonts.tiny, f"{p * 100:4.0f}%", (bar.right + 8, y), rgb(pal.text))
                y += 15
        elif sel.action:
            text(surface, fonts.small, f"action: {sel.action}  (this controller does not expose probabilities)", (x0, y), rgb(pal.text_dim)); y += 18
        if sel.reward is not None:
            tot = cumulative(sel.id, index)
            text(surface, fonts.small, f"reward this step {sel.reward:+.2f}    return so far {tot:+.2f}", (x0, y), rgb(pal.text)); y += 17
            comps = sorted(sel.components.items(), key=lambda kv: -abs(kv[1]))[:4]
            if comps:
                text(surface, fonts.tiny, "  ".join(f"{k} {v:+.2f}" for k, v in comps), (x0, y), rgb(pal.text_dim)); y += 16
    y += 4

    heading("Events"); y += 18
    log = [(i, e) for i, f in enumerate(frames[: index + 1]) for e in f.events if e["type"] in EVENT_LABEL][-5:]
    if not log:
        text(surface, fonts.tiny, "nothing yet", (x0, y), rgb(pal.text_dim)); y += 16
    for i, e in log:
        who = e.get("agent_id", "?")
        team = "RED" if str(who).startswith("R") else "BLUE"
        extra = f" (by {e['by_agent_id']})" if e.get("by_agent_id") else ""
        text(surface, fonts.tiny, f"t={i:<3}", (x0, y), rgb(pal.text_dim))
        text(surface, fonts.tiny_bold, f"{who} {EVENT_LABEL.get(e['type'], e['type'])}{extra}", (x0 + 40, y), rgb(cfg.team(team).light)); y += 15
    y = max(y + 6, lay.panel.bottom - 56)

    heading("Legend"); y += 17
    x = x0
    for name in ("attacker", "defender", "scout"):
        role = cfg.role(name)
        pygame.draw.circle(surface, rgb(role.color), (x + 7, y + 8), 7)
        text(surface, fonts.tiny_bold, role.letter, (x + 7, y + 8), (15, 15, 15), "center")
        r = text(surface, fonts.tiny, role.label, (x + 18, y + 1), rgb(pal.text_dim))
        x = r.right + 14
    text(surface, fonts.tiny, "gold ring = carrying a flag   ·   X = eliminated", (x0, y + 20), rgb(pal.text_dim))
    return spots


def _draw_editor_panel(surface, cfg, fonts, lay, geo: Geometry, view: ViewState) -> list[Hotspot]:
    pal, keys = cfg.palette, cfg.keys
    spots: list[Hotspot] = []
    x0, x1 = lay.panel.x + 14, lay.panel.right - 14
    y = lay.panel.y + 10
    text(surface, fonts.small_bold, "MAP MAKER", (x0, y), rgb(pal.warn)); y += 22
    text(surface, fonts.label_bold, f"{view.map_name or 'unsaved map'}   {geo.width} × {geo.height}", (x0, y), rgb(pal.text)); y += 28
    text(surface, fonts.tiny, "Left-click paints with the current tool. Right-click erases. Drag to paint many cells.", (x0, y), rgb(pal.text_dim)); y += 22
    tools = [("tool_wall", "wall", "Wall"), ("tool_erase", "erase", "Erase"), ("tool_red_base", "red_base", "RED base"),
             ("tool_blue_base", "blue_base", "BLUE base"), ("tool_red_spawn", "red_spawn", "RED spawn"), ("tool_blue_spawn", "blue_spawn", "BLUE spawn")]
    for cmd, name, label in tools:
        r = pygame.Rect(x0, y, x1 - x0, 28)
        _button(surface, fonts, r, label, active=view.tool == name, cfg=cfg, hint=keys[cmd].upper())
        spots.append(Hotspot(r, "command", cmd)); y += 32
    y += 6
    for cmd, label, on in (("toggle_mirror", "Mirror edits left <-> right", view.mirror),):
        r = pygame.Rect(x0, y, x1 - x0, 28)
        _button(surface, fonts, r, f"{label}: {'ON' if on else 'off'}", active=on, cfg=cfg, hint=keys[cmd].upper())
        spots.append(Hotspot(r, "command", cmd)); y += 32
    half = (x1 - x0 - 8) // 2
    for row in ((("shrink_width", "Width −"), ("grow_width", "Width +")), (("shrink_height", "Height −"), ("grow_height", "Height +"))):
        for i, (cmd, label) in enumerate(row):
            r = pygame.Rect(x0 + i * (half + 8), y, half, 28)
            _button(surface, fonts, r, label, cfg=cfg, hint=keys[cmd].upper())
            spots.append(Hotspot(r, "command", cmd))
        y += 32
    y += 4
    for cmd, label in (("validate_map", "Check map"), ("save_map", "Save map"), ("clear_map", "Clear all walls"), ("play_on_map", "Play on this map")):
        r = pygame.Rect(x0, y, x1 - x0, 28)
        _button(surface, fonts, r, label, cfg=cfg, hint=keys[cmd].upper())
        spots.append(Hotspot(r, "command", cmd)); y += 32
    for i, (cmd, label) in enumerate((("prev_map", "< Prev map"), ("next_map", "Next map >"))):
        r = pygame.Rect(x0 + i * (half + 8), y, half, 28)
        _button(surface, fonts, r, label, cfg=cfg)
        spots.append(Hotspot(r, "command", cmd))
    y += 40
    colors = {"error": pal.loss, "warning": pal.warn, "info": pal.text_dim}
    for issue in view.issues[:7]:
        mark = {"error": "x", "warning": "!", "info": "·"}[issue.level]
        text(surface, fonts.tiny, f"{mark} {issue.message}", (x0, y), rgb(colors[issue.level])); y += 16
    return spots


def draw_toast(surface, cfg, fonts, lay, view: ViewState, now: float) -> None:
    if view.message and now < view.message_until:
        img = fonts.label_bold.render(view.message, True, (15, 15, 15))
        box = img.get_rect(midbottom=(lay.board.centerx, lay.board.bottom - 10)).inflate(26, 12)
        pygame.draw.rect(surface, rgb(cfg.palette.accent), box, border_radius=8)
        surface.blit(img, img.get_rect(center=box.center))


def draw_help(surface, cfg, fonts, size: tuple[int, int]) -> None:
    pal = cfg.palette
    veil = _alpha(size)
    veil.fill((6, 8, 11, 232))
    surface.blit(veil, (0, 0))
    text(surface, fonts.title, "Controls", (size[0] // 2, 26), rgb(pal.text), "midtop")
    text(surface, fonts.small, "All keys can be changed in configs/viz/default.yaml  ·  press H to close", (size[0] // 2, 56), rgb(pal.text_dim), "midtop")
    cols = 3
    col_w, y_start = (size[0] - 80) // cols, 96
    ys = [y_start] * cols
    for title, cmds in GROUPS.items():
        c = ys.index(min(ys))  # put the next group in the shortest column
        x, y = 40 + c * col_w, ys[c]
        text(surface, fonts.label_bold, title.upper(), (x, y), rgb(pal.accent)); y += 24
        for cmd in cmds:
            kr = pygame.Rect(x, y, 78, 20)
            pygame.draw.rect(surface, rgb(pal.panel_alt), kr, border_radius=5)
            text(surface, fonts.tiny_bold, cfg.keys[cmd].upper(), kr.center, rgb(pal.text), "center")
            text(surface, fonts.small, DESCRIPTIONS[cmd], (x + 88, y + 1), rgb(pal.text))
            y += 23
        ys[c] = y + 14
    text(surface, fonts.small, "Mouse: click a player to select it · click the timeline to seek · in the map maker, click/drag on the board to paint.", (size[0] // 2, size[1] - 34), rgb(pal.text_dim), "midtop")


# ------------------------------------------------------------------------------ scene

def make_cumulative(frames: list[Frame]):
    def cumulative(agent_id: str, index: int) -> float:
        total = 0.0
        for f in frames[1 : index + 1]:
            for a in f.agents:
                if a.id == agent_id and a.reward:
                    total += a.reward
        return total
    return cumulative


def render_scene(surface, cfg: VizConfig, fonts: Fonts, geo: Geometry, meta: RecordingMeta | None, frames: list[Frame],
                 index: int, view: ViewState, now: float = 0.0):
    """Draw one complete window. Returns (layout, clickable spots, agent click rectangles)."""
    surface.fill(rgb(cfg.palette.background))
    lay = compute_layout(surface.get_size(), (geo.width, geo.height), cfg)
    editing = view.mode == "edit"
    clicks = draw_board(surface, cfg, fonts, lay, geo, [] if editing else frames, index, view, show_spawns=editing)
    spots = draw_panel(surface, cfg, fonts, lay, geo, meta, frames, index, view, make_cumulative(frames))
    tl = draw_timeline(surface, cfg, fonts, lay, [] if editing else frames, index, view, geo.max_steps)
    draw_header(surface, cfg, fonts, lay, meta, view)
    draw_toast(surface, cfg, fonts, lay, view, now)
    if view.show_help:
        draw_help(surface, cfg, fonts, surface.get_size())
    return lay, spots + ([tl] if tl else []), clicks


def draw_board_panel(surface, cfg: VizConfig, fonts: Fonts, geo: Geometry, frames: list[Frame], index: int,
                     view: ViewState, rect: pygame.Rect) -> None:
    """The arena only, scaled to fit ``rect`` (used for side-by-side time-lapse panels)."""
    cell = max(8, min(rect.w // geo.width, rect.h // geo.height))
    board = pygame.Rect(0, 0, cell * geo.width, cell * geo.height)
    board.center = rect.center
    lay = Layout(rect, board, pygame.Rect(0, 0, 0, 0), pygame.Rect(0, 0, 0, 0), cell)
    draw_board(surface, cfg, fonts, lay, geo, frames, index, view)
