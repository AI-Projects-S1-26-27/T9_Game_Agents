"""The interactive viewer: replay a recording, play live, edit maps. All state lives in ``Viewer``;
drawing is in draw.py, so everything here can be driven (and tested) without a window."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pygame

from ctf.agent import Team
from ctf.config import CTFConfig, load_config
from rl.schema import ConfigError
from viz.commands import COMMANDS
from viz.controllers import Controller, ControllerError, ControllerFactory, ControllerSpec
from viz.draw import Fonts, Geometry, Hotspot, Layout, ViewState, cell_rect, pixel_to_cell, render_scene
from viz.frames import MEANINGFUL_EVENTS, Recording
from viz.mapmaker import MapEditor, MapError, Tool
from viz.session import Session
from viz.theme import ViewTeam, VizConfig

_RANDOM = ControllerSpec("Random", "random", "", True, {})


class Viewer:
    def __init__(
        self,
        cfg: VizConfig,
        *,
        env_config: CTFConfig,
        specs: list[ControllerSpec],
        recording: Recording | None = None,
        mode: str = "live",
        seed: int = 0,
        map_path: Path | None = None,
        red_index: int = 0,
        blue_index: int = 1,
    ) -> None:
        self.cfg = cfg
        self.env_config = env_config
        self.specs = specs or [_RANDOM]
        self.factory = ControllerFactory()
        self.fonts: Fonts | None = None
        self.seed = seed
        self.map_path = map_path
        self.red_i = red_index % len(self.specs)
        self.blue_i = blue_index % len(self.specs)
        o, p = cfg.overlays, cfg.playback
        self.view = ViewState(
            fog=o.fog, belief=o.belief, vision=o.vision, trails=o.trails, probabilities=o.probabilities, arrows=o.arrows,
            labels=o.labels, coordinates=o.coordinates, spawns=o.spawns, view_team=o.view_team, mode=mode,
            speed_index=p.default_speed_index, autoloop=p.autoloop,
        )
        self.recording = recording
        self.session: Session | None = None
        self.index = 0
        self._acc = 0.0
        self._painting: int | None = None
        self._last_cell = None
        self.hotspots: list[Hotspot] = []
        self.agent_rects: dict = {}
        self.layout: Layout | None = None
        self._surface: pygame.Surface | None = None
        self.map_files = sorted(Path(cfg.paths.maps_dir).glob("*.yaml")) if Path(cfg.paths.maps_dir).exists() else []
        self.map_i = self._index_of_map(map_path)
        self.editor = self._new_editor(env_config)
        self.keymap = self._build_keymap()
        self.should_quit = False
        if mode == "live":
            self._start_session(seed)
            self.view.playing = p.live_start_playing
        elif mode == "edit":
            self._enter_edit()
        elif recording is not None:
            self.view.live_labels = dict(recording.meta.teams)

    # ---------------------------------------------------------------------- setup

    def _build_keymap(self) -> dict[int, str]:
        pygame.display.init()
        out: dict[int, str] = {}
        for cmd in COMMANDS:
            name = self.cfg.keys[cmd]
            try:
                out[pygame.key.key_code(name)] = cmd
            except ValueError as exc:
                raise ConfigError(f"keys.{cmd}: {name!r} is not a pygame key name") from exc
        return out

    def _new_editor(self, env_config: CTFConfig) -> MapEditor:
        e = self.cfg.editor
        return MapEditor(env_config, min_size=e.min_size, max_size=e.max_size)

    def _index_of_map(self, path: Path | None) -> int:
        if path is not None:
            for i, f in enumerate(self.map_files):
                if f.resolve() == Path(path).resolve():
                    return i
        return -1

    # ---------------------------------------------------------------------- state

    @property
    def frames(self) -> list:
        if self.view.mode == "edit":
            return []
        if self.session is not None:
            return self.session.frames
        return self.recording.frames if self.recording else []

    @property
    def meta(self):
        if self.session is not None:
            return self.session.recording().meta
        return self.recording.meta if self.recording else None

    def toast(self, message: str, seconds: float = 2.6) -> None:
        self.view.message = message
        self.view.message_until = time.monotonic() + seconds

    def _geometry(self) -> Geometry:
        if self.view.mode == "edit":
            c = self.env_config
            return Geometry.from_editor(self.editor, c.visibility_radius, c.max_episode_steps)
        if self.recording is not None and self.session is None:
            return Geometry.from_meta(self.recording.meta)
        c = self.env_config
        return Geometry(c.width, c.height, set(c.obstacles), tuple(c.red_base), tuple(c.blue_base),
                        list(c.default_spawns(Team.RED)), list(c.default_spawns(Team.BLUE)), c.visibility_radius, c.max_episode_steps)

    def _controller(self, spec: ControllerSpec, seed: int) -> Controller:
        try:
            return self.factory.build(spec, self.env_config, seed)
        except (ControllerError, ConfigError, ValueError, OSError) as exc:
            self.toast(f"{spec.label}: {exc}", 7.0)
            fallback = next((s for s in self.specs if s.kind != "rl"), _RANDOM)
            return self.factory.build(fallback, self.env_config, seed)

    def _start_session(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        red = self._controller(self.specs[self.red_i], self.seed)
        blue = self._controller(self.specs[self.blue_i], self.seed + 1)
        self.session = Session(self.env_config, red, blue, seed=self.seed)
        self.recording = None
        self.view.live_labels = {"RED": red.label, "BLUE": blue.label}
        self.index = 0
        self._acc = 0.0
        self.view.flash_t = 0.0
        self.view.selected = None

    def _require_live(self) -> bool:
        if self.view.mode == "replay" and self.session is None:
            self.toast("Replay mode: start the viewer with `viz play` for live episodes")
            return False
        return True

    # ------------------------------------------------------------------- playback

    def _set_index(self, i: int) -> None:
        n = len(self.frames)
        new = max(0, min(i, n - 1)) if n else 0
        if new != self.index:
            self.view.flash_t = 0.0
        self.index = new

    def _advance(self) -> bool:
        n = len(self.frames)
        if self.index < n - 1:
            self._set_index(self.index + 1)
            return True
        if self.session is not None and not self.session.done:
            self.session.step()
            self._set_index(len(self.session.frames) - 1)
            return True
        if self.session is not None and self.view.autoloop:
            self._start_session(self.seed + 1)
            return True
        self.view.playing = False
        return False

    def _event_indices(self) -> list[int]:
        return [i for i, f in enumerate(self.frames) if any(e["type"] in MEANINGFUL_EVENTS for e in f.events)]

    def update(self, dt: float) -> None:
        self.view.flash_t += dt
        if self.view.playing and self.view.mode != "edit" and self.frames:
            self._acc += dt * self.cfg.playback.speeds[self.view.speed_index]
            while self._acc >= 1.0 and self.view.playing:
                self._acc -= 1.0
                self._advance()

    # ------------------------------------------------------------------- commands

    def execute(self, command: str) -> None:
        if command not in COMMANDS:
            raise ValueError(f"unknown command {command!r}")
        getattr(self, f"_cmd_{command}")()

    def _cmd_quit(self) -> None:
        self.should_quit = True

    def _cmd_help(self) -> None:
        self.view.show_help = not self.view.show_help

    def _cmd_play_pause(self) -> None:
        if self.view.mode == "edit":
            return
        if not self.view.playing and self.index >= len(self.frames) - 1 and (self.session is None or self.session.done):
            self._set_index(0)
        self.view.playing = not self.view.playing

    def _cmd_step_forward(self) -> None:
        self.view.playing = False
        self._advance()

    def _cmd_step_back(self) -> None:
        self.view.playing = False
        self._set_index(self.index - 1)

    def _cmd_jump_start(self) -> None:
        self._set_index(0)

    def _cmd_jump_end(self) -> None:
        if self.session is not None:
            self.session.run_to_end()
        self._set_index(len(self.frames) - 1)

    def _cmd_speed_up(self) -> None:
        self.view.speed_index = min(self.view.speed_index + 1, len(self.cfg.playback.speeds) - 1)
        self.toast(f"speed x{self.cfg.playback.speeds[self.view.speed_index]:g}", 1.2)

    def _cmd_speed_down(self) -> None:
        self.view.speed_index = max(self.view.speed_index - 1, 0)
        self.toast(f"speed x{self.cfg.playback.speeds[self.view.speed_index]:g}", 1.2)

    def _cmd_next_event(self) -> None:
        if self.session is not None:
            self.session.run_to_end()
        later = [i for i in self._event_indices() if i > self.index]
        if later:
            self.view.playing = False
            self._set_index(later[0])

    def _cmd_prev_event(self) -> None:
        earlier = [i for i in self._event_indices() if i < self.index]
        if earlier:
            self.view.playing = False
            self._set_index(earlier[-1])

    def _toggle(self, name: str, label: str) -> None:
        setattr(self.view, name, not getattr(self.view, name))
        self.toast(f"{label}: {'on' if getattr(self.view, name) else 'off'}", 1.2)

    def _cmd_toggle_fog(self) -> None:
        if self.view.view_team is ViewTeam.NONE and not self.view.fog:
            self.view.view_team = ViewTeam.RED
        self._toggle("fog", "fog of war")

    def _cmd_toggle_belief(self) -> None: self._toggle("belief", "belief map")
    def _cmd_toggle_vision(self) -> None: self._toggle("vision", "vision range")
    def _cmd_toggle_trails(self) -> None: self._toggle("trails", "trails")
    def _cmd_toggle_probabilities(self) -> None: self._toggle("probabilities", "action probabilities")
    def _cmd_toggle_arrows(self) -> None: self._toggle("arrows", "action arrows")
    def _cmd_toggle_labels(self) -> None: self._toggle("labels", "labels")
    def _cmd_toggle_coordinates(self) -> None: self._toggle("coordinates", "coordinates")
    def _cmd_toggle_spawns(self) -> None: self._toggle("spawns", "spawn markers")

    def _cmd_cycle_view_team(self) -> None:
        order = [ViewTeam.NONE, ViewTeam.RED, ViewTeam.BLUE]
        self.view.view_team = order[(order.index(self.view.view_team) + 1) % 3]
        self.toast(f"viewing as {self.view.view_team.value.upper() if self.view.view_team is not ViewTeam.NONE else 'everyone'}", 1.5)

    def _cycle_agent(self, step: int) -> None:
        if not self.frames:
            return
        ids = [a.id for a in self.frames[self.index].agents]
        cur = ids.index(self.view.selected) if self.view.selected in ids else (-1 if step > 0 else 0)
        self.view.selected = ids[(cur + step) % len(ids)]

    def _cmd_next_agent(self) -> None: self._cycle_agent(1)
    def _cmd_prev_agent(self) -> None: self._cycle_agent(-1)
    def _cmd_deselect(self) -> None: self.view.selected = None

    def _cmd_new_episode(self) -> None:
        if self._require_live():
            self._start_session(self.seed)
            self.toast(f"new episode (seed {self.seed})")

    def _cmd_new_seed(self) -> None:
        if self._require_live():
            self._start_session(self.seed + 1)
            self.toast(f"new episode (seed {self.seed})")

    def _cmd_swap_sides(self) -> None:
        if self._require_live():
            self.red_i, self.blue_i = self.blue_i, self.red_i
            self._start_session(self.seed)
            self.toast("sides swapped")

    def _cmd_cycle_red_controller(self) -> None:
        if self._require_live():
            self.red_i = (self.red_i + 1) % len(self.specs)
            self._start_session(self.seed)
            self.toast(f"RED = {self.view.live_labels['RED']}")

    def _cmd_cycle_blue_controller(self) -> None:
        if self._require_live():
            self.blue_i = (self.blue_i + 1) % len(self.specs)
            self._start_session(self.seed)
            self.toast(f"BLUE = {self.view.live_labels['BLUE']}")

    def _cmd_toggle_autoloop(self) -> None: self._toggle("autoloop", "auto-restart")

    # ------------------------------------------------------------------- map maker

    def _enter_edit(self) -> None:
        self.view.mode = "edit"
        self.view.playing = False
        self.editor = self._new_editor(self.env_config)
        self.editor.mirror = self.view.mirror
        self.view.map_name = self.map_path.name if self.map_path else ""
        self._refresh_issues()

    def _refresh_issues(self) -> None:
        self.view.issues = sorted(self.editor.issues(), key=lambda i: {"error": 0, "warning": 1, "info": 2}[i.level])

    def _cmd_toggle_edit(self) -> None:
        if self.view.mode == "edit":
            self.view.mode = "live" if self.session is not None else "replay"
            self.toast("map maker closed (the map is not applied until you press play-on-map)")
        else:
            self._enter_edit()

    def _tool(self, tool: Tool) -> None:
        self.editor.tool = tool
        self.view.tool = tool.value

    def _cmd_tool_wall(self) -> None: self._tool(Tool.WALL)
    def _cmd_tool_erase(self) -> None: self._tool(Tool.ERASE)
    def _cmd_tool_red_base(self) -> None: self._tool(Tool.RED_BASE)
    def _cmd_tool_blue_base(self) -> None: self._tool(Tool.BLUE_BASE)
    def _cmd_tool_red_spawn(self) -> None: self._tool(Tool.RED_SPAWN)
    def _cmd_tool_blue_spawn(self) -> None: self._tool(Tool.BLUE_SPAWN)

    def _cmd_toggle_mirror(self) -> None:
        self.editor.mirror = not self.editor.mirror
        self.view.mirror = self.editor.mirror
        if self.editor.mirror:
            self.editor.symmetrize()
        self._refresh_issues()
        self.toast(f"mirror: {'on (right half now mirrors the left)' if self.editor.mirror else 'off'}")

    def _resize(self, dw: int, dh: int) -> None:
        if self.view.mode != "edit":
            return
        msg = self.editor.resize(dw, dh)
        self._refresh_issues()
        if msg:
            self.toast(msg)

    def _cmd_grow_width(self) -> None: self._resize(1, 0)
    def _cmd_shrink_width(self) -> None: self._resize(-1, 0)
    def _cmd_grow_height(self) -> None: self._resize(0, 1)
    def _cmd_shrink_height(self) -> None: self._resize(0, -1)

    def _cmd_clear_map(self) -> None:
        if self.view.mode == "edit":
            self.editor.clear_walls()
            self._refresh_issues()

    def _cmd_validate_map(self) -> None:
        self._refresh_issues()
        errors = sum(i.level == "error" for i in self.view.issues)
        warns = sum(i.level == "warning" for i in self.view.issues)
        self.toast("map OK" if not errors else f"{errors} problem(s) to fix", 2.2) if not warns else self.toast(f"{errors} error(s), {warns} warning(s)", 2.2)

    def _cmd_save_map(self) -> None:
        if self.view.mode != "edit":
            return
        path = self.map_path
        if path is None:
            folder = Path(self.cfg.paths.maps_dir)
            n = 1
            while (folder / f"custom_{n}.yaml").exists():
                n += 1
            path = folder / f"custom_{n}.yaml"
        try:
            saved = self.editor.save(path)
        except MapError as exc:
            self.toast(f"can't save: {exc}", 4.0)
            return
        self.map_path = saved
        self.view.map_name = saved.name
        self.map_files = sorted(Path(self.cfg.paths.maps_dir).glob("*.yaml"))
        self.map_i = self._index_of_map(saved)
        self.toast(f"saved → {saved}", 4.0)

    def _load_map(self, step: int) -> None:
        if not self.map_files:
            self.toast(f"no .yaml maps in {self.cfg.paths.maps_dir}")
            return
        self.map_i = (self.map_i + step) % len(self.map_files)
        path = self.map_files[self.map_i]
        try:
            self.env_config = load_config(path)
        except (ValueError, OSError) as exc:
            self.toast(f"can't load {path.name}: {exc}", 5.0)
            return
        self.map_path = path
        if self.view.mode == "edit":
            self._enter_edit()
        else:
            self._start_session(self.seed)
        self.toast(f"map: {path.name}")

    def _cmd_next_map(self) -> None: self._load_map(1)
    def _cmd_prev_map(self) -> None: self._load_map(-1)

    def _cmd_play_on_map(self) -> None:
        if self.view.mode != "edit":
            return
        try:
            self.env_config = self.editor.to_config()
        except MapError as exc:
            self.toast(f"fix the map first: {exc}", 5.0)
            return
        self.view.mode = "live"
        self._start_session(self.seed)
        self.view.playing = True

    # --------------------------------------------------------------------- export

    def _out(self, stem: str, ext: str) -> Path:
        folder = Path(self.cfg.paths.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.{ext}"

    def _current_recording(self) -> Recording | None:
        if self.session is not None:
            self.session.run_to_end()
            return self.session.recording()
        return self.recording

    def _cmd_screenshot(self) -> None:
        if self._surface is None:
            return
        path = self._out("screenshot", "png")
        pygame.image.save(self._surface, str(path))
        self.toast(f"saved → {path}", 3.5)

    def _export(self, kind: str) -> None:
        from viz.export import export_gif, export_mp4

        rec = self._current_recording()
        if rec is None:
            return
        self.toast(f"exporting {kind.upper()}…", 30.0)
        path = self._out("episode", kind)
        try:
            (export_gif if kind == "gif" else export_mp4)(rec, path, self.cfg, self.view)
        except Exception as exc:  # export problems must not kill the viewer
            self.toast(f"export failed: {exc}", 6.0)
            return
        self.toast(f"saved → {path}", 4.0)

    def _cmd_export_gif(self) -> None: self._export("gif")
    def _cmd_export_mp4(self) -> None: self._export("mp4")

    def _cmd_save_recording(self) -> None:
        rec = self._current_recording()
        if rec is None:
            return
        path = self._out("episode", "json.gz")
        rec.save(path)
        self.toast(f"saved → {path}", 4.0)

    # --------------------------------------------------------------------- input

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.should_quit = True
        elif event.type == pygame.KEYDOWN:
            cmd = self.keymap.get(event.key)
            if cmd:
                self.execute(cmd)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
            self._mouse_down(event.pos, event.button)
        elif event.type == pygame.MOUSEBUTTONUP:
            self._painting, self._last_cell = None, None
        elif event.type == pygame.MOUSEMOTION and self._painting is not None:
            self._paint_at(event.pos)

    def _mouse_down(self, pos, button: int) -> None:
        if self.view.show_help:
            self.view.show_help = False
            return
        if button == 1:
            for spot in self.hotspots:
                if spot.rect.collidepoint(pos):
                    if spot.kind == "command":
                        self.execute(spot.payload)
                    elif spot.kind == "agent":
                        self.view.selected = spot.payload
                    elif spot.kind == "timeline":
                        x0, w, span = (int(v) for v in spot.payload.split(","))
                        self.view.playing = False
                        if self.session is not None:
                            self.session.run_to_end()
                        self._set_index(round((pos[0] - x0) / max(w, 1) * span))
                    return
        if self.view.mode == "edit":
            self._painting = button
            self._paint_at(pos)
            return
        if button == 1:
            hit = next((aid for aid, r in self.agent_rects.items() if r.collidepoint(pos)), None)
            self.view.selected = hit

    def _paint_at(self, pos) -> None:
        if self.layout is None:
            return
        cell = pixel_to_cell(self.layout, pos, (self.editor.width, self.editor.height))
        if cell is None or cell == self._last_cell:
            return
        tool = self.editor.tool if self._painting == 1 else Tool.ERASE
        if self._last_cell is not None and tool not in (Tool.WALL, Tool.ERASE):
            return  # dragging only paints walls/erases; spawns and bases are single clicks
        self._last_cell = cell
        msg = self.editor.paint(cell, tool)
        self._refresh_issues()
        if msg:
            self.toast(msg)

    # ------------------------------------------------------------------ rendering

    def render(self, surface: pygame.Surface) -> None:
        if self.fonts is None:
            self.fonts = Fonts(self.cfg)
        geo = self._geometry()
        frames = self.frames
        if frames:
            self.index = min(self.index, len(frames) - 1)
        lay, spots, clicks = render_scene(surface, self.cfg, self.fonts, geo, self.meta, frames, self.index, self.view, time.monotonic())
        self.layout, self.hotspots, self.agent_rects = lay, spots, clicks
        self._surface = surface

    def run(self) -> None:
        pygame.init()
        flags = pygame.RESIZABLE if self.cfg.window.resizable else 0
        screen = pygame.display.set_mode((self.cfg.window.width, self.cfg.window.height), flags)
        pygame.display.set_caption(self.cfg.window.title)
        clock = pygame.time.Clock()
        while not self.should_quit:
            dt = clock.tick(self.cfg.window.fps) / 1000.0
            for event in pygame.event.get():
                self.handle_event(event)
            self.update(dt)
            self.render(screen)
            pygame.display.flip()
        pygame.quit()
