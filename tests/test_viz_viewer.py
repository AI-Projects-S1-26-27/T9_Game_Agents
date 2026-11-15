"""Viewer, exports, CLI, reports and time-lapse, driven headlessly with simulated input."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

pygame = pytest.importorskip("pygame")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from ctf.config import load_config  # noqa: E402
from viz.__main__ import main  # noqa: E402
from viz.commands import COMMANDS  # noqa: E402
from viz.controllers import ControllerSpec  # noqa: E402
from viz.draw import Fonts, ViewState, cell_rect  # noqa: E402
from viz.export import export_gif, render_frames  # noqa: E402
from viz.frames import Recording  # noqa: E402
from viz.mapmaker import MapEditor  # noqa: E402
from viz.report import build_report, final_summary, smooth  # noqa: E402
from viz.session import record_episode  # noqa: E402
from viz.theme import ViewTeam, load_viz_config, rgb  # noqa: E402
from viz.viewer import Viewer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTED = {"roles": ["attacker", "defender", "scout"], "guard_mode": "leash", "guard_leash": 2, "scout_behavior": "random"}
SPECS = [
    ControllerSpec("Scripted A", "scripted", "", True, SCRIPTED),
    ControllerSpec("Scripted B", "scripted", "", True, {**SCRIPTED, "guard_mode": "on_flag", "guard_leash": 0}),
    ControllerSpec("Random", "random", "", True, {}),
]


@pytest.fixture()
def cfg(tmp_path):
    base = load_viz_config(ROOT / "configs" / "viz" / "default.yaml")
    return replace(base, paths=replace(base.paths, output_dir=str(tmp_path / "out"), maps_dir=str(tmp_path / "maps")))


def _env():
    return replace(load_config(ROOT / "configs" / "default.yaml"), spawn_mode="near_base")


def _viewer(cfg, mode="live", **kw):
    v = Viewer(cfg, env_config=_env(), specs=SPECS, mode=mode, seed=2, **kw)
    v.view.playing = False
    return v


def _surface(v, size=(1360, 800)):
    surf = pygame.Surface(size)
    v.render(surf)
    return surf


def _key(key):
    return pygame.event.Event(pygame.KEYDOWN, key=key)


def _click(pos, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button)


# ------------------------------------------------------------------------------ commands

def test_every_configured_key_is_a_real_pygame_key(cfg) -> None:
    v = _viewer(cfg)
    assert len(v.keymap) == len(COMMANDS)


def test_keyboard_drives_playback_and_overlays(cfg) -> None:
    v = _viewer(cfg)
    assert len(v.frames) == 1 and v.index == 0
    v.handle_event(_key(pygame.K_RIGHT)); v.handle_event(_key(pygame.K_RIGHT))
    assert v.index == 2
    v.handle_event(_key(pygame.K_LEFT))
    assert v.index == 1
    v.handle_event(_key(pygame.K_END))
    assert v.session.done and v.index == len(v.frames) - 1
    v.handle_event(_key(pygame.K_HOME))
    assert v.index == 0
    before = v.view.trails
    v.handle_event(_key(pygame.K_t))
    assert v.view.trails != before
    v.handle_event(_key(pygame.K_TAB)); assert v.view.view_team is ViewTeam.RED
    v.handle_event(_key(pygame.K_TAB)); assert v.view.view_team is ViewTeam.BLUE
    v.handle_event(_key(pygame.K_TAB)); assert v.view.view_team is ViewTeam.NONE
    v.handle_event(_key(pygame.K_f))  # fog needs a team to view as -> picks RED
    assert v.view.fog and v.view.view_team is ViewTeam.RED
    v.handle_event(_key(pygame.K_h)); assert v.view.show_help
    v.handle_event(_key(pygame.K_ESCAPE)); assert v.should_quit


def test_playback_speed_and_event_jumping(cfg) -> None:
    v = _viewer(cfg)
    v.execute("play_pause")
    assert v.view.playing
    speed = cfg.playback.speeds[v.view.speed_index]
    v.update(2.0 / speed)  # two steps worth of time
    assert v.index == 2
    v.execute("speed_up")
    assert v.view.speed_index == cfg.playback.default_speed_index + 1
    v.execute("play_pause")
    v.execute("jump_end")
    events = [i for i, f in enumerate(v.frames) if any(e["type"] != "episode_end" for e in f.events)]
    v.execute("jump_start")
    assert events, "a scripted episode should contain at least one pickup/tag/capture"
    v.execute("next_event")
    assert v.index == events[0]
    v.execute("jump_end")
    last = len(v.frames) - 1
    v.execute("prev_event")
    assert v.index == max((i for i in events if i < last), default=last)


def test_selection_new_episodes_swap_and_controller_cycling(cfg) -> None:
    v = _viewer(cfg)
    v.execute("next_agent"); assert v.view.selected == "B1"
    v.execute("prev_agent"); assert v.view.selected == "R3"
    v.execute("deselect"); assert v.view.selected is None
    assert v.view.live_labels == {"RED": "Scripted A", "BLUE": "Scripted B"}
    v.execute("swap_sides"); assert v.view.live_labels == {"RED": "Scripted B", "BLUE": "Scripted A"}
    seed = v.seed
    v.execute("new_seed"); assert v.seed == seed + 1 and v.index == 0
    v.execute("cycle_red_controller"); assert v.view.live_labels["RED"] == "Random"
    v.execute("cycle_blue_controller"); assert v.view.live_labels["BLUE"] == "Scripted B"  # was Scripted A after the swap


def test_autoloop_restarts_finished_episodes(cfg) -> None:
    v = _viewer(cfg)
    v.view.autoloop = True
    v.view.playing = True
    first_seed = v.seed
    for _ in range(400):
        v.update(1.0)
        if v.seed != first_seed:
            break
    assert v.seed == first_seed + 1


# ------------------------------------------------------------------------------ mouse + render

def test_clicking_the_panel_the_board_and_the_timeline(cfg) -> None:
    v = _viewer(cfg)
    v.execute("jump_end")
    _surface(v)
    row = next(s for s in v.hotspots if s.kind == "agent" and s.payload == "R2")
    v.handle_event(_click(row.rect.center))
    assert v.view.selected == "R2"
    rect = v.agent_rects["B3"]
    v.handle_event(_click(rect.center))
    assert v.view.selected == "B3"
    track = next(s for s in v.hotspots if s.kind == "timeline")
    x0, w, span = (int(t) for t in track.payload.split(","))
    v.handle_event(_click((x0 + w // 2, track.rect.centery)))
    assert abs(v.index - span // 2) <= 1
    v.handle_event(_click((v.layout.board.x + 2, v.layout.board.y + 2)))  # empty floor deselects
    assert v.view.selected is None


@pytest.mark.parametrize("size", [(800, 600), (1360, 800), (1920, 1080)])
def test_every_mode_renders_at_several_window_sizes(cfg, size) -> None:
    v = _viewer(cfg)
    for _ in range(6):
        v.execute("step_forward")
    v.view.selected = "R1"
    for toggle in ("fog", "belief", "vision", "coordinates", "spawns"):
        setattr(v.view, toggle, True)
    v.view.view_team = ViewTeam.BLUE
    surf = _surface(v, size)
    assert len({tuple(p) for p in pygame.surfarray.array3d(surf)[:: 17, :: 13].reshape(-1, 3)}) > 12
    v.view.show_help = True
    _surface(v, size)
    v.view.show_help = False
    v.execute("toggle_edit"); _surface(v, size)
    v.execute("toggle_edit"); _surface(v, size)


def test_players_are_drawn_in_their_team_colours_with_role_labels(cfg) -> None:
    v = _viewer(cfg)
    surf = _surface(v)
    red, blue = rgb(cfg.teams.red.color), rgb(cfg.teams.blue.color)
    pixels = {tuple(p) for p in pygame.surfarray.array3d(surf).reshape(-1, 3)}
    assert red in pixels and blue in pixels
    for role in ("attacker", "defender", "scout"):
        assert rgb(cfg.role(role).color) in pixels  # each role's ring / label colour is on screen


# ------------------------------------------------------------------------------ map maker

def test_map_maker_with_the_mouse_validation_save_load_and_play(cfg, tmp_path) -> None:
    v = _viewer(cfg)
    v.execute("toggle_edit")
    assert v.view.mode == "edit" and v.frames == []
    _surface(v)
    wall_btn = next(s for s in v.hotspots if s.payload == "tool_wall")
    v.handle_event(_click(wall_btn.rect.center))
    target = cell_rect(v.layout, 3, 1).center
    v.handle_event(_click(target))
    assert (3, 1) in v.editor.obstacles
    v.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=target, button=1))
    v.handle_event(_click(cell_rect(v.layout, 3, 1).center, button=3))  # right click erases
    assert (3, 1) not in v.editor.obstacles
    v.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=target, button=3))
    # drag paints a line of walls
    v.handle_event(_click(cell_rect(v.layout, 2, 5).center))
    v.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=cell_rect(v.layout, 3, 5).center, rel=(0, 0), buttons=(1, 0, 0)))
    v.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=target, button=1))
    assert {(2, 5), (3, 5)} <= v.editor.obstacles
    # walling off the bases is an error and blocks play
    for y in range(7):
        v.editor.obstacles.add((5, y))
    v._refresh_issues()
    assert any(i.level == "error" for i in v.view.issues)
    v.execute("play_on_map")
    assert v.view.mode == "edit"
    v.editor.obstacles = {(5, y) for y in range(1, 6)} | {(2, 5), (3, 5)}
    v.execute("save_map")
    saved = sorted(Path(cfg.paths.maps_dir).glob("custom_*.yaml"))
    assert len(saved) == 1 and v.view.map_name == saved[0].name
    assert (2, 5) in load_config(saved[0]).obstacles
    v.execute("play_on_map")
    assert v.view.mode == "live" and (2, 5) in v.env_config.obstacles and v.session is not None
    v.execute("next_map")  # the saved map is the only file in the maps dir
    assert v.map_path == saved[0]


def test_resize_and_mirror_commands(cfg) -> None:
    v = _viewer(cfg)
    v.execute("toggle_edit")
    v.execute("grow_width"); assert v.editor.width == 12
    v.execute("shrink_height"); assert v.editor.height == 6
    v.execute("toggle_mirror"); assert v.editor.mirror and v.editor.is_symmetric()
    for _ in range(30):
        v.execute("shrink_width")
    assert v.editor.width == cfg.editor.min_size and "between" in v.view.message


def test_replay_mode_shows_a_recording_and_refuses_live_commands(cfg) -> None:
    from viz.controllers import ControllerFactory

    env = _env()
    f = ControllerFactory()
    rec = record_episode(env, f.build(SPECS[0], env, 1), f.build(SPECS[2], env, 2), seed=9)
    v = Viewer(cfg, env_config=env, specs=SPECS, recording=rec, mode="replay")
    _surface(v)
    assert v.frames is rec.frames and v.view.live_labels["RED"] == "Scripted A"
    v.execute("new_seed")
    assert "Replay mode" in v.view.message and v.session is None
    v.execute("jump_end"); _surface(v)
    assert v.index == len(rec.frames) - 1


# ------------------------------------------------------------------------------ export / cli / report

def test_exports_are_even_sized_and_match_the_frame_count(cfg, tmp_path) -> None:
    from viz.controllers import ControllerFactory

    env = _env()
    f = ControllerFactory()
    rec = record_episode(env, f.build(SPECS[0], env, 1), f.build(SPECS[2], env, 2), seed=4)
    view = ViewState(fog=False, belief=False, vision=False, trails=True, probabilities=True, arrows=True, labels=True,
                     coordinates=False, spawns=False, view_team=ViewTeam.NONE, selected="R1")
    arrays = render_frames(rec, cfg, view, size=(801, 601))
    assert len(arrays) == len(rec.frames) and arrays[0].shape == (600, 800, 3)
    gif = export_gif(rec, tmp_path / "e.gif", cfg, view, hold_last=2)
    from PIL import Image
    img = Image.open(gif)
    assert img.n_frames == len(rec.frames)  # identical hold frames are merged into one longer frame...
    img.seek(img.n_frames - 1)
    assert img.info["duration"] >= 2.5 * int(1000 / cfg.export.fps)  # ...so the last frame lingers (1 + hold_last=2)
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        pytest.skip("imageio-ffmpeg not installed")
    from viz.export import export_mp4
    mp4 = export_mp4(rec, tmp_path / "e.mp4", cfg, view, hold_last=1)
    assert mp4.stat().st_size > 2000


def test_cli_record_export_and_errors(cfg, tmp_path, capsys) -> None:
    ctrl = tmp_path / "controllers.yaml"
    ctrl.write_text(yaml.safe_dump({"controllers": [
        {"label": s.label, "kind": s.kind, "checkpoint": s.checkpoint, "deterministic": True, "params": s.params} for s in SPECS]}))
    out = tmp_path / "ep.json.gz"
    base = ["--set", f"paths.output_dir={tmp_path / 'o'}"]
    code = main(["record", "--controllers", str(ctrl), "--red", "Scripted A", "--blue", "Random", "--seed", "3",
                 "--out", str(out), "--gif", str(tmp_path / "ep.gif"), *base])
    assert code == 0 and out.exists() and (tmp_path / "ep.gif").exists()
    rec = Recording.load(out)
    assert rec.meta.teams == {"RED": "Scripted A", "BLUE": "Random"}
    assert main(["export", str(out), "--png", str(tmp_path / "f.png"), "--frame", "0", "--select", "R1", *base]) == 0
    assert (tmp_path / "f.png").stat().st_size > 5000
    assert main(["record", "--controllers", str(ctrl), "--red", "no such controller", *base]) == 2
    assert "no controller matches" in capsys.readouterr().err
    assert main(["export", str(tmp_path / "missing.json"), "--png", "x.png", *base]) == 2


def _fake_run(tmp_path, n=40):
    run = tmp_path / "myrun"
    run.mkdir()
    lines = []
    for i in range(1, n + 1):
        win = min(1.0, i / n)
        lines.append({"step": i * 2048, "time": 0.0, "episode/win_rate": win, "episode/draw_rate": 1 - win, "episode/loss_rate": 0.0,
                      "episode/win_rate/opp_stay": win, "episode/win_rate/opp_random": None if i % 3 else 0.5,
                      "episode/episode_length": 200 - i, "episode/pickups": 1.0, "episode/tags_inflicted": 0.5, "episode/tags_suffered": 0.5,
                      "episode/return/slot0_attacker": win * 10, "episode/return/slot1_defender": win * 8,
                      "episode/reward/outcome": win * 10, "episode/reward/step": -0.3,
                      "train/entropy": 1.4 - i / 100, "train/approx_kl": 0.003, "train/explained_variance": 0.8, "train/value_loss": 0.4,
                      "perf/steps_per_sec": 900.0, "curriculum/phase": float(i > n // 2), "schedule/scout_scale": 1 - i / n})
    lines.insert(n // 2, {"step": (n // 2) * 2048, "time": 0.0, "event": "phase_advance", "phase_index": 1, "phase": "second_phase"})
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    return run


def test_training_report_builds_a_png_and_summarises_the_tail(cfg, tmp_path, capsys) -> None:
    run = _fake_run(tmp_path)
    png = build_report(run, cfg, tmp_path / "r.png")
    assert png.stat().st_size > 30_000
    from viz.report import load_run
    updates, phases = load_run(run)
    assert len(phases) == 1 and phases[0]["phase"] == "second_phase"
    s = final_summary(updates, 10)
    assert s["win"] > 0.8 and abs(s["win"] + s["draw"] + s["loss"] - 1) < 1e-9
    assert main(["report", "--run", str(run), "--out", str(tmp_path / "cli.png")]) == 0
    assert "last" in capsys.readouterr().out
    assert main(["report", "--run", str(tmp_path / "nope")]) == 2


def test_smoothing_keeps_gaps_and_averages_a_trailing_window() -> None:
    out = smooth([1, 3, np.nan, 5, np.nan, np.nan, 7], 2)
    assert out[0] == 1 and out[1] == 2 and np.isnan(out[2]) and out[3] == 5 and np.isnan(out[4]) and out[6] == 7


# ------------------------------------------------------------------------------ maps + time-lapse

@pytest.mark.parametrize("path", sorted((ROOT / "configs" / "maps").glob("*.yaml")), ids=lambda p: p.stem)
def test_preset_maps_are_valid_symmetric_and_connected(path) -> None:
    ed = MapEditor.from_file(path, min_size=5, max_size=25)
    assert (ed.width, ed.height) == (11, 7)
    assert ed.is_valid and ed.is_symmetric() and ed.path_length() is not None
    assert load_config(path).spawn_mode.value == "fixed"


def test_default_preset_matches_the_environment_default() -> None:
    assert load_config(ROOT / "configs" / "maps" / "default.yaml").obstacles == load_config(ROOT / "configs" / "default.yaml").obstacles


def test_timelapse_tiles_checkpoints_side_by_side(cfg, tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from rl.checkpoint import save_checkpoint
    from rl.config import load_base_env_config
    from rl.trainer import Trainer
    from rl_helpers import tiny_config
    from viz.timelapse import build_timelapse, list_checkpoints, pick_evenly

    rcfg = tiny_config(str(tmp_path))
    run = tmp_path / "run"
    for steps in (1000, 500_000, 2_000_000):
        t = Trainer(rcfg, load_base_env_config(rcfg, ROOT), run)
        t.counters.update(total_steps=steps, update=steps // 16)
        save_checkpoint(run / "checkpoints" / f"step_{steps:010d}.pt", **t._state())
    found = list_checkpoints(run)
    assert [s for s, _, _ in found] == [1000, 500_000, 2_000_000]
    assert [s for s, _, _ in pick_evenly(found, 2)] == [1000, 2_000_000]
    out = build_timelapse(found, SPECS[0], _env(), cfg, tmp_path / "t.gif", seed=1)
    from PIL import Image
    img = Image.open(out)
    assert img.size[0] == min(cfg.timelapse.columns, 3) * cfg.timelapse.panel_width and img.n_frames > 5
    del torch


def test_rl_controller_frames_carry_probabilities_values_and_rewards(cfg, tmp_path) -> None:
    pytest.importorskip("torch")
    from rl.checkpoint import save_checkpoint
    from rl.config import load_base_env_config
    from rl.trainer import Trainer
    from rl_helpers import tiny_config
    from viz.controllers import ControllerFactory

    rcfg = tiny_config(str(tmp_path))
    t = Trainer(rcfg, load_base_env_config(rcfg, ROOT), tmp_path / "run")
    t.counters.update(total_steps=250_000, update=10)
    ck = tmp_path / "ck.pt"
    save_checkpoint(ck, **t._state())
    env = _env()
    f = ControllerFactory()
    red = f.build(ControllerSpec("RL", "rl", str(ck), True, {}), env, 1)
    rec = record_episode(env, red, f.build(SPECS[0], env, 2), seed=5)
    assert red.label == "RL @ 0.25M steps"
    first = {a.id: a for a in rec.frames[0].agents}
    assert len(first["R1"].probs) == 5 and abs(sum(first["R1"].probs) - 1) < 1e-3 and first["R1"].value is not None
    assert first["B1"].probs == [] and first["B1"].role == "attacker" and first["R2"].role == "defender"
    later = rec.frames[1]
    assert later.agents[3].reward is not None and "step" in later.agents[3].components
    v = Viewer(cfg, env_config=env, specs=[SPECS[0]], recording=rec, mode="replay")
    v.view.selected = "R1"
    v.index = 2
    _surface(v)  # the probability bars / critic value / reward breakdown path renders
