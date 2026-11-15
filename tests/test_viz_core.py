"""Pure-logic parts of the visualisation: theme, recordings, map editor, controllers, sessions."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from ctf import CTFConfig, Team
from ctf.config import load_config
from rl.schema import ConfigError
from viz.commands import COMMANDS
from viz.controllers import ControllerFactory, ControllerSpec, load_controller_specs
from viz.frames import Recording
from viz.mapmaker import MapEditor, MapError, Tool
from viz.session import Session, record_episode
from viz.theme import load_viz_config, rgb

ROOT = Path(__file__).resolve().parents[1]
VIZ_YAML = ROOT / "configs" / "viz" / "default.yaml"
DEFAULT_MAP = ROOT / "configs" / "default.yaml"
SCRIPTED = {"roles": ["attacker", "defender", "scout"], "guard_mode": "on_flag", "guard_leash": 0, "scout_behavior": "stay", "chase_radius": 0, "escort": False}


def _editor(**kw) -> MapEditor:
    return MapEditor(load_config(DEFAULT_MAP), min_size=5, max_size=25, **kw)


def _controller(kind="random", params=None):
    spec = ControllerSpec(kind.title(), kind, "", True, params or {})
    return spec


# ------------------------------------------------------------------------------ theme

def test_theme_loads_and_every_command_has_a_distinct_key() -> None:
    cfg = load_viz_config(VIZ_YAML)
    assert set(cfg.keys) == set(COMMANDS) and len(set(cfg.keys.values())) == len(COMMANDS)
    assert rgb(cfg.teams.red.color) == (229, 72, 77)
    assert cfg.role("scout").letter == "S" and cfg.role("nonsense").label == "Agent"


def test_theme_rejects_bad_colours_duplicate_keys_and_unknown_commands() -> None:
    raw = yaml.safe_load(VIZ_YAML.read_text())
    bad = yaml.safe_load(VIZ_YAML.read_text())
    bad["palette"]["text"] = "white"
    with pytest.raises(ConfigError, match="#rrggbb"):
        from rl.schema import from_dict
        from viz.theme import VizConfig
        from_dict(VizConfig, bad)
    dup = yaml.safe_load(VIZ_YAML.read_text())
    dup["keys"]["step_back"] = dup["keys"]["step_forward"]
    with pytest.raises(ConfigError, match="same key"):
        from_dict(VizConfig, dup)
    unknown = yaml.safe_load(VIZ_YAML.read_text())
    unknown["keys"]["fly"] = "f5"
    with pytest.raises(ConfigError, match="unknown commands"):
        from_dict(VizConfig, unknown)
    assert raw["window"]["width"] > 0


# ------------------------------------------------------------------------- map maker

def test_painting_walls_bases_and_spawns_with_rules() -> None:
    ed = _editor()
    assert ed.paint((3, 3), Tool.WALL) == "" and (3, 3) in ed.obstacles
    assert "base" in ed.paint(ed.red_base, Tool.WALL)
    assert "spawn" in ed.paint(ed.spawns[Team.RED][0], Tool.WALL)
    ed.paint((3, 3), Tool.ERASE)
    assert (3, 3) not in ed.obstacles
    ed.paint((2, 0), Tool.RED_BASE)
    assert ed.red_base == (2, 0)
    ed.paint((4, 0), Tool.RED_SPAWN)  # team is full -> replaces the last spawn
    assert len(ed.spawns[Team.RED]) == 3 and (4, 0) in ed.spawns[Team.RED]
    ed.paint((4, 0), Tool.RED_SPAWN)  # painting again toggles it off
    assert (4, 0) not in ed.spawns[Team.RED]
    assert ed.paint((99, 99), Tool.WALL) == ""  # out of bounds is ignored


def test_mirror_painting_keeps_the_map_symmetric() -> None:
    ed = _editor()
    assert ed.is_symmetric()
    ed.mirror = True
    ed.paint((3, 1), Tool.WALL)
    assert (3, 1) in ed.obstacles and (7, 1) in ed.obstacles and ed.is_symmetric()
    ed.paint((2, 5), Tool.RED_BASE)
    assert ed.red_base == (2, 5) and ed.blue_base == (8, 5)
    ed.paint((0, 6), Tool.RED_SPAWN)
    assert (0, 6) in ed.spawns[Team.RED] and (10, 6) in ed.spawns[Team.BLUE]
    assert ed.is_symmetric()


def test_validation_finds_real_problems() -> None:
    ed = _editor()
    assert ed.is_valid and any("path" in i.message for i in ed.issues())
    for y in range(ed.height):  # wall off the bases completely
        ed.obstacles.add((5, y))
    assert not ed.is_valid and any("walled off" in i.message for i in ed.issues())
    with pytest.raises(MapError, match="walled off"):
        ed.to_config()
    ed2 = _editor()
    ed2.spawns[Team.RED].pop()
    assert any("needs 3 spawns" in i.message for i in ed2.issues())
    ed3 = _editor()
    ed3.obstacles.add((3, 3))
    assert any("symmetric" in i.message and i.level == "warning" for i in ed3.issues()) and ed3.is_valid


def test_resize_respects_limits_and_symmetrize_repairs_the_map() -> None:
    ed = _editor()
    assert ed.resize(-1, 0) == "" and ed.width == 10
    assert "between" in ed.resize(-50, 0)
    ed.mirror = True
    ed.resize(1, 0)
    assert ed.width == 11 and ed.is_symmetric()
    small = _editor()
    small.resize(-4, -2)
    assert all(small.in_bounds(c) for c in small.obstacles) and small.in_bounds(small.blue_base)


def test_save_and_load_round_trip_as_a_normal_env_config(tmp_path) -> None:
    ed = _editor()
    ed.mirror = True
    ed.paint((3, 2), Tool.WALL)
    path = ed.save(tmp_path / "maps" / "mine.yaml")
    loaded = load_config(path)  # the same loader the environment/training uses
    assert (3, 2) in loaded.obstacles and (7, 2) in loaded.obstacles
    again = MapEditor.from_file(path, min_size=5, max_size=25)
    assert again.obstacles == ed.obstacles and again.is_symmetric()


# ------------------------------------------------------------------ controllers/session

def test_controller_specs_are_validated() -> None:
    specs = load_controller_specs(ROOT / "configs" / "viz" / "controllers.yaml")
    assert [s.kind for s in specs][:2] == ["rl", "scripted"]
    with pytest.raises(ConfigError, match="needs a checkpoint"):
        ControllerSpec("x", "rl", "", True, {})
    with pytest.raises(ConfigError, match="kind"):
        ControllerSpec("x", "wizard", "", True, {})


def test_session_records_a_complete_episode_with_labels() -> None:
    cfg = load_config(DEFAULT_MAP)
    factory = ControllerFactory()
    red = factory.build(ControllerSpec("Scripted", "scripted", "", True, SCRIPTED), cfg, 1)
    blue = factory.build(_controller("stay"), cfg, 2)
    rec = record_episode(cfg, red, blue, seed=3, note="test")
    assert rec.frames[0].t == 0 and rec.frames[-1].done and rec.outcome in {"draw", "RED wins", "BLUE wins"}
    first = rec.frames[0]
    assert [a.id for a in first.agents] == ["B1", "B2", "B3", "R1", "R2", "R3"]
    roles = {a.id: a.role for a in first.agents}
    assert roles["R1"] == "attacker" and roles["R2"] == "defender" and roles["B1"] == "unknown"
    assert all(a.action for a in first.agents if a.active)  # a decision exists for every active agent
    assert rec.frames[-1].agents[0].action == ""  # terminal frame: no decision
    assert set(first.belief) == {"RED", "BLUE"} and len(first.belief["RED"]["enemy_prob"]) == cfg.height
    assert rec.meta.teams["RED"].startswith("Scripted") and rec.meta.seed == 3


def test_session_is_reproducible_and_resettable() -> None:
    cfg = replace(load_config(DEFAULT_MAP), spawn_mode="near_base")
    factory = ControllerFactory()

    def run(seed):
        s = Session(cfg, factory.build(_controller("random"), cfg, 5), factory.build(_controller("random"), cfg, 6), seed=seed)
        s.run_to_end(40)
        return [[(a.x, a.y) for a in f.agents] for f in s.frames]

    assert run(7) == run(7) and run(7) != run(8)
    s = Session(cfg, factory.build(_controller("stay"), cfg, 1), factory.build(_controller("stay"), cfg, 2), seed=1)
    s.step(); s.step()
    assert len(s.frames) == 3
    s.reset()
    assert len(s.frames) == 1 and not s.done


def test_recording_json_round_trip_plain_and_gzip(tmp_path) -> None:
    cfg = load_config(DEFAULT_MAP)
    f = ControllerFactory()
    rec = record_episode(cfg, f.build(_controller("random"), cfg, 1), f.build(_controller("stay"), cfg, 2), seed=1)
    for name in ("ep.json", "ep.json.gz"):
        rec.save(tmp_path / name)
        back = Recording.load(tmp_path / name)
        assert back == rec
    (tmp_path / "bad.json").write_text('{"format_version": 99}')
    with pytest.raises(ConfigError, match="format"):
        Recording.load(tmp_path / "bad.json")
    assert isinstance(rec.event_steps(), list)


def test_session_with_events_marks_the_timeline() -> None:
    cfg = replace(load_config(DEFAULT_MAP), red_spawns=[(8, 3), (1, 2), (1, 4)], blue_spawns=[(9, 2), (9, 1), (9, 4)])
    f = ControllerFactory()
    red = f.build(ControllerSpec("Scripted", "scripted", "", True, {**SCRIPTED, "guard_mode": "leash", "guard_leash": 2}), cfg, 1)
    rec = record_episode(cfg, red, f.build(_controller("stay"), cfg, 2), seed=0)
    kinds = {k for _, k in rec.event_steps()}
    assert "flag_pickup" in kinds  # the attacker one step from the flag picks it up
