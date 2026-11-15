"""Visualisation of the v2 flag rules: recording metadata, old recordings, dropped-flag drawing."""

import gzip
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pytest.importorskip("numpy")
pytest.importorskip("pygame")

from ctf.config import load_config  # noqa: E402
from viz.controllers import ControllerFactory, ControllerSpec  # noqa: E402
from viz.draw import Geometry, ViewState  # noqa: E402
from viz.export import render_frames  # noqa: E402
from viz.frames import FORMAT_VERSION, Recording  # noqa: E402
from viz.session import record_episode  # noqa: E402
from viz.theme import load_viz_config, rgb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTED = {"roles": ["attacker", "defender", "scout"], "guard_mode": "on_flag", "guard_leash": 0, "scout_behavior": "stay"}


def _record(config_name: str) -> Recording:
    cfg = load_config(ROOT / "configs" / config_name)
    f = ControllerFactory()
    red = f.build(ControllerSpec("Scripted", "scripted", "", True, SCRIPTED), cfg, 1)
    blue = f.build(ControllerSpec("Stay", "stay", "", True, {}), cfg, 2)
    return record_episode(cfg, red, blue, seed=0)


def _view(cfg) -> ViewState:
    o = cfg.overlays
    return ViewState(fog=False, belief=False, vision=False, trails=False, probabilities=False, arrows=False,
                     labels=o.labels, coordinates=False, spawns=False, view_team=o.view_team)


@pytest.fixture(scope="module")
def v2_recording() -> Recording:
    return _record("ctf_v2.yaml")


@pytest.fixture()
def viz_cfg():
    return load_viz_config(ROOT / "configs" / "viz" / "default.yaml")


def test_recording_stores_the_rules_it_was_played_under(v2_recording) -> None:
    m = v2_recording.meta
    assert (m.carrier_slots, m.tagging_rule, m.drop_flag_on_tag) == ([1], "headcount", True)
    old = _record("default.yaml").meta
    assert (old.carrier_slots, old.tagging_rule, old.drop_flag_on_tag) == ([], "any_enemy", False)
    assert FORMAT_VERSION == 2


def test_recording_shows_a_dropped_flag_then_its_return(v2_recording) -> None:
    statuses = [next(fl.status for fl in f.flags if fl.team == "BLUE") for f in v2_recording.frames]
    drop = statuses.index("DROPPED")
    # Pickup and tag happen in the same step, so no frame ever shows the flag CARRIED here.
    assert statuses[drop - 1] == "AT_BASE" and statuses[drop + 1] == "AT_BASE"  # B2 stands on it: sent home
    kinds = [e["type"] for e in v2_recording.frames[drop].events]
    assert kinds == ["flag_pickup", "tag", "flag_drop"]
    assert ("flag_return" in {k for _, k in v2_recording.event_steps()})  # return is a timeline marker


def test_round_trip_keeps_the_rules(v2_recording, tmp_path) -> None:
    v2_recording.save(tmp_path / "ep.json.gz")
    assert Recording.load(tmp_path / "ep.json.gz") == v2_recording


def test_version_1_recordings_still_load_with_the_original_rules(tmp_path) -> None:
    rec = _record("default.yaml")
    rec.save(tmp_path / "new.json")
    data = json.loads((tmp_path / "new.json").read_text())
    data["format_version"] = 1
    for key in ("carrier_slots", "tagging_rule", "drop_flag_on_tag"):
        del data["meta"][key]  # exactly what a version-1 file looked like
    (tmp_path / "old.json").write_text(json.dumps(data))
    with gzip.open(tmp_path / "old.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(data, fh)
    for name in ("old.json", "old.json.gz"):
        back = Recording.load(tmp_path / name)
        assert back == rec and back.format_version == FORMAT_VERSION


def test_rules_text_describes_both_rule_sets(v2_recording) -> None:
    text = Geometry.from_meta(v2_recording.meta).rules_text()
    assert "only slot 1 carries" in text and "headcount tagging" in text and "flag drops where tagged" in text
    assert "any agent carries" in Geometry.from_meta(_record("default.yaml").meta).rules_text()
    assert "slot 1/3" in replace(Geometry.from_meta(v2_recording.meta), carrier_slots=(1, 3)).rules_text()


def test_a_dropped_flag_is_drawn_where_it_lies_and_only_then(v2_recording, viz_cfg) -> None:
    statuses = [next(fl.status for fl in f.flags if fl.team == "BLUE") for f in v2_recording.frames]
    drop, before = statuses.index("DROPPED"), statuses.index("DROPPED") - 1
    frames = render_frames(v2_recording, viz_cfg, _view(viz_cfg), size=(1360, 800), indices=[before, drop])
    ring = rgb(viz_cfg.palette.warn)

    def has_ring(img) -> bool:
        return bool(((img == ring).all(axis=2)).any())

    assert not has_ring(frames[0]) and has_ring(frames[1])  # the marker appears with the dropped flag


def test_an_unseen_dropped_flag_is_hidden_under_the_viewing_teams_fog(v2_recording, viz_cfg) -> None:
    drop = [next(fl.status for fl in f.flags if fl.team == "BLUE") for f in v2_recording.frames].index("DROPPED")
    # The carrier was tagged on BLUE's flag cell, so RED (whose carrier is now dead) has no one near it.
    # With RED's fog on, the dropped flag must not be drawn; with BLUE's fog on (its own flag), it must.
    ring = rgb(viz_cfg.palette.warn)
    for team, expected in (("red", False), ("blue", True)):
        view = replace(_view(viz_cfg), fog=True, view_team=type(_view(viz_cfg).view_team)(team))
        img = render_frames(v2_recording, viz_cfg, view, size=(1360, 800), indices=[drop])[0]
        assert bool(((img == ring).all(axis=2)).any()) is expected, team


def test_live_viewer_and_cli_run_on_the_v2_config(viz_cfg, tmp_path) -> None:
    import pygame

    from viz.__main__ import main
    from viz.viewer import Viewer

    specs = [
        ControllerSpec("Scripted: leash guard", "scripted", "", True, {**SCRIPTED, "guard_mode": "leash", "guard_leash": 2}),
        ControllerSpec("Random", "random", "", True, {}),
    ]
    env = load_config(ROOT / "configs" / "ctf_v2.yaml")
    v = Viewer(viz_cfg, env_config=env, specs=specs, mode="live", seed=1, red_index=0, blue_index=1)
    v.view.playing = False
    for _ in range(25):
        v.execute("step_forward")
    surf = pygame.Surface((1360, 800))
    v.render(surf)  # the side panel prints the rules and, when relevant, "DROPPED at (x,y)"
    assert Geometry.from_meta(v.session.recording().meta).rules_text().startswith("only slot 1 carries")

    out = tmp_path / "ep.json.gz"
    code = main(["record", "--map", str(ROOT / "configs" / "ctf_v2.yaml"), "--red", "scripted", "--blue", "random",
                 "--seed", "3", "--out", str(out)])
    assert code == 0 and Recording.load(out).meta.tagging_rule == "headcount"
