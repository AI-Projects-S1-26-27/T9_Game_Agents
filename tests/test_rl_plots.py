"""scripts/make_rl_plots.py: the numbers it draws and a small end-to-end run on synthetic files."""

from __future__ import annotations

import json
import sys

import pytest

pytest.importorskip("matplotlib")  # the plotting extra; the other tests do not need it

from rl.schema import ConfigError, from_dict, read_yaml
from rl_helpers import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import make_rl_plots as plots  # noqa: E402


def _cell(wins: int, n: int) -> dict:
    lo, hi = plots.wilson(wins, n)
    return {"all": {"win": {"count": wins, "n": n, "rate": wins / n, "lo": lo, "hi": hi}}}


def test_wilson_matches_the_interval_the_evaluation_stores() -> None:
    # 186 wins of 200 games: the lo / hi that rl.evaluate wrote into outputs/eval/v2_run2_9M.json
    lo, hi = plots.wilson(186, 200)
    assert lo == pytest.approx(0.8859451911438936, abs=1e-6)
    assert hi == pytest.approx(0.9578478281716571, abs=1e-6)
    assert plots.wilson(0, 0) == (0.0, 0.0)
    assert plots.wilson(0, 40)[0] == 0.0 and plots.wilson(40, 40)[1] == 1.0


def test_rolling_mean_and_series_order() -> None:
    assert plots.rolling([1.0, 2.0, 3.0, 4.0], 2) == [1.0, 1.5, 2.5, 3.5]
    rows = [{"step": 30, "k": 3.0}, {"step": 10, "k": 1.0}, {"step": 20}, {"step": 20, "k": None}]
    xs, ys = plots.series(rows, "k", 1)
    assert xs == [10, 30] and ys == [1.0, 3.0]  # sorted by step, rows without the key skipped


def test_held_out_seen_flag_and_the_fallback_for_old_files() -> None:
    held = {"center_gate": {"a": _cell(2, 10), "b": _cell(3, 10)}, "default": {"a": _cell(9, 10)}}
    new = plots.held_out_rates({"held_out": held, "held_out_seen": {"center_gate": True, "default": True}})
    assert new["center_gate"] == (5, 20, True)
    old = plots.held_out_rates({"held_out": held})  # no held_out_seen: only the default layout counts as seen
    assert old["center_gate"][2] is False and old["default"][2] is True


def test_plot_config_is_strict(tmp_path) -> None:
    cfg = from_dict(plots.PlotCfg, read_yaml(ROOT / "configs" / "plots" / "rl_plots.yaml"))
    assert cfg.smoothing_window > 0
    assert len(cfg.colors) >= 7 and len(set(cfg.colors)) == len(cfg.colors)  # the 7-map pool needs 7 distinguishable lines
    text = (ROOT / "configs" / "plots" / "rl_plots.yaml").read_text() + "surprise: 1\n"
    bad = tmp_path / "bad.yaml"
    bad.write_text(text)
    with pytest.raises(ConfigError):
        from_dict(plots.PlotCfg, read_yaml(bad))


def test_main_writes_all_four_pictures(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {"step": 1000 * i, "episode/win_rate": min(1.0, i / 10), "train/entropy": 1.0 / (i + 1),
         "episode/win_rate/map_default": 0.5, "episode/win_rate/map_open_field": 0.4}
        for i in range(1, 30)
    ]
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    ev = {
        "results": {"random": _cell(18, 20), "minimax": _cell(10, 20)},
        "held_out": {"open_field": {"leash": _cell(5, 10)}, "center_gate": {"leash": _cell(1, 10)}},
        "held_out_seen": {"open_field": True, "center_gate": False},
    }
    (tmp_path / "ev.json").write_text(json.dumps(ev))
    out = tmp_path / "plots"
    assert plots.main(["--run", f"R={run}", "--eval", f"E={tmp_path / 'ev.json'}", "--out", str(out)]) == 0
    for name in ("training_curves", "training_per_map", "eval_by_opponent", "eval_held_out_maps"):
        assert (out / f"{name}.png").stat().st_size > 1000, name
    # a run without map keys makes no per-map picture, the others are unaffected
    (run / "metrics.jsonl").write_text("\n".join(json.dumps({"step": i, "episode/win_rate": 0.5, "train/entropy": 1.0}) for i in range(5)))
    out2 = tmp_path / "plots2"
    plots.main(["--run", f"R={run}", "--out", str(out2)])
    assert (out2 / "training_curves.png").exists() and not (out2 / "training_per_map.png").exists()
