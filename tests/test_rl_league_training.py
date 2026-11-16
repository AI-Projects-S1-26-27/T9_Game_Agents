"""The trainer writes league snapshots, prunes them, logs league metrics and plays them in-process."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from league_helpers import ROOT, env_config_for, file_spec, league_config, league_spec, write_snapshot  # noqa: E402
from rl.checkpoint import load_policy  # noqa: E402
from rl.config import OpponentSpec  # noqa: E402
from rl.league import LeagueStore  # noqa: E402
from rl.metrics import read_metrics  # noqa: E402
from rl.schema import ConfigError  # noqa: E402
from rl.trainer import Trainer  # noqa: E402

STAY = OpponentSpec("stay", "stay", 1.0, {})


def _train(cfg, run_dir, resume=None):
    trainer = Trainer(cfg, env_config_for(cfg), run_dir, resume)
    return trainer, trainer.run()


def _rows(run_dir):
    return [r for r in read_metrics(run_dir / "metrics.jsonl") if "event" not in r]


def test_trainer_writes_snapshots_at_the_configured_cadence_and_prunes(tmp_path) -> None:
    cfg = league_config(tmp_path, snapshot_every=16, keep=2, total_steps=80)  # 5 updates of 16 steps
    _train(cfg, tmp_path / "run")
    snaps = LeagueStore(tmp_path / "run" / "league", 1).snapshots()
    assert [s for s, _ in snaps] == [64, 80]  # newest two of 16, 32, 48, 64, 80
    model, meta = load_policy(snaps[-1][1], "cpu")  # plays like any checkpoint
    assert meta["counters"]["total_steps"] == 80
    assert not list((tmp_path / "run" / "league").glob("*.tmp"))


def test_snapshot_cadence_can_be_coarser_than_an_update(tmp_path) -> None:
    cfg = league_config(tmp_path, snapshot_every=32, total_steps=80)
    _train(cfg, tmp_path / "run")
    assert [s for s, _ in LeagueStore(tmp_path / "run" / "league", 1).snapshots()] == [32, 64]


def test_league_off_writes_nothing_and_logs_no_league_metrics(tmp_path) -> None:
    cfg = league_config(tmp_path, opponents=[STAY], snapshot_every=0)
    trainer, _ = _train(cfg, tmp_path / "run")
    assert trainer.league is None and not (tmp_path / "run" / "league").exists()
    assert not any(k.startswith("league/") for k in _rows(tmp_path / "run")[0])


def test_league_metrics_are_logged_and_league_games_are_played(tmp_path) -> None:
    cfg = league_config(
        tmp_path, opponents=[STAY, league_spec(weight=30.0)], snapshot_every=16, total_steps=800,
        rollout_length=100, num_envs=2,
    )
    run = tmp_path / "run"
    write_snapshot(run / "league", 1, cfg)  # the league is not empty from the first update
    _train(cfg, run)
    rows = _rows(run)
    assert all("league/num_snapshots" in r for r in rows) and rows[-1]["league/num_snapshots"] >= 2
    played = [r for r in rows if "league/opponent_age_steps" in r]
    assert played, "no league game finished in the whole run"
    assert all(r["league/opponent_age_steps"] >= 0 and 0 < r["league/games_fraction"] <= 1 for r in played)
    assert any(k.endswith("/opp_league") for r in rows for k in r)  # per-label metrics cover league games


def test_resuming_keeps_the_league_and_continues_the_cadence(tmp_path) -> None:
    run = tmp_path / "run"
    _train(league_config(tmp_path, snapshot_every=32, total_steps=32), run)
    latest = run / "checkpoints" / "latest.pt"
    _train(league_config(tmp_path, snapshot_every=32, total_steps=96), run, latest)
    assert [s for s, _ in LeagueStore(run / "league", 1).snapshots()] == [32, 64, 96]


def test_a_missing_opponent_file_stops_the_run_before_training(tmp_path) -> None:
    cfg = league_config(tmp_path, opponents=[STAY, file_spec(tmp_path / "gone.pt")], snapshot_every=0)
    with pytest.raises(ConfigError, match="file not found"):
        Trainer(cfg, env_config_for(cfg), tmp_path / "run")


def test_file_opponent_trains_in_process(tmp_path) -> None:
    cfg0 = league_config(tmp_path, opponents=[STAY], snapshot_every=0)
    ckpt = write_snapshot(tmp_path / "pre", 16, cfg0)
    cfg = league_config(tmp_path, opponents=[STAY, file_spec(ckpt, weight=5.0)], snapshot_every=0, total_steps=64)
    _, counters = _train(cfg, tmp_path / "run")
    assert counters["total_steps"] == 64


def test_spawned_workers_load_and_play_snapshots(tmp_path) -> None:
    import numpy as np

    from rl.vec_env import make_vec_env

    cfg = league_config(tmp_path, opponents=[STAY, league_spec(weight=60.0)], num_envs=2, num_workers=2)
    league = tmp_path / "run" / "league"
    write_snapshot(league, 5, cfg)
    vec = make_vec_env(cfg, env_config_for(cfg), 0, league)
    try:
        vec.set_phase(0)
        vec.reset()
        infos = []
        for _ in range(205):  # an idle learner: every episode ends by the step limit (200)
            infos += vec.step(np.full((2, 3), 4, dtype=np.int64)).infos
    finally:
        vec.close()
    assert infos and any(i["snapshot_step"] == 5 and i["opponent"] == "league" for i in infos)
