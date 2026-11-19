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


# ------------------------------------------------- C1.1: league.directory


def test_the_league_folder_defaults_to_the_run_and_can_be_moved(tmp_path) -> None:
    from rl.league import resolve_league_dir

    cfg = league_config(tmp_path)
    assert resolve_league_dir(cfg, tmp_path / "run") == tmp_path / "run" / "league"
    moved = league_config(tmp_path, directory=str(tmp_path / "local"))
    assert resolve_league_dir(moved, tmp_path / "run") == tmp_path / "local"


def test_snapshots_go_to_league_directory_and_are_played_from_there(tmp_path) -> None:
    local = tmp_path / "local_league"
    cfg = league_config(
        tmp_path, opponents=[STAY, league_spec(weight=30.0)], snapshot_every=16, total_steps=800,
        rollout_length=100, num_envs=2, directory=str(local),
    )
    run = tmp_path / "run"
    trainer, _ = _train(cfg, run)
    assert trainer.league_dir == local
    assert [s for s, _ in LeagueStore(local, 1).snapshots()]  # written there ...
    assert not (run / "league").exists()  # ... and not into the run folder
    rows = _rows(run)
    assert any("league/opponent_age_steps" in r for r in rows)  # league games were played from the moved folder
    # the metric is logged before the same update's snapshot is written, so it can lag by one
    assert 2 <= rows[-1]["league/num_snapshots"] <= len(LeagueStore(local, 1).snapshots())


def test_moved_snapshots_still_play_like_any_checkpoint(tmp_path) -> None:
    from rl.rl_strategy import RLStrategy

    cfg = league_config(tmp_path, snapshot_every=16, total_steps=32, directory=str(tmp_path / "local"))
    _train(cfg, tmp_path / "run")
    path = LeagueStore(tmp_path / "local", 1).snapshots()[-1][1]
    model, meta = load_policy(path, "cpu")  # what the viewer uses
    assert meta["config"].league.directory == str(tmp_path / "local")
    assert RLStrategy.from_checkpoint(path).roles  # what rl.evaluate uses


def test_a_new_run_refuses_a_league_directory_that_holds_another_runs_snapshots(tmp_path) -> None:
    local = tmp_path / "local"
    cfg = league_config(tmp_path, snapshot_every=16, total_steps=16, directory=str(local))
    write_snapshot(local, 5, cfg)
    with pytest.raises(ConfigError, match="already holds snapshots"):
        Trainer(cfg, env_config_for(cfg), tmp_path / "run")
    assert [s for s, _ in LeagueStore(local, 1).snapshots()] == [5]  # nothing was touched
    # an empty or not yet existing folder is fine
    fresh = league_config(tmp_path, snapshot_every=16, total_steps=16, directory=str(tmp_path / "empty"))
    Trainer(fresh, env_config_for(fresh), tmp_path / "run2")


def test_resuming_keeps_snapshots_in_the_moved_league_directory(tmp_path) -> None:
    local, run = tmp_path / "local", tmp_path / "run"
    _train(league_config(tmp_path, snapshot_every=32, total_steps=32, directory=str(local)), run)
    latest = run / "checkpoints" / "latest.pt"
    _train(league_config(tmp_path, snapshot_every=32, total_steps=96, directory=str(local)), run, latest)  # not refused
    assert [s for s, _ in LeagueStore(local, 1).snapshots()] == [32, 64, 96]


def test_resuming_after_the_local_league_folder_was_lost_just_starts_with_an_empty_league(tmp_path) -> None:
    """Colab's local disk is wiped when the runtime restarts; the run must go on, refilling its league."""
    local, run = tmp_path / "local", tmp_path / "run"
    _train(league_config(tmp_path, snapshot_every=32, total_steps=32, directory=str(local)), run)
    import shutil

    shutil.rmtree(local)
    cfg = league_config(tmp_path, opponents=[STAY, league_spec(weight=30.0)], snapshot_every=32, total_steps=96, directory=str(local))
    _train(cfg, run, run / "checkpoints" / "latest.pt")
    assert [s for s, _ in LeagueStore(local, 1).snapshots()] == [64, 96]


def test_spawned_workers_play_the_moved_league_folder(tmp_path) -> None:
    local = tmp_path / "local"
    cfg = league_config(
        tmp_path, opponents=[STAY, league_spec(weight=60.0)], snapshot_every=16, total_steps=400,
        rollout_length=100, num_envs=2, num_workers=2, directory=str(local),
    )
    cfg_seed = league_config(tmp_path, opponents=[STAY])
    trainer = Trainer(cfg, env_config_for(cfg), tmp_path / "run")
    write_snapshot(local, 1, cfg_seed)  # the workers' folder is the moved one (a fresh run's trainer already checked it empty)
    trainer.run()
    rows = _rows(tmp_path / "run")
    assert any("league/opponent_age_steps" in r for r in rows)
