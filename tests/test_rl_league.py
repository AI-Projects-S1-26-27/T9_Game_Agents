"""League opponents: params and config checks, snapshot store, loading, opponent play, episode draw."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from random import Random

import pytest
import yaml

torch = pytest.importorskip("torch")

from ctf import CTFEnvironment, ObservationMode, Team  # noqa: E402
from ctf.actions import Action  # noqa: E402
from league_helpers import (  # noqa: E402
    ROOT,
    env_config_for,
    file_spec,
    league_config,
    league_spec,
    write_snapshot,
)
from rl.checkpoint import CheckpointError, CheckpointManager, migrate_config, restore_training_state  # noqa: E402
from rl.config import LeagueConfig, OpponentSpec, RLConfig, load_rl_config  # noqa: E402
from rl.env_wrapper import CTFTrainEnv  # noqa: E402
from rl.league import (  # noqa: E402
    CheckpointParams,
    LeagueStore,
    LeagueWriter,
    get_store,
    parse_params,
    uses_league_source,
)
from rl.opponents import build_opponent  # noqa: E402
from rl.rl_strategy import RLStrategy  # noqa: E402
from rl.schema import ConfigError, from_dict, to_dict  # noqa: E402
from rl_helpers import rl_config  # noqa: E402

STAY = OpponentSpec("stay", "stay", 1.0, {})


# ------------------------------------------------------------------------- params / config


def test_params_accept_both_sources_and_are_strict() -> None:
    p = parse_params({"source": "league", "window": 4, "path": "", "deterministic": False})
    assert isinstance(p, CheckpointParams) and p.window == 4
    assert parse_params({"source": "file", "window": 0, "path": "a.pt", "deterministic": True}).path == "a.pt"
    with pytest.raises(ConfigError, match="unknown key"):
        parse_params({"source": "league", "window": 0, "path": "", "deterministic": False, "extra": 1})
    with pytest.raises(ConfigError, match="missing required"):
        parse_params({"source": "league", "window": 0, "path": ""})
    with pytest.raises(ConfigError, match="not one of"):
        parse_params({"source": "pool", "window": 0, "path": "", "deterministic": True})


@pytest.mark.parametrize(
    "params, text",
    [
        ({"source": "league", "window": -1, "path": "", "deterministic": True}, "window must be >= 0"),
        ({"source": "league", "window": 0, "path": "x.pt", "deterministic": True}, "takes no path"),
        ({"source": "file", "window": 0, "path": "", "deterministic": True}, "needs a path"),
        ({"source": "file", "window": 3, "path": "x.pt", "deterministic": True}, "window only applies"),
    ],
)
def test_params_reject_inconsistent_combinations(params: dict, text: str) -> None:
    with pytest.raises(ConfigError, match=text):
        parse_params(params)


def test_config_needs_snapshots_and_a_non_league_opponent(tmp_path) -> None:
    ok = league_config(tmp_path)
    assert uses_league_source(ok)
    with pytest.raises(ConfigError, match="league would stay empty"):
        league_config(tmp_path, snapshot_every=0)
    only_league = [league_spec("a"), league_spec("b")]
    with pytest.raises(ConfigError, match="not a league opponent"):
        league_config(tmp_path, opponents=only_league)
    with pytest.raises(ConfigError, match="window must be"):  # parameters are checked when the config loads
        league_config(tmp_path, opponents=[STAY, league_spec(window=-2)])


def test_league_settings_are_validated(tmp_path) -> None:
    cfg = rl_config()
    for bad in (LeagueConfig(-1, 0, 1), LeagueConfig(0, -1, 1), LeagueConfig(0, 0, 0)):
        with pytest.raises(ConfigError):
            replace(cfg, league=bad)


def test_shipped_default_has_the_league_off_and_the_section_is_required() -> None:
    cfg = load_rl_config(ROOT / "configs" / "rl" / "default.yaml")
    assert cfg.league.snapshot_every_steps == 0 and not uses_league_source(cfg)
    data = yaml.safe_load((ROOT / "configs" / "rl" / "default.yaml").read_text())
    del data["league"]
    with pytest.raises(ConfigError, match="missing required"):
        from_dict(RLConfig, data)


def test_old_checkpoint_configs_get_the_league_off() -> None:
    plain = to_dict(rl_config())
    del plain["league"]
    migrated = migrate_config(plain)
    assert migrated["league"] == {"snapshot_every_steps": 0, "keep_snapshots": 0, "cache_size": 4}
    assert from_dict(RLConfig, migrated).league.snapshot_every_steps == 0
    assert migrate_config(migrated) == migrated  # idempotent


# ----------------------------------------------------------------------------- the store


def test_snapshots_are_listed_oldest_first_and_junk_is_ignored(tmp_path) -> None:
    cfg = league_config(tmp_path)
    league = tmp_path / "league"
    for step in (300, 100, 200):
        write_snapshot(league, step, cfg)
    (league / "snapshot_0000000400.pt.tmp").write_bytes(b"half written")
    (league / "notes.txt").write_text("x")
    store = LeagueStore(league, cache_size=2)
    assert [s for s, _ in store.snapshots()] == [100, 200, 300]
    assert LeagueStore(tmp_path / "missing", 1).snapshots() == []


def test_draw_respects_the_window_and_returns_none_when_empty(tmp_path) -> None:
    cfg = league_config(tmp_path)
    league = tmp_path / "league"
    store = LeagueStore(league, cache_size=4)
    assert store.draw(0, Random(0)) is None
    for step in (100, 200, 300, 400):
        write_snapshot(league, step, cfg)
    rng = Random(1)
    recent = {store.draw(2, rng)[0] for _ in range(60)}
    assert recent == {300, 400}
    assert {store.draw(0, rng)[0] for _ in range(80)} == {100, 200, 300, 400}
    assert {store.draw(10, rng)[0] for _ in range(80)} == {100, 200, 300, 400}  # window larger than the league


def test_a_snapshot_pruned_while_being_drawn_is_drawn_again(tmp_path, monkeypatch) -> None:
    cfg = league_config(tmp_path)
    league = tmp_path / "league"
    for step in (100, 200):
        write_snapshot(league, step, cfg)
    store = LeagueStore(league, cache_size=2)
    real_get, calls = store.cache.get, []

    def flaky(path):
        calls.append(path)
        if len(calls) == 1:
            path.unlink()  # pruned by the trainer between listing and loading
            raise FileNotFoundError(path)
        return real_get(path)

    monkeypatch.setattr(store.cache, "get", flaky)
    step, _, _ = store.draw(0, Random(0))
    assert len(calls) >= 2 and step in (100, 200) and len(store.snapshots()) == 1


def test_cache_keeps_only_the_newest_policies_and_notices_rewritten_files(tmp_path) -> None:
    cfg = league_config(tmp_path)
    league = tmp_path / "league"
    paths = [write_snapshot(league, s, cfg) for s in (100, 200, 300)]
    store = LeagueStore(league, cache_size=2)
    first = store.cache.get(paths[0])
    assert store.cache.get(paths[0]) is first  # cached
    store.cache.get(paths[1])
    store.cache.get(paths[2])
    assert len(store.cache._items) == 2
    assert store.cache.get(paths[0]) is not first  # evicted, loaded again
    write_snapshot(league, 100, cfg)  # same name, new content and modification time
    assert len(store.cache._items) <= 2
    assert get_store(league, 2) is get_store(league, 2)  # one store per folder and process


# ------------------------------------------------------------------------------ the writer


def test_writer_prunes_to_the_newest_and_snapshots_are_loadable_everywhere(tmp_path) -> None:
    cfg = league_config(tmp_path)
    league = tmp_path / "league"
    for step in (16, 32, 48, 64):
        write_snapshot(league, step, cfg, keep=2)
    assert [s for s, _ in LeagueStore(league, 1).snapshots()] == [48, 64]
    assert not list(league.glob("*.tmp"))
    latest = league / "snapshot_0000000064.pt"
    strat = RLStrategy.from_checkpoint(latest, deterministic=True)  # a snapshot plays like any checkpoint
    assert strat.checkpoint_step == 64
    env = CTFEnvironment(env_config_for(cfg))
    env.reset(seed=0)
    acts = strat.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
    assert set(acts) == {"R1", "R2", "R3"}


def test_training_cannot_resume_from_a_snapshot(tmp_path) -> None:
    from rl.curriculum import Curriculum
    from rl.mappo import MAPPO, ValueNorm
    from rl.networks import ActorCritic, NetworkSpec

    cfg = league_config(tmp_path)
    snap = write_snapshot(tmp_path / "league", 16, cfg)
    payload = torch.load(snap, weights_only=True)
    assert payload["weights_only"] is True and "optimizer" not in payload
    env_config = env_config_for(cfg)
    model = ActorCritic(NetworkSpec.from_env(env_config), cfg.model)
    algo = MAPPO(model, cfg.ppo, ValueNorm(True))
    with pytest.raises(CheckpointError, match="league snapshot"):
        restore_training_state(
            payload, model=model, optimizer=algo.optimizer, value_norm=ValueNorm(True),
            curriculum=Curriculum(cfg.curriculum.phases),
        )


# ------------------------------------------------------------------------ opponent play


def _play(env, red, blue, limit=60):
    for _ in range(limit):
        if env.done:
            break
        acts = red.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
        acts.update(blue.act(env.observe(ObservationMode.PARTIAL, team=Team.BLUE), Team.BLUE))
        env.step(acts)
    return env.state


def test_file_opponent_plays_from_either_side_and_only_for_active_agents(tmp_path) -> None:
    cfg = league_config(tmp_path)
    ckpt = write_snapshot(tmp_path / "league", 16, cfg)
    env_config = env_config_for(cfg)
    for side in (Team.RED, Team.BLUE):
        opp = build_opponent(file_spec(ckpt), env_config, Random(0))
        stay = build_opponent(STAY, env_config, Random(0))
        env = CTFEnvironment(env_config)
        env.reset(seed=3)
        acts = opp.act(env.observe(ObservationMode.PARTIAL, team=side), side)
        prefix = "R" if side is Team.RED else "B"
        assert set(acts) == {f"{prefix}1", f"{prefix}2", f"{prefix}3"} and all(isinstance(a, Action) for a in acts.values())
        red, blue = (opp, stay) if side is Team.RED else (stay, opp)
        _play(env, red, blue)  # a few steps without errors


def test_league_source_needs_a_store_and_a_missing_file_is_a_config_error(tmp_path) -> None:
    cfg = league_config(tmp_path)
    env_config = env_config_for(cfg)
    with pytest.raises(ConfigError, match="needs a training run"):
        build_opponent(league_spec(), env_config, Random(0))
    with pytest.raises(ConfigError, match="file not found"):
        build_opponent(file_spec(tmp_path / "nope.pt"), env_config, Random(0))
    empty = LeagueStore(tmp_path / "league", 1)
    with pytest.raises(ConfigError, match="league is empty"):
        build_opponent(league_spec(), env_config, Random(0), empty)


def test_opponent_for_a_different_team_size_is_refused(tmp_path) -> None:
    cfg = league_config(tmp_path)  # 3 agents per team
    ckpt = write_snapshot(tmp_path / "league", 16, cfg)
    four = env_config_for(cfg)
    four = replace(four, agents_per_team=4, red_spawns=None, blue_spawns=None, carrier_slots=None)
    with pytest.raises(ConfigError, match="3 agents per team"):
        build_opponent(file_spec(ckpt), four, Random(0))


def test_deterministic_flag_controls_argmax_versus_sampling(tmp_path) -> None:
    cfg = league_config(tmp_path)
    ckpt = write_snapshot(tmp_path / "league", 16, cfg)
    env_config = env_config_for(cfg)
    env = CTFEnvironment(env_config)
    env.reset(seed=1)
    obs = env.observe(ObservationMode.PARTIAL, team=Team.RED)
    greedy = {tuple(sorted(build_opponent(file_spec(ckpt, deterministic=True), env_config, Random(s)).act(obs, Team.RED).items())) for s in range(8)}
    assert len(greedy) == 1
    sampled = {tuple(sorted(build_opponent(file_spec(ckpt, deterministic=False), env_config, Random(s)).act(obs, Team.RED).items())) for s in range(30)}
    assert len(sampled) > 1  # an untrained network is nearly uniform: sampling varies


# ------------------------------------------------------------------- drawing in an episode


def _labels(env: CTFTrainEnv, n: int) -> Counter:
    seen = Counter()
    for _ in range(n):
        env.reset()
        seen[env._opponent_label] += 1
    return seen


def test_league_opponents_are_skipped_while_the_league_is_empty(tmp_path) -> None:
    cfg = league_config(tmp_path)
    env = CTFTrainEnv(cfg, env_config_for(cfg), seed=0, league_dir=tmp_path / "league")
    assert _labels(env, 25) == Counter({"stay": 25})


def test_league_games_appear_with_the_snapshot_step_once_the_league_has_one(tmp_path) -> None:
    cfg = league_config(tmp_path, opponents=[STAY, league_spec(window=1, weight=4.0)])
    league = tmp_path / "league"
    env = CTFTrainEnv(cfg, env_config_for(cfg), seed=0, league_dir=league)
    write_snapshot(league, 100, cfg)
    write_snapshot(league, 200, cfg)
    steps, labels = Counter(), Counter()
    for _ in range(40):
        env.reset()
        labels[env._opponent_label] += 1
        steps[env._opponent_snapshot_step] += 1
    assert labels["league"] > labels["stay"] > 0  # weights 4 : 1, so most games, not all
    assert set(steps) == {None, 200}  # window 1: only the newest snapshot, never the older one


def test_episode_summary_reports_the_snapshot_step(tmp_path) -> None:
    cfg = league_config(tmp_path, opponents=[STAY, league_spec(weight=50.0)])
    league = tmp_path / "league"
    write_snapshot(league, 123, cfg)
    env = CTFTrainEnv(cfg, env_config_for(cfg), seed=2, league_dir=league)
    step = env.reset()
    n = 0
    while not step.done and n < 400:
        step = env.step(Random(n).choices(range(5), k=env.num_agents))
        n += 1
    assert step.done
    assert step.info["opponent"] in {"stay", "league"}
    assert step.info["snapshot_step"] == (123 if step.info["opponent"] == "league" else None)


def test_a_league_config_without_a_league_dir_is_an_error(tmp_path) -> None:
    cfg = league_config(tmp_path)
    with pytest.raises(ValueError, match="league directory"):
        CTFTrainEnv(cfg, env_config_for(cfg), seed=0)


def test_old_configs_build_envs_without_a_league_dir() -> None:
    cfg = rl_config()
    env = CTFTrainEnv(cfg, env_config_for(cfg), seed=0)
    env.reset()
    assert env._league is None
