"""Multi-map training: the maps pool of a phase (config, env wrapper, metrics, startup checks, seen/unseen evaluation)."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from random import Random

import pytest
import yaml

import rl.env_wrapper as env_wrapper
from ctf.agent import Team
from ctf.config import load_config
from rl import evaluate as ev
from rl.checkpoint import migrate_config
from rl.config import PhaseMapSpec, RLConfig, load_base_env_config, load_rl_config, phase_map_bases
from rl.env_wrapper import CTFTrainEnv
from rl.eval_config import load_eval_config
from rl.eval_stats import rate
from rl.maps import bases_connected, layout_key, map_env_config, same_file, symmetry_problem, training_map_env_config
from rl.metrics import summarize_episodes
from rl.schema import ConfigError, from_dict, to_dict
from rl.train_eval import validate_map_pools
from rl.trainer import Trainer
from rl_helpers import ROOT, rl_config

REAL_PHASE = 2  # real_setting: the phase that carries the pool in configs/rl/default.yaml


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)  # the YAML files name maps and the eval config relative to the repository root


def v2_config() -> RLConfig:
    return load_rl_config(ROOT / "configs" / "rl" / "default.yaml")


def with_pool(cfg: RLConfig, maps: list[tuple[str, str, float]], phase: int = REAL_PHASE) -> RLConfig:
    """``cfg`` with the given ``(path, label, weight)`` pool on one phase."""
    phases = list(cfg.curriculum.phases)
    phases[phase] = replace(phases[phase], maps=[PhaseMapSpec(*m) for m in maps])
    return replace(cfg, curriculum=replace(cfg.curriculum, phases=phases))


def map_file(name: str) -> str:
    return f"configs/maps/{name}.yaml"


def write_map(tmp_path: Path, name: str, **changes) -> str:
    """A copy of the default map with some keys changed; returns its path."""
    data = yaml.safe_load((ROOT / "configs" / "maps" / "default.yaml").read_text())
    data.update(changes)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def make_env(cfg: RLConfig, seed: int = 1) -> CTFTrainEnv:
    env = CTFTrainEnv(cfg, load_base_env_config(cfg, ROOT), seed=seed)
    env.set_phase(REAL_PHASE)
    return env


# --------------------------------------------------------------------------- config and compat


def test_shipped_default_has_a_pool_only_in_the_last_phase():
    cfg = v2_config()
    assert [len(p.maps) for p in cfg.curriculum.phases] == [0, 0, 5]
    labels = [m.label for m in cfg.curriculum.phases[REAL_PHASE].maps]
    assert labels == ["default", "center_pillar", "open_field", "pillars", "two_corridors"]
    assert "center_gate" not in labels and "long_wall_south" not in labels  # the held-out maps stay unseen


def test_config_without_maps_is_rejected_by_the_strict_schema_but_migrates():
    plain = to_dict(v2_config())
    for phase in plain["curriculum"]["phases"]:
        del phase["maps"]  # what a checkpoint written before multi-map training holds
    with pytest.raises(ConfigError, match="maps"):
        from_dict(RLConfig, plain)
    migrated_plain = migrate_config(plain)
    assert all(p.get("maps") == [] for p in migrated_plain["curriculum"]["phases"])  # filled with the "no pool" value
    migrated = from_dict(RLConfig, migrated_plain)
    assert all(p.maps == [] for p in migrated.curriculum.phases)


def test_legacy_test_yaml_still_loads_with_empty_pools():
    assert all(p.maps == [] for p in rl_config().curriculum.phases)


def test_pool_validation_errors():
    cfg = v2_config()
    ok = [(map_file("default"), "default", 1.0)]
    with pytest.raises(ConfigError, match="labels"):
        with_pool(cfg, [(map_file("default"), "Default", 1.0)])  # replace() re-runs the RLConfig validation
    with pytest.raises(ConfigError, match="unique"):
        with_pool(cfg, ok + [(map_file("pillars"), "default", 1.0)])
    with pytest.raises(ConfigError, match="weights"):
        with_pool(cfg, [(map_file("default"), "default", 0.0)])


def test_pool_with_map_keys_in_env_overrides_is_rejected():
    cfg = v2_config()
    phases = list(cfg.curriculum.phases)
    phases[REAL_PHASE] = replace(phases[REAL_PHASE], env_overrides={**phases[REAL_PHASE].env_overrides, "obstacles": []})
    with pytest.raises(ConfigError, match="obstacles"):
        replace(cfg, curriculum=replace(cfg.curriculum, phases=phases))
    # the same override is fine without a pool (phases 1 and 2 use spawn overrides that way)
    phases[REAL_PHASE] = replace(phases[REAL_PHASE], maps=[])
    replace(cfg, curriculum=replace(cfg.curriculum, phases=phases))


# --------------------------------------------------------------------------- map checks


def test_every_shipped_map_is_a_valid_training_map():
    base = load_config(ROOT / "configs" / "ctf_v2.yaml")
    for path in sorted((ROOT / "configs" / "maps").glob("*.yaml")):
        config = training_map_env_config(base, path)
        assert symmetry_problem(config) is None and bases_connected(config)


def test_bad_training_maps_are_rejected(tmp_path):
    base = load_config(ROOT / "configs" / "ctf_v2.yaml")
    default_walls = yaml.safe_load((ROOT / "configs" / "maps" / "default.yaml").read_text())["obstacles"]
    asym = write_map(tmp_path, "asym", obstacles=default_walls + [[3, 1]])
    with pytest.raises(ConfigError, match="symmetric"):
        training_map_env_config(base, asym)
    wrong_size = write_map(tmp_path, "narrow", width=9)
    with pytest.raises(ConfigError, match="width"):
        training_map_env_config(base, wrong_size)
    sealed = write_map(tmp_path, "sealed", obstacles=[[5, y] for y in range(7)])  # symmetric, but no way across
    with pytest.raises(ConfigError, match="no path"):
        training_map_env_config(base, sealed)
    missing_key = tmp_path / "nokey.yaml"
    data = yaml.safe_load((ROOT / "configs" / "maps" / "default.yaml").read_text())
    del data["blue_spawns"]
    missing_key.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="missing key"):
        training_map_env_config(base, missing_key)


def test_same_file_and_layout_key(tmp_path):
    copy = tmp_path / "copy.yaml"
    shutil.copy(ROOT / "configs" / "maps" / "default.yaml", copy)
    assert same_file(copy, copy) and not same_file(copy, ROOT / "configs" / "maps" / "default.yaml")
    base = load_config(ROOT / "configs" / "ctf_v2.yaml")
    # default.yaml is the layout of ctf_v2.yaml; a copy under another name is still the same layout
    assert layout_key(map_env_config(base, copy)) == layout_key(base)
    assert layout_key(map_env_config(base, map_file("pillars"))) != layout_key(base)


def test_phase_map_bases_without_pool_is_the_base_itself():
    cfg = v2_config()
    base = load_base_env_config(cfg, ROOT)
    assert phase_map_bases(base, cfg.curriculum.phases[0]) == [("", 1.0, base)]
    pooled = phase_map_bases(base, cfg.curriculum.phases[REAL_PHASE])
    assert [label for label, _, _ in pooled] == ["default", "center_pillar", "open_field", "pillars", "two_corridors"]
    assert [w for _, w, _ in pooled] == [2.0, 1.0, 1.0, 1.0, 1.0]


# --------------------------------------------------------------------------- env wrapper


def test_a_map_is_drawn_per_episode_by_weight():
    cfg = with_pool(v2_config(), [(map_file("default"), "heavy", 9.0), (map_file("pillars"), "light", 1.0)])
    env = make_env(cfg, seed=5)
    drawn = []
    for _ in range(400):
        env.reset()
        drawn.append(env._map.label)
    heavy = drawn.count("heavy")
    assert 0 < drawn.count("light") < 100  # both maps appear, the light one rarely (expected 40 of 400)
    assert 330 <= heavy <= 390  # expected 360 of 400


class _SpyRandom(Random):
    """Counts the draws that pick a MAP (population of map runtimes), nothing else."""

    map_draws = 0

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        if population and isinstance(population[0], env_wrapper._MapRuntime):
            self.map_draws += 1
        return super().choices(population, weights, cum_weights=cum_weights, k=k)


def test_single_map_and_no_pool_draw_no_map_so_old_seeds_still_reproduce():
    episodes = 25
    draws = {}
    for name, pool in (("none", []), ("one", [(map_file("default"), "only", 1.0)]),
                       ("two", [(map_file("default"), "a", 1.0), (map_file("pillars"), "b", 1.0)])):
        env = make_env(with_pool(v2_config(), pool), seed=7)
        env._rng = _SpyRandom(7)
        for _ in range(episodes):
            env.reset()
        draws[name] = env._rng.map_draws
    assert draws == {"none": 0, "one": 0, "two": episodes}  # a map is drawn only when there is a choice
    # and a one-map pool plays exactly like no pool: same team / opponent / spawn-style draws, same generator state
    a, b = make_env(with_pool(v2_config(), []), seed=7), make_env(with_pool(v2_config(), [(map_file("default"), "only", 1.0)]), seed=7)
    for _ in range(episodes):
        a.reset()
        b.reset()
        assert (a._team, a._opponent_label, a._fixed_spawns) == (b._team, b._opponent_label, b._fixed_spawns)
    assert a._rng.getstate() == b._rng.getstate()


def test_episode_summary_carries_the_map_label():
    pooled = make_env(with_pool(v2_config(), [(map_file("pillars"), "pillars", 1.0), (map_file("open_field"), "open_field", 1.0)]))
    plain = make_env(with_pool(v2_config(), []))
    for env, allowed in ((pooled, {"pillars", "open_field"}), (plain, {""})):
        env.reset()
        for _ in range(env.base_env_config.max_episode_steps + 1):
            step = env.step([0] * env.num_agents)
            if step.done:
                break
        assert step.done and step.info["map"] in allowed


def test_each_map_has_its_own_path_distances_for_the_rewards():
    cfg = with_pool(v2_config(), [(map_file("default"), "default", 1.0), (map_file("long_wall_south"), "long_wall", 1.0)])
    env = make_env(cfg)
    env.reset()
    by_label = {m.label: m for m in env._runtime.maps}
    right, left = (6, 0), (4, 0)  # either side of the wall's top gap: 2 steps on default, a long detour via the bottom gap on long_wall_south
    d_default = by_label["default"].rewards[Team.RED].nav.distance(right, left)
    d_long = by_label["long_wall"].rewards[Team.RED].nav.distance(right, left)
    assert d_default == 2 and d_long == 14
    assert by_label["long_wall"].env_config.obstacles != by_label["default"].env_config.obstacles


def test_the_opponent_is_built_for_the_drawn_map(monkeypatch):
    cfg = with_pool(v2_config(), [(map_file("default"), "default", 1.0), (map_file("pillars"), "pillars", 1.0)])
    env = make_env(cfg, seed=3)
    seen_configs = []
    real_build = env_wrapper.build_opponent

    def spy(spec, env_config, rng):
        seen_configs.append(env_config)
        return real_build(spec, env_config, rng)

    monkeypatch.setattr(env_wrapper, "build_opponent", spy)
    labels = set()
    for _ in range(30):
        seen_configs.clear()
        env.reset()
        labels.add(env._map.label)
        assert seen_configs and seen_configs[-1] is env._map.env_config
        assert tuple(seen_configs[-1].obstacles) == tuple(env._env.config.obstacles)
    assert labels == {"default", "pillars"}


# --------------------------------------------------------------------------- metrics


def _info(map_label: str, outcome: str, opponent: str = "random") -> dict:
    return {
        "outcome": outcome, "length": 10, "learner_team": "RED", "opponent": opponent, "fixed_spawns": False, "map": map_label,
        "pickups": 1, "tags_inflicted": 0, "tags_suffered": 0, "return": [0.0, 0.0], "components": {"step": [0.0, 0.0]},
    }


def test_per_map_win_rate_series():
    out = summarize_episodes(
        [_info("a", "win"), _info("a", "loss"), _info("b", "win"), _info("b", "win")], ["attacker", "scout"]
    )
    assert "win_rate/map_a" in out and "win_rate/map_b" in out
    assert out["win_rate/map_a"] == 0.5 and out["win_rate/map_b"] == 1.0
    assert out["win_rate/opp_random"] == 0.75  # the per-opponent series are still there
    plain = summarize_episodes([_info("", "win")], ["attacker", "scout"])
    assert not any(k.startswith("win_rate/map_") for k in plain)  # no pool = no per-map series


# --------------------------------------------------------------------------- startup check


def _eval_yaml_with_held_out(tmp_path: Path, held: list[tuple[str, str]]) -> str:
    data = yaml.safe_load((ROOT / "configs" / "rl" / "eval_train.yaml").read_text())
    data["held_out"]["maps"] = [{"label": label, "path": path} for label, path in held]
    path = tmp_path / "eval_overlap.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_shipped_default_passes_the_pool_check():
    cfg = v2_config()
    validate_map_pools(cfg, load_base_env_config(cfg, ROOT), load_eval_config(ROOT / "configs" / "rl" / "eval_train.yaml"))


def test_pool_overlapping_the_held_out_maps_is_rejected(tmp_path):
    cfg = v2_config()
    base = load_base_env_config(cfg, ROOT)
    eval_cfg = load_eval_config(_eval_yaml_with_held_out(tmp_path, [("center_pillar", map_file("center_pillar"))]))
    with pytest.raises(ConfigError, match="center_pillar") as err:
        validate_map_pools(cfg, base, eval_cfg)
    assert "real_setting" in str(err.value)
    # the same layout under another file name counts as the same map
    copy = tmp_path / "pillars_copy.yaml"
    shutil.copy(ROOT / "configs" / "maps" / "pillars.yaml", copy)
    eval_cfg = load_eval_config(_eval_yaml_with_held_out(tmp_path, [("copy", str(copy))]))
    with pytest.raises(ConfigError, match="never trains on"):
        validate_map_pools(cfg, base, eval_cfg)


def test_missing_or_invalid_pool_map_is_rejected_at_startup(tmp_path):
    base = load_base_env_config(v2_config(), ROOT)
    with pytest.raises(ConfigError, match="not found"):
        validate_map_pools(with_pool(v2_config(), [("configs/maps/nope.yaml", "nope", 1.0)]), base)
    asym = write_map(tmp_path, "asym", obstacles=[[3, 1]])
    with pytest.raises(ConfigError, match="symmetric"):
        validate_map_pools(with_pool(v2_config(), [(asym, "asym", 1.0)]), base)


def test_trainer_runs_the_pool_check_at_startup(tmp_path):
    cfg = v2_config()
    cfg = replace(
        cfg,
        run=replace(cfg.run, output_dir=str(tmp_path), name="t", tensorboard=False, device="cpu"),
        model=replace(cfg.model, conv_channels=[4], hidden_size=16),
        noise_free_eval=replace(
            cfg.noise_free_eval, config_path=_eval_yaml_with_held_out(tmp_path, [("center_pillar", map_file("center_pillar"))])
        ),
    )
    with pytest.raises(ConfigError, match="held-out map"):
        Trainer(cfg, load_base_env_config(cfg, ROOT), tmp_path / "run")


# --------------------------------------------------------------------------- seen / unseen evaluation


class _Learner:
    def __init__(self, config: RLConfig) -> None:
        self.config = config


def test_held_out_maps_are_marked_seen_or_unseen():
    eval_cfg = load_eval_config(ROOT / "configs" / "rl" / "eval.yaml")  # lists all six maps
    pooled = v2_config()
    base = load_base_env_config(pooled, ROOT)
    seen = ev.held_out_seen(_Learner(pooled), base, eval_cfg)
    assert seen == {
        "center_gate": False, "center_pillar": True, "long_wall_south": False,
        "open_field": True, "pillars": True, "two_corridors": True,
    }
    old = ev.held_out_seen(_Learner(with_pool(pooled, [])), base, eval_cfg)  # a Run 1 style checkpoint: nothing but the base map
    assert not any(old.values())
    assert ev.held_out_seen(_Learner(pooled), base, replace(eval_cfg, held_out=replace(eval_cfg.held_out, maps=[]))) is None


def test_report_groups_seen_and_unseen_maps():
    stat = lambda k: {"n": 40, "win": rate(k, 40, 0.95), "draw": rate(40 - k, 40, 0.95), "loss": rate(0, 40, 0.95)}  # noqa: E731
    held = {"pillars": {"scripted_leash": {"all": stat(30)}}, "center_gate": {"scripted_leash": {"all": stat(10)}}}
    meta = {"confidence": 0.95}
    text = ev.format_extras({"held_out": held, "held_out_seen": {"pillars": True, "center_gate": False}}, meta)
    seen_part, unseen_part = text.split("unseen maps:")
    assert "seen maps:" in seen_part and "pillars" in seen_part and "center_gate" not in seen_part
    assert "center_gate" in unseen_part and "pillars" not in unseen_part
    # an old JSON without the flags: every map is reported as unseen
    old = ev.format_extras({"held_out": held}, meta)
    assert "unseen maps:" in old and "\nseen maps:" not in old


def test_evaluate_extras_returns_the_seen_flags():
    eval_cfg = load_eval_config(ROOT / "configs" / "rl" / "eval.yaml")
    cfg = v2_config()
    base = load_base_env_config(cfg, ROOT)
    skipped = ev.evaluate_extras(_Learner(cfg), list(cfg.roles.slots), base, eval_cfg, skip=["held_out", "role_swap", "policy_check"])
    assert skipped["held_out"] is None and skipped["held_out_seen"] is None


# --------------------------------------------------------------------------- end to end


def test_a_short_training_run_on_the_pool_logs_every_map(tmp_path):
    from rl.metrics import read_metrics

    cfg = with_pool(v2_config(), [(map_file("default"), "default", 1.0), (map_file("pillars"), "pillars", 1.0)])
    cfg = replace(
        cfg,
        run=replace(cfg.run, output_dir=str(tmp_path), name="t", tensorboard=False, device="cpu", torch_threads=1),
        model=replace(cfg.model, conv_channels=[4], hidden_size=16),
        rollout=replace(cfg.rollout, num_envs=4, num_workers=0, rollout_length=100, total_steps=1200),
        curriculum=replace(cfg.curriculum, phases=[cfg.curriculum.phases[REAL_PHASE]]),  # only the phase with the pool
        noise_free_eval=replace(cfg.noise_free_eval, enabled=False),
    )
    run_dir = tmp_path / "run"
    Trainer(cfg, load_base_env_config(cfg, ROOT), run_dir).run()
    keys = {k for rec in read_metrics(run_dir / "metrics.jsonl") for k in rec}
    assert "episode/win_rate/map_default" in keys and "episode/win_rate/map_pillars" in keys
