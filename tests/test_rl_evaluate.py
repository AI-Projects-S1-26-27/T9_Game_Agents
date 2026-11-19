"""Evaluation: statistics, strict eval config, per-episode metrics on hand-checkable games, determinism, CLI."""

from __future__ import annotations

import json
from dataclasses import replace
from random import Random

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from ctf import CTFConfig, CTFEnvironment  # noqa: E402
from ctf.actions import Action  # noqa: E402
from ctf.agent import Team  # noqa: E402
from ctf.config import load_config  # noqa: E402
from rl import evaluate as ev  # noqa: E402
from rl.checkpoint import save_checkpoint  # noqa: E402
from rl.config import load_base_env_config  # noqa: E402
from rl.eval_config import (  # noqa: E402
    EvalConfig, EvalOpponent, HeldOutConfig, HeldOutMap, PolicyCheckConfig, PolicyMode, RoleSwapConfig, Side,
    load_eval_config,
)
from rl.eval_stats import mean_stat, rate, wilson_interval, z_value  # noqa: E402
from rl.mappo import ValueNorm  # noqa: E402
from rl.networks import ActorCritic, NetworkSpec  # noqa: E402
from rl.opponents import GuardMode, ScoutBehavior, ScriptedParams, ScriptedTeamStrategy  # noqa: E402
from rl.rl_strategy import RLStrategy  # noqa: E402
from rl.schema import ConfigError  # noqa: E402
from rl.trainer import Trainer  # noqa: E402
from rl_helpers import ROOT, rl_config, small_env_config, tiny_config  # noqa: E402
from strategies import BaseStrategy, Role, StayStrategy  # noqa: E402

ROLES3 = [Role.ATTACKER, Role.DEFENDER, Role.SCOUT]


# ----------------------------------------------------------------------------- statistics


def test_wilson_matches_textbook_values() -> None:
    lo, hi = wilson_interval(5, 10, 0.95)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = wilson_interval(0, 10, 0.95)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson_interval(10, 10, 0.95)
    assert lo == pytest.approx(0.7225, abs=1e-3) and hi == 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_is_narrower_with_more_data_and_wider_at_higher_confidence() -> None:
    small = wilson_interval(15, 30, 0.95)
    large = wilson_interval(150, 300, 0.95)
    assert (large[1] - large[0]) < (small[1] - small[0])
    wide = wilson_interval(15, 30, 0.99)
    assert (wide[1] - wide[0]) > (small[1] - small[0])
    assert z_value(0.95) == pytest.approx(1.95996, abs=1e-4)


def test_wilson_and_z_reject_bad_input() -> None:
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
    with pytest.raises(ValueError):
        z_value(1.0)


def test_rate_and_mean_stat() -> None:
    r = rate(3, 4, 0.95)
    assert r["rate"] == 0.75 and r["count"] == 3 and r["n"] == 4 and r["lo"] < 0.75 < r["hi"]
    assert rate(0, 0, 0.95)["rate"] is None
    m = mean_stat([1.0, 2.0, 3.0, 4.0], 0.95)
    assert m["mean"] == 2.5 and m["se"] == pytest.approx(0.6455, abs=1e-3)
    assert m["lo"] < 2.5 < m["hi"]
    assert mean_stat([5.0], 0.95)["lo"] is None and mean_stat([], 0.95)["mean"] is None


# -------------------------------------------------------------------------- eval config


def _eval_yaml() -> dict:
    return yaml.safe_load((ROOT / "configs" / "rl" / "eval.yaml").read_text())


def test_shipped_eval_configs_load_and_build_their_opponents() -> None:
    for name, env_yaml in (("eval.yaml", "ctf_v2.yaml"), ("eval_legacy.yaml", "default.yaml")):
        cfg = load_eval_config(ROOT / "configs" / "rl" / name)
        env_config = load_config(ROOT / "configs" / env_yaml)
        assert cfg.episodes == 200 and cfg.sides == [Side.RED, Side.BLUE] and cfg.policy_mode is PolicyMode.ARGMAX
        for opp in cfg.opponents:
            assert ev.build_eval_opponent(opp, env_config, Random(0)) is not None
    v2 = load_eval_config(ROOT / "configs" / "rl" / "eval.yaml")
    assert {o.label for o in v2.opponents} >= {"random", "stay", "scripted_chaser", "minimax"}


def test_eval_config_is_strict(tmp_path) -> None:
    def write(data: dict):
        path = tmp_path / "e.yaml"
        path.write_text(yaml.safe_dump(data))
        return path

    base = _eval_yaml()
    assert load_eval_config(write(base)).episodes == 200
    with pytest.raises(ConfigError, match="opponent_noise"):  # evaluation has no noise setting at all
        load_eval_config(write({**base, "opponent_noise": 0.1}))
    missing = {k: v for k, v in base.items() if k != "seed"}
    with pytest.raises(ConfigError, match="seed"):
        load_eval_config(write(missing))
    with pytest.raises(ConfigError, match="divisible"):
        load_eval_config(write({**base, "episodes": 7}))
    with pytest.raises(ConfigError, match="unique"):
        load_eval_config(write({**base, "sides": ["red", "red"]}))
    with pytest.raises(ConfigError, match="confidence"):
        load_eval_config(write({**base, "confidence": 1.0}))
    with pytest.raises(ConfigError, match="fixed_spawn_fraction"):
        load_eval_config(write({**base, "fixed_spawn_fraction": 1.5}))
    dup = [*base["opponents"], base["opponents"][0]]
    with pytest.raises(ConfigError, match="unique"):
        load_eval_config(write({**base, "opponents": dup}))
    with pytest.raises(ConfigError, match="labels"):
        load_eval_config(write({**base, "opponents": [{"kind": "stay", "label": "Bad Label", "params": {}}]}))


def test_overrides_go_through_the_same_validation() -> None:
    path = ROOT / "configs" / "rl" / "eval.yaml"
    assert load_eval_config(path, {"episodes": 40, "policy_mode": "sampled"}).policy_mode is PolicyMode.SAMPLED
    with pytest.raises(ConfigError):
        load_eval_config(path, {"episodes": 41})
    with pytest.raises(ConfigError, match="unknown key"):
        load_eval_config(path, {"nonsense": 1})


def test_unknown_opponent_kind_and_external_registry(monkeypatch) -> None:
    env_config = small_env_config()
    odd = EvalOpponent(kind="human", label="human", params={})  # not built in: an outside opponent goes through EXTERNAL_OPPONENTS
    with pytest.raises(ConfigError, match="human"):
        ev.build_eval_opponent(odd, env_config, Random(0))
    monkeypatch.setitem(ev.EXTERNAL_OPPONENTS, "human", lambda spec, cfg, rng: StayStrategy())
    assert isinstance(ev.build_eval_opponent(odd, env_config, Random(0)), StayStrategy)


# -------------------------------------------------------------------------- environments


def test_eval_env_config_applies_overrides_and_fixed_spawns() -> None:
    base = load_config(ROOT / "configs" / "ctf_v2.yaml")
    random_cfg = ev.eval_env_config(base, {"spawn_mode": "near_base", "visibility_radius": 6}, fixed_spawns=False)
    assert random_cfg.spawn_mode.value == "near_base" and random_cfg.visibility_radius == 6
    fixed_cfg = ev.eval_env_config(base, {"spawn_mode": "near_base"}, fixed_spawns=True)
    assert fixed_cfg.spawn_mode.value == "fixed"
    with pytest.raises(ConfigError, match="env_overrides"):
        ev.eval_env_config(base, {"no_such_key": 1}, fixed_spawns=False)


def test_mirror_symmetry_detection() -> None:
    assert ev.is_mirror_symmetric(load_config(ROOT / "configs" / "ctf_v2.yaml"))
    lopsided = CTFConfig(obstacles=[(2, 1)], red_base=(1, 3), blue_base=(9, 3))
    assert not ev.is_mirror_symmetric(lopsided)


# ------------------------------------------------- one game with hand-checkable numbers


class _Script(BaseStrategy):
    """Plays a fixed action list per agent (agents not listed STAY; the list's last action repeats)."""

    def __init__(self, plans: dict[str, list[Action]]) -> None:
        self.plans, self.t = plans, 0

    def reset(self) -> None:
        self.t = 0

    def act(self, observation, team):
        out = {}
        for aid, agent in observation.agents.items():
            if agent.team is team and agent.active:
                plan = self.plans.get(aid, [Action.STAY])
                out[aid] = plan[min(self.t, len(plan) - 1)]
        self.t += 1
        return out


def _play(env_config: CTFConfig, learner, opponent, team=Team.RED, route_roles=frozenset({Role.ATTACKER}), mirrored=False):
    env = CTFEnvironment(env_config)
    return ev.play_episode(env, learner, ROLES3, opponent, team, set(route_roles), 0, mirrored)


def test_capture_game_metrics() -> None:
    # R1 walks (3,1) -> BLUE flag (4,1) -> home (0,1): pickup at step 1, capture at step 5.
    learner = _Script({"R1": [Action.RIGHT, Action.LEFT, Action.LEFT, Action.LEFT, Action.LEFT]})
    rec = _play(small_env_config(), learner, StayStrategy())
    assert (rec.outcome, rec.length, rec.side) == ("win", 5, "red")
    assert (rec.captures, rec.pickups, rec.first_pickup) == (1, 1, 1)
    assert (rec.tags_inflicted, rec.tags_suffered) == (0, 0)
    # action counts per role (attacker, defender, scout): attacker 1 RIGHT + 4 LEFT, the rest STAY x5
    names = [a.name for a in Action]
    att = dict(zip(names, rec.action_counts[0]))
    assert att["RIGHT"] == 1 and att["LEFT"] == 4 and att["STAY"] == 0
    assert rec.action_counts[1][names.index("STAY")] == 5 and rec.action_counts[2][names.index("STAY")] == 5
    # route: R1 enters the centre column (x=2) once, on the middle row, while carrying the flag
    assert rec.route == {"top": 0, "middle": 1, "bottom": 0}
    assert rec.route_carrying == {"top": 0, "middle": 1, "bottom": 0}
    # the defender sits at (3,0): 4 cells from its own base/flag (0,1) the whole game
    assert rec.defender_dist_base == 4.0 and rec.defender_dist_flag == 4.0
    # the scout sits at (3,2), radius 2 on a 5x3 board: 8 of 15 cells in sight
    assert rec.scout_coverage == pytest.approx(8 / 15)
    assert rec.heat[0].sum() == 6  # initial position + one per step


def test_tag_game_counts_tags_and_pickup_without_capture() -> None:
    # B1 steps onto the BLUE flag cell in the same step R1 picks the flag up: R1 is tagged at once.
    cfg = small_env_config(blue_spawns=[(4, 0), (0, 2), (1, 2)], max_episode_steps=6)
    learner = _Script({"R1": [Action.RIGHT]})
    opponent = _Script({"B1": [Action.DOWN]})
    rec = _play(cfg, learner, opponent)
    assert (rec.pickups, rec.captures, rec.first_pickup) == (1, 0, 1)
    assert (rec.tags_suffered, rec.tags_inflicted) == (1, 0)
    assert rec.outcome == "draw" and rec.length == 6
    # seen from BLUE's side the same game has the tag inflicted instead
    swapped = _play(cfg, _Script({"B1": [Action.DOWN]}), _Script({"R1": [Action.RIGHT]}), team=Team.BLUE)
    assert (swapped.tags_inflicted, swapped.tags_suffered, swapped.pickups) == (1, 0, 0)


def test_route_lanes_and_blue_mirroring() -> None:
    # 7-wide, 5-high open board; centre column x=3, middle row y=2. R1 goes up to y=0, then right.
    cfg = CTFConfig(width=7, height=5, agents_per_team=3, red_base=(0, 2), blue_base=(6, 2),
                    red_spawns=[(1, 0), (1, 1), (1, 2)], blue_spawns=[(5, 0), (5, 1), (5, 2)],
                    visibility_radius=2, max_episode_steps=4)
    rec = _play(cfg, _Script({"R1": [Action.RIGHT]}), StayStrategy())
    assert rec.route == {"top": 1, "middle": 0, "bottom": 0}  # entered x=3 on row 0 (< 2)
    assert rec.route_carrying["top"] == 0
    # BLUE learner standing still on a symmetric map: team-relative heatmap flips x
    blue = _play(cfg, StayStrategy(), StayStrategy(), team=Team.BLUE, mirrored=True)
    s = ev.summarize([blue], ["attacker", "defender", "scout"], 0.95, team_relative=True)
    assert s["heatmaps"]["attacker"][0][1] == pytest.approx(1.0)  # B1 at (5,0) -> team-relative x = 1
    raw = ev.summarize([blue], ["attacker", "defender", "scout"], 0.95, team_relative=False)
    assert raw["heatmaps"]["attacker"][0][5] == pytest.approx(1.0)  # one side only: environment coordinates


def test_summary_rates_and_no_heatmap_for_mixed_sides_without_symmetry() -> None:
    win = _play(small_env_config(), _Script({"R1": [Action.RIGHT, Action.LEFT]}), StayStrategy())
    draw = _play(small_env_config(max_episode_steps=3), StayStrategy(), StayStrategy())
    draw_blue = _play(small_env_config(max_episode_steps=3), StayStrategy(), StayStrategy(), team=Team.BLUE)
    names = ["attacker", "defender", "scout"]
    s = ev.summarize([win, draw, draw_blue], names, 0.95, team_relative=False)
    assert s["n"] == 3 and s["win"]["count"] == 1 and s["draw"]["count"] == 2 and s["loss"]["count"] == 0
    assert s["win"]["lo"] < 1 / 3 < s["win"]["hi"]
    assert s["heatmaps"] is None  # two sides, no symmetry: no honest single frame
    assert s["pickup_to_capture"]["n"] == 1  # one pickup in total (the winning game)
    assert s["episodes_with_pickup"]["count"] == 1
    assert s["stay_fraction"]["scout"] == 1.0 and 0 < s["stay_fraction"]["all"] < 1
    assert sum(s["action_histograms"]["attacker"].values()) == pytest.approx(1.0)


def test_no_scout_role_gives_no_coverage() -> None:
    env = CTFEnvironment(small_env_config(max_episode_steps=2))
    rec = ev.play_episode(env, StayStrategy(), [Role.ATTACKER, Role.ATTACKER, Role.DEFENDER], StayStrategy(),
                          Team.RED, {Role.ATTACKER}, 0)
    assert rec.scout_coverage is None
    s = ev.summarize([rec], ["attacker", "defender"], 0.95, team_relative=False)
    assert s["scout_coverage"] is None and "scout" not in s["stay_fraction"]


# ---------------------------------------------------------------- driver: determinism etc.


def _cfg(**kw) -> EvalConfig:
    base = dict(
        episodes=6, seed=7, confidence=0.95, sides=[Side.RED, Side.BLUE], policy_mode=PolicyMode.ARGMAX,
        fixed_spawn_fraction=0.0, env_overrides={"spawn_mode": "near_base"}, route_roles=[Role.ATTACKER],
        opponents=[EvalOpponent("random", "random", {}), EvalOpponent("stay", "stay", {})],
        held_out=HeldOutConfig(episodes=2, opponents=[], maps=[]),
        role_swap=RoleSwapConfig(enabled=False, episodes=2, opponents=[], swaps=[]),
        policy_check=PolicyCheckConfig(enabled=False, episodes=2, opponents=[]),
    )
    base.update(kw)
    return EvalConfig(**base)


def _scripted_learner(env_config: CTFConfig) -> ScriptedTeamStrategy:
    params = ScriptedParams(ROLES3, GuardMode.LEASH, 1, ScoutBehavior.RANDOM, 0, False)
    return ScriptedTeamStrategy(env_config, params, Random(3))


def test_evaluate_policy_structure_and_determinism() -> None:
    env_config = small_env_config(max_episode_steps=30)
    cfg = _cfg()
    first = ev.evaluate_policy(_scripted_learner(env_config), ROLES3, env_config, cfg)
    again = ev.evaluate_policy(_scripted_learner(env_config), ROLES3, env_config, cfg)
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)  # same seed, same games
    assert list(first) == ["random", "stay"]
    block = first["random"]
    assert set(block) == {"all", "red", "blue"}
    assert block["all"]["n"] == 6 and block["red"]["n"] == 3 and block["blue"]["n"] == 3
    for side in block.values():
        total = side["win"]["count"] + side["draw"]["count"] + side["loss"]["count"]
        assert total == side["n"]
    other_seed = ev.evaluate_policy(_scripted_learner(env_config), ROLES3, env_config, _cfg(seed=8))
    assert json.dumps(other_seed, sort_keys=True) != json.dumps(first, sort_keys=True)


def test_single_side_and_opponent_filter_and_errors() -> None:
    env_config = small_env_config(max_episode_steps=10)
    cfg = _cfg(sides=[Side.BLUE], episodes=3)
    out = ev.evaluate_policy(_scripted_learner(env_config), ROLES3, env_config, cfg, only=["stay"])
    assert list(out) == ["stay"] and set(out["stay"]) == {"all", "blue"} and out["stay"]["blue"]["n"] == 3
    with pytest.raises(ConfigError, match="unknown opponent label"):
        ev.evaluate_policy(_scripted_learner(env_config), ROLES3, env_config, cfg, only=["nope"])
    with pytest.raises(ConfigError, match="roles"):
        ev.evaluate_policy(_scripted_learner(env_config), ROLES3[:2], env_config, cfg)


def test_fixed_spawn_fraction_uses_fixed_spawns() -> None:
    env_config = small_env_config(max_episode_steps=2)
    cfg = _cfg(fixed_spawn_fraction=1.0, sides=[Side.RED], episodes=2, opponents=[EvalOpponent("stay", "stay", {})])
    out = ev.evaluate_policy(StayStrategy(), ROLES3, env_config, cfg)
    heat = out["stay"]["red"]["heatmaps"]
    # every episode started on the fixed spawns: the attacker (slot 1) only ever stood on (3,1)
    assert heat["attacker"][1][3] == pytest.approx(1.0)


def test_report_text_has_a_row_per_opponent_and_side() -> None:
    env_config = small_env_config(max_episode_steps=10)
    cfg = _cfg()
    results = ev.evaluate_policy(_scripted_learner(env_config), ROLES3, env_config, cfg)
    meta = {"checkpoint": "x.pt", "checkpoint_step": 1, "env_summary": "5x3", "policy_mode": "argmax",
            "episodes": 6, "confidence": 0.95}
    text = ev.format_report(results, meta)
    for word in ("random", "stay", "red", "blue", "win%", "route", "defender"):
        assert word in text


# ----------------------------------------------------------------------- RL policy + CLI


def _tiny_policy(env_config: CTFConfig, mode: PolicyMode) -> RLStrategy:
    cfg = rl_config()
    torch.manual_seed(0)  # same weights every call, so two evaluations can be compared
    model = ActorCritic(NetworkSpec.from_env(env_config), cfg.model)
    return RLStrategy(model, cfg, env_config, ValueNorm(cfg.ppo.value_normalization),
                      deterministic=mode is PolicyMode.ARGMAX, seed=1)


@pytest.mark.parametrize("mode", [PolicyMode.ARGMAX, PolicyMode.SAMPLED])
def test_rl_policy_evaluates_on_both_sides(mode) -> None:
    env_config = small_env_config(max_episode_steps=12)
    cfg = _cfg(policy_mode=mode, episodes=4)
    a = ev.evaluate_policy(_tiny_policy(env_config, mode), list(rl_config().roles.slots), env_config, cfg)
    b = ev.evaluate_policy(_tiny_policy(env_config, mode), list(rl_config().roles.slots), env_config, cfg)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)  # a sampled policy is seeded too
    assert a["random"]["all"]["n"] == 4


@pytest.fixture()
def checkpoint(tmp_path):
    cfg = tiny_config(str(tmp_path))
    trainer = Trainer(cfg, load_base_env_config(cfg, ROOT), tmp_path / "run")
    trainer.counters.update(total_steps=1000, update=3)
    path = tmp_path / "runs" / "demo" / "checkpoints" / "ck.pt"
    save_checkpoint(path, **trainer._state())
    return path


def test_cli_writes_json_report(checkpoint, tmp_path, capsys) -> None:
    out = tmp_path / "result.json"
    code = ev.main([
        "--checkpoint", str(checkpoint), "--eval-config", str(ROOT / "configs" / "rl" / "eval_legacy.yaml"),
        "--set", "episodes=4", "--opponents", "random", "scripted_chaser", "--out", str(out),
        "--env-set", "max_episode_steps=15", "--skip", "held_out", "role_swap", "policy_check",
    ])
    assert code == 0
    data = json.loads(out.read_text())
    assert list(data["results"]) == ["random", "scripted_chaser"]
    assert data["meta"]["checkpoint_step"] == 1000 and data["meta"]["episodes"] == 4
    assert data["meta"]["env_config"]["max_episode_steps"] == 15
    assert data["results"]["random"]["all"]["n"] == 4
    assert "random" in capsys.readouterr().out
    assert data["held_out"] is None and data["role_swap"] is None and data["policy_check"] is None


def test_cli_reports_errors_with_exit_code_2(checkpoint, tmp_path, capsys) -> None:
    common = ["--checkpoint", str(checkpoint), "--eval-config", str(ROOT / "configs" / "rl" / "eval_legacy.yaml"),
              "--out", str(tmp_path / "x.json")]
    assert ev.main([*common, "--opponents", "nope"]) == 2
    assert ev.main([*common, "--env-set", "no_such_key=1"]) == 2
    assert ev.main([*common, "--set", "episodes=3"]) == 2  # not divisible by two sides
    # the 4-agent opponents of eval.yaml do not fit a 3-agent checkpoint: clear error, no traceback
    assert ev.main(["--checkpoint", str(checkpoint), "--opponents", "scripted_leash",
                    "--out", str(tmp_path / "y.json")]) == 2
    assert "evaluation error" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()
