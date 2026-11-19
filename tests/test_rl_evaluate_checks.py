"""Stage B2b-2: held-out maps, role-swap check, argmax-vs-sampled check (rl.evaluate), their config keys."""

from __future__ import annotations

import json
from dataclasses import replace
from random import Random
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from ctf import CTFConfig  # noqa: E402
from ctf.actions import Action  # noqa: E402
from ctf.agent import Team  # noqa: E402
from ctf.config import load_config  # noqa: E402
from rl import evaluate as ev  # noqa: E402
from rl.canonical import ACTION_INDEX  # noqa: E402
from rl.checkpoint import save_checkpoint  # noqa: E402
from rl.config import load_base_env_config, load_rl_config  # noqa: E402
from rl.eval_config import (  # noqa: E402
    EvalConfig, EvalOpponent, HeldOutConfig, HeldOutMap, PolicyCheckConfig, PolicyMode, RoleSwapConfig, Side,
    load_eval_config,
)
from rl.eval_stats import rate_difference  # noqa: E402
from rl.features import ROLE_ORDER  # noqa: E402
from rl.mappo import ValueNorm  # noqa: E402
from rl.networks import ActorCritic, NetworkSpec  # noqa: E402
from rl.rl_strategy import Decision, RLStrategy  # noqa: E402
from rl.schema import ConfigError  # noqa: E402
from rl.trainer import Trainer  # noqa: E402
from rl_helpers import ROOT, rl_config, small_env_config, tiny_config  # noqa: E402
from strategies import BaseStrategy, Role, StayStrategy  # noqa: E402

ROLES3 = [Role.ATTACKER, Role.DEFENDER, Role.SCOUT]
STAY = ACTION_INDEX[Action.STAY]
EVAL_YAMLS = ("eval.yaml", "eval_legacy.yaml", "eval_train.yaml")


def _cfg(**kw) -> EvalConfig:
    base = dict(
        episodes=4, seed=7, confidence=0.95, sides=[Side.RED, Side.BLUE], policy_mode=PolicyMode.ARGMAX,
        fixed_spawn_fraction=0.0, env_overrides={"spawn_mode": "near_base"}, route_roles=[Role.ATTACKER],
        opponents=[EvalOpponent("random", "random", {}), EvalOpponent("stay", "stay", {})],
        held_out=HeldOutConfig(episodes=2, opponents=["random"], maps=[]),
        role_swap=RoleSwapConfig(enabled=True, episodes=2, opponents=["random"],
                                 swaps=[[Role.ATTACKER, Role.DEFENDER], [Role.ATTACKER, Role.SCOUT]]),
        policy_check=PolicyCheckConfig(enabled=True, episodes=2, opponents=["random", "stay"]),
    )
    base.update(kw)
    return EvalConfig(**base)


def _peaked_policy(env_config: CTFConfig, *, role_preferences: dict[int, int] | None, mode=PolicyMode.ARGMAX) -> RLStrategy:
    """A policy whose output depends ONLY on the role head's bias: role r always prefers action ``role_preferences[r]``
    (every role prefers STAY when None, i.e. the policy ignores its role)."""
    cfg = rl_config()
    torch.manual_seed(0)
    model = ActorCritic(NetworkSpec.from_env(env_config), cfg.model)
    n_actions = model.spec.num_actions
    with torch.no_grad():
        model.role_heads.weight.zero_()
        model.role_heads.bias.zero_()
        for role in range(model.spec.num_roles):
            model.role_heads.bias[role * n_actions + (STAY if role_preferences is None else role_preferences[role])] = 12.0
    return RLStrategy(model, cfg, env_config, ValueNorm(cfg.ppo.value_normalization),
                      deterministic=mode is PolicyMode.ARGMAX, seed=1)


# ------------------------------------------------------------------------------- statistics


def test_rate_difference_matches_hand_computation() -> None:
    d = rate_difference(60, 100, 40, 100, 0.95)
    assert d["diff"] == pytest.approx(0.2)
    # Newcombe 95% interval for 60/100 vs 40/100 is about (0.065, 0.324)
    assert d["lo"] == pytest.approx(0.065, abs=0.005) and d["hi"] == pytest.approx(0.324, abs=0.005)
    same = rate_difference(30, 60, 30, 60, 0.95)
    assert same["diff"] == 0.0 and same["lo"] == pytest.approx(-same["hi"])
    flipped = rate_difference(40, 100, 60, 100, 0.95)
    assert flipped["diff"] == pytest.approx(-0.2) and flipped["lo"] == pytest.approx(-d["hi"])
    assert rate_difference(0, 0, 1, 2, 0.95) == {"diff": None, "lo": None, "hi": None}
    edge = rate_difference(10, 10, 0, 10, 0.95)
    assert edge["diff"] == 1.0 and edge["hi"] <= 1.0 and -1.0 <= edge["lo"]


# ----------------------------------------------------------------------------- config keys


@pytest.mark.parametrize("name", EVAL_YAMLS)
def test_shipped_eval_configs_have_the_new_sections(name) -> None:
    cfg = load_eval_config(ROOT / "configs" / "rl" / name)
    labels = {o.label for o in cfg.opponents}
    assert set(cfg.held_out.opponents) <= labels and set(cfg.role_swap.opponents) <= labels
    assert set(cfg.policy_check.opponents) <= labels
    assert [Role.ATTACKER, Role.DEFENDER] in cfg.role_swap.swaps


def test_shipped_held_out_maps_fit_the_v2_environment_and_are_not_the_training_map() -> None:
    cfg = load_eval_config(ROOT / "configs" / "rl" / "eval.yaml")
    base = load_config(ROOT / "configs" / "ctf_v2.yaml")
    assert {m.label for m in cfg.held_out.maps} == {
        "center_gate", "center_pillar", "long_wall_south", "open_field", "pillars", "two_corridors"}
    for held in cfg.held_out.maps:
        out = ev.held_out_env_config(base, ROOT / held.path)
        assert (out.width, out.height, out.agents_per_team) == (11, 7, 4)
        assert set(out.obstacles) != set(base.obstacles)  # a genuinely different layout
        for rule in ("carrier_slots", "tagging_rule", "drop_flag_on_tag", "visibility_radius", "max_episode_steps"):
            assert getattr(out, rule) == getattr(base, rule)  # rules are the policy's, not the map file's
    assert load_eval_config(ROOT / "configs" / "rl" / "eval_legacy.yaml").held_out.maps == []  # 3 agents: no 4-agent maps


def _eval_yaml(name: str = "eval.yaml") -> dict:
    return yaml.safe_load((ROOT / "configs" / "rl" / name).read_text())


def test_new_eval_sections_are_strict(tmp_path) -> None:
    def load(data: dict):
        path = tmp_path / "e.yaml"
        path.write_text(yaml.safe_dump(data))
        return load_eval_config(path)

    base = _eval_yaml()
    assert load(base).held_out.episodes == 40
    for section in ("held_out", "role_swap", "policy_check"):
        with pytest.raises(ConfigError, match=section):
            load({k: v for k, v in base.items() if k != section})
    with pytest.raises(ConfigError, match="unknown key"):
        load({**base, "role_swap": {**base["role_swap"], "threshold": 0.5}})
    with pytest.raises(ConfigError, match="unknown opponent label"):
        load({**base, "held_out": {**base["held_out"], "opponents": ["nope"]}})
    with pytest.raises(ConfigError, match="unknown opponent label"):
        load({**base, "role_swap": {**base["role_swap"], "opponents": ["nope"]}})
    with pytest.raises(ConfigError, match="unknown opponent label"):
        load({**base, "policy_check": {**base["policy_check"], "opponents": ["nope"]}})
    with pytest.raises(ConfigError, match="divisible"):
        load({**base, "held_out": {**base["held_out"], "episodes": 3}})
    with pytest.raises(ConfigError, match="divisible"):
        load({**base, "role_swap": {**base["role_swap"], "episodes": 5}})
    with pytest.raises(ConfigError, match="min_change_rate"):  # the old pass line is gone: role-swap is report-only
        load({**base, "role_swap": {**base["role_swap"], "min_change_rate": 0.1}})
    with pytest.raises(ConfigError, match="two different roles"):
        load({**base, "role_swap": {**base["role_swap"], "swaps": [["attacker", "attacker"]]}})
    with pytest.raises(ConfigError, match="is not one of"):
        load({**base, "role_swap": {**base["role_swap"], "swaps": [["attacker", "goalie"]]}})
    maps = base["held_out"]["maps"]
    with pytest.raises(ConfigError, match="unique"):
        load({**base, "held_out": {**base["held_out"], "maps": [maps[0], maps[0]]}})
    with pytest.raises(ConfigError, match="labels"):
        load({**base, "held_out": {**base["held_out"], "maps": [{"label": "Bad Map", "path": "x.yaml"}]}})
    with pytest.raises(ConfigError, match="must not be empty"):
        load({**base, "held_out": {**base["held_out"], "opponents": []}})
    with pytest.raises(ConfigError, match="at least one opponent"):
        load({**base, "role_swap": {**base["role_swap"], "opponents": []}})
    with pytest.raises(ConfigError, match="must not be empty"):
        load({**base, "policy_check": {**base["policy_check"], "opponents": []}})
    # a disabled check may have empty lists
    off = load({**base, "role_swap": {**base["role_swap"], "enabled": False, "opponents": [], "swaps": []},
                "policy_check": {**base["policy_check"], "enabled": False, "opponents": []}})
    assert not off.role_swap.enabled and not off.policy_check.enabled
    assert load_eval_config(ROOT / "configs" / "rl" / "eval.yaml", {"role_swap.episodes": 4}).role_swap.episodes == 4


# ----------------------------------------------------------------------------- held-out maps


def _map_yaml(tmp_path, name="m.yaml", **over) -> str:
    data = dict(width=5, height=3, agents_per_team=3, obstacles=[[2, 0]], red_base=[0, 1], blue_base=[4, 1],
                red_spawns=[[2, 1], [2, 2], [1, 2]], blue_spawns=[[3, 1], [3, 2], [4, 2]])
    data.update(over)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_held_out_env_config_takes_geometry_only_and_checks_the_size(tmp_path) -> None:
    base = small_env_config(max_episode_steps=33, visibility_radius=2)
    out = ev.held_out_env_config(base, _map_yaml(tmp_path))
    assert out.obstacles == [(2, 0)] and out.red_spawns == [(2, 1), (2, 2), (1, 2)]
    assert out.max_episode_steps == 33 and out.visibility_radius == 2  # not taken from the map file
    for key, value, text in (("width", 7, "width"), ("height", 5, "height"), ("agents_per_team", 4, "agents_per_team")):
        with pytest.raises(ConfigError, match=text):
            ev.held_out_env_config(base, _map_yaml(tmp_path, "bad.yaml", **{key: value}))
    data = yaml.safe_load(open(_map_yaml(tmp_path)))
    del data["obstacles"]
    (tmp_path / "nokey.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="missing key"):
        ev.held_out_env_config(base, tmp_path / "nokey.yaml")
    with pytest.raises(ConfigError, match="invalid"):  # a spawn on a wall
        ev.held_out_env_config(base, _map_yaml(tmp_path, "wall.yaml", obstacles=[[2, 1]]))
    with pytest.raises(FileNotFoundError):
        ev.held_out_env_config(base, tmp_path / "missing.yaml")


def test_evaluate_held_out_plays_on_the_map_file_and_is_seeded(tmp_path) -> None:
    env_config = small_env_config(max_episode_steps=6)
    path = _map_yaml(tmp_path)
    cfg = _cfg(fixed_spawn_fraction=1.0, sides=[Side.RED], episodes=2,
               held_out=HeldOutConfig(episodes=2, opponents=["stay"], maps=[HeldOutMap("tiny", path)]))
    out = ev.evaluate_held_out(StayStrategy(), ROLES3, env_config, cfg)
    assert list(out) == ["tiny"] and list(out["tiny"]) == ["stay"]  # only the held-out opponents, not all
    block = out["tiny"]["stay"]
    assert set(block) == {"all", "red"} and block["all"]["n"] == 2
    # the stay-put attacker stood on the MAP's first red spawn (2,1); the training-env spawn would be (3,1)
    assert block["red"]["heatmaps"]["attacker"][1][2] == pytest.approx(1.0)
    again = ev.evaluate_held_out(StayStrategy(), ROLES3, env_config, cfg)
    assert json.dumps(out, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_held_out_does_not_change_the_main_results_and_none_without_maps(tmp_path) -> None:
    env_config = small_env_config(max_episode_steps=8)
    plain = _cfg()
    with_maps = _cfg(held_out=HeldOutConfig(episodes=2, opponents=["random"], maps=[HeldOutMap("tiny", _map_yaml(tmp_path))]))
    assert ev.evaluate_held_out(StayStrategy(), ROLES3, env_config, plain) is None
    a = ev.evaluate_policy(StayStrategy(), ROLES3, env_config, plain)
    b = ev.evaluate_policy(StayStrategy(), ROLES3, env_config, with_maps)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)  # seeds of the main table never moved
    held = ev.evaluate_held_out(StayStrategy(), ROLES3, env_config, with_maps)
    assert held["tiny"]["random"]["all"]["n"] == 2


def test_wrong_sized_held_out_map_fails_before_any_game(tmp_path) -> None:
    cfg = _cfg(held_out=HeldOutConfig(episodes=2, opponents=["random"], maps=[HeldOutMap("big", _map_yaml(tmp_path, width=7))]))
    with pytest.raises(ConfigError, match="width"):
        ev.evaluate_held_out(StayStrategy(), ROLES3, small_env_config(), cfg)


# ---------------------------------------------------------------------- policy mode / probes


def test_set_policy_mode_switches_and_reseeds() -> None:
    env_config = small_env_config(max_episode_steps=10)
    policy = _peaked_policy(env_config, role_preferences=None)
    assert policy.deterministic
    policy.set_policy_mode(False, seed=5)
    assert not policy.deterministic
    policy.set_policy_mode(True)
    assert policy.deterministic
    # identical seeds give identical sampled games
    cfg = _cfg(policy_mode=PolicyMode.SAMPLED)
    runs = []
    for _ in range(2):
        policy.set_policy_mode(False, seed=5)
        runs.append(json.dumps(ev.evaluate_policy(policy, ROLES3, env_config, cfg), sort_keys=True))
    assert runs[0] == runs[1]


def test_greedy_actions_swaps_role_one_hot_and_head_without_side_effects() -> None:
    env_config = small_env_config(max_episode_steps=10)
    policy = _peaked_policy(env_config, role_preferences={0: 0, 1: 1, 2: 2})
    seen = []
    real = policy.model.policy

    def spy(grid, vec, mask, role_index):
        seen.append((vec.clone(), role_index.clone()))
        return real(grid, vec, mask, role_index)

    policy.model.policy = spy  # type: ignore[method-assign]
    from ctf import CTFEnvironment
    from ctf.observations import ObservationMode

    env = CTFEnvironment(env_config)
    env.reset(seed=1)
    policy.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
    actual_vec, actual_roles = seen[-1]
    cols = policy._role_columns
    assert all(c >= 0 for c in cols)  # the current layout has a one-hot column for every role
    before = policy.greedy_actions(Team.RED)
    swapped_roles = np.array([ROLE_ORDER.index(Role.DEFENDER), ROLE_ORDER.index(Role.ATTACKER), ROLE_ORDER.index(Role.SCOUT)])
    policy.greedy_actions(Team.RED, swapped_roles)
    vec2, roles2 = seen[-1]
    assert roles2.tolist() == swapped_roles.tolist()
    for slot, new in enumerate(swapped_roles):  # exactly one role column set per agent, the new one
        assert vec2[slot, cols[new]] == 1.0 and vec2[slot, [c for c in cols if c != cols[new]]].sum() == 0.0
    rest = [i for i in range(vec2.shape[1]) if i not in cols]
    assert torch.equal(vec2[:, rest], actual_vec[:, rest])  # everything else is what the agent saw
    assert np.array_equal(policy.greedy_actions(Team.RED), before)  # the stored inputs were not modified
    again = policy.greedy_actions(Team.RED, np.array([ROLE_ORDER.index(r) for r in ROLES3]))
    assert np.array_equal(again, before)  # "swapping" to the real roles is the identity


class _FakeRoleNet(BaseStrategy):
    """Stands in for RLStrategy in the probe: records the role arrays it is asked about."""

    def __init__(self, alive) -> None:
        self.alive, self.asked = np.array(alive), []
        self.last_decision = {}

    def reset(self) -> None: ...

    def act(self, observation, team):
        self.last_decision[team] = Decision(["R1", "R2", "R3"], np.full((3, 5), 0.2), self.alive, {})
        return {}

    def greedy_actions(self, team, role_index=None):
        if role_index is None:
            return np.array([9, 9, 9])
        self.asked.append(list(role_index))
        return np.array(role_index)  # never equals 9: every counted decision "changes"


def test_role_swap_probe_counts_only_alive_agents_of_the_swapped_roles() -> None:
    fake = _FakeRoleNet([True, False, True])
    probe = ev._RoleSwapProbe(fake, ROLES3, [(Role.ATTACKER, Role.DEFENDER), (Role.ATTACKER, Role.SCOUT)])
    probe.act(None, Team.RED)
    a, d, s = (ROLE_ORDER.index(r) for r in ROLES3)
    assert fake.asked == [[d, a, s], [s, d, a]]  # swapped for the whole team, other roles untouched
    assert probe.counts[(Role.ATTACKER, Role.DEFENDER)] == [1, 1]  # slot 1 (defender) is dead: only the attacker counts
    assert probe.counts[(Role.ATTACKER, Role.SCOUT)] == [2, 2]


# ---------------------------------------------------------------------------- role-swap check


def _has_no_verdict(node) -> bool:
    """True when no pass/fail flag or threshold appears anywhere in a role-swap result."""
    if isinstance(node, dict):
        return not ({"passed", "min_change_rate"} & set(node)) and all(_has_no_verdict(v) for v in node.values())
    return True


def test_role_swap_check_detects_a_policy_that_uses_roles() -> None:
    env_config = small_env_config(max_episode_steps=12)
    policy = _peaked_policy(env_config, role_preferences={0: 0, 1: 1, 2: 2})  # each role prefers its own action
    out = ev.role_swap_check(policy, ROLES3, env_config, _cfg())
    assert set(out["pairs"]) == {"attacker<->defender", "attacker<->scout"}
    assert out["all"]["n"] > 0 and out["all"]["rate"] > 0.25  # masked moves (walls) cap it below 1
    for stat in out["pairs"].values():
        assert stat["lo"] <= stat["rate"] <= stat["hi"] and stat["n"] > 0
    assert _has_no_verdict(out)


def test_role_swap_check_scores_zero_for_a_policy_that_ignores_roles() -> None:
    env_config = small_env_config(max_episode_steps=12)
    out = ev.role_swap_check(_peaked_policy(env_config, role_preferences=None), ROLES3, env_config, _cfg())
    assert out["all"]["count"] == 0 and out["all"]["rate"] == 0.0 and out["all"]["n"] > 0
    assert _has_no_verdict(out)  # a zero is reported as a number, not as a FAIL


def test_role_swap_check_is_deterministic_and_errors_are_clear() -> None:
    env_config = small_env_config(max_episode_steps=12)
    a = ev.role_swap_check(_peaked_policy(env_config, role_preferences={0: 0, 1: 1, 2: 2}), ROLES3, env_config, _cfg())
    b = ev.role_swap_check(_peaked_policy(env_config, role_preferences={0: 0, 1: 1, 2: 2}), ROLES3, env_config, _cfg())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    with pytest.raises(ConfigError, match="neural policy"):
        ev.role_swap_check(StayStrategy(), ROLES3, env_config, _cfg())
    stub = SimpleNamespace(greedy_actions=lambda *a, **k: None)
    with pytest.raises(ConfigError, match="does not have"):
        ev.role_swap_check(stub, [Role.ATTACKER, Role.ATTACKER, Role.DEFENDER], env_config, _cfg())  # no scout in the team


# -------------------------------------------------------------------- argmax vs sampled check


def test_policy_check_agrees_for_a_peaked_policy_and_restores_the_mode() -> None:
    env_config = small_env_config(max_episode_steps=12)
    policy = _peaked_policy(env_config, role_preferences=None)  # (almost) all probability on STAY
    out = ev.policy_mode_check(policy, ROLES3, env_config, _cfg())
    assert list(out) == ["random", "stay"]
    for c in out.values():
        assert c["argmax"]["win"]["n"] == 2 and c["sampled"]["win"]["n"] == 2
        assert c["argmax"]["top_action_prob"] > 0.99 and c["sampled"]["top_action_prob"] > 0.99
        assert c["win_difference"]["diff"] == 0.0  # same games, same actions
        assert c["argmax"]["win"]["rate"] == c["sampled"]["win"]["rate"]
    assert policy.deterministic  # the policy's own mode is back
    policy.set_policy_mode(False)
    ev.policy_mode_check(policy, ROLES3, env_config, _cfg())
    assert not policy.deterministic


def test_policy_check_sees_a_difference_for_a_random_policy() -> None:
    env_config = small_env_config(max_episode_steps=12)
    cfg = rl_config()
    torch.manual_seed(0)
    policy = RLStrategy(ActorCritic(NetworkSpec.from_env(env_config), cfg.model), cfg, env_config,
                        ValueNorm(cfg.ppo.value_normalization), seed=1)  # untrained: nearly uniform actions
    out = ev.policy_mode_check(policy, ROLES3, env_config, _cfg(policy_check=PolicyCheckConfig(True, 2, ["random"])))
    top = out["random"]["sampled"]["top_action_prob"]
    assert top < 0.5 and out["random"]["win_difference"]["diff"] is not None
    again = ev.policy_mode_check(policy, ROLES3, env_config, _cfg(policy_check=PolicyCheckConfig(True, 2, ["random"])))
    assert json.dumps(out, sort_keys=True) == json.dumps(again, sort_keys=True)  # seeded in both modes
    with pytest.raises(ConfigError, match="neural policy"):
        ev.policy_mode_check(StayStrategy(), ROLES3, env_config, _cfg())


# ------------------------------------------------------------------- aggregation + report + CLI


def test_evaluate_extras_runs_enabled_checks_and_skips_the_rest(tmp_path) -> None:
    env_config = small_env_config(max_episode_steps=8)
    policy = _peaked_policy(env_config, role_preferences=None)
    cfg = _cfg(held_out=HeldOutConfig(2, ["random"], [HeldOutMap("tiny", _map_yaml(tmp_path))]))
    full = ev.evaluate_extras(policy, ROLES3, env_config, cfg)
    assert set(full) == {"held_out", "held_out_seen", "role_swap", "policy_check"} and all(v is not None for v in full.values())
    assert full["held_out_seen"] == {"tiny": False}  # a test double has only its base map, which is not the "tiny" layout
    skipped = ev.evaluate_extras(policy, ROLES3, env_config, cfg, skip=["held_out", "role_swap"])
    assert skipped["held_out"] is None and skipped["held_out_seen"] is None
    assert skipped["role_swap"] is None and skipped["policy_check"] is not None
    off = _cfg(role_swap=replace(cfg.role_swap, enabled=False), policy_check=replace(cfg.policy_check, enabled=False))
    assert ev.evaluate_extras(StayStrategy(), ROLES3, env_config, off) == {"held_out": None, "held_out_seen": None, "role_swap": None, "policy_check": None}
    with pytest.raises(ConfigError, match="unknown check"):
        ev.evaluate_extras(policy, ROLES3, env_config, cfg, skip=["nope"])
    with pytest.raises(ConfigError, match="roles"):
        ev.evaluate_extras(policy, ROLES3[:2], env_config, cfg)


def test_format_extras_prints_every_section(tmp_path) -> None:
    env_config = small_env_config(max_episode_steps=8)
    cfg = _cfg(held_out=HeldOutConfig(2, ["random"], [HeldOutMap("tiny", _map_yaml(tmp_path))]))
    extras = ev.evaluate_extras(_peaked_policy(env_config, role_preferences=None), ROLES3, env_config, cfg)
    text = ev.format_extras(extras, {"confidence": 0.95})
    for word in ("held-out maps", "tiny", "role-swap check", "attacker<->defender", "REPORT ONLY", "argmax vs sampled", "diff CI"):
        assert word in text
    assert "PASS" not in text and "FAIL" not in text  # role-swap numbers carry no verdict (an untrained net scores high)
    assert ev.format_extras({"held_out": None, "role_swap": None, "policy_check": None}, {"confidence": 0.95}) == ""


@pytest.fixture()
def checkpoint(tmp_path):
    cfg = tiny_config(str(tmp_path))
    trainer = Trainer(cfg, load_base_env_config(cfg, ROOT), tmp_path / "run")
    trainer.counters.update(total_steps=1000, update=3)
    path = tmp_path / "runs" / "demo" / "checkpoints" / "ck.pt"
    save_checkpoint(path, **trainer._state())
    return path


def test_cli_runs_extras_and_writes_them_to_json(checkpoint, tmp_path, capsys) -> None:
    out = tmp_path / "r.json"
    code = ev.main([
        "--checkpoint", str(checkpoint), "--eval-config", str(ROOT / "configs" / "rl" / "eval_legacy.yaml"),
        "--set", "episodes=2", "--set", "role_swap.episodes=2", "--set", "policy_check.episodes=2",
        "--opponents", "random", "--env-set", "max_episode_steps=12", "--out", str(out),
    ])
    assert code == 0
    data = json.loads(out.read_text())
    assert data["held_out"] is None  # the legacy config lists no held-out maps
    assert set(data["role_swap"]["pairs"]) == {"attacker<->defender", "attacker<->scout", "defender<->scout"}
    assert set(data["policy_check"]) == {"scripted_leash", "scripted_chaser"}
    assert data["meta"]["eval_config"]["role_swap"]["swaps"][0] == ["attacker", "defender"]
    text = capsys.readouterr().out
    assert "role-swap check" in text and "argmax vs sampled" in text
    code = ev.main(["--checkpoint", str(checkpoint), "--eval-config", str(ROOT / "configs" / "rl" / "eval_legacy.yaml"),
                    "--set", "episodes=2", "--opponents", "random", "--env-set", "max_episode_steps=12",
                    "--skip", "role_swap", "policy_check", "--out", str(tmp_path / "s.json")])
    assert code == 0 and json.loads((tmp_path / "s.json").read_text())["role_swap"] is None


def test_cli_gives_a_clear_error_for_held_out_maps_of_the_wrong_team_size(checkpoint, tmp_path, capsys) -> None:
    # the checkpoint plays 3 agents per team; configs/maps files hold 4: report it, do not crash
    maps = "[{label: pillars, path: %s}]" % (ROOT / "configs" / "maps" / "pillars.yaml")
    code = ev.main(["--checkpoint", str(checkpoint), "--eval-config", str(ROOT / "configs" / "rl" / "eval_legacy.yaml"),
                    "--set", "episodes=2", "--set", f"held_out.maps={maps}", "--set", "held_out.episodes=2",
                    "--opponents", "random", "--env-set", "max_episode_steps=12", "--out", str(tmp_path / "x.json")])
    assert code == 2
    assert "agents_per_team" in capsys.readouterr().err


def test_v2_four_agent_policy_end_to_end_with_the_shipped_eval_yaml() -> None:
    """The real eval.yaml (4 roles, 6 held-out 11x7 maps) on an untrained 4-agent v2 policy, short games."""
    rl_cfg = load_rl_config(ROOT / "configs" / "rl" / "default.yaml")
    env_config = load_base_env_config(rl_cfg, ROOT)
    torch.manual_seed(0)
    policy = RLStrategy(ActorCritic(NetworkSpec.from_env(env_config), rl_cfg.model), rl_cfg, env_config,
                        ValueNorm(rl_cfg.ppo.value_normalization), seed=0)
    cfg = load_eval_config(ROOT / "configs" / "rl" / "eval.yaml", {
        "episodes": 2, "held_out.episodes": 2, "role_swap.episodes": 2, "policy_check.episodes": 2,
        "env_overrides.spawn_mode": "near_base",
    })
    cfg = replace(cfg, env_overrides={**cfg.env_overrides, "max_episode_steps": 10},
                  held_out=replace(cfg.held_out, maps=[m for m in cfg.held_out.maps if m.label in ("center_pillar", "two_corridors")],
                                   opponents=["scripted_leash"]),
                  role_swap=replace(cfg.role_swap, opponents=["scripted_chaser"]),
                  policy_check=replace(cfg.policy_check, opponents=["scripted_escort"]))
    roles = list(policy.roles)
    assert len(roles) == 4
    extras = ev.evaluate_extras(policy, roles, env_config, cfg)
    assert set(extras["held_out"]) == {"center_pillar", "two_corridors"}
    assert extras["held_out"]["two_corridors"]["scripted_leash"]["all"]["n"] == 2
    assert extras["role_swap"]["all"]["n"] > 0 and "attacker<->scout" in extras["role_swap"]["pairs"]
    assert extras["policy_check"]["scripted_escort"]["win_difference"]["diff"] is not None
    main_table = ev.evaluate_policy(policy, roles, env_config, cfg, only=["random", "scripted_parked"])
    assert main_table["random"]["all"]["n"] == 2


def test_legacy_checkpoint_still_loads_and_runs_the_new_checks() -> None:
    """runs/colab_run1 (3 agents, old rules, trained before any of this existed) works with the new checks."""
    ckpt = ROOT / "runs" / "colab_run1" / "checkpoints" / "latest.pt"
    if not ckpt.exists():
        pytest.skip("legacy checkpoint not in this checkout")
    policy = RLStrategy.from_checkpoint(ckpt, deterministic=True, seed=0)
    assert policy.config.noise_free_eval.enabled is False  # filled in by the compat table
    cfg = load_eval_config(ROOT / "configs" / "rl" / "eval_legacy.yaml", {
        "role_swap.episodes": 2, "policy_check.episodes": 2, "role_swap.opponents": ["scripted_chaser"],
        "policy_check.opponents": ["scripted_leash"],
    })
    cfg = replace(cfg, env_overrides={**cfg.env_overrides, "max_episode_steps": 25})
    extras = ev.evaluate_extras(policy, list(policy.roles), policy.env_config, cfg)
    assert extras["role_swap"]["all"]["n"] > 0 and extras["policy_check"]["scripted_leash"]["argmax"]["win"]["n"] == 2
