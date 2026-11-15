"""Minimax opponent: forward model == environment, alpha-beta == plain minimax, fair information, config."""

from __future__ import annotations

from dataclasses import replace
from random import Random

import pytest

from ctf import CTFConfig, CTFEnvironment, ObservationMode, Team
from ctf.actions import Action
from ctf.game_state import FlagStatus
from rl.config import OpponentSpec
from rl.minimax import ACTIONS, AT_BASE, CARRIED, DROPPED, MinimaxParams, MinimaxStrategy, MinimaxWeights
from rl.opponents import build_opponent
from rl.schema import ConfigError, from_dict

WEIGHTS = dict(
    win=1000.0,
    carry_progress=4.0,
    pickup=20.0,
    approach=1.0,
    approach_others=0.5,
    escort=2.0,
    enemy_carry_progress=4.0,
    enemy_pickup=20.0,
    defend=1.0,
    alive=3.0,
    pressure=1.5,
)


def params(**kw) -> MinimaxParams:
    base = dict(
        depth=1,
        own_moves_per_agent=2,
        enemy_moves_per_agent=2,
        danger_radius=3,
        max_nodes=20000,
        alpha_beta=True,
        weights=MinimaxWeights(**WEIGHTS),
    )
    base.update(kw)
    return MinimaxParams(**base)


def small_cfg(**kw) -> CTFConfig:
    base = dict(
        width=7,
        height=3,
        agents_per_team=2,
        red_base=(0, 1),
        blue_base=(6, 1),
        red_spawns=[(1, 0), (1, 2)],
        blue_spawns=[(5, 0), (5, 2)],
        visibility_radius=20,
        carrier_slots=[1],
        tagging_rule="headcount",
        drop_flag_on_tag=True,
    )
    base.update(kw)
    return CTFConfig(**base)


RULE_SETS = {
    "original": dict(carrier_slots=None, tagging_rule="any_enemy", drop_flag_on_tag=False),
    "v2": dict(carrier_slots=[1, 2], tagging_rule="headcount", drop_flag_on_tag=True),
    "v2_no_return": dict(
        carrier_slots=[1, 2], tagging_rule="headcount", drop_flag_on_tag=True, dropped_flag_return_on_touch=False
    ),
    "v2_no_repickup": dict(
        carrier_slots=[1, 2], tagging_rule="headcount", drop_flag_on_tag=True, dropped_flag_enemy_pickup=False
    ),
    "headcount_no_drop": dict(carrier_slots=[1, 2], tagging_rule="headcount", drop_flag_on_tag=False),
    "any_enemy_drop": dict(carrier_slots=[1, 2], tagging_rule="any_enemy", drop_flag_on_tag=True),
}


@pytest.mark.parametrize("rules", sorted(RULE_SETS))
def test_forward_model_matches_environment_step_by_step(rules: str) -> None:
    """Random play on a small board: the fast model must agree with CTFEnvironment after every step."""
    seen = {"flag_pickup": 0, "tag": 0, "flag_drop": 0, "flag_return": 0, "flag_capture": 0}
    for seed in range(40):
        cfg = small_cfg(width=5, agents_per_team=3, red_spawns=[(2, 1), (2, 0), (3, 2)],
                        blue_spawns=[(2, 1), (2, 2), (1, 0)], red_base=(0, 1), blue_base=(4, 1),
                        spawn_mode="shuffle", **RULE_SETS[rules])
        env = CTFEnvironment(cfg)
        env.reset(seed=seed)
        rng = Random(seed)
        strat = MinimaxStrategy(cfg, params(), Random(0))
        team = Team.RED if seed % 2 == 0 else Team.BLUE
        for _ in range(60):
            if env.done or not any(a.active and a.team is team for a in env.state.agents.values()):
                break  # a team without active agents has no view at all
            fr, st, ids = strat._build(env.observe(ObservationMode.PARTIAL, team=team), team)
            assert len(ids) == sum(ag.active for ag in env.state.agents.values())  # radius 20: all active visible
            actions = {a: rng.choice(list(Action)) for a, ag in env.state.agents.items() if ag.active}
            acts = [ACTIONS.index(actions[ids[i]]) if st.alive[i] else 4 for i in range(len(ids))]
            nxt = strat._sim.step(fr, st, acts)
            _, _, _, info = env.step(actions)
            for event in info["events"]:
                if event["type"] in seen:
                    seen[event["type"]] += 1
            _compare_relative(strat._sim, fr, nxt, ids, env.state, team)
    assert seen["flag_pickup"] > 0 and seen["tag"] > 0 and seen["flag_capture"] > 0
    if RULE_SETS[rules]["drop_flag_on_tag"]:
        assert seen["flag_drop"] > 0


def _compare_relative(sim, fr, st, ids, state, team: Team) -> None:
    """Like _compare, but flag index 0 is the observing team's flag, 1 the other team's."""
    status = {FlagStatus.AT_BASE: AT_BASE, FlagStatus.CARRIED: CARRIED, FlagStatus.DROPPED: DROPPED}
    for i, aid in enumerate(ids):
        agent = state.agents[aid]
        assert sim.xy(st.pos[i]) == (agent.x, agent.y), aid
        assert st.alive[i] == agent.active, aid
        assert st.carry[i] == agent.carrying_flag, aid
    for f, flag_team in enumerate((team, team.opponent)):
        flag = state.flags[flag_team]
        assert st.fstat[f] == status[flag.status]
        assert sim.xy(st.fcell[f]) == (flag.x, flag.y)
        assert (ids[st.fcar[f]] if st.fcar[f] >= 0 else None) == flag.carrier_id
    expected = 0 if state.winner is None else (1 if state.winner is team else 2)
    assert st.winner == expected


def test_forward_model_follows_pickup_tag_drop_and_return_in_a_scripted_sequence() -> None:
    cfg = small_cfg(red_spawns=[(5, 1), (0, 0)], blue_spawns=[(6, 0), (6, 2)])
    env = CTFEnvironment(cfg)
    env.reset()
    strat = MinimaxStrategy(cfg, params(), Random(0))
    stay = {"R1": Action.STAY, "R2": Action.STAY, "B1": Action.STAY, "B2": Action.STAY}
    plan = [
        {**stay, "R1": Action.RIGHT},  # R1 takes BLUE's flag at (6, 1)
        {**stay, "B2": Action.UP},  # B2 steps onto the lone carrier: tagged, flag dropped at (6, 1)
        stay,  # B2 stands on the dropped flag: it goes home
    ]
    events: list[str] = []
    for actions in plan:
        obs = env.observe(ObservationMode.PARTIAL, team=Team.RED)
        fr, st, ids = strat._build(obs, Team.RED)
        nxt = strat._sim.step(fr, st, [ACTIONS.index(actions[ids[i]]) for i in range(len(ids))])
        _, _, _, info = env.step(actions)
        events += [e["type"] for e in info["events"]]
        _compare_relative(strat._sim, fr, nxt, ids, env.state, Team.RED)
    assert events == ["flag_pickup", "tag", "flag_drop", "flag_return"]
    assert env.state.flags[Team.BLUE].status is FlagStatus.AT_BASE


# ------------------------------------------------------------------------------------ search


def _midgame(seed: int, steps: int, cfg: CTFConfig):
    env = CTFEnvironment(cfg)
    env.reset(seed=seed)
    rng = Random(seed)
    for _ in range(steps):
        if env.done:
            break
        env.step({a: rng.choice(list(Action)) for a, ag in env.state.agents.items() if ag.active})
    return env


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_alpha_beta_gives_the_same_value_as_plain_minimax_with_fewer_nodes(depth: int) -> None:
    cfg = small_cfg(red_spawns=[(1, 0), (2, 2)], blue_spawns=[(5, 0), (4, 2)])
    saved = checked = 0
    for seed in range(12):
        env = _midgame(seed, seed * 2, cfg)
        if env.done:
            continue
        obs = env.observe(ObservationMode.PARTIAL, team=Team.RED)
        a = MinimaxStrategy(cfg, params(depth=depth, alpha_beta=True, max_nodes=10**9), Random(1))
        b = MinimaxStrategy(cfg, params(depth=depth, alpha_beta=False, max_nodes=10**9), Random(1))
        a.act(obs, Team.RED)
        b.act(obs, Team.RED)
        assert a.last_stats.value == pytest.approx(b.last_stats.value)
        assert a.last_stats.nodes <= b.last_stats.nodes
        saved += b.last_stats.nodes - a.last_stats.nodes
        checked += 1
    assert checked >= 8
    if depth >= 2:
        assert saved > 0  # pruning actually removes work


def test_node_budget_is_a_hard_cap_and_a_move_is_still_returned() -> None:
    cfg = small_cfg()
    env = _midgame(3, 4, cfg)
    obs = env.observe(ObservationMode.PARTIAL, team=Team.RED)
    strat = MinimaxStrategy(cfg, params(depth=3, max_nodes=30), Random(0))
    actions = strat.act(obs, Team.RED)
    assert set(actions) == {a for a, ag in env.state.agents.items() if ag.team is Team.RED and ag.active}
    assert strat.last_stats.nodes <= 30 and strat.last_stats.budget_hit


def test_attacker_next_to_the_flag_takes_it_and_the_carrier_heads_home() -> None:
    cfg = small_cfg(red_spawns=[(5, 1), (0, 0)], blue_spawns=[(3, 0), (3, 2)])
    env = CTFEnvironment(cfg)
    env.reset()
    strat = MinimaxStrategy(cfg, params(), Random(0))
    first = strat.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
    assert first["R1"] is Action.RIGHT  # onto the enemy flag at (6, 1)
    env.step({**first, "B1": Action.STAY, "B2": Action.STAY})
    assert env.state.agents["R1"].carrying_flag
    second = strat.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
    assert second["R1"] is Action.LEFT  # straight home


def test_defender_touches_a_dropped_flag_to_send_it_home() -> None:
    cfg = small_cfg(red_spawns=[(5, 1), (2, 1)], blue_spawns=[(6, 0), (6, 2)])
    env = CTFEnvironment(cfg)
    env.reset()
    # R1 steps onto BLUE's flag, then BLUE's B1 steps on R1 and tags it (lone carrier): flag drops at (6, 1).
    env.step({"R1": Action.RIGHT, "R2": Action.STAY, "B1": Action.STAY, "B2": Action.STAY})
    env.step({"R1": Action.STAY, "R2": Action.STAY, "B1": Action.DOWN, "B2": Action.STAY})
    assert env.state.flags[Team.BLUE].status is FlagStatus.DROPPED
    strat = MinimaxStrategy(cfg, params(), Random(0))
    # BLUE (we, minimax) must send its dropped flag home with a defender-slot agent: B2 is slot 2.
    acts = strat.act(env.observe(ObservationMode.PARTIAL, team=Team.BLUE), Team.BLUE)
    assert "B2" in acts and acts["B2"] in (Action.UP, Action.STAY)  # B2 is at (6,2): the flag lies at (6,1)
    assert acts["B2"] is Action.UP


def test_beats_a_random_team_most_of_the_time() -> None:
    cfg = CTFConfig(
        width=9, height=5, agents_per_team=3, red_base=(0, 2), blue_base=(8, 2),
        red_spawns=[(1, 1), (1, 3), (1, 2)], blue_spawns=[(7, 1), (7, 3), (7, 2)],
        carrier_slots=[1, 2], tagging_rule="headcount", drop_flag_on_tag=True, visibility_radius=3,
        max_episode_steps=120,
    )
    wins = 0
    games = 10
    for seed in range(games):
        env = CTFEnvironment(cfg)
        env.reset(seed=seed)
        team = Team.RED if seed % 2 == 0 else Team.BLUE
        mm = MinimaxStrategy(cfg, params(), Random(seed))
        rnd = build_opponent(OpponentSpec("random", "random", 1.0, {}), cfg, Random(seed))
        while not env.done:
            acts = mm.act(env.observe(ObservationMode.PARTIAL, team=team), team)
            acts.update(rnd.act(env.observe(ObservationMode.PARTIAL, team=team.opponent), team.opponent))
            env.step(acts)
        wins += env.state.winner is team
    assert wins >= 6


# ---------------------------------------------------------------------------- information


def test_decisions_do_not_depend_on_enemies_it_cannot_see() -> None:
    cfg = small_cfg(width=9, red_spawns=[(1, 0), (1, 2)], blue_spawns=[(8, 0), (8, 2)], blue_base=(8, 1),
                    visibility_radius=2)
    results = []
    for hidden in [(8, 0), (7, 2)]:  # far outside RED's radius-2 view
        c = replace(cfg, blue_spawns=[hidden, (8, 2)]) if hidden != (7, 2) else replace(cfg, blue_spawns=[(7, 2), (8, 0)])
        env = CTFEnvironment(c)
        env.reset()
        obs = env.observe(ObservationMode.PARTIAL, team=Team.RED)
        assert not any(a.team is Team.BLUE for a in obs.agents.values())
        results.append(MinimaxStrategy(c, params(), Random(5)).act(obs, Team.RED))
    assert results[0] == results[1]


def test_global_and_partial_observations_give_the_same_actions() -> None:
    cfg = small_cfg(visibility_radius=2)
    env = _midgame(2, 5, cfg)
    glob = env.observe(ObservationMode.GLOBAL)
    part = env.observe(ObservationMode.PARTIAL, team=Team.RED)
    a = MinimaxStrategy(cfg, params(), Random(7)).act(glob, Team.RED)
    b = MinimaxStrategy(cfg, params(), Random(7)).act(part, Team.RED)
    assert a == b


def test_an_unseen_enemy_carrier_is_located_by_our_flag() -> None:
    cfg = small_cfg(width=9, blue_base=(8, 1), red_spawns=[(7, 0), (7, 2)], blue_spawns=[(0, 1), (8, 0)],
                    visibility_radius=1)
    env = CTFEnvironment(cfg)
    env.reset()  # B1 stands on RED's flag at (0, 1): it picks it up on the first step
    env.step({a: Action.STAY for a in ("R1", "R2", "B1", "B2")})
    obs = env.observe(ObservationMode.PARTIAL, team=Team.RED)
    assert "B1" not in obs.agents and obs.flags[Team.RED].status is FlagStatus.CARRIED
    strat = MinimaxStrategy(cfg, params(), Random(0))
    fr, st, ids = strat._build(obs, Team.RED)
    j = ids.index("B1")
    assert st.carry[j] and st.alive[j] and st.fstat[0] == CARRIED and st.fcar[0] == j
    assert strat._sim.xy(st.pos[j]) == (0, 1)


def test_dropped_enemy_flag_out_of_sight_is_remembered_until_its_cell_is_seen_empty() -> None:
    cfg = small_cfg(visibility_radius=1)
    env = CTFEnvironment(cfg)
    env.reset()
    strat = MinimaxStrategy(cfg, params(), Random(0))
    # Pretend the enemy flag was last seen lying at (4, 1); our agents are far away now.
    strat._enemy_flag_memory = (4, 1)
    obs = env.observe(ObservationMode.PARTIAL, team=Team.RED)
    assert Team.BLUE not in obs.flags
    status, carrier, cell = strat._enemy_flag_belief(obs, Team.RED)
    assert (status, carrier, cell) == (DROPPED, None, (4, 1))
    # Once a teammate is next to that cell and the flag is not in view, it must have gone home.
    near = replace(cfg, red_spawns=[(3, 1), (0, 0)])
    env2 = CTFEnvironment(near)
    env2.reset()
    obs2 = env2.observe(ObservationMode.PARTIAL, team=Team.RED)
    assert Team.BLUE not in obs2.flags
    status2, _, cell2 = strat._enemy_flag_belief(obs2, Team.RED)
    assert (status2, cell2) == (AT_BASE, (6, 1))


def test_a_restarted_episode_clears_the_memory() -> None:
    cfg = small_cfg()
    env = CTFEnvironment(cfg)
    env.reset()
    strat = MinimaxStrategy(cfg, params(), Random(0))
    strat._enemy_flag_memory = (4, 1)
    strat._last_t = 9
    strat.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)  # timestep 0 < 9: new episode
    assert strat._enemy_flag_memory is None


# ------------------------------------------------------------------------------ configuration


def _raw() -> dict:
    return dict(depth=1, own_moves_per_agent=2, enemy_moves_per_agent=2, danger_radius=3, max_nodes=100,
                alpha_beta=True, weights=dict(WEIGHTS))


def test_build_opponent_makes_a_minimax_strategy_from_yaml_params() -> None:
    strat = build_opponent(OpponentSpec("minimax", "minimax", 1.0, _raw()), small_cfg(), Random(0))
    assert isinstance(strat, MinimaxStrategy)


@pytest.mark.parametrize(
    "change, text",
    [
        ({"depth": 0}, "depth"),
        ({"own_moves_per_agent": 6}, "own_moves_per_agent"),
        ({"enemy_moves_per_agent": 0}, "enemy_moves_per_agent"),
        ({"danger_radius": -1}, "danger_radius"),
        ({"max_nodes": 0}, "max_nodes"),
        ({"weights": {**WEIGHTS, "win": 0.0}}, "weights.win"),
    ],
)
def test_invalid_minimax_params_are_rejected(change: dict, text: str) -> None:
    raw = {**_raw(), **change}
    with pytest.raises(ConfigError, match=text):
        build_opponent(OpponentSpec("minimax", "minimax", 1.0, raw), small_cfg(), Random(0))


def test_minimax_params_are_strict_unknown_and_missing_keys_fail() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        from_dict(MinimaxParams, {**_raw(), "extra": 1})
    raw = _raw()
    del raw["weights"]["escort"]
    with pytest.raises(ConfigError, match="missing required"):
        from_dict(MinimaxParams, raw)
    raw = _raw()
    del raw["alpha_beta"]
    with pytest.raises(ConfigError, match="missing required"):
        from_dict(MinimaxParams, raw)


def test_works_with_the_original_three_agent_rules() -> None:
    cfg = CTFConfig(width=7, height=3, agents_per_team=3, red_base=(0, 1), blue_base=(6, 1), visibility_radius=3)
    env = CTFEnvironment(cfg)
    env.reset()
    strat = MinimaxStrategy(cfg, params(), Random(0))
    for _ in range(30):
        if env.done:
            break
        acts = strat.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
        acts.update({a: Action.STAY for a, ag in env.state.agents.items() if ag.team is Team.BLUE and ag.active})
        env.step(acts)


def test_the_shipped_viewer_controller_builds_and_plays() -> None:
    from pathlib import Path

    from ctf.config import load_config
    from viz.controllers import ControllerFactory, load_controller_specs

    root = Path(__file__).resolve().parents[1]
    env_cfg = load_config(root / "configs" / "ctf_v2.yaml")
    spec = next(s for s in load_controller_specs(root / "configs" / "viz" / "controllers.yaml") if s.kind == "minimax")
    ctrl = ControllerFactory().build(spec, env_cfg, seed=1)
    assert ctrl.roles == ["attacker", "attacker", "defender", "scout"]
    env = CTFEnvironment(env_cfg)
    env.reset(seed=1)
    for _ in range(5):
        acts = ctrl.strategy.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
        acts.update({a: Action.STAY for a, ag in env.state.agents.items() if ag.team is Team.BLUE})
        env.step(acts)
    assert ctrl.strategy.last_stats.nodes >= 1
