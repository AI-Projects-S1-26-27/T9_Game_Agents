"""Fixes for the classical AI (rl/classical_fixes.py): each switch changes behaviour exactly where it should."""

from __future__ import annotations

import dataclasses
from random import Random

import pytest

from classical_ai.classical_strategy import ClassicalStrategy
from classical_ai.roles.scout import ScoutRole
from coordination.shared_map import SharedTeamMap
from ctf import CTFEnvironment, ObservationMode, Team
from ctf.actions import Action
from ctf.config import load_config
from ctf.game_state import Flag, FlagStatus
from rl.classical_fixes import FixedClassicalStrategy, FixedScoutRole, FixSettings, _Influence, _path_len, ambush_cell
from rl.classical_opponent import ClassicalOpponent, ClassicalParams
from rl.config import OpponentSpec
from rl.eval_config import load_eval_config
from rl.opponents import build_opponent
from rl.schema import ConfigError
from rl_helpers import ROOT
from viz.controllers import load_controller_specs

V2 = load_config(ROOT / "configs" / "ctf_v2.yaml")
NEAR = dataclasses.replace(V2, spawn_mode=type(V2.spawn_mode)("near_base"))
ROLES = ["attacker", "attacker", "defender", "scout"]
OFF = {"roles": ROLES}
ALL_ON = {
    "roles": ROLES,
    "remember_dropped_flag": True,
    "defend_dropped_flag": True,
    "buddy_system": True,
    "intercept_in_pairs": True,
}
FAR = {  # bottom-right corner: at least 3 cells from the blue flag (9, 3) and from the top lane used below
    "B1": (10, 6),
    "B2": (9, 6),
    "B3": (8, 6),
    "B4": (10, 5),
}


def _ai(params: dict) -> ClassicalOpponent:
    return build_opponent(OpponentSpec("classical", "classical", 1.0, params), V2, Random(0))


def _env(red: dict[str, tuple[int, int]], blue: dict[str, tuple[int, int]] | None = None) -> CTFEnvironment:
    """A reset v2 game with the given agents moved (the rest of each team goes to a far corner / the other end)."""
    env = CTFEnvironment(V2)
    env.reset(seed=1)
    spots = {**{f"R{i}": (0, i - 1) for i in range(1, 5)}, **red}  # unlisted red agents wait in column 0
    spots.update(FAR if blue is None else {**FAR, **blue})
    for aid, (x, y) in spots.items():
        env._state.agents[aid] = env._state.agents[aid].with_position(x, y)
    return env


def _obs(env: CTFEnvironment, team: Team = Team.RED):
    return env.observe(ObservationMode.PARTIAL, team=team)


def _carry(env: CTFEnvironment, carrier: str, cell: tuple[int, int]) -> None:
    team = env._state.agents[carrier].team
    env._state.agents[carrier] = env._state.agents[carrier].with_position(*cell).with_carrying_flag(True)
    env._state.flags[team.opponent] = Flag(team.opponent, FlagStatus.CARRIED, cell[0], cell[1], carrier)


# ------------------------------------------------------------------------------------ params


def test_old_params_mean_the_original_code_and_the_fixes_are_opt_in() -> None:
    ai = _ai(OFF)
    env = _env({})
    ai.act(_obs(env), Team.RED)
    assert type(ai._inner) is ClassicalStrategy  # the verbatim code, untouched
    fixed = _ai(ALL_ON)
    fixed.act(_obs(env), Team.RED)
    assert isinstance(fixed._inner, FixedClassicalStrategy)
    # a single non-default number is enough to leave the verbatim code
    tuned = _ai({**OFF, "enemy_penalty": 9.0})
    tuned.act(_obs(env), Team.RED)
    assert isinstance(tuned._inner, FixedClassicalStrategy)


def test_params_stay_strict_and_validated() -> None:
    with pytest.raises(ConfigError):
        _ai({**OFF, "teleport": True})  # unknown key
    with pytest.raises(ConfigError, match="pair_wait_steps"):
        _ai({**OFF, "pair_wait_steps": -1})
    with pytest.raises(ConfigError, match="enemy_penalty"):
        _ai({**OFF, "enemy_penalty": -0.5})


def test_eval_and_viewer_yaml_list_the_fixed_variant_with_every_key_spelled_out() -> None:
    keys = {f.name for f in dataclasses.fields(ClassicalParams)}
    cfg = load_eval_config(ROOT / "configs" / "rl" / "eval.yaml")
    entry = next(o for o in cfg.opponents if o.label == "classical_fixed")
    assert set(entry.params) == keys
    specs = [s for s in load_controller_specs(ROOT / "configs" / "viz" / "controllers.yaml") if s.kind == "classical"]
    assert len(specs) == 2 and all(set(s.params) >= {"roles"} for s in specs)
    assert any(set(s.params) == keys for s in specs)
    assert isinstance(_ai(dict(entry.params)), ClassicalOpponent)


# -------------------------------------------------------------------------------- dropped flags


def test_defender_walks_onto_a_dropped_own_flag() -> None:
    env = _env({"R3": (1, 3)})  # R3 stands on the red base
    env._state.flags[Team.RED] = Flag(Team.RED, FlagStatus.DROPPED, 3, 3)
    assert _ai(OFF).act(_obs(env), Team.RED)["R3"] is Action.STAY  # the original ignores it (probed earlier)
    assert _ai({**OFF, "defend_dropped_flag": True}).act(_obs(env), Team.RED)["R3"] is Action.RIGHT


def test_a_dropped_enemy_flag_out_of_sight_is_remembered_until_the_cell_is_seen_empty() -> None:
    settings = ClassicalParams(**{**ALL_ON, "roles": ROLES}).fix_settings()
    table = {Team.RED: {f"R{i}": r for i, r in enumerate(ROLES, 1)}, Team.BLUE: {}}
    strategy = FixedClassicalStrategy(table, settings)
    cell = (3, 5)
    env = _env({"R1": (3, 4)})  # R1 is next to the cell: the flag is in view
    env._state.flags[Team.BLUE] = Flag(Team.BLUE, FlagStatus.DROPPED, *cell)
    seen = strategy._with_remembered_flag(_obs(env), Team.RED)
    assert seen.flags[Team.BLUE].status is FlagStatus.DROPPED
    # everybody walks away: the engine no longer reports the flag, the strategy puts it back
    env = _env({})
    env._state.flags[Team.BLUE] = Flag(Team.BLUE, FlagStatus.DROPPED, *cell)
    assert Team.BLUE not in _obs(env).flags
    remembered = strategy._with_remembered_flag(_obs(env), Team.RED).flags.get(Team.BLUE)
    assert remembered is not None and (remembered.x, remembered.y) == cell and remembered.status is FlagStatus.DROPPED
    # an agent looks at the cell and the flag is not reported there (somebody sent it home): memory is dropped
    env = _env({"R1": (3, 4)})
    env._state.flags[Team.BLUE] = Flag(Team.BLUE, FlagStatus.AT_BASE, 9, 3)
    strategy._with_remembered_flag(_obs(env), Team.RED)
    far = _env({})
    assert Team.BLUE not in strategy._with_remembered_flag(_obs(far), Team.RED).flags


def test_an_attacker_heads_for_the_remembered_dropped_flag_not_the_enemy_base() -> None:
    cell = (3, 5)

    def second_step(params: dict) -> Action:
        ai = _ai(params)
        look = _env({"R1": (3, 4)})
        look._state.flags[Team.BLUE] = Flag(Team.BLUE, FlagStatus.DROPPED, *cell)
        ai.act(_obs(look), Team.RED)
        gone = _env({"R2": (3, 1)})  # nobody sees the cell now
        gone._state.flags[Team.BLUE] = Flag(Team.BLUE, FlagStatus.DROPPED, *cell)
        return ai.act(_obs(gone), Team.RED)["R2"]

    assert second_step({**OFF, "remember_dropped_flag": True}) is Action.DOWN  # toward (3, 5)
    assert second_step(OFF) is not Action.DOWN  # the original walks to the enemy base instead


# ------------------------------------------------------------------------------- buddy system


def _guarded_entry(partner: tuple[int, int], **extra) -> Action:
    env = _env({"R1": (8, 3), "R2": partner}, blue={"B3": (9, 3)})  # B3 stands on the blue flag
    return _ai({**ALL_ON, **extra}).act(_obs(env), Team.RED)["R1"]


def test_an_attacker_next_to_a_guarded_flag_waits_for_its_partner() -> None:
    assert _guarded_entry(partner=(3, 3)) is Action.STAY  # partner far away: wait (non-carriers cannot be tagged)
    assert _guarded_entry(partner=(9, 2)) is Action.RIGHT  # partner next to the flag too: both step in
    # the original walks in alone and is tagged on the spot
    env = _env({"R1": (8, 3), "R2": (3, 3)}, blue={"B3": (9, 3)})
    assert _ai(OFF).act(_obs(env), Team.RED)["R1"] is Action.RIGHT


def test_the_wait_has_a_budget_and_an_unguarded_flag_is_never_waited_for() -> None:
    env = _env({"R1": (8, 3), "R2": (3, 3)}, blue={"B3": (9, 3)})
    ai = _ai({**ALL_ON, "pair_wait_steps": 2})
    moves = [ai.act(_obs(env), Team.RED)["R1"] for _ in range(4)]
    assert moves[:2] == [Action.STAY, Action.STAY] and moves[2] is Action.RIGHT and moves[3] is Action.RIGHT
    unguarded = _env({"R1": (8, 3), "R2": (3, 3)})  # the blue defender sits in the far corner
    assert _ai(ALL_ON).act(_obs(unguarded), Team.RED)["R1"] is Action.RIGHT


def test_a_carrier_waits_one_step_for_an_adjacent_escort_when_an_enemy_is_close() -> None:
    def carrier_move(escort: tuple[int, int], enemy: tuple[int, int], **extra) -> Action:
        env = _env({"R2": escort}, blue={"B1": enemy})
        _carry(env, "R1", (6, 0))
        return _ai({**ALL_ON, **extra}).act(_obs(env), Team.RED)["R1"]

    assert carrier_move(escort=(6, 1), enemy=(7, 0)) is Action.STAY  # escort adjacent, enemy next to us
    assert carrier_move(escort=(6, 0), enemy=(7, 0)) is Action.LEFT  # escort already on the cell: go
    assert carrier_move(escort=(6, 1), enemy=(10, 0)) is Action.LEFT  # no enemy close: go
    assert carrier_move(escort=(2, 1), enemy=(7, 0)) is Action.LEFT  # nobody can join: go
    assert carrier_move(escort=(6, 1), enemy=(7, 0), buddy_system=False) is Action.LEFT  # switch off: original


def test_an_escort_on_the_carriers_cell_repeats_the_carriers_move() -> None:
    env = _env({"R2": (6, 0)})
    _carry(env, "R1", (6, 0))
    fixed = _ai(ALL_ON).act(_obs(env), Team.RED)
    assert fixed["R1"] is Action.LEFT and fixed["R2"] is fixed["R1"]
    assert _ai(OFF).act(_obs(env), Team.RED)["R2"] is Action.STAY  # the original escort stands on the flag cell


# ---------------------------------------------------------------------------- pair interception


def test_ambush_cell_is_on_the_carriers_way_home_and_reachable_in_time_by_everybody() -> None:
    env = _env({"R3": (6, 2), "R4": (7, 1)}, blue={"B1": (4, 1)})  # the interceptors are AHEAD of the carrier
    obs = _obs(env)
    carrier = obs.agents["B1"]
    interceptors = [obs.agents["R3"], obs.agents["R4"]]
    cell = ambush_cell(obs, Team.RED, carrier, interceptors)
    k = _path_len((4, 1), cell, obs)
    assert cell == (6, 0) and k == 3  # (5, 0) is 3 steps away for R3, so too early; (6, 0) is 2 and 2
    assert _path_len((4, 1), obs.blue_base, obs) == k + _path_len(cell, obs.blue_base, obs)  # on a shortest way home
    assert all(_path_len((i.x, i.y), cell, obs) <= k for i in interceptors)
    # nobody can get there in time: plain chase of the carrier's own cell
    near_home = dataclasses.replace(carrier, x=8, y=3)  # one step from its base, interceptors are far away
    assert ambush_cell(obs, Team.RED, near_home, interceptors) == (8, 3)


def test_the_scout_joins_the_defender_against_a_visible_carrier() -> None:
    env = _env({"R3": (1, 3), "R4": (1, 1)}, blue={"B1": (3, 3)})
    env._state.agents["B1"] = env._state.agents["B1"].with_carrying_flag(True)
    env._state.flags[Team.RED] = Flag(Team.RED, FlagStatus.CARRIED, 3, 3, "B1")
    obs = _obs(env)
    assert "B1" in obs.agents  # visible: R3 is two cells away
    fixed = _ai(ALL_ON).act(obs, Team.RED)
    off = _ai(OFF).act(obs, Team.RED)
    cell = ambush_cell(obs, Team.RED, obs.agents["B1"], [obs.agents["R3"], obs.agents["R4"]])
    before = _path_len((1, 1), cell, obs)
    dx, dy = fixed["R4"].delta()
    assert _path_len((1 + dx, 1 + dy), cell, obs) < before or before == 0
    assert fixed["R4"] != off["R4"]  # the original scout goes exploring instead


# ----------------------------------------------------------------------- state, literals, radius


def test_a_new_episode_is_noticed_without_reset() -> None:
    settings = ClassicalParams(**{**ALL_ON, "roles": ROLES}).fix_settings()
    table = {t: {f"{p}{i}": r for i, r in enumerate(ROLES, 1)} for t, p in ((Team.RED, "R"), (Team.BLUE, "B"))}
    strategy = FixedClassicalStrategy(table, settings)
    env = CTFEnvironment(V2)
    env.reset(seed=2)
    for _ in range(4):
        strategy.act(_obs(env), Team.RED)
        env.step({aid: Action.STAY for aid, a in env.state.agents.items() if a.active})
    old = strategy.coordinator
    assert env.state.timestep == 4
    env.reset(seed=3)  # nobody calls strategy.reset()
    strategy.act(_obs(env), Team.RED)
    assert strategy.coordinator is not old
    fresh = FixedClassicalStrategy(table, settings)
    fresh.act(_obs(env), Team.RED)
    assert strategy.coordinator.team_maps.for_team(Team.RED).observed == fresh.coordinator.team_maps.for_team(Team.RED).observed


def test_influence_literals_are_parameters() -> None:
    def settings(**kw) -> FixSettings:
        base = dataclasses.asdict(ClassicalParams(roles=ROLES).fix_settings())
        return FixSettings(**{**base, **kw})

    env = _env({"R1": (2, 3)}, blue={"B1": (4, 3)})
    obs = _obs(env)
    plain = _Influence(settings()).build(obs, Team.RED, None)
    assert plain[3][4] == 5.0 and plain[3][2] == 5.0 * 2 / 4  # penalty 5, radius 3, distance 2 -> 5 * (4 - 2) / 4
    tuned = _Influence(settings(enemy_penalty=9.0, influence_radius=1)).build(obs, Team.RED, None)
    assert tuned[3][4] == 9.0 and tuned[3][3] == 9.0 * 1 / 2 and tuned[3][2] == 0.0
    shared = SharedTeamMap(obs.width, obs.height, Team.RED)
    shared.enemy_probability[0][0] = 1.0
    assert _Influence(settings(probability_weight=7.0)).build(obs, Team.RED, shared)[0][0] == 7.0


def test_the_scout_uses_the_observations_radius_for_global_observations() -> None:
    env = _env({"R4": (3, 3)})
    obs = dataclasses.replace(env.observe(ObservationMode.GLOBAL, team=Team.RED), visibility_radius=1)

    def seen(role) -> int:
        shared = SharedTeamMap(obs.width, obs.height, Team.RED)
        settings = ClassicalParams(roles=ROLES).fix_settings()
        controller = role(shared) if role is ScoutRole else role(shared, settings, lambda team: [])
        controller.update_knowledge("R4", obs, Team.RED)
        return sum(map(sum, shared.observed))

    assert seen(FixedScoutRole) == 5  # radius 1 diamond
    assert seen(ScoutRole) == 25  # the original's literal 3 (clipped by the walls) ignores the radius


# ------------------------------------------------------------------------------------ end to end


def _wins(params: dict, opponent: OpponentSpec, games: int) -> int:
    wins = 0
    for seed in range(games):
        for team in (Team.RED, Team.BLUE):
            env = CTFEnvironment(NEAR)
            env.reset(seed=seed)
            ai = build_opponent(OpponentSpec("classical", "classical", 1.0, params), NEAR, Random(seed))
            other = build_opponent(opponent, NEAR, Random(seed + 1))
            ai.reset()
            other.reset()
            done = False
            while not done:
                mine = ai.act(env.observe(ObservationMode.PARTIAL, team=team), team)
                theirs = other.act(env.observe(ObservationMode.PARTIAL, team=team.opponent), team.opponent)
                state, _, done, _ = env.step({**mine, **theirs})
            wins += state.winner is team
    return wins


def test_fixed_classical_plays_whole_games_and_beats_a_motionless_and_a_parked_team() -> None:
    stay = OpponentSpec("stay", "stay", 1.0, {})
    parked = OpponentSpec(
        "scripted",
        "parked",
        1.0,
        {"roles": ROLES, "guard_mode": "on_flag", "guard_leash": 0, "scout_behavior": "stay", "chase_radius": 0, "escort": False},
    )
    # a stationary agent that spawned on its flag cell tags every lone carrier: only pairs get in
    # (measured with these seeds: stay 32 -> 39 wins of 40, parked 5 -> 34 of 40; the games are deterministic)
    assert _wins(ALL_ON, stay, 20) >= _wins(OFF, stay, 20) + 4
    assert _wins(ALL_ON, parked, 10) >= _wins(OFF, parked, 10) + 8
