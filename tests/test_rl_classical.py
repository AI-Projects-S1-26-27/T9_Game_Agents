"""The classical AI as an opponent (rl/classical_opponent.py): fixed roles, per-episode reset, rule checks, hookup."""

from __future__ import annotations

from random import Random

import pytest

from ctf import CTFEnvironment, ObservationMode, Team
from ctf.actions import Action
from ctf.config import load_config
from rl.classical_opponent import ClassicalOpponent
from rl.config import OpponentSpec
from rl.eval_config import load_eval_config
from rl.opponents import KNOWN_KINDS, build_opponent
from rl.schema import ConfigError
from rl_helpers import ROOT
from rl import evaluate as ev
from viz.controllers import ControllerFactory, load_controller_specs
from viz.session import record_episode

V2 = load_config(ROOT / "configs" / "ctf_v2.yaml")
ROLES = ["attacker", "attacker", "defender", "scout"]
TABLE = {
    Team.RED: {"R1": "attacker", "R2": "attacker", "R3": "defender", "R4": "scout"},
    Team.BLUE: {"B1": "attacker", "B2": "attacker", "B3": "defender", "B4": "scout"},
}


def _classical(params=None, cfg=V2) -> ClassicalOpponent:
    return build_opponent(OpponentSpec("classical", "classical", 1.0, {"roles": ROLES} if params is None else params), cfg, Random(0))


def _random(seed: int):
    return build_opponent(OpponentSpec("random", "random", 1.0, {}), V2, Random(seed))


def _game(team: Team, seed: int, opponent=None, classical=None) -> tuple[str, list[dict[str, str]]]:
    """Classical plays ``team`` against a random team; returns the outcome and its action trace."""
    env = CTFEnvironment(V2)
    env.reset(seed=seed)
    ai = classical or _classical()
    ai.reset()
    other = opponent or _random(seed + 1)
    other.reset()
    trace, done = [], False
    while not done:
        mine = ai.act(env.observe(ObservationMode.PARTIAL, team=team), team)
        theirs = other.act(env.observe(ObservationMode.PARTIAL, team=team.opponent), team.opponent)
        trace.append({k: v.name for k, v in sorted(mine.items())})
        state, _, done, _ = env.step({**mine, **theirs})
    outcome = "win" if state.winner is team else "draw" if state.winner is None else "loss"
    return outcome, trace


# ----------------------------------------------------------------------------- schema / rules


def test_classical_is_a_built_in_kind_with_strict_params() -> None:
    assert "classical" in KNOWN_KINDS
    assert isinstance(_classical(), ClassicalOpponent)
    with pytest.raises(ConfigError):
        _classical({"roles": ROLES, "depth": 3})  # unknown key
    with pytest.raises(ConfigError):
        _classical({})  # roles are required
    with pytest.raises(ConfigError):
        _classical({"roles": ["attacker", "attacker", "defender", "goalie"]})  # unknown role


def test_roles_must_match_the_team_size_and_the_carrier_slots() -> None:
    with pytest.raises(ConfigError, match="roles"):
        _classical({"roles": ["attacker", "defender", "scout"]})  # 3 roles, 4 agents
    with pytest.raises(ConfigError, match="carry"):
        _classical({"roles": ["defender", "scout", "attacker", "attacker"]})  # slots 3-4 cannot carry in v2


# ---------------------------------------------------------------------------- roles and reset


def test_role_table_is_explicit_and_does_not_shift_when_an_agent_is_tagged() -> None:
    env = CTFEnvironment(V2)
    env.reset(seed=3)
    ai = _classical()
    first = ai.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
    assert set(first) == {"R1", "R2", "R3", "R4"}
    assert ai._inner.coordinator.role_assignments == TABLE
    env._state.agents["R1"] = env._state.agents["R1"].with_active(False)  # R1 tagged out
    later = ai.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
    assert set(later) == {"R2", "R3", "R4"}  # only active agents act ...
    assert ai._inner.coordinator.role_assignments == TABLE  # ... and nobody changes role


def test_the_defender_slot_returns_to_its_base_while_the_default_assignment_would_not() -> None:
    env = CTFEnvironment(V2)
    env.reset(seed=3)
    env._state.agents["R3"] = env._state.agents["R3"].with_position(3, 3)  # base is (1, 3)
    acts = _classical().act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
    assert acts["R3"] is Action.LEFT


def test_reset_gives_every_episode_a_fresh_belief_map() -> None:
    def observed_after_first_step(ai: ClassicalOpponent, seed: int):
        env = CTFEnvironment(V2)
        env.reset(seed=seed)
        ai.reset()
        ai.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
        return [row[:] for row in ai._inner.coordinator.team_maps.for_team(Team.RED).observed]

    reused = _classical()
    _game(Team.RED, 5, classical=reused)  # a whole episode fills the shared map
    old = reused._inner
    assert observed_after_first_step(reused, 6) == observed_after_first_step(_classical(), 6)
    assert reused._inner is not old
    reused.reset()
    assert reused._inner is None


def test_a_reused_instance_replays_the_same_game() -> None:
    ai = _classical()
    a = _game(Team.BLUE, 11, opponent=_random(11), classical=ai)
    b = _game(Team.BLUE, 11, opponent=_random(11), classical=ai)
    assert a == b


# ------------------------------------------------------------------------------- play / hookup


def test_plays_both_sides_without_errors_and_beats_a_random_team() -> None:
    wins = 0
    for team in (Team.RED, Team.BLUE):
        for seed in range(6):
            outcome, trace = _game(team, 100 + seed)
            assert trace and all(a in {x.name for x in Action} for step in trace for a in step.values())
            wins += outcome == "win"
    assert wins >= 8  # measured 12 of 12 with these seeds; a broken team table gives about 0


def test_eval_yaml_and_viewer_yaml_list_the_classical_ai() -> None:
    cfg = load_eval_config(ROOT / "configs" / "rl" / "eval.yaml")
    entry = next(o for o in cfg.opponents if o.kind == "classical")
    assert entry.label == "classical" and entry.params == {"roles": ROLES}
    assert isinstance(ev.build_eval_opponent(entry, V2, Random(0)), ClassicalOpponent)

    specs = [s for s in load_controller_specs(ROOT / "configs" / "viz" / "controllers.yaml") if s.kind == "classical"]
    assert len(specs) == 1
    ctrl = ControllerFactory().build(specs[0], V2, 0)
    assert isinstance(ctrl.strategy, ClassicalOpponent) and ctrl.roles == ROLES


def test_the_viewer_records_a_classical_game_and_shows_its_roles() -> None:
    spec = next(s for s in load_controller_specs(ROOT / "configs" / "viz" / "controllers.yaml") if s.kind == "classical")
    factory = ControllerFactory()
    rec = record_episode(V2, factory.build(spec, V2, 1), factory.build(spec, V2, 2), seed=0)
    assert rec.frames[-1].done
    roles = {a.id: a.role for a in rec.frames[0].agents}
    assert [roles[f"R{i}"] for i in range(1, 5)] == ROLES and [roles[f"B{i}"] for i in range(1, 5)] == ROLES
