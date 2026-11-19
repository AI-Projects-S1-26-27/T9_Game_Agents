"""The LLM commander as a team of the opponent registry (kind ``llm``): evaluation harness, viewer, v2 classical baseline."""

from __future__ import annotations

from random import Random

import pytest

from ctf import CTFEnvironment, ObservationMode, Team
from ctf.config import load_config
from rl import evaluate as ev
from rl.config import OpponentSpec
from rl.eval_config import EvalOpponent
from rl.opponents import KNOWN_KINDS, build_opponent
from rl.schema import ConfigError
from rl_helpers import ROOT
from strategies.llm_commander import LLMCommanderStrategy
from strategies.llm_commander.conditions import build_condition
from strategies.llm_commander.opponents import make_opponent
from strategies.llm_commander.team import LLMTeamParams, make_classical, slot_roles
from viz.controllers import ControllerFactory, load_controller_specs
from viz.session import record_episode

V1 = load_config(ROOT / "configs" / "default.yaml")
V2 = load_config(ROOT / "configs" / "ctf_v2.yaml")
ROLES_V2 = ["attacker", "attacker", "defender", "scout"]
CONTROLLERS = ROOT / "configs" / "viz" / "controllers.yaml"


def _llm(params: dict | None = None, cfg=V2):
    return build_opponent(OpponentSpec("llm", "llm", 1.0, params or {}), cfg, Random(0))


def _llm_specs():
    return [s for s in load_controller_specs(CONTROLLERS) if s.kind == "llm"]


# ----------------------------------------------------------------------------------------- registry / params


def test_llm_is_a_built_in_kind_and_its_params_are_strict() -> None:
    assert "llm" in KNOWN_KINDS
    assert isinstance(_llm(), LLMCommanderStrategy)  # every key is optional
    assert isinstance(_llm({"condition": "llm_classical", "every_k_turns": 3}), LLMCommanderStrategy)
    for bad in (
        {"depth": 3},  # unknown key
        {"condition": "llm_telepathy"},  # unknown condition
        {"every_k_turns": 0},
        {"rpm": 0},
        {"condition": "llm_rl"},  # needs rl_checkpoint
    ):
        with pytest.raises(ConfigError):
            _llm(bad)


def test_a_provider_that_cannot_be_built_is_a_config_error_the_viewer_can_show() -> None:
    # the provider code raises a bare RuntimeError; the viewer only catches ConfigError / ValueError / OSError / ControllerError
    with pytest.raises(ConfigError, match="provider"):
        _llm({"provider": "no-such-provider"})


def test_roles_follow_the_carrier_slots_of_the_map() -> None:
    assert slot_roles(V2) == ROLES_V2
    assert slot_roles(V1) == ["attacker", "defender", "scout"]


def test_the_evaluation_harness_can_build_an_llm_opponent() -> None:
    entry = EvalOpponent(kind="llm", label="llm", params={"condition": "llm"})
    assert isinstance(ev.build_eval_opponent(entry, V2, Random(0)), LLMCommanderStrategy)


# ---------------------------------------------------------------------------------------------------- viewer


def test_viewer_yaml_lists_llm_controllers_with_the_roles_of_the_map() -> None:
    specs = _llm_specs()
    assert [s.params["condition"] for s in specs] == ["llm", "llm_classical"]
    factory = ControllerFactory()
    for spec in specs:
        ctrl = factory.build(spec, V2, 1)
        assert isinstance(ctrl.strategy, LLMCommanderStrategy) and ctrl.roles == ROLES_V2 and ctrl.rl is None
    assert factory.build(specs[1], V2, 1).strategy.controller_kind == "classical"


@pytest.mark.parametrize("opponent", ["Classical AI (A*", "Classical AI, fixed", "Scripted: parked guard", "LLM commander (mock model, classical", "Minimax"])
def test_an_llm_team_plays_whole_games_against_the_other_controllers_on_both_sides(opponent: str) -> None:
    factory = ControllerFactory()
    specs = {s.label: s for s in load_controller_specs(CONTROLLERS)}
    llm = next(s for label, s in specs.items() if label.startswith("LLM commander (mock model, simple"))
    other = next(s for label, s in specs.items() if opponent in label)
    for seed, (red, blue) in enumerate(((llm, other), (other, llm))):
        rec = record_episode(V2, factory.build(red, V2, seed), factory.build(blue, V2, seed + 1), seed=seed)
        assert rec.frames[-1].done and len(rec.frames) > 5
        agents = {a.id: a.role for a in rec.frames[0].agents}
        assert len(agents) == 8 and sorted(set(agents.values())) == ["attacker", "defender", "scout"]


def test_a_reused_llm_controller_replays_the_same_game() -> None:
    factory = ControllerFactory()
    specs = {s.label: s for s in load_controller_specs(CONTROLLERS)}
    llm = factory.build(next(s for l, s in specs.items() if l.startswith("LLM commander (mock model, simple")), V2, 1)
    other = factory.build(next(s for l, s in specs.items() if "Classical AI (A*" in l), V2, 2)

    def run():
        rec = record_episode(V2, llm, other, seed=3)
        return len(rec.frames), [[(a.id, a.x, a.y) for a in f.agents] for f in rec.frames]

    assert run() == run()  # reset() gives each game a fresh commander memory and a fresh classical belief map


def test_the_cache_lets_a_game_be_replayed_without_the_service(tmp_path) -> None:
    cache = tmp_path / "answers.json"
    params = {"condition": "llm", "provider": "mock", "cache": str(cache), "every_k_turns": 2}
    hits = []
    for _ in range(2):
        team = _llm(params)
        blue = build_opponent(OpponentSpec("stay", "stay", 1.0, {}), V2, Random(0))
        env = CTFEnvironment(V2)
        env.reset(seed=4)
        team.reset()
        done = False
        while not done:
            acts = team.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
            acts.update(blue.act(env.observe(ObservationMode.PARTIAL, team=Team.BLUE), Team.BLUE))
            _, _, done, _ = env.step(acts)
        hits.append(team.commander.llm.hits)
    assert cache.exists() and cache.stat().st_size > 2
    assert hits[0] == 0 and hits[1] > 0  # the second game is answered from the cache, without asking the model again


# --------------------------------------------------------------------------------- v2 classical baselines


def test_the_llm_packages_classical_baseline_uses_the_fixed_v2_roles() -> None:
    from classical_ai.classical_strategy import ClassicalStrategy
    from rl.classical_opponent import ClassicalOpponent

    for team in (make_classical(V2), make_opponent("classical", V2), build_condition("classical", llm=None, max_steps=200, config=V2)):
        assert isinstance(team, ClassicalOpponent)
    env = CTFEnvironment(V2)
    env.reset(seed=3)
    team = make_classical(V2)
    team.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
    assert team._inner.coordinator.role_assignments[Team.RED] == dict(zip(["R1", "R2", "R3", "R4"], ROLES_V2))
    # the original rules keep the plain classical strategy exactly as before
    assert type(make_classical(V1)) is ClassicalStrategy and type(make_classical(None)) is ClassicalStrategy
    assert type(make_opponent("classical", V1)) is ClassicalStrategy
