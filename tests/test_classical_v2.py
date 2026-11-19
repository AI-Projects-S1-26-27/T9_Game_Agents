"""``classical_ai.v2``: the classical AI's v2 layer lives in the classical package and plays the v2 game as it did before the move."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from random import Random

import pytest

from ctf.agent import Team
from ctf.config import load_config
from ctf.environment import CTFEnvironment
from ctf.observations import ObservationMode
from rl.config import OpponentSpec
from rl.opponents import build_opponent
from rl_helpers import ROOT

ROLES = ["attacker", "attacker", "defender", "scout"]
FIXED = dict(roles=ROLES, remember_dropped_flag=True, defend_dropped_flag=True, buddy_system=True, intercept_in_pairs=True,
             pair_guard_radius=2, pair_wait_steps=20, carrier_danger_radius=2, influence_radius=3, enemy_penalty=5.0,
             probability_weight=3.0)
VARIANTS = {"original": dict(roles=ROLES), "fixed": FIXED}
OPPONENTS = {
    "stay": ("stay", {}),
    "leash": ("scripted", dict(roles=ROLES, guard_mode="leash", guard_leash=2, scout_behavior="random", chase_radius=0, escort=False)),
    "parked": ("scripted", dict(roles=ROLES, guard_mode="on_flag", guard_leash=0, scout_behavior="stay", chase_radius=0, escort=False)),
    "minimax": ("minimax", dict(
        depth=2, own_moves_per_agent=2, enemy_moves_per_agent=2, danger_radius=3, max_nodes=4000, alpha_beta=True,
        weights={"win": 1000.0, "carry_progress": 4.0, "pickup": 20.0, "approach": 1.0, "approach_others": 0.5, "escort": 2.0,
                 "enemy_carry_progress": 4.0, "enemy_pickup": 20.0, "defend": 1.0, "alive": 3.0, "pressure": 1.5})),
}


def _play(map_path: str, variant: str, opponent: str, team: Team, seed: int) -> tuple[str, int]:
    base = load_config(ROOT / map_path)
    cfg = replace(base, spawn_mode="near_base", seed=seed)
    env = CTFEnvironment(cfg)
    env.reset(seed=seed)
    me = build_opponent(OpponentSpec("classical", "c", 1.0, VARIANTS[variant]), cfg, Random(seed))
    kind, params = OPPONENTS[opponent]
    other = build_opponent(OpponentSpec(kind, kind, 1.0, params), cfg, Random(seed + 1))
    me.reset(), other.reset()
    done, steps = False, 0
    while not done:
        mine = me.act(env.observe(ObservationMode.PARTIAL, team=team), team)
        theirs = other.act(env.observe(ObservationMode.PARTIAL, team=team.opponent), team.opponent)
        state, _, done, _ = env.step({**mine, **theirs})
        steps += 1
    return ("win" if state.winner is team else "draw" if state.winner is None else "loss"), steps


def test_the_v2_layer_lives_in_the_classical_package_and_the_old_import_paths_still_work() -> None:
    import classical_ai.v2 as v2
    import rl.classical_fixes as old_fixes
    import rl.classical_opponent as old_opponent

    assert v2.ClassicalOpponent.__module__ == "classical_ai.v2" and v2.FixedClassicalStrategy.__module__ == "classical_ai.v2"
    assert old_opponent.ClassicalOpponent is v2.ClassicalOpponent and old_opponent.ClassicalParams is v2.ClassicalParams
    assert old_fixes.FixedClassicalStrategy is v2.FixedClassicalStrategy and old_fixes.FixSettings is v2.FixSettings


def test_the_classical_package_does_not_need_pytorch() -> None:
    code = "import sys, classical_ai.v2; assert 'torch' not in sys.modules, 'classical_ai.v2 imported torch'"
    done = subprocess.run([sys.executable, "-c", code], env={"PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


# Outcome and length of whole games. They were recorded with the code BEFORE it moved from rl/ to classical_ai/v2.py (the full
# action traces of 240 games were identical before and after), so they pin the behaviour. If the classical AI is changed
# on purpose, regenerate them (and the README / docs/classical.md numbers that depend on them).
GOLDEN = [
    ("configs/ctf_v2.yaml", "fixed", "minimax", Team.RED, [("win", 32), ("win", 28), ("win", 32), ("win", 26)]),
    ("configs/ctf_v2.yaml", "fixed", "minimax", Team.BLUE, [("draw", 200), ("win", 31), ("draw", 200), ("draw", 200)]),
    ("configs/ctf_v2.yaml", "fixed", "parked", Team.RED, [("win", 32), ("loss", 27), ("loss", 27), ("win", 26)]),
    ("configs/ctf_v2.yaml", "original", "stay", Team.RED, [("win", 28), ("win", 26), ("win", 26), ("draw", 200)]),
    ("configs/ctf_v2.yaml", "original", "leash", Team.BLUE, [("win", 34), ("win", 27), ("win", 32), ("win", 27)]),
    ("configs/maps/center_gate.yaml", "fixed", "leash", Team.RED, [("win", 24), ("win", 22), ("win", 21), ("win", 22)]),
]


@pytest.mark.parametrize("map_path,variant,opponent,team,expected", GOLDEN, ids=lambda x: x.value if isinstance(x, Team) else None)
def test_classical_games_are_unchanged_by_the_move(map_path, variant, opponent, team, expected) -> None:
    assert [_play(map_path, variant, opponent, team, seed) for seed in range(len(expected))] == expected


def test_the_v2_layer_also_runs_on_the_original_three_agent_rules() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.agents_per_team == 3 and cfg.carrier_slots is None
    roles = ["attacker", "defender", "scout"]
    for params in ({"roles": roles}, {**FIXED, "roles": roles}):
        for team in (Team.RED, Team.BLUE):
            env = CTFEnvironment(replace(cfg, seed=1))
            env.reset(seed=1)
            me = build_opponent(OpponentSpec("classical", "c", 1.0, params), cfg, Random(1))
            other = build_opponent(OpponentSpec("random", "random", 1.0, {}), cfg, Random(2))
            me.reset(), other.reset()
            done = False
            while not done:
                mine = me.act(env.observe(ObservationMode.PARTIAL, team=team), team)
                theirs = other.act(env.observe(ObservationMode.PARTIAL, team=team.opponent), team.opponent)
                assert set(mine) <= {f"{'R' if team is Team.RED else 'B'}{i}" for i in (1, 2, 3)}
                _, _, done, _ = env.step({**mine, **theirs})
