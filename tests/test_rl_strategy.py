import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ctf import CTFConfig, CTFEnvironment, ObservationMode, Team  # noqa: E402
from ctf.actions import Action  # noqa: E402
from rl.checkpoint import save_checkpoint  # noqa: E402
from rl.config import load_base_env_config  # noqa: E402
from rl.rl_strategy import RLStrategy  # noqa: E402
from rl.trainer import Trainer  # noqa: E402
from rl_helpers import ROOT, tiny_config  # noqa: E402
from strategies import Role  # noqa: E402


@pytest.fixture()
def checkpoint(tmp_path):
    cfg = tiny_config(str(tmp_path))
    trainer = Trainer(cfg, load_base_env_config(cfg, ROOT), tmp_path / "run")
    trainer.counters.update(total_steps=1000, update=3)
    path = tmp_path / "ck.pt"
    save_checkpoint(path, **trainer._state())
    return path


def _env():
    env = CTFEnvironment(CTFConfig(
        obstacles=[(5, y) for y in range(1, 6)], visibility_radius=3,
        red_spawns=[(1, 2), (1, 3), (1, 4)], blue_spawns=[(9, 2), (9, 3), (9, 4)]))
    env.reset(seed=0)
    return env


def test_acts_for_active_agents_and_exposes_probabilities(checkpoint) -> None:
    strat = RLStrategy.from_checkpoint(checkpoint, deterministic=True)
    env = _env()
    for team in (Team.RED, Team.BLUE):
        actions = strat.act(env.observe(ObservationMode.PARTIAL, team=team), team)
        prefix = "R" if team is Team.RED else "B"
        assert set(actions) == {f"{prefix}1", f"{prefix}2", f"{prefix}3"}
        d = strat.last_decision[team]
        assert d.probs.shape == (3, 5) and np.allclose(d.probs.sum(1), 1.0, atol=1e-5)
        for slot, aid in enumerate(d.agent_ids):  # chosen action = most likely action (env order)
            assert list(Action)[int(d.probs[slot].argmax())] is actions[aid]
    assert strat.checkpoint_step == 1000 and 0.0 < strat.scout_scale <= 1.0


def test_global_and_partial_observations_give_identical_decisions(checkpoint) -> None:
    a = RLStrategy.from_checkpoint(checkpoint)
    b = RLStrategy.from_checkpoint(checkpoint)
    env = _env()
    got_partial = a.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
    got_global = b.act(env.observe(ObservationMode.GLOBAL), Team.RED)  # must not leak hidden info
    assert got_partial == got_global
    np.testing.assert_allclose(a.last_decision[Team.RED].probs, b.last_decision[Team.RED].probs)


def test_blue_probabilities_are_reported_in_environment_orientation(checkpoint) -> None:
    strat = RLStrategy.from_checkpoint(checkpoint, deterministic=True)
    env = _env()
    red = strat.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
    blue = strat.act(env.observe(ObservationMode.PARTIAL, team=Team.BLUE), Team.BLUE)
    left, right = list(Action).index(Action.LEFT), list(Action).index(Action.RIGHT)
    pr, pb = strat.last_decision[Team.RED].probs, strat.last_decision[Team.BLUE].probs
    # on a mirror-symmetric start, BLUE's environment-orientation LEFT/RIGHT are swapped relative to RED's
    np.testing.assert_allclose(pr[:, left], pb[:, right], atol=1e-6)
    np.testing.assert_allclose(pr[:, right], pb[:, left], atol=1e-6)
    assert len(red) == len(blue) == 3


def test_controlled_roles_limit_the_returned_actions(checkpoint) -> None:
    strat = RLStrategy.from_checkpoint(checkpoint, controlled_roles={Role.ATTACKER})
    actions = strat.act(_env().observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
    assert set(actions) == {"R1"}


def test_belief_resets_when_a_new_episode_starts_and_value_is_finite(checkpoint) -> None:
    strat = RLStrategy.from_checkpoint(checkpoint)
    env = _env()
    for _ in range(5):
        strat.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
        env.step({aid: Action.RIGHT for aid, a in env.state.agents.items() if a.active})
    builder = strat._builders[Team.RED]
    assert np.asarray(builder.belief.observed).sum() > 0
    env.reset(seed=1)
    strat.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)  # timestep went back to 0
    assert strat._last_t[Team.RED] == 0
    v = strat.value(env.observe(ObservationMode.GLOBAL), Team.RED)
    assert v.shape == (3,) and np.isfinite(v).all()
    calc = strat.make_reward_calculator(Team.RED)
    assert calc.team is Team.RED


def test_map_size_mismatch_is_refused(checkpoint) -> None:
    with pytest.raises(ValueError, match="grid"):
        RLStrategy.from_checkpoint(checkpoint, env_config=CTFConfig(width=9, height=5, red_base=(1, 2), blue_base=(7, 2)))
