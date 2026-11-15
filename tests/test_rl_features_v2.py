"""Features for the v2 rules (dropped flags, carrier eligibility) and old-checkpoint compatibility."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ctf import CTFConfig, CTFEnvironment, ObservationMode, Team  # noqa: E402
from ctf.actions import Action  # noqa: E402
from ctf.config import load_config  # noqa: E402
from ctf.game_state import FlagStatus  # noqa: E402
from rl.checkpoint import (  # noqa: E402
    CheckpointError,
    load_checkpoint,
    load_policy,
    migrate_config,
    observation_spec,
)
from rl.config import RLConfig  # noqa: E402
from rl.features import ACTOR_CHANNELS, ACTOR_VECTOR, CRITIC_VECTOR, FeatureBuilder, ObservationLayout  # noqa: E402
from rl.networks import NetworkSpec  # noqa: E402
from rl.rl_strategy import RLStrategy  # noqa: E402
from rl.schema import from_dict  # noqa: E402
from rl_helpers import rl_config  # noqa: E402
from strategies.roles import Role  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OLD_CHECKPOINT = ROOT / "runs" / "colab_run1" / "checkpoints" / "latest.pt"
ROLES = [Role.ATTACKER, Role.ATTACKER, Role.DEFENDER, Role.SCOUT]
STAY = Action.STAY


def _env(**kw) -> CTFEnvironment:
    """9x5 open board, 4 agents per team, v2 rules; RED attackers can step onto the BLUE flag at (8, 2)."""
    base = dict(
        width=9, height=5, agents_per_team=4, red_base=(0, 2), blue_base=(8, 2),
        red_spawns=[(7, 2), (7, 3), (0, 1), (0, 3)], blue_spawns=[(8, 0), (8, 4), (5, 2), (8, 3)],
        visibility_radius=3, carrier_slots=[1, 2], tagging_rule="headcount", drop_flag_on_tag=True,
        tagging_enabled=True, max_episode_steps=100,
    )
    base.update(kw)
    env = CTFEnvironment(CTFConfig(**base))
    env.reset()
    return env


def _builder(env: CTFEnvironment, team: Team = Team.RED, layout=None) -> FeatureBuilder:
    return FeatureBuilder(env.config, team, ROLES, rl_config().features, layout)


def _step(env, **moves):
    acts = {a: STAY for a, ag in env.state.agents.items() if ag.active}
    acts.update({k: v for k, v in moves.items() if k in acts})
    env.step(acts)


def _drop_flag(env: CTFEnvironment) -> None:
    """R1 takes the blue flag and walks into B3 at (5, 2): tagged, flag DROPPED at (5, 2)."""
    _step(env, R1=Action.RIGHT)
    for _ in range(3):
        _step(env, R1=Action.LEFT)
    assert env.state.flags[Team.BLUE].status is FlagStatus.DROPPED


def _channel(feats, name: str) -> np.ndarray:
    return feats.grid[0, ACTOR_CHANNELS.index(name)]


def _scalar(feats, name: str, slot: int = 0) -> float:
    return float(feats.vec[slot, ACTOR_VECTOR.index(name)])


# ----------------------------------------------------------------------------------- layout


def test_new_features_are_appended_so_old_layouts_are_prefixes() -> None:
    assert ACTOR_CHANNELS[-2:] == ("enemy_flag_dropped", "enemy_flag_memory")
    assert ACTOR_VECTOR[-3:] == ("own_flag_dropped", "enemy_flag_dropped", "self_can_carry")
    assert CRITIC_VECTOR == ACTOR_VECTOR


def test_layout_from_spec_rejects_unknown_features() -> None:
    spec = {k: list(v) for k, v in observation_spec().items() if k.startswith(("actor", "critic"))}
    assert ObservationLayout.from_spec(spec).is_current
    spec["actor_channels"] = spec["actor_channels"] + ["telepathy"]
    with pytest.raises(ValueError, match="telepathy"):
        ObservationLayout.from_spec(spec)


def test_an_older_layout_sees_exactly_the_matching_slice_of_the_new_features() -> None:
    env = _env()
    _drop_flag(env)
    old = ObservationLayout(ACTOR_CHANNELS[:-2], ACTOR_VECTOR[:-3],
                            ObservationLayout.current().critic_channels, CRITIC_VECTOR[:-3])
    obs, glob = env.observe(ObservationMode.PARTIAL, team=Team.RED), env.observe(ObservationMode.GLOBAL)
    full, small = _builder(env), _builder(env, layout=old)
    f_full, f_small = full.actor_features(obs), small.actor_features(obs)
    assert f_small.grid.shape[1] == len(ACTOR_CHANNELS) - 2 and f_small.vec.shape[1] == len(ACTOR_VECTOR) - 3
    np.testing.assert_array_equal(f_small.grid, f_full.grid[:, : len(ACTOR_CHANNELS) - 2])
    np.testing.assert_array_equal(f_small.vec, f_full.vec[:, : len(ACTOR_VECTOR) - 3])
    c_full, c_small = full.critic_features(glob), small.critic_features(glob)
    np.testing.assert_array_equal(c_small.vec, c_full.vec[:, : len(CRITIC_VECTOR) - 3])
    np.testing.assert_array_equal(c_small.grid, c_full.grid)
    assert NetworkSpec.from_env(env.config, old).actor_channels == len(ACTOR_CHANNELS) - 2


# ------------------------------------------------------------------------------- flag features


def test_carrier_eligibility_follows_the_env_rule_per_slot() -> None:
    f = _builder(_env()).actor_features(_env().observe(ObservationMode.PARTIAL, team=Team.RED))
    assert [_scalar(f, "self_can_carry", s) for s in range(4)] == [1.0, 1.0, 0.0, 0.0]
    env = _env(carrier_slots=None)
    f = _builder(env).actor_features(env.observe(ObservationMode.PARTIAL, team=Team.RED))
    assert [_scalar(f, "self_can_carry", s) for s in range(4)] == [1.0] * 4


def test_a_visible_dropped_enemy_flag_is_marked_and_remembered() -> None:
    env = _env()
    b = _builder(env)
    b.actor_features(env.observe(ObservationMode.PARTIAL, team=Team.RED))
    _drop_flag(env)  # flag lies at (5, 2); R2 stands at (7, 3), within radius 3 of it
    f = b.actor_features(env.observe(ObservationMode.PARTIAL, team=Team.RED))
    assert _channel(f, "enemy_flag_dropped")[2, 5] == 1.0
    assert _channel(f, "enemy_flag_memory")[2, 5] == 1.0
    assert _scalar(f, "enemy_flag_dropped") == 1.0 and _scalar(f, "own_flag_dropped") == 0.0


def test_the_memory_outlives_the_view_fades_and_clears_when_the_flag_is_gone() -> None:
    # Radius 2 keeps the geometry tight: from (7, 4) RED sees neither (5, 2) nor the blue base (8, 2).
    env = _env(dropped_flag_return_on_touch=False, visibility_radius=2,
               red_spawns=[(7, 2), (6, 3), (0, 1), (0, 3)])
    b = _builder(env)
    b.actor_features(env.observe(ObservationMode.PARTIAL, team=Team.RED))
    _drop_flag(env)  # the flag lies at (5, 2), two cells from R2 at (6, 3)
    f = b.actor_features(env.observe(ObservationMode.PARTIAL, team=Team.RED))
    assert _channel(f, "enemy_flag_dropped")[2, 5] == 1.0
    _step(env, R2=Action.RIGHT)
    _step(env, R2=Action.DOWN)  # R2 is at (7, 4): out of sight of the flag
    f = b.actor_features(env.observe(ObservationMode.PARTIAL, team=Team.RED))
    assert _channel(f, "enemy_flag_dropped").sum() == 0.0  # not visible now ...
    fade = _channel(f, "enemy_flag_memory")[2, 5]
    assert 0.0 < fade < 1.0 and _scalar(f, "enemy_flag_dropped") == 1.0  # ... but still remembered, fading
    # The flag is sent home while nobody on RED is looking (env.state is a copy: edit the live state).
    live = env._state
    live.flags[Team.BLUE] = live.flags[Team.BLUE].with_at_base(env.config)
    for _ in range(4):
        _step(env, R3=Action.RIGHT)  # R3 walks (0, 1) -> (4, 1): (5, 2) comes into view, the base stays hidden
    f = b.actor_features(env.observe(ObservationMode.PARTIAL, team=Team.RED))
    assert _channel(f, "enemy_flag_visible").sum() == 0.0  # the base flag is still unseen, so only the empty cell cleared it
    assert _channel(f, "enemy_flag_memory").sum() == 0.0 and _scalar(f, "enemy_flag_dropped") == 0.0


def test_reset_forgets_a_remembered_flag() -> None:
    env = _env(dropped_flag_return_on_touch=False)
    b = _builder(env)
    _drop_flag(env)
    b.actor_features(env.observe(ObservationMode.PARTIAL, team=Team.RED))
    b.reset()
    assert b._flag_memory is None


def test_own_dropped_flag_is_a_scalar_for_the_flags_team_and_the_critic_sees_both() -> None:
    env = _env(dropped_flag_return_on_touch=False)
    _drop_flag(env)
    blue = FeatureBuilder(env.config, Team.BLUE, ROLES, rl_config().features)
    f = blue.actor_features(env.observe(ObservationMode.PARTIAL, team=Team.BLUE))
    assert _scalar(f, "own_flag_dropped") == 1.0 and _scalar(f, "own_flag_taken") == 0.0
    cf = _builder(env).critic_features(env.observe(ObservationMode.GLOBAL))
    assert cf.vec[0, CRITIC_VECTOR.index("enemy_flag_dropped")] == 1.0


# --------------------------------------------------------- config migration and old checkpoint


@pytest.mark.skipif(not OLD_CHECKPOINT.exists(), reason="the 2M-step Colab checkpoint is not in this checkout")
class TestOldCheckpoint:
    def test_it_still_loads_for_playing_but_not_for_resuming(self) -> None:
        model, meta = load_policy(OLD_CHECKPOINT)
        assert not meta["layout"].is_current and meta["config"].roles.slots == [Role.ATTACKER, Role.DEFENDER, Role.SCOUT]
        assert all(o.label for p in meta["config"].curriculum.phases for o in p.opponents)  # labels were filled in
        assert model.spec.actor_channels == len(ACTOR_CHANNELS) - 2
        with pytest.raises(CheckpointError, match="observation layout differs"):
            load_checkpoint(OLD_CHECKPOINT)  # resuming training needs the exact layout

    def test_the_stored_config_parses_after_migration_only(self) -> None:
        plain = load_checkpoint(OLD_CHECKPOINT, exact_layout=False)["config"]
        with pytest.raises(Exception):
            from_dict(RLConfig, plain)  # the strict schema rejects the old dict as is
        cfg = from_dict(RLConfig, migrate_config(plain))
        assert cfg.rewards.escort_bonus == 0.0 and cfg.rewards.outcome_share_scout == 1.0
        assert [o.label for o in cfg.curriculum.phases[0].opponents] == ["stay", "random"]

    def test_it_plays_a_full_episode_on_its_own_original_rules(self) -> None:
        strategy = RLStrategy.from_checkpoint(OLD_CHECKPOINT, deterministic=True)
        env = CTFEnvironment(replace(strategy.env_config, seed=3))
        env.reset()
        done, steps = False, 0
        while not done and steps < 60:
            acts = strategy.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
            blue = {a: STAY for a, ag in env.state.agents.items() if ag.team is Team.BLUE and ag.active}
            _, _, done, _ = env.step({**acts, **blue})
            steps += 1
        assert steps > 0 and strategy.last_decision[Team.RED].probs.shape == (3, 5)

    def test_it_is_refused_on_a_four_agent_map_with_a_clear_error(self) -> None:
        strategy = RLStrategy.from_checkpoint(OLD_CHECKPOINT, env_config=load_config(ROOT / "configs" / "ctf_v2.yaml"))
        env = CTFEnvironment(strategy.env_config)
        env.reset()
        with pytest.raises(ValueError, match="one role per team slot"):
            strategy.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)


def test_migrate_config_is_idempotent_and_does_not_touch_current_configs() -> None:
    from rl.config import config_to_plain

    current = config_to_plain(rl_config())
    assert migrate_config(current) == current
    assert migrate_config(migrate_config(current)) == current
