"""Rewards, scripted opponents and the shipped default config under the v2 rules (4 agents per team)."""

from dataclasses import replace
from pathlib import Path
from random import Random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ctf import CTFConfig, CTFEnvironment, ObservationMode, Team  # noqa: E402
from ctf.actions import Action  # noqa: E402
from ctf.config import load_config  # noqa: E402
from rl.config import OpponentSpec, load_base_env_config, load_rl_config, phase_env_config  # noqa: E402
from rl.env_wrapper import CTFTrainEnv  # noqa: E402
from rl.metrics import read_metrics, summarize_episodes  # noqa: E402
from rl.navigation import GridNav  # noqa: E402
from rl.opponents import GuardMode, ScoutBehavior, ScriptedParams, ScriptedTeamStrategy, build_opponent  # noqa: E402
from rl.rewards import RewardCalculator  # noqa: E402
from rl.schema import ConfigError  # noqa: E402
from rl.trainer import Trainer  # noqa: E402
from rl_helpers import rl_config  # noqa: E402
from strategies import Role, StayStrategy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ROLES = [Role.ATTACKER, Role.ATTACKER, Role.DEFENDER, Role.SCOUT]
STAY = Action.STAY


def _env(**kw) -> CTFEnvironment:
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


def _step(env, **moves):
    acts = {a: STAY for a, ag in env.state.agents.items() if ag.active}
    acts.update({k: v for k, v in moves.items() if k in acts})
    return env.step(acts)[3]["events"]


def _calc(env, **reward_kw) -> RewardCalculator:
    base = rl_config().rewards
    zero = {f: 0.0 for f in base.__dataclass_fields__
            if f not in ("scout_anneal_steps", "escort_radius", "escort_roles")}
    cfg = replace(base, **{**zero, "escort_radius": 0, "escort_roles": [], "outcome_share_attacker": 1.0,
                           "outcome_share_defender": 1.0, "outcome_share_scout": 1.0, **reward_kw})
    c = env.config
    nav = GridNav([[(x, y) not in set(c.obstacles) for x in range(c.width)] for y in range(c.height)])
    calc = RewardCalculator(c, Team.RED, ROLES, cfg, 1.0, nav)
    calc.reset(env.observe(ObservationMode.GLOBAL))
    return calc


def _advance(env, calc, **moves):
    prev = dict(env.observe(ObservationMode.GLOBAL).agents)
    events = _step(env, **moves)
    return calc.step(env.observe(ObservationMode.GLOBAL), prev, events, 1.0)


# ------------------------------------------------------------------------------- rewards


def test_outcome_is_shared_per_role() -> None:
    env = _env()
    calc = _calc(env, win=10.0, loss=-10.0, timeout=-1.0, outcome_share_attacker=1.0,
                 outcome_share_defender=0.7, outcome_share_scout=0.5)
    glob = env.observe(ObservationMode.GLOBAL)
    prev = dict(glob.agents)
    for winner, base in ((Team.RED, 10.0), (Team.BLUE, -10.0), (None, -1.0)):
        res = calc.step(replace(glob, terminated=True, winner=winner), prev, [], 1.0)
        np.testing.assert_allclose(res.components["outcome"], [base, base, 0.7 * base, 0.5 * base])


def test_escort_bonus_goes_to_non_carrying_escort_roles_near_the_carrier() -> None:
    env = _env(red_spawns=[(7, 2), (7, 2), (7, 3), (6, 2)], blue_spawns=[(8, 0), (8, 4), (0, 0), (0, 4)])
    calc = _calc(env, escort_bonus=0.5, escort_radius=1, escort_roles=[Role.ATTACKER, Role.SCOUT])
    res = _advance(env, calc, R1=Action.RIGHT, R2=Action.RIGHT, R3=Action.RIGHT, R4=Action.RIGHT)
    # R1 (carrier) and R2 (attacker) are now on the flag cell, R3 (defender) too; R4 (scout) went (6,2)->(7,2)
    assert env.state.agents["R1"].carrying_flag and not env.state.agents["R2"].carrying_flag
    np.testing.assert_allclose(res.components["escort"], [0.0, 0.5, 0.0, 0.5])  # defender is not an escort role


def test_escort_radius_zero_needs_the_same_cell_and_no_carrier_means_no_bonus() -> None:
    env = _env(red_spawns=[(7, 2), (7, 2), (7, 3), (7, 2)], blue_spawns=[(8, 0), (8, 4), (0, 0), (0, 4)])
    calc = _calc(env, escort_bonus=0.5, escort_radius=0, escort_roles=[Role.ATTACKER, Role.SCOUT])
    assert not _advance(env, calc).components["escort"].any()  # nobody carries yet
    res = _advance(env, calc, R1=Action.RIGHT, R2=Action.RIGHT)  # R4 stays one cell away
    np.testing.assert_allclose(res.components["escort"], [0.0, 0.5, 0.0, 0.0])


def test_defender_intercept_potential_pulls_toward_the_nearest_enemy_carrier() -> None:
    env = _env(blue_spawns=[(1, 2), (8, 4), (5, 2), (8, 3)], red_spawns=[(7, 2), (7, 3), (3, 2), (0, 3)])
    _step(env, B1=Action.LEFT)  # B1 takes the red flag at (0, 2)
    assert env.state.agents["B1"].carrying_flag
    glob = env.observe(ObservationMode.GLOBAL)
    guard = _calc(env, defender_potential=0.02, defender_intercept_potential=0.0)
    intercept = _calc(env, defender_potential=0.02, defender_intercept_potential=0.06)
    d_carrier = guard.nav.distance((3, 2), (0, 2))
    d_flag = guard.nav.distance((3, 2), (glob.flags[Team.RED].x, glob.flags[Team.RED].y))
    assert d_carrier == d_flag  # the flag moves with its carrier, so both targets coincide ...
    assert guard._potentials(glob)[2] == pytest.approx(-0.02 * d_flag)  # ... only the weight differs
    assert intercept._potentials(glob)[2] == pytest.approx(-0.06 * d_carrier)


def test_defender_guards_a_dropped_own_flag_where_it_lies() -> None:
    env = _env(red_spawns=[(7, 2), (7, 3), (3, 2), (0, 3)], blue_spawns=[(8, 0), (8, 4), (5, 2), (8, 3)])
    # RED's own flag gets dropped at (2, 2): BLUE's carrier is tagged there by R3 ... use live state for brevity
    live = env._state
    live.flags[Team.RED] = live.flags[Team.RED].with_dropped(2, 2)
    calc = _calc(env, defender_potential=0.1, defender_intercept_potential=0.0)
    phi = calc._potentials(env.observe(ObservationMode.GLOBAL))
    assert phi[2] == pytest.approx(-0.1 * calc.nav.distance((3, 2), (2, 2)))  # R3 at (3, 2), flag lies at (2, 2)


def test_scout_bonus_keeps_a_floor_after_the_anneal() -> None:
    env = _env()
    for floor, expected in ((0.0, 0.0), (0.3, 0.3)):
        calc = _calc(env, scout_new_cell=1.0, scout_floor_fraction=floor)
        prev = dict(env.observe(ObservationMode.GLOBAL).agents)
        _step(env, R4=Action.RIGHT)
        res = calc.step(env.observe(ObservationMode.GLOBAL), prev, [], 0.0)  # fully annealed
        new_cells = calc.cfg.scout_new_cell  # 1.0, so the bonus equals (floor share) x (cells newly seen)
        cells = res.components["scout"][3] / (new_cells * expected) if expected else None
        if expected == 0.0:
            assert res.components["scout"][3] == 0.0
        else:
            assert cells > 0 and cells == pytest.approx(round(cells))  # a whole number of cells at 30% weight
        env = _env()


# ----------------------------------------------------------------------------- opponents


def _params(**kw) -> ScriptedParams:
    base = dict(roles=list(ROLES), guard_mode=GuardMode.ON_FLAG, guard_leash=0, scout_behavior=ScoutBehavior.STAY,
                chase_radius=0, escort=False)
    base.update(kw)
    return ScriptedParams(**base)


def _play(env, red, blue, steps=200) -> str:
    for _ in range(steps):
        acts = {**red.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED),
                **blue.act(env.observe(ObservationMode.PARTIAL, team=Team.BLUE), Team.BLUE)}
        _, _, done, _ = env.step(acts)
        if done:
            break
    w = env.state.winner
    return "draw" if w is None else w.value


def _v2_match(escort: bool) -> str:
    cfg = replace(load_config(ROOT / "configs" / "ctf_v2.yaml"), seed=0)
    env = CTFEnvironment(cfg)
    env.reset()
    red = ScriptedTeamStrategy(cfg, _params(escort=escort), Random(0))
    return _play(env, red, StayStrategy())


def test_a_parked_defender_is_only_beaten_by_an_escorted_carrier() -> None:
    assert _v2_match(escort=False) == "draw"  # the pair arrives staggered; each carrier is alone and tagged
    assert _v2_match(escort=True) == "RED"  # arriving together, the second attacker is the escort


def test_unescorted_scripted_attackers_still_capture_an_undefended_flag() -> None:
    # Blue sits in the far corners: nobody is near the flag, so a lone carrier is not outnumbered.
    cfg = replace(load_config(ROOT / "configs" / "ctf_v2.yaml"), seed=0,
                  blue_spawns=[(10, 0), (10, 6), (8, 0), (8, 6)])
    env = CTFEnvironment(cfg)
    env.reset()
    red = ScriptedTeamStrategy(cfg, _params(), Random(0))
    assert _play(env, red, StayStrategy()) == "RED"


def test_the_escort_leader_waits_for_its_partner_then_both_move_as_one() -> None:
    cfg = replace(load_config(ROOT / "configs" / "ctf_v2.yaml"), red_spawns=[(1, 2), (3, 2), (1, 3), (2, 3)])
    env = CTFEnvironment(cfg)
    env.reset()
    strat = ScriptedTeamStrategy(cfg, _params(escort=True), Random(0))
    acts = strat.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
    assert acts["R1"] is STAY and acts["R2"] is not STAY  # R1 waits, R2 closes the gap
    for _ in range(6):  # R2 reaches R1's cell; from then on they share it and share their moves
        env.step({**strat.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED),
                  **{a: STAY for a in cfg.agent_ids() if a.startswith("B")}})
    s = env.state.agents
    assert (s["R1"].x, s["R1"].y) == (s["R2"].x, s["R2"].y) and s["R1"].x > 1


def test_chaser_leaves_its_flag_for_a_close_enemy_and_ignores_a_far_one() -> None:
    chaser = _params(guard_mode=GuardMode.CHASER, chase_radius=3)
    near = _env(red_spawns=[(7, 3), (0, 0), (0, 1), (0, 4)], blue_spawns=[(8, 0), (8, 4), (8, 2), (8, 3)],
                carrier_slots=[1, 2])
    far = _env(red_spawns=[(2, 3), (0, 0), (0, 1), (0, 4)], blue_spawns=[(8, 0), (8, 4), (8, 2), (8, 3)])
    for env, moves in ((near, True), (far, False)):
        strat = ScriptedTeamStrategy(env.config, chaser, Random(0))
        acts = strat.act(env.observe(ObservationMode.PARTIAL, team=Team.BLUE), Team.BLUE)
        assert (acts["B3"] is not STAY) is moves  # B3 stands on its flag at (8, 2)


def test_defender_walks_to_its_dropped_flag_to_send_it_home() -> None:
    env = _env(red_spawns=[(0, 1), (0, 3), (0, 0), (0, 4)], blue_spawns=[(8, 0), (8, 4), (8, 3), (8, 1)])
    live = env._state
    live.flags[Team.BLUE] = live.flags[Team.BLUE].with_dropped(5, 3)
    strat = ScriptedTeamStrategy(env.config, _params(), Random(0))
    acts = strat.act(env.observe(ObservationMode.PARTIAL, team=Team.BLUE), Team.BLUE)
    assert acts["B3"] is Action.LEFT  # B3 at (8, 3), the flag at (5, 3)


def test_scripted_attackers_must_sit_in_carrier_slots() -> None:
    cfg = load_config(ROOT / "configs" / "ctf_v2.yaml")  # slots 1 and 2 carry
    bad = _params(roles=[Role.DEFENDER, Role.ATTACKER, Role.ATTACKER, Role.SCOUT])
    with pytest.raises(ConfigError, match="attacker in slot 3"):
        ScriptedTeamStrategy(cfg, bad, Random(0))
    ScriptedTeamStrategy(cfg, _params(roles=[Role.ATTACKER, Role.DEFENDER, Role.SCOUT, Role.SCOUT]), Random(0))


def test_chaser_needs_a_radius_and_radii_cannot_be_negative() -> None:
    cfg = load_config(ROOT / "configs" / "ctf_v2.yaml")
    with pytest.raises(ConfigError, match="chase_radius >= 1"):
        ScriptedTeamStrategy(cfg, _params(guard_mode=GuardMode.CHASER, chase_radius=0), Random(0))
    with pytest.raises(ConfigError, match="chase_radius must be >= 0"):
        ScriptedTeamStrategy(cfg, _params(chase_radius=-1), Random(0))


# ---------------------------------------------------------------------------- metrics + config


def _info(outcome, opponent, fixed=False, team="RED"):
    return {"outcome": outcome, "length": 10, "learner_team": team, "opponent": opponent, "pickups": 0,
            "tags_inflicted": 0, "tags_suffered": 0, "phase": "p", "fixed_spawns": fixed,
            "return": [0.0] * 4, "components": {"outcome": [0.0] * 4}}


def test_metrics_split_outcomes_per_opponent_label_and_per_spawn_style() -> None:
    infos = [_info("win", "scripted_leash", True), _info("draw", "scripted_leash"), _info("loss", "scripted_chaser"),
             _info("win", "scripted_chaser", True)]
    s = summarize_episodes(infos, ["attacker", "attacker", "defender", "scout"])
    assert s["win_rate/opp_scripted_leash"] == 0.5 and s["draw_rate/opp_scripted_leash"] == 0.5
    assert s["loss_rate/opp_scripted_chaser"] == 0.5 and s["loss_rate/opp_scripted_leash"] == 0.0
    assert s["win_rate/spawns_fixed"] == 1.0 and s["win_rate/spawns_random"] == 0.0


def test_opponent_labels_must_be_unique_and_plain() -> None:
    cfg = load_rl_config(ROOT / "configs" / "rl" / "default.yaml")
    phase = cfg.curriculum.phases[0]
    dup = replace(phase, opponents=[phase.opponents[0], replace(phase.opponents[1], label=phase.opponents[0].label)])
    with pytest.raises(ConfigError, match="unique"):
        replace(cfg, curriculum=replace(cfg.curriculum, phases=[dup]))
    bad = replace(phase, opponents=[replace(phase.opponents[0], label="Has Space")])
    with pytest.raises(ConfigError, match="labels must match"):
        replace(cfg, curriculum=replace(cfg.curriculum, phases=[bad]))


def test_reward_settings_are_validated() -> None:
    cfg = load_rl_config(ROOT / "configs" / "rl" / "default.yaml")
    for field, value in (("scout_floor_fraction", 1.5), ("escort_radius", -1), ("outcome_share_scout", -0.1)):
        with pytest.raises(ConfigError):
            replace(cfg, rewards=replace(cfg.rewards, **{field: value}))


# --------------------------------------------------------------------------- shipped default


def test_default_rl_config_matches_the_v2_environment() -> None:
    cfg = load_rl_config(ROOT / "configs" / "rl" / "default.yaml")
    env = load_base_env_config(cfg, ROOT)
    assert env.agents_per_team == len(cfg.roles.slots) == 4 and env.tagging_rule.value == "headcount"
    # exactly the carrier slots are attackers, so no role is asked to do what the rules forbid
    assert [r is Role.ATTACKER for r in cfg.roles.slots] == [env.can_carry(i) for i in range(1, 5)]
    assert env.red_spawns == [(1, 2), (1, 4), (1, 3), (2, 3)] and env.blue_spawns == [(9, 2), (9, 4), (9, 3), (8, 3)]


def test_every_opponent_of_every_phase_builds_and_plays_one_episode() -> None:
    cfg = load_rl_config(ROOT / "configs" / "rl" / "default.yaml")
    base = load_base_env_config(cfg, ROOT)
    for idx, phase in enumerate(cfg.curriculum.phases):
        env_cfg = phase_env_config(base, phase, fixed_spawns=False)
        for spec in phase.opponents:
            build_opponent(spec, env_cfg, Random(0))  # raises ConfigError for any invalid entry
        wrapper = CTFTrainEnv(cfg, base, seed=idx)
        wrapper.set_phase(idx)
        wrapper.reset()
        actions = np.full(base.agents_per_team, 4, dtype=np.int64)  # STAY
        for _ in range(5):
            wrapper.step(actions)


@pytest.mark.parametrize("phase_index", [0, 1, 2])
def test_each_default_phase_trains_for_a_few_updates(tmp_path, phase_index) -> None:
    cfg = load_rl_config(ROOT / "configs" / "rl" / "default.yaml")
    cfg = replace(
        cfg,
        run=replace(cfg.run, output_dir=str(tmp_path), name="t", tensorboard=False, checkpoint_every_steps=16),
        model=replace(cfg.model, conv_channels=[4], hidden_size=16),
        rollout=replace(cfg.rollout, num_envs=2, num_workers=0, rollout_length=8, total_steps=48),
        curriculum=replace(cfg.curriculum, phases=[cfg.curriculum.phases[phase_index]]),
    )
    trainer = Trainer(cfg, load_base_env_config(cfg, ROOT), tmp_path / "run")
    assert trainer.run()["total_steps"] == 48
    rows = [r for r in read_metrics(tmp_path / "run" / "metrics.jsonl") if "event" not in r]
    assert len(rows) == 3 and all(np.isfinite(r["train/policy_loss"]) for r in rows)
