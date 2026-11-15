"""The v2 flag rules: slot-restricted carrying, headcount tagging, dropped flags.

Every test uses a small open grid. Red base is (0, 2) and blue base is (8, 2). Agents are placed
with explicit spawns, so each test states exactly who stands where.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctf.actions import Action
from ctf.agent import Team
from ctf.config import CTFConfig, TaggingRule, config_from_dict, config_to_dict, load_config
from ctf.environment import CTFEnvironment
from ctf.game_state import FlagStatus
from ctf.observations import ObservationMode

ROOT = Path(__file__).resolve().parents[1]

V2 = dict(
    carrier_slots=[1],
    tagging_rule="headcount",
    drop_flag_on_tag=True,
    dropped_flag_enemy_pickup=True,
    dropped_flag_return_on_touch=True,
)


def make_env(red, blue, **rules) -> CTFEnvironment:
    """3 agents per team on a 9x5 open grid. ``red`` / ``blue`` are spawn cells for slots 1..3."""
    kwargs = dict(
        width=9, height=5, agents_per_team=3,
        red_base=(0, 2), blue_base=(8, 2),
        red_spawns=list(red), blue_spawns=list(blue),
        tagging_enabled=True, max_episode_steps=100, visibility_radius=3,
    )
    kwargs.update(V2)
    kwargs.update(rules)
    env = CTFEnvironment(CTFConfig(**kwargs))
    env.reset()
    return env


def step(env: CTFEnvironment, **moves: Action):
    """One step: every active agent STAYs unless named, e.g. ``step(env, R1=Action.RIGHT)``."""
    actions = {aid: Action.STAY for aid, a in env.state.agents.items() if a.active}
    actions.update({k: v for k, v in moves.items() if k in actions})
    _, _, _, info = env.step(actions)
    return info["events"]


def kinds(events) -> list[str]:
    return [e["type"] for e in events]


RED_FAR = [(0, 0), (0, 1), (0, 4)]
BLUE_FAR = [(8, 0), (8, 1), (8, 4)]


# ------------------------------------------------------------------ slot-restricted carrying


@pytest.mark.parametrize("slot, can", [(1, True), (2, False), (3, False)])
def test_only_carrier_slots_pick_up_the_flag(slot: int, can: bool) -> None:
    red = [(0, 0), (0, 1), (0, 4)]
    red[slot - 1] = (7, 2)
    env = make_env(red, BLUE_FAR)  # nobody stands on the blue flag at (8, 2)
    events = step(env, **{f"R{slot}": Action.RIGHT})
    assert env.state.agents[f"R{slot}"].carrying_flag is can
    assert ("flag_pickup" in kinds(events)) is can
    assert (env.state.flags[Team.BLUE].status is FlagStatus.CARRIED) is can


def test_non_carrier_standing_on_flag_does_nothing_for_many_steps() -> None:
    env = make_env([(8, 2), (0, 1), (0, 4)], BLUE_FAR)  # R1 (slot 1) starts ON the blue flag cell
    # The pickup check runs after every move, so R1 grabs it on the first step even if it stays.
    step(env)
    assert env.state.agents["R1"].carrying_flag is True
    env = make_env([(0, 0), (8, 2), (0, 4)], BLUE_FAR)  # R2 (slot 2) starts on it instead
    for _ in range(3):
        step(env)
    assert env.state.flags[Team.BLUE].status is FlagStatus.AT_BASE


def test_carrier_slots_null_lets_everyone_carry() -> None:
    env = make_env([(0, 0), (7, 2), (0, 4)], BLUE_FAR, carrier_slots=None)
    step(env, R2=Action.RIGHT)
    assert env.state.agents["R2"].carrying_flag is True


# ---------------------------------------------------------------------------- headcount tagging


def test_lone_attacker_is_tagged_by_one_parked_defender() -> None:
    env = make_env([(7, 2), (0, 1), (0, 4)], [(8, 0), (8, 2), (8, 4)])  # B2 parked on the flag
    events = step(env, R1=Action.RIGHT)
    assert kinds(events) == ["flag_pickup", "tag", "flag_drop"]
    tag = events[1]
    assert (tag["agent_id"], tag["by_agent_id"], tag["flag_team"]) == ("R1", "B2", "BLUE")
    assert env.state.agents["R1"].active is False
    assert env.state.agents["R1"].carrying_flag is False


def test_attacker_with_scout_beats_one_defender() -> None:
    env = make_env([(7, 2), (0, 1), (7, 2)], [(8, 0), (8, 2), (8, 4)])  # R1 + R3 arrive together
    events = step(env, R1=Action.RIGHT, R3=Action.RIGHT)
    assert kinds(events) == ["flag_pickup"]
    assert env.state.agents["R1"].active and env.state.agents["R1"].carrying_flag


def test_two_defenders_beat_attacker_with_scout() -> None:
    env = make_env([(7, 2), (0, 1), (7, 2)], [(8, 2), (8, 2), (8, 4)])  # B1 and B2 on the flag
    events = step(env, R1=Action.RIGHT, R3=Action.RIGHT)
    assert kinds(events) == ["flag_pickup", "tag", "flag_drop"]
    assert events[1]["by_agent_id"] == "B1"  # lowest id among the taggers, so it is deterministic
    assert env.state.agents["R3"].active  # only the carrier is eliminated


def test_scout_or_defender_of_the_enemy_counts_as_a_tagger() -> None:
    env = make_env([(7, 2), (0, 1), (0, 4)], [(8, 0), (8, 1), (8, 2)])  # B3 (scout) on the flag
    events = step(env, R1=Action.RIGHT)
    assert "tag" in kinds(events)


def test_dead_escort_does_not_protect_a_carrier() -> None:
    env = _drop_scenario(dropped_flag_return_on_touch=False)  # R1's body lies on (5, 2) with B2
    assert not env.state.agents["R1"].active
    events = step(env, R2=Action.LEFT)  # R2 re-picks the flag on (5, 2); B2 is still there
    assert kinds(events) == ["flag_pickup", "tag", "flag_drop"]  # the dead R1 is no escort


def test_headcount_with_escort_allows_capture_past_a_defender() -> None:
    # Red carrier + scout walk home together; a blue defender shadows them, sharing their cell.
    env = make_env([(7, 2), (0, 1), (7, 2)], [(8, 0), (8, 2), (8, 4)])
    step(env, R1=Action.RIGHT, R3=Action.RIGHT)  # both on (8, 2); R1 picks up, B2 cannot tag
    for x in range(7, 0, -1):
        step(env, R1=Action.LEFT, R3=Action.LEFT, B2=Action.LEFT)
        assert env.state.agents["R1"].active, f"carrier tagged at x={x}"
    events = step(env, R1=Action.LEFT, R3=Action.LEFT, B2=Action.LEFT)
    assert "flag_capture" in kinds(events)
    assert env.state.winner is Team.RED


def _corridor(**rules) -> CTFEnvironment:
    """5x1 corridor: red flag (0,0), blue flag (4,0). R1 starts at x=3, B1 at x=1."""
    cfg = dict(
        width=5, height=1, agents_per_team=1, red_base=(0, 0), blue_base=(4, 0),
        red_spawns=[(3, 0)], blue_spawns=[(1, 0)], max_episode_steps=100,
    )
    cfg.update({**V2, "carrier_slots": [1]})
    cfg.update(rules)
    env = CTFEnvironment(CTFConfig(**cfg))
    env.reset()
    return env


def test_headcount_judges_two_carriers_on_one_snapshot() -> None:
    env = _corridor()
    step(env, R1=Action.RIGHT, B1=Action.LEFT)  # R1 takes blue's flag at x=4, B1 takes red's at x=0
    assert env.state.agents["R1"].carrying_flag and env.state.agents["B1"].carrying_flag
    step(env, R1=Action.LEFT, B1=Action.RIGHT)  # R1 -> 3, B1 -> 1
    events = step(env, R1=Action.LEFT, B1=Action.RIGHT)  # both -> 2: same cell, no helpers
    assert kinds(events) == ["tag", "flag_drop", "tag", "flag_drop"]  # each is outnumbered 1 to 0
    assert not env.state.agents["R1"].active and not env.state.agents["B1"].active
    assert env.state.flags[Team.RED].status is FlagStatus.DROPPED
    assert env.state.flags[Team.BLUE].status is FlagStatus.DROPPED


def test_any_enemy_rule_is_unchanged_and_order_dependent() -> None:
    env = _corridor(tagging_rule="any_enemy", drop_flag_on_tag=False, carrier_slots=None)
    step(env, R1=Action.RIGHT, B1=Action.LEFT)
    step(env, R1=Action.LEFT, B1=Action.RIGHT)
    events = step(env, R1=Action.LEFT, B1=Action.RIGHT)
    assert kinds(events) == ["tag"]  # original behaviour: R1 is judged first, then B1 is safe
    assert env.state.agents["B1"].active and not env.state.agents["R1"].active


# ---------------------------------------------------------------------------------- dropped flags


def test_flag_drops_where_the_carrier_was_tagged_not_at_base() -> None:
    # R1 picks up at (8, 2) with no defender around, then B2 (waiting at (5, 2)) catches it there.
    env = make_env([(7, 2), (0, 1), (0, 4)], [(8, 0), (5, 2), (8, 4)])
    step(env, R1=Action.RIGHT)
    assert env.state.agents["R1"].carrying_flag
    step(env, R1=Action.LEFT)
    step(env, R1=Action.LEFT)
    events = step(env, R1=Action.LEFT)  # R1 reaches (5, 2)
    assert kinds(events) == ["tag", "flag_drop"]
    flag = env.state.flags[Team.BLUE]
    assert (flag.status, flag.x, flag.y, flag.carrier_id) == (FlagStatus.DROPPED, 5, 2, None)
    assert (events[1]["x"], events[1]["y"]) == (5, 2)


def test_original_pickup_event_has_no_extra_key() -> None:
    env = make_env([(7, 2), (0, 1), (0, 4)], BLUE_FAR)
    events = step(env, R1=Action.RIGHT)
    assert events == [{"type": "flag_pickup", "agent_id": "R1", "flag_team": "BLUE"}]


def _drop_scenario(**rules) -> CTFEnvironment:
    """R1 is tagged at (5, 2) by B2, who stays there. The blue flag lies DROPPED at (5, 2).

    Slots 1 and 2 may carry, so R2 (at (6, 2)) can try to pick the flag up again.
    """
    env = make_env([(7, 2), (6, 2), (0, 4)], [(8, 0), (5, 2), (8, 4)], carrier_slots=[1, 2], **rules)
    step(env, R1=Action.RIGHT)
    for _ in range(3):
        step(env, R1=Action.LEFT)
    assert env.state.flags[Team.BLUE].status is FlagStatus.DROPPED
    return env


def test_dropped_flag_can_be_picked_up_again_by_an_eligible_enemy() -> None:
    env = _drop_scenario(dropped_flag_return_on_touch=False)
    step(env, B2=Action.UP)  # B2 leaves the flag cell
    events = step(env, R2=Action.LEFT)  # R2 (6, 2) -> (5, 2), onto the dropped flag
    assert kinds(events) == ["flag_pickup"]
    assert events[0]["from_dropped"] is True and events[0]["agent_id"] == "R2"
    assert env.state.agents["R2"].carrying_flag
    flag = env.state.flags[Team.BLUE]
    assert (flag.status, flag.carrier_id) == (FlagStatus.CARRIED, "R2")


def test_enemy_pickup_of_dropped_flag_can_be_disabled() -> None:
    env = _drop_scenario(dropped_flag_enemy_pickup=False)
    events = step(env, B2=Action.UP, R2=Action.LEFT)  # B2 leaves, R2 arrives in the same step
    assert "flag_pickup" not in kinds(events)
    assert not env.state.agents["R2"].carrying_flag
    assert env.state.flags[Team.BLUE].status is FlagStatus.DROPPED  # touching it does not return it either


def test_re_pickup_after_return_comes_from_the_base_without_the_extra_key() -> None:
    env = _drop_scenario()
    step(env)  # B2 is on the flag: it goes home to (8, 2)
    step(env, R2=Action.RIGHT)  # R2 (6 -> 7)
    events = step(env, R2=Action.RIGHT)  # R2 reaches the flag at its base
    assert events == [{"type": "flag_pickup", "agent_id": "R2", "flag_team": "BLUE"}]


# ----------------------------------------------------------------------------- return on touch


def test_own_team_touching_a_dropped_flag_sends_it_home() -> None:
    env = _drop_scenario()
    events = step(env)  # B2 is still standing on the dropped flag
    assert events == [{"type": "flag_return", "agent_id": "B2", "flag_team": "BLUE"}]
    flag = env.state.flags[Team.BLUE]
    assert (flag.status, flag.x, flag.y) == (FlagStatus.AT_BASE, 8, 2)


def test_return_is_checked_before_pickup() -> None:
    env = _drop_scenario()
    events = step(env, R2=Action.LEFT)  # R2 arrives on the cell where B2 stands
    assert kinds(events) == ["flag_return"]  # sent home first, so R2 cannot pick it up
    assert not env.state.agents["R2"].carrying_flag


def test_return_on_touch_can_be_disabled() -> None:
    env = _drop_scenario(dropped_flag_return_on_touch=False)
    for _ in range(3):
        assert "flag_return" not in kinds(step(env))
    assert env.state.flags[Team.BLUE].status is FlagStatus.DROPPED


# --------------------------------------------------------------------- old rules stay unchanged


def test_new_keys_default_to_the_original_rules() -> None:
    cfg = CTFConfig()
    assert cfg.carrier_slots is None
    assert cfg.tagging_rule is TaggingRule.ANY_ENEMY
    assert cfg.drop_flag_on_tag is False
    d = load_config(ROOT / "configs" / "default.yaml")
    assert (d.carrier_slots, d.tagging_rule, d.drop_flag_on_tag) == (None, TaggingRule.ANY_ENEMY, False)
    assert d.dropped_flag_enemy_pickup is cfg.dropped_flag_enemy_pickup
    assert d.dropped_flag_return_on_touch is cfg.dropped_flag_return_on_touch


def test_old_rules_tag_returns_flag_to_base_and_emits_no_new_events() -> None:
    env = CTFEnvironment(CTFConfig(
        width=7, height=5, agents_per_team=1, red_spawns=[(5, 2)], blue_spawns=[(6, 2)],
        red_base=(0, 2), blue_base=(6, 2),
    ))
    env.reset()
    _, _, _, info = env.step({"R1": Action.RIGHT, "B1": Action.STAY})
    assert [e["type"] for e in info["events"]] == ["flag_pickup", "tag"]
    flag = env.state.flags[Team.BLUE]
    assert (flag.status, flag.x, flag.y) == (FlagStatus.AT_BASE, 6, 2)


def test_old_rules_any_agent_may_carry() -> None:
    env = CTFEnvironment(CTFConfig(
        width=7, height=5, agents_per_team=3, red_spawns=[(0, 0), (5, 2), (0, 4)],
        blue_spawns=[(6, 0), (6, 1), (6, 4)], red_base=(0, 2), blue_base=(6, 2),
    ))
    env.reset()
    env.step({a: (Action.RIGHT if a == "R2" else Action.STAY) for a in env.config.agent_ids()})
    assert env.state.agents["R2"].carrying_flag is True


# ----------------------------------------------------------------------- configuration checks


def test_v2_yaml_loads_with_the_new_rules_and_the_default_map() -> None:
    v2 = load_config(ROOT / "configs" / "ctf_v2.yaml")
    old = load_config(ROOT / "configs" / "default.yaml")
    assert v2.carrier_slots == [1]
    assert v2.tagging_rule is TaggingRule.HEADCOUNT
    assert v2.drop_flag_on_tag is True
    # Same map and spawns as the original config.
    for name in ("width", "height", "agents_per_team", "obstacles", "red_base", "blue_base",
                 "red_spawns", "blue_spawns", "visibility_radius", "max_episode_steps", "spawn_mode"):
        assert getattr(v2, name) == getattr(old, name), name


def test_round_trip_keeps_the_new_keys() -> None:
    cfg = load_config(ROOT / "configs" / "ctf_v2.yaml")
    assert config_from_dict(config_to_dict(cfg)) == cfg
    assert config_to_dict(cfg)["carrier_slots"] == [1]
    assert config_to_dict(CTFConfig())["carrier_slots"] is None


def test_old_checkpoint_style_dicts_without_the_new_keys_still_load() -> None:
    d = config_to_dict(CTFConfig())
    for key in ("carrier_slots", "tagging_rule", "drop_flag_on_tag",
                "dropped_flag_enemy_pickup", "dropped_flag_return_on_touch"):
        d.pop(key)
    assert config_from_dict(d) == CTFConfig()


@pytest.mark.parametrize(
    "bad",
    [
        {"carrier_slots": []},
        {"carrier_slots": [1, 1]},
        {"carrier_slots": [0]},
        {"carrier_slots": [4]},
        {"carrier_slots": [True]},
        {"carrier_slots": "1"},
        {"carrier_slots": [1.5]},
        {"tagging_rule": "majority"},
        {"drop_flag_on_tag": "yes"},
        {"drop_flag_on_tag": True, "dropped_flag_enemy_pickup": False, "dropped_flag_return_on_touch": False},
    ],
)
def test_bad_rule_settings_are_rejected(bad: dict) -> None:
    with pytest.raises(ValueError):
        config_from_dict(bad)


def test_both_dropped_options_off_is_fine_when_flags_are_never_dropped() -> None:
    config_from_dict({"dropped_flag_enemy_pickup": False, "dropped_flag_return_on_touch": False})


# -------------------------------------------------------------------------------- observations


def test_enemy_dropped_flag_is_hidden_beyond_the_radius_and_visible_inside_it() -> None:
    env = _drop_scenario(dropped_flag_return_on_touch=False, visibility_radius=1)
    # Blue flag DROPPED at (5, 2). RED's active agents: R2 at (6, 2), R3 at (0, 4).
    near = env.observe(ObservationMode.PARTIAL, team=Team.RED)
    assert near.flags[Team.BLUE].status is FlagStatus.DROPPED  # R2 is one cell away
    step(env, R2=Action.UP)
    step(env, R2=Action.UP)  # R2 now at (6, 0), three cells from the flag
    far = env.observe(ObservationMode.PARTIAL, team=Team.RED)
    assert Team.BLUE not in far.flags  # unseen: a team does not know where a dropped flag lies
    assert env.observe(ObservationMode.GLOBAL).flags[Team.BLUE].status is FlagStatus.DROPPED


def test_own_dropped_flag_is_always_visible_to_its_team() -> None:
    env = _drop_scenario(dropped_flag_return_on_touch=False)
    view = env.observe(ObservationMode.PARTIAL, team=Team.BLUE)
    assert view.flags[Team.BLUE].status is FlagStatus.DROPPED
    assert (view.flags[Team.BLUE].x, view.flags[Team.BLUE].y) == (5, 2)
    assert view.to_dict()["flags"]["BLUE"]["status"] == "DROPPED"


def test_agent_slot_property() -> None:
    env = make_env(RED_FAR, BLUE_FAR)
    assert [env.state.agents[a].slot for a in ("R1", "R2", "R3", "B1", "B2", "B3")] == [1, 2, 3, 1, 2, 3]
