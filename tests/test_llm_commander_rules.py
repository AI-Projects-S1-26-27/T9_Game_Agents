"""The commander follows the flag rules of the environment config (original rules and the RL agent's v2 rules)."""
import dataclasses
from pathlib import Path

import pytest

from ctf.actions import Action
from ctf.agent import Team
from ctf.config import load_config
from ctf.environment import CTFEnvironment
from ctf.game_state import Flag
from strategies.llm_commander import LLMCommanderStrategy, MockLLM
from strategies.llm_commander.commands import Rules, assign_roles, resolve_target, rules_from_config
from strategies.llm_commander.controller import Controller
from strategies.llm_commander.prompt import build_system_prompt
from strategies.llm_commander.validator import validate
from strategies.llm_commander.view import build_view, view_to_text

ROOT = Path(__file__).resolve().parents[1]


def env_for(name):
    cfg = load_config(ROOT / "configs" / name)
    env = CTFEnvironment(cfg)
    env.reset(seed=1)
    return cfg, env


def view_for(name="ctf_v2.yaml", team=Team.RED):
    cfg, env = env_for(name)
    rules = rules_from_config(cfg)
    obs = env.observe()
    ids = [a for a in obs.agents if obs.agents[a].team is team]
    return cfg, env, build_view(obs, team, assign_roles(ids, rules), {}, None, 200, None, rules)


def test_rules_are_read_from_the_config():
    assert rules_from_config(None) == Rules()
    v1, v2 = rules_from_config(load_config(ROOT / "configs/default.yaml")), rules_from_config(load_config(ROOT / "configs/ctf_v2.yaml"))
    assert v1.carrier_slots is None and v1.tagging_rule == "any_enemy" and not v1.drop_flag_on_tag
    assert v2.carrier_slots == (1, 2) and v2.tagging_rule == "headcount" and v2.drop_flag_on_tag


def test_roles_follow_the_carrier_slots():
    ids4 = ["R4", "R1", "R3", "R2"]
    assert assign_roles(ids4, Rules(carrier_slots=(1, 2))) == {"R1": "attacker", "R2": "attacker", "R3": "defender", "R4": "scout"}
    assert assign_roles(["R1", "R2", "R3"]) == {"R1": "attacker", "R2": "defender", "R3": "scout"}      # original behaviour


def test_prompt_states_the_rules_of_the_game_being_played():
    _, _, v2 = view_for("ctf_v2.yaml")
    text = build_system_prompt(v2)
    assert "OUTNUMBER" in text and "R1, R2 can pick up" in text and "stays lying" in text and "own_carrier" in text
    _, _, v1 = view_for("default.yaml")
    text1 = build_system_prompt(v1)
    assert "pickup is checked first" in text1 and "OUTNUMBER" not in text1


def test_dropped_flags_are_described_and_targetable():
    cfg, env, _ = view_for("ctf_v2.yaml")
    st = env._state
    st.flags[Team.RED] = Flag(Team.RED, st.flags[Team.RED].status.__class__.DROPPED, 3, 3)
    st.flags[Team.BLUE] = Flag(Team.BLUE, st.flags[Team.BLUE].status.__class__.DROPPED, 7, 3)
    rules = rules_from_config(cfg)
    obs = env.observe()
    view = build_view(obs, Team.RED, assign_roles(["R1", "R2", "R3", "R4"], rules), {}, None, 200, None, rules)
    text = view_to_text(view)
    assert "lying at (3, 3), away from our base" in text and "Enemy flag: lying at (7, 3)" in text
    assert resolve_target(view, "own_flag") == (3, 3) and resolve_target(view, "enemy_flag") == (7, 3)


def test_controller_picks_up_and_returns_dropped_flags():
    cfg, env, _ = view_for("ctf_v2.yaml")
    st = env._state
    st.flags[Team.RED] = Flag(Team.RED, st.flags[Team.RED].status.__class__.DROPPED, 4, 3)    # our flag lies in front of the wall
    st.flags[Team.BLUE] = Flag(Team.BLUE, st.flags[Team.BLUE].status.__class__.DROPPED, 6, 3)
    for aid, pos in {"R1": (3, 3), "R3": (3, 2)}.items():
        st.agents[aid] = dataclasses.replace(st.agents[aid], x=pos[0], y=pos[1])
    rules = rules_from_config(cfg)
    view = build_view(env.observe(), Team.RED, assign_roles(["R1", "R2", "R3", "R4"], rules), {}, None, 200, None, rules)
    ctrl = Controller()
    moves = ctrl.actions(view)                       # defaults: R1 attacker captures, R3 defender defends
    assert moves["R1"] in (Action.RIGHT, Action.UP, Action.DOWN)        # heads for the enemy flag lying at (6, 3), around the wall
    assert moves["R3"] in (Action.DOWN, Action.RIGHT)                   # defender goes to touch our dropped flag at (4, 3)


def test_escort_target_follows_our_carrier():
    cfg, env, _ = view_for("ctf_v2.yaml")
    st = env._state
    st.agents["R1"] = dataclasses.replace(st.agents["R1"], x=6, y=0, carrying_flag=True)
    st.flags[Team.BLUE] = st.flags[Team.BLUE].with_carried("R1", 6, 0)
    rules = rules_from_config(cfg)
    view = build_view(env.observe(), Team.RED, assign_roles(["R1", "R2", "R3", "R4"], rules), {}, None, 200, None, rules)
    assert resolve_target(view, "own_carrier") == (6, 0)
    clean, errors = validate(view, {"orders": [{"agent_id": "R2", "command": "intercept", "target": "own_carrier"}]})
    assert errors == [] and clean[0]["target"] == "own_carrier"
    _, _, no_carrier = view_for("ctf_v2.yaml")
    assert resolve_target(no_carrier, "own_carrier") is None


@pytest.mark.parametrize("config", ["default.yaml", "ctf_v2.yaml"])
def test_commander_plays_both_rule_sets_on_both_sides(config):
    pytest.importorskip("rl.opponents")
    pytest.importorskip("classical_ai.evaluation")
    from strategies.llm_commander.opponents import make_opponent
    from strategies.llm_commander.runner import play_game

    cfg = load_config(ROOT / "configs" / config)
    for team in (Team.RED, Team.BLUE):
        for opp in ("balanced", "camper", "escort", "chaser"):
            mine = LLMCommanderStrategy(MockLLM(), env_config=cfg, max_steps=cfg.max_episode_steps)
            res = play_game(cfg, mine, make_opponent(opp, cfg, 1), team, seed=3)
            assert res.result in ("win", "loss", "draw") and res.turns > 0
