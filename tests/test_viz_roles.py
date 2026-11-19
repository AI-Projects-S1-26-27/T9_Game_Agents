"""Role labels in the viewer: every controller on either side shows a role on each slot."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pytest.importorskip("numpy")
pytest.importorskip("pygame")

from ctf.config import load_config  # noqa: E402
from rl.schema import ConfigError  # noqa: E402
from viz.controllers import ControllerError, ControllerFactory, ControllerSpec, load_controller_specs  # noqa: E402
from viz.session import record_episode  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ENV = load_config(ROOT / "configs" / "default.yaml")
LAYOUT = ["attacker", "defender", "scout"]


def _spec(kind: str, roles=None) -> ControllerSpec:
    return ControllerSpec(kind.title(), kind, "", True, {}, roles if roles is not None else [])


def _roles_by_team(rec) -> dict[str, list[str]]:
    first = rec.frames[0]
    return {t: [a.role for a in sorted(first.agents, key=lambda a: a.id) if a.team == t] for t in ("RED", "BLUE")}


@pytest.mark.parametrize("red_kind, blue_kind", [("random", "stay"), ("stay", "random"), ("random", "random")])
def test_random_and_stay_teams_show_their_roles_on_both_sides(red_kind, blue_kind) -> None:
    f = ControllerFactory()
    red = f.build(_spec(red_kind, LAYOUT), ENV, 1)
    blue = f.build(_spec(blue_kind, LAYOUT), ENV, 2)
    rec = record_episode(ENV, red, blue, seed=0)
    assert _roles_by_team(rec) == {"RED": LAYOUT, "BLUE": LAYOUT}


def test_a_controller_without_roles_still_shows_unknown() -> None:
    c = ControllerFactory().build(_spec("random"), ENV, 1)
    assert c.roles == ["unknown"] * ENV.agents_per_team


def test_roles_must_match_the_number_of_agents() -> None:
    with pytest.raises(ControllerError, match="agents per team"):
        ControllerFactory().build(_spec("random", ["attacker", "defender"]), ENV, 1)


def test_unknown_role_names_are_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown role"):
        _spec("stay", ["attacker", "goalkeeper", "scout"])


def test_every_shipped_non_rl_controller_has_roles_to_show() -> None:
    env = load_config(ROOT / "configs" / "ctf_v2.yaml")  # the map the shipped controllers are written for
    specs = [s for s in load_controller_specs(ROOT / "configs" / "viz" / "controllers.yaml") if s.kind != "rl"]
    assert specs
    for spec in specs:
        roles = ControllerFactory().build(spec, env, 1).roles
        assert "unknown" not in roles, spec.label
        assert roles == ["attacker", "attacker", "defender", "scout"], spec.label
