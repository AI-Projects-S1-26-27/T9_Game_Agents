"""The command vocabulary: what the LLM may order, who may use what, and named zones."""
from __future__ import annotations

from dataclasses import dataclass, field

from ctf.agent import Team

from .pathing import Cell, nearest_free

# Commands the LLM may issue
COMMANDS = ("go_to", "capture_flag", "defend_base", "intercept", "scout")
NEEDS_TARGET = {"go_to", "intercept", "scout"}
ROLE_ALLOWED = {
    "attacker": {"go_to", "capture_flag", "intercept", "scout"},
    "defender": {"go_to", "defend_base", "intercept"},
    "scout": {"go_to", "scout", "intercept"},
}
ROLE_CYCLE = ("attacker", "defender", "scout")
# Used when an agent has no order. "hold" is internal (the LLM cannot order it).
ROLE_DEFAULT = {"attacker": "capture_flag", "defender": "defend_base", "scout": "hold"}
DEFAULT_ORDER_LIFETIME = 25  # turns before an order expires and the agent reverts to its default


def agent_index(agent_id: str) -> int:
    digits = "".join(ch for ch in agent_id if ch.isdigit())
    return int(digits) if digits else 0


@dataclass(frozen=True)
class Rules:
    """The flag rules that matter for ordering players (read from the environment config)."""

    carrier_slots: tuple[int, ...] | None = None    # 1-based slots that may carry the flag; None = everyone
    tagging_rule: str = "any_enemy"                  # "any_enemy" or "headcount"
    drop_flag_on_tag: bool = False
    dropped_flag_enemy_pickup: bool = True
    dropped_flag_return_on_touch: bool = True


def rules_from_config(config) -> Rules:
    """Rules of an environment config (the original rules when no config is given)."""
    if config is None:
        return Rules()
    slots = getattr(config, "carrier_slots", None)
    rule = getattr(config, "tagging_rule", "any_enemy")
    return Rules(
        carrier_slots=tuple(slots) if slots else None,
        tagging_rule=str(getattr(rule, "value", rule)),
        drop_flag_on_tag=bool(getattr(config, "drop_flag_on_tag", False)),
        dropped_flag_enemy_pickup=bool(getattr(config, "dropped_flag_enemy_pickup", True)),
        dropped_flag_return_on_touch=bool(getattr(config, "dropped_flag_return_on_touch", True)),
    )


def assign_roles(agent_ids: list[str], rules: Rules | None = None) -> dict[str, str]:
    """Original rules: R1 attacker, R2 defender, R3 scout (repeating for more agents).
    With carrier slots: the carrier slots are attackers, then one defender, the rest scouts
    (4 agents, carriers 1 and 2: attacker, attacker, defender, scout)."""
    ordered = sorted(agent_ids, key=agent_index)
    if rules is None or rules.carrier_slots is None:
        return {aid: ROLE_CYCLE[i % len(ROLE_CYCLE)] for i, aid in enumerate(ordered)}
    roles, defenders = {}, 0
    for i, aid in enumerate(ordered, start=1):
        if i in rules.carrier_slots:
            roles[aid] = "attacker"
        elif defenders == 0:
            roles[aid], defenders = "defender", 1
        else:
            roles[aid] = "scout"
    return roles


@dataclass
class AgentView:
    id: str
    role: str
    pos: Cell
    carrying: bool
    active: bool


@dataclass
class FlagView:
    pos: Cell
    carried: bool
    carrier_id: str | None
    dropped: bool = False       # lying on the cell where its carrier was tagged (not at its base)


@dataclass
class OrderStatus:
    agent_id: str
    command: str
    target: object
    status: str  # active | done | failed | blocked | expired


@dataclass
class View:
    """Everything the commander is allowed to know (built from the observation only)."""

    team: Team
    timestep: int
    max_steps: int | None
    width: int
    height: int
    obstacles: set[Cell]
    observed: set[Cell] | None  # None = whole map visible (global observability)
    own_base: Cell
    enemy_base: Cell
    own_flag: FlagView
    enemy_flag: FlagView | None
    agents: list[AgentView]
    enemies: list[tuple[str, Cell, int]]  # (id, last seen position, turn last seen)
    order_status: list[OrderStatus] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    rules: Rules = field(default_factory=Rules)

    def agent(self, agent_id: str) -> AgentView | None:
        return next((a for a in self.agents if a.id == agent_id), None)


def zone_table(view: View) -> dict[str, Cell]:
    w, h, ob = view.width, view.height, view.obstacles
    raw = {
        "north": (w // 2, 0), "south": (w // 2, h - 1),
        "west": (0, h // 2), "east": (w - 1, h // 2),
        "northwest": (0, 0), "northeast": (w - 1, 0),
        "southwest": (0, h - 1), "southeast": (w - 1, h - 1),
        "center": (w // 2, h // 2),
        "own_base": view.own_base, "enemy_base": view.enemy_base,
    }
    return {k: nearest_free(w, h, ob, v) for k, v in raw.items()}


DYNAMIC_TARGETS = ("enemy_carrier", "own_carrier")  # resolved fresh every turn


def resolve_target(view: View, target) -> Cell | None:
    """Named zone / [x, y] -> cell. 'enemy_carrier' -> where our stolen flag is (if stolen);
    'own_carrier' -> where our flag carrier is (if one of us carries the enemy flag)."""
    if isinstance(target, str):
        name = target.strip().lower()
        if name == "enemy_carrier":
            return view.own_flag.pos if view.own_flag.carried else None
        if name == "own_carrier":       # our player carrying the enemy flag (to escort it)
            return view.enemy_flag.pos if view.enemy_flag is not None and view.enemy_flag.carried else None
        if name == "enemy_flag":
            return view.enemy_flag.pos if view.enemy_flag else view.enemy_base
        if name == "own_flag":
            return view.own_flag.pos
        return zone_table(view).get(name)
    if (isinstance(target, (list, tuple)) and len(target) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in target)):
        return (target[0], target[1])
    return None
