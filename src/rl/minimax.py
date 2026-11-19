"""Minimax opponent with alpha-beta pruning.

A classical game-tree searcher that plugs into the same ``BaseStrategy`` interface as every
other controller. It is meant as a fixed, non-learning opponent for evaluation (and optionally
for training).

How it works, in short
----------------------
* **Fair information.** It only reads the team's fog-of-war observation. Visible enemies are
  simulated. The enemy carrier of our flag is always located (our flag moves with it), so it
  is simulated even outside our view. Enemies we cannot see are ignored.
* **Fast forward model.** ``CTFEnvironment.step`` copies dataclasses and is too slow to call
  thousands of times per decision. ``_Sim`` re-implements the same rules on tuples of ints
  (moves, carried flags, dropped-flag returns, pickups, tagging, scoring) in the same order.
  ``tests/test_rl_minimax.py`` checks it against the real environment step by step.
* **Search.** One *level* is one game step: our team picks a joint move (MAX), then the
  opponent team picks its worst-case reply (MIN), then both are applied together. ``depth``
  is the number of levels (``depth: 1`` = one step lookahead). Replying after seeing our move
  is pessimistic (real moves are simultaneous), the usual approximation for minimax.
* **Branching control.** Five moves per agent and four agents per team would be 625 joint
  moves per side. Each agent therefore only tries its ``*_moves_per_agent`` best moves, ranked
  by path distance to a role-based target. Joint moves are tried best-first, which makes
  alpha-beta cut a lot. ``max_nodes`` bounds the work per decision (hard time limit).
* **Evaluation.** A weighted sum of simple features: progress of our carrier towards home,
  distance of our nearest attacker to the enemy flag, escorts staying with the carrier,
  the same terms for the enemy (negative), the defenders heading for the threat, number of
  agents alive. Which slots attack and which defend is taken from the environment's
  ``carrier_slots`` (slots that may carry are attackers; the rest defend/support).
  All weights are in the YAML (``MinimaxParams``).
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from random import Random

from ctf.actions import Action
from ctf.agent import Team
from ctf.config import CTFConfig, TaggingRule
from ctf.game_state import FlagStatus
from ctf.observations import PartialTeamObservation, TeamObservation, manhattan_distance, restrict_to_team_view
from rl.navigation import GridNav
from rl.schema import ConfigError
from strategies import BaseStrategy

ACTIONS: tuple[Action, ...] = tuple(Action)  # UP, DOWN, LEFT, RIGHT, STAY
STAY = ACTIONS.index(Action.STAY)

AT_BASE, CARRIED, DROPPED = 0, 1, 2
_STATUS = {FlagStatus.AT_BASE: AT_BASE, FlagStatus.CARRIED: CARRIED, FlagStatus.DROPPED: DROPPED}
_INF = float("inf")


# ------------------------------------------------------------------------------ configuration


@dataclass(frozen=True)
class MinimaxWeights:
    win: float  # terminal value of a capture (a loss is the negative)
    carry_progress: float  # per cell our carrier has already gained towards our base
    pickup: float  # per carrier of ours (we hold the enemy flag)
    approach: float  # per cell between our nearest free attacker and the enemy flag (subtracted)
    approach_others: float  # same for the other free attackers
    escort: float  # per teammate near our carrier: weight / (1 + path distance to the carrier)
    enemy_carry_progress: float  # per cell an enemy carrier has gained towards its base (subtracted)
    enemy_pickup: float  # per enemy carrier of our flag (subtracted)
    defend: float  # per cell between each of our defenders and the threat (subtracted)
    alive: float  # per active agent of ours
    pressure: float  # per cell an enemy attacker is inside danger_radius of our flag (subtracted)


@dataclass(frozen=True)
class MinimaxParams:
    depth: int  # game steps looked ahead (each: our joint move, then the opponent's worst reply), >= 1
    own_moves_per_agent: int  # best moves tried per own agent, 1..5
    enemy_moves_per_agent: int  # best moves tried per visible enemy agent, 1..5
    danger_radius: int  # an enemy attacker closer than this to our flag counts as a threat, >= 0
    max_nodes: int  # hard cap on simulated steps per decision; the search stops expanding after it
    alpha_beta: bool  # false = plain minimax (same result, more nodes; for the report's pruning comparison)
    weights: MinimaxWeights


def validate_params(p: MinimaxParams) -> None:
    if p.depth < 1:
        raise ConfigError("minimax depth must be >= 1")
    for name in ("own_moves_per_agent", "enemy_moves_per_agent"):
        if not 1 <= getattr(p, name) <= len(ACTIONS):
            raise ConfigError(f"minimax {name} must be between 1 and {len(ACTIONS)}")
    if p.danger_radius < 0:
        raise ConfigError("minimax danger_radius must be >= 0")
    if p.max_nodes < 1:
        raise ConfigError("minimax max_nodes must be >= 1")
    if p.weights.win <= 0:
        raise ConfigError("minimax weights.win must be > 0 (it must dominate every heuristic term)")


@dataclass(frozen=True)
class SearchStats:
    """What the last decision cost (for the evaluation report)."""

    nodes: int  # simulated steps
    value: float  # minimax value of the chosen move
    seconds: float
    budget_hit: bool


# ----------------------------------------------------------------------------- forward model


class _State:
    """Immutable-by-convention snapshot; ``_Sim.step`` returns a new one."""

    __slots__ = ("pos", "alive", "carry", "fstat", "fcell", "fcar", "winner")

    def __init__(self, pos, alive, carry, fstat, fcell, fcar, winner=0) -> None:
        self.pos = pos  # cell index of every simulated agent
        self.alive = alive
        self.carry = carry
        self.fstat = fstat  # per flag owner side (0 = us, 1 = them): AT_BASE / CARRIED / DROPPED
        self.fcell = fcell
        self.fcar = fcar  # carrier's agent index, or -1
        self.winner = winner  # 0 running, 1 we captured, 2 they captured


class _Frame:
    """Per-decision constants: which simulated agent belongs to which side / slot."""

    __slots__ = ("n", "side", "slot", "can_carry", "attacker", "base", "us", "them")

    def __init__(
        self,
        sides: list[int],
        slots: list[int],
        can_carry: list[bool],
        attacker: list[bool],
        base: tuple[int, int],
    ) -> None:
        self.n = len(sides)
        self.side = sides  # 0 = our team, 1 = the opponent
        self.slot = slots
        self.can_carry = can_carry  # the environment lets this agent pick up the flag (rule)
        self.attacker = attacker  # the search treats this agent as an attacker (role), the rest defend
        self.base = base  # base cell index of side 0 and side 1
        self.us = [i for i, s in enumerate(sides) if s == 0]
        self.them = [i for i, s in enumerate(sides) if s == 1]


class _Sim:
    """The environment's rules on ints. Mirrors ``CTFEnvironment.step`` (same phase order)."""

    def __init__(self, cfg: CTFConfig) -> None:
        self.cfg = cfg
        w, h = cfg.width, cfg.height
        self.width, self.height = w, h
        obstacles = set(cfg.obstacles)
        free = [[(x, y) not in obstacles for x in range(w)] for y in range(h)]
        nav = GridNav(free)
        self.nav = nav
        self.max_dist = nav.max_distance
        n_cells = w * h
        self.next: list[list[int]] = []  # next[cell][action] -> cell after the move (walls: stay put)
        for c in range(n_cells):
            x, y = c % w, c // w
            row = []
            for action in ACTIONS:
                dx, dy = action.delta()
                nx, ny = x + dx, y + dy
                row.append(ny * w + nx if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in obstacles else c)
            self.next.append(row)
        self.dist: list[list[int]] = [[nav.unreachable] * n_cells for _ in range(n_cells)]
        for y in range(h):
            for x in range(w):
                if free[y][x]:
                    self.dist[y * w + x] = [int(v) for v in nav.distance_map((x, y)).ravel()]

    def cell(self, x: int, y: int) -> int:
        return y * self.width + x

    def xy(self, cell: int) -> tuple[int, int]:
        return cell % self.width, cell // self.width

    def step(self, fr: _Frame, st: _State, acts: list[int]) -> _State:
        cfg = self.cfg
        n, side, alive = fr.n, fr.side, list(st.alive)
        pos, carry = list(st.pos), list(st.carry)
        nxt = self.next
        for i in range(n):
            if alive[i] and acts[i] != STAY:
                pos[i] = nxt[pos[i]][acts[i]]
        fstat, fcell, fcar = list(st.fstat), list(st.fcell), list(st.fcar)

        for f in (0, 1):  # carried flags travel with their carrier
            c = fcar[f]
            if fstat[f] == CARRIED and c >= 0 and alive[c]:
                fcell[f] = pos[c]
        if cfg.dropped_flag_return_on_touch:  # returns run before pickups
            for f in (0, 1):
                if fstat[f] != DROPPED:
                    continue
                for i in range(n):
                    if alive[i] and side[i] == f and pos[i] == fcell[f]:
                        fstat[f], fcell[f], fcar[f] = AT_BASE, fr.base[f], -1
                        break
        for i in range(n):  # pickups
            if not alive[i] or carry[i] or not fr.can_carry[i]:
                continue
            f = 1 - side[i]
            if fstat[f] == CARRIED or (fstat[f] == DROPPED and not cfg.dropped_flag_enemy_pickup):
                continue
            if pos[i] == fcell[f]:
                carry[i] = True
                fstat[f], fcell[f], fcar[f] = CARRIED, pos[i], i

        if cfg.tagging_enabled:
            carriers = [i for i in range(n) if alive[i] and carry[i]]
            if cfg.tagging_rule is TaggingRule.HEADCOUNT:
                tagged = []
                for c in carriers:  # every carrier is judged on the same snapshot
                    enemies = [o for o in range(n) if o != c and alive[o] and pos[o] == pos[c] and side[o] != side[c]]
                    escorts = sum(
                        1 for o in range(n) if o != c and alive[o] and pos[o] == pos[c] and side[o] == side[c] and not carry[o]
                    )
                    if len(enemies) > escorts:
                        tagged.append(c)
                for c in tagged:
                    self._tag(fr, c, alive, carry, pos, fstat, fcell, fcar)
            else:  # any_enemy: one enemy on the cell tags at once, carriers are processed in turn
                for c in carriers:
                    if any(o != c and alive[o] and side[o] != side[c] and pos[o] == pos[c] for o in range(n)):
                        self._tag(fr, c, alive, carry, pos, fstat, fcell, fcar)

        winner = 0
        for i in range(n):
            if alive[i] and carry[i] and pos[i] == fr.base[side[i]]:
                winner = side[i] + 1
                break
        return _State(tuple(pos), tuple(alive), tuple(carry), tuple(fstat), tuple(fcell), tuple(fcar), winner)

    def _tag(self, fr: _Frame, c: int, alive, carry, pos, fstat, fcell, fcar) -> None:
        alive[c], carry[c] = False, False
        f = 1 - fr.side[c]
        if self.cfg.drop_flag_on_tag:
            fstat[f], fcell[f], fcar[f] = DROPPED, pos[c], -1
        else:
            fstat[f], fcell[f], fcar[f] = AT_BASE, fr.base[f], -1


# --------------------------------------------------------------------------------- strategy


class MinimaxStrategy(BaseStrategy):
    """Depth-limited alpha-beta search over joint team moves (see the module docstring)."""

    def __init__(self, env_config: CTFConfig, params: MinimaxParams, rng: Random) -> None:
        validate_params(params)
        self._cfg = env_config
        self._p = params
        self._w = params.weights
        self._rng = rng
        self._sim = _Sim(env_config)
        # Slots that may not carry are the defenders. If every slot may carry (original rules) the
        # last slot is made the defender so the team still keeps a guard.
        n = env_config.agents_per_team
        all_carry = all(env_config.can_carry(s) for s in range(1, n + 1))
        self._guard_slot: int | None = n if all_carry and n >= 2 else None
        self._combos: dict[tuple[int, int], list[tuple[int, ...]]] = {}
        self.last_stats: SearchStats | None = None
        self.reset()

    def reset(self) -> None:
        self._last_t = -1
        self._enemy_flag_memory: tuple[int, int] | None = None  # cell where we last saw it lying dropped
        self._our_carrier_cell: tuple[int, int] | None = None  # where our carrier stood a step ago
        self._nodes = 0
        self._budget_hit = False

    # ------------------------------------------------------------------ observation -> state

    def _enemy_flag_belief(
        self, obs: PartialTeamObservation, team: Team
    ) -> tuple[int, str | None, tuple[int, int]]:
        """What we believe about the enemy flag: (status, carrier id, cell).

        Seen: exactly what we see. Not seen: probably still where it was last seen lying dropped,
        or where our carrier vanished (it was tagged there); otherwise safe at its base.
        """
        flag = obs.flags.get(team.opponent)
        base = obs.red_base if team.opponent is Team.RED else obs.blue_base
        if flag is not None:
            self._enemy_flag_memory = (flag.x, flag.y) if flag.status is FlagStatus.DROPPED else None
            return _STATUS[flag.status], flag.carrier_id, (flag.x, flag.y)
        remembered = self._enemy_flag_memory
        if remembered is None and self._our_carrier_cell is not None:
            remembered = self._our_carrier_cell  # our carrier vanished: it was tagged and dropped the flag there
        self._our_carrier_cell = None
        if remembered is not None:
            seen_now = any(
                a.team is team and a.active and manhattan_distance(a.x, a.y, *remembered) <= obs.visibility_radius
                for a in obs.agents.values()
            )
            if not seen_now:  # still out of sight: it is probably still lying there
                self._enemy_flag_memory = remembered
                return DROPPED, None, remembered
            self._enemy_flag_memory = None  # we can see the cell and the flag is not there: it went home
        return AT_BASE, None, base

    def _build(self, obs: PartialTeamObservation, team: Team) -> tuple[_Frame, _State, list[str]]:
        sim = self._sim
        agents = [a for a in obs.agents.values() if a.active]
        flags = obs.flags
        own_flag = flags[team]
        visible_ids = {a.agent_id for a in agents}
        phantom: list[tuple[str, int, int]] = []  # unseen enemy carrier of our flag, located by the flag
        if own_flag.status is FlagStatus.CARRIED and own_flag.carrier_id and own_flag.carrier_id not in visible_ids:
            phantom.append((own_flag.carrier_id, own_flag.x, own_flag.y))

        items = [(a.agent_id, a.team, a.slot, a.x, a.y, a.carrying_flag) for a in agents]
        for aid, x, y in phantom:
            items.append((aid, team.opponent, int(aid[1:]), x, y, True))
        items.sort(key=lambda it: (0 if it[1] is Team.RED else 1, it[2]))  # environment order: RED then BLUE

        ids = [it[0] for it in items]
        sides = [0 if it[1] is team else 1 for it in items]
        slots = [it[2] for it in items]
        can_carry = [self._cfg.can_carry(s) for s in slots]
        attacker = [c and s != self._guard_slot for c, s in zip(can_carry, slots)]
        own_base = obs.red_base if team is Team.RED else obs.blue_base
        enemy_base = obs.blue_base if team is Team.RED else obs.red_base
        fr = _Frame(sides, slots, can_carry, attacker, (sim.cell(*own_base), sim.cell(*enemy_base)))

        status, e_carrier, e_cell = self._enemy_flag_belief(obs, team)
        index = {aid: i for i, aid in enumerate(ids)}
        fstat = (_STATUS[own_flag.status], status)
        fcell = (sim.cell(own_flag.x, own_flag.y), sim.cell(*e_cell))
        fcar = (index.get(own_flag.carrier_id, -1), index.get(e_carrier, -1))
        st = _State(
            tuple(sim.cell(it[3], it[4]) for it in items),
            tuple(True for _ in items),
            tuple(it[5] for it in items),
            fstat,
            fcell,
            fcar,
        )
        carrier_cells = [(it[3], it[4]) for it in items if it[1] is team and it[5]]
        self._our_carrier_cell = carrier_cells[0] if carrier_cells else None
        return fr, st, ids

    # --------------------------------------------------------------------- move generation

    def _our_threat_targets(self, fr: _Frame, st: _State) -> list[int]:
        """Cells a defender should head for, most urgent first source only."""
        carriers = [st.pos[j] for j in fr.them if st.alive[j] and st.carry[j]]
        if carriers:
            return carriers
        if st.fstat[0] == DROPPED:
            return [st.fcell[0]]
        near = [
            st.pos[j]
            for j in fr.them
            if st.alive[j] and fr.attacker[j] and self._sim.dist[st.pos[j]][fr.base[0]] <= self._p.danger_radius
        ]
        return near or [fr.base[0]]

    def _own_target(self, fr: _Frame, st: _State, i: int, threats: list[int]) -> int:
        dist = self._sim.dist
        here = st.pos[i]
        if st.carry[i]:
            return fr.base[0]
        our_carriers = [st.pos[j] for j in fr.us if st.alive[j] and st.carry[j]]
        if fr.attacker[i]:
            if our_carriers:  # escort the carrier
                return min(our_carriers, key=lambda c: dist[here][c])
            return st.fcell[1]
        return min(threats, key=lambda c: dist[here][c])

    def _enemy_target(self, fr: _Frame, st: _State, j: int) -> int:
        dist = self._sim.dist
        here = st.pos[j]
        if st.carry[j]:
            return fr.base[1]
        their_carriers = [st.pos[k] for k in fr.them if st.alive[k] and st.carry[k]]
        our_carriers = [st.pos[k] for k in fr.us if st.alive[k] and st.carry[k]]
        if fr.attacker[j]:
            if their_carriers:
                return min(their_carriers, key=lambda c: dist[here][c])
            return st.fcell[0]
        if our_carriers:  # a defender intercepts our carrier
            return min(our_carriers, key=lambda c: dist[here][c])
        ours = [st.pos[k] for k in fr.us if st.alive[k]]
        return min(ours, key=lambda c: dist[here][c]) if ours else fr.base[1]

    def _ranked_moves(self, st: _State, i: int, target: int, limit: int) -> list[int]:
        """Up to ``limit`` actions, best first: by path distance to the target after the move,
        STAY before a real move on ties; moves leading to the same cell are listed once."""
        dist_row = self._sim.dist[target]
        nxt = self._sim.next[st.pos[i]]
        seen: set[int] = set()
        scored = []
        for a in (STAY, 0, 1, 2, 3):
            cell = nxt[a]
            if cell in seen:
                continue
            seen.add(cell)
            scored.append((dist_row[cell], a != STAY, a))
        scored.sort()
        return [a for _, _, a in scored[:limit]]

    def _combos_for(self, n: int, k: int) -> list[tuple[int, ...]]:
        key = (n, k)
        if key not in self._combos:
            combos = list(itertools.product(range(k), repeat=n))
            combos.sort(key=lambda c: (sum(c), c))  # best-first: low total rank = most promising joint move
            self._combos[key] = combos
        return self._combos[key]

    def _joint(self, ranked: list[tuple[int, list[int]]], n_agents: int, k: int, shuffle: bool = False) -> list[list[int]]:
        """Joint moves as full action lists (one entry per simulated agent, STAY for the rest)."""
        out: list[list[int]] = []
        combos = self._combos_for(len(ranked), k)
        if shuffle:  # root only: break ties between equally promising joint moves at random
            combos = list(combos)
            groups: list[list[tuple[int, ...]]] = []
            for _, grp in itertools.groupby(combos, key=sum):
                g = list(grp)
                self._rng.shuffle(g)
                groups.append(g)
            combos = [c for g in groups for c in g]
        for combo in combos:
            if any(r >= len(ranked[x][1]) for x, r in enumerate(combo)):
                continue
            acts = [STAY] * n_agents
            for x, r in enumerate(combo):
                acts[ranked[x][0]] = ranked[x][1][r]
            out.append(acts)
        return out

    def _our_candidates(self, fr: _Frame, st: _State, shuffle: bool = False) -> list[list[int]]:
        threats = self._our_threat_targets(fr, st)
        k = self._p.own_moves_per_agent
        ranked = [
            (i, self._ranked_moves(st, i, self._own_target(fr, st, i, threats), k))
            for i in fr.us
            if st.alive[i]
        ]
        return self._joint(ranked, fr.n, k, shuffle) if ranked else [[STAY] * fr.n]

    def _enemy_candidates(self, fr: _Frame, st: _State) -> list[list[int]]:
        k = self._p.enemy_moves_per_agent
        ranked = [
            (j, self._ranked_moves(st, j, self._enemy_target(fr, st, j), k))
            for j in fr.them
            if st.alive[j]
        ]
        return self._joint(ranked, fr.n, k) if ranked else [[STAY] * fr.n]

    # ------------------------------------------------------------------------- evaluation

    def _evaluate(self, fr: _Frame, st: _State) -> float:
        w, dist, big_d = self._w, self._sim.dist, self._sim.max_dist
        if st.winner:
            return w.win if st.winner == 1 else -w.win
        pos, alive, carry = st.pos, st.alive, st.carry
        score = 0.0
        ours = [i for i in fr.us if alive[i]]
        score += w.alive * len(ours)
        our_carriers = [i for i in ours if carry[i]]
        for i in our_carriers:
            score += w.carry_progress * (big_d - dist[pos[i]][fr.base[0]]) + w.pickup
        if our_carriers:
            for i in ours:
                if not carry[i]:
                    d = min(dist[pos[i]][pos[c]] for c in our_carriers)
                    score += w.escort / (1 + d)
        elif st.fstat[1] != CARRIED:
            free = sorted(dist[pos[i]][st.fcell[1]] for i in ours if fr.attacker[i])
            if free:
                score -= w.approach * free[0] + w.approach_others * sum(free[1:])

        enemy_carriers = [j for j in fr.them if alive[j] and carry[j]]
        for j in enemy_carriers:
            score -= w.enemy_carry_progress * (big_d - dist[pos[j]][fr.base[1]]) + w.enemy_pickup
        radius = self._p.danger_radius
        for j in fr.them:
            if alive[j] and not carry[j] and fr.attacker[j]:
                gap = radius - dist[pos[j]][fr.base[0]]
                if gap > 0:
                    score -= w.pressure * gap

        defenders = [i for i in ours if not fr.attacker[i] and not carry[i]]
        if defenders:
            targets = self._our_threat_targets(fr, st)
            # every defender heads for the threat: two defenders are needed to tag an escorted carrier
            score -= w.defend * sum(min(dist[pos[i]][t] for t in targets) for i in defenders)
        return score

    # ----------------------------------------------------------------------------- search

    def _value(self, fr: _Frame, st: _State, depth: int, alpha: float, beta: float) -> float:
        """Value for us of ``st`` with ``depth`` game steps left: MAX over our joint moves,
        MIN over the opponent's replies, both applied together."""
        if st.winner or depth == 0 or self._nodes >= self._p.max_nodes:
            if self._nodes >= self._p.max_nodes:
                self._budget_hit = True
            return self._evaluate(fr, st)
        prune = self._p.alpha_beta
        best = -_INF
        for ours in self._our_candidates(fr, st):
            worst = _INF
            for theirs in self._enemy_candidates(fr, st):
                acts = list(ours)
                for i in fr.them:  # enemy agents take their entry from `theirs`, ours keep theirs from `ours`
                    acts[i] = theirs[i]
                nxt = self._sim.step(fr, st, acts)
                self._nodes += 1
                v = self._evaluate(fr, nxt) if depth == 1 else self._value(fr, nxt, depth - 1, alpha, min(beta, worst))
                if v < worst:
                    worst = v
                if prune and worst <= alpha:
                    break
                if self._nodes >= self._p.max_nodes:
                    self._budget_hit = True
                    break
            if worst > best:
                best = worst
            if best > alpha:
                alpha = best
            if prune and alpha >= beta:
                break
            if self._nodes >= self._p.max_nodes:
                self._budget_hit = True
                break
        return best

    def _decide(self, fr: _Frame, st: _State) -> tuple[list[int], float]:
        """Root of the search: best joint move for our team and its minimax value."""
        self._nodes = 0
        self._budget_hit = False
        best_acts: list[int] | None = None
        best = -_INF
        alpha = -_INF
        prune = self._p.alpha_beta
        depth = self._p.depth
        for ours in self._our_candidates(fr, st, shuffle=True):
            worst = _INF
            for theirs in self._enemy_candidates(fr, st):
                acts = list(ours)
                for i in fr.them:
                    acts[i] = theirs[i]
                nxt = self._sim.step(fr, st, acts)
                self._nodes += 1
                v = self._evaluate(fr, nxt) if depth == 1 else self._value(fr, nxt, depth - 1, alpha, min(_INF, worst))
                if v < worst:
                    worst = v
                if prune and worst <= alpha:
                    break
                if self._nodes >= self._p.max_nodes:
                    self._budget_hit = True
                    break
            if best_acts is None or worst > best:
                best, best_acts = worst, ours
            alpha = max(alpha, best)
            if self._nodes >= self._p.max_nodes:
                self._budget_hit = True
                break
        assert best_acts is not None
        return best_acts, best

    # ------------------------------------------------------------------------------- act

    def act(self, observation: TeamObservation, team: Team) -> dict[str, Action]:
        started = time.perf_counter()
        partial = (
            observation
            if isinstance(observation, PartialTeamObservation)
            else restrict_to_team_view(observation, team)
        )
        if partial.timestep < self._last_t:  # a new episode started without reset()
            self.reset()
        self._last_t = partial.timestep

        fr, st, ids = self._build(partial, team)
        acts, value = self._decide(fr, st)
        own_ids = {a.agent_id for a in partial.agents.values() if a.team is team and a.active}
        out = {ids[i]: ACTIONS[acts[i]] for i in fr.us if ids[i] in own_ids}
        self.last_stats = SearchStats(self._nodes, float(value), time.perf_counter() - started, self._budget_hit)
        return out
