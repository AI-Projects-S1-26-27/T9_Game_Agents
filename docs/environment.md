# Multi-Agent CTF Environment Specification

Prototype 1 as implemented in `src/ctf/`.

## 1. Purpose

Lightweight testbed for multi-agent coordination and adversarial play on a discrete CTF grid. The environment is not the research contribution—strategies and experiments live outside `src/ctf/`.

## 2. Environment Overview

| Aspect | Prototype 1 behavior |
|--------|----------------------|
| Teams | **2** fixed teams: `RED` and `BLUE` |
| Agents | Default **3 per team** (`agents_per_team` configurable) |
| World | **2D discrete grid** (`width` × `height`), no continuous physics |
| Obstacles | Configurable blocked cells |
| Bases | One base cell per team (spawn / scoring reference) |
| Flags | **One flag per team**, initially at that team’s base |
| Parameters | Grid size, spawns, spawn randomization, obstacles, tagging, episode cap, visibility radius, seed — via `CTFConfig` / YAML |

The shipped demo configuration is `configs/default.yaml` (11×7 grid, center wall obstacle column, explicit spawns).

---

## 3. Game Configuration

Configuration is defined by `CTFConfig` and loaded with `load_config(path)` from YAML. Values below list **code defaults**; **`configs/default.yaml`** overrides spawns and obstacles as noted.

### 3.1 Teams

- Teams are the enum `Team.RED` and `Team.BLUE`.
- Exactly two teams are supported in Prototype 1 (not configurable beyond RED/BLUE).

### 3.2 Agents

- Count per team: `agents_per_team` (default **3**, minimum 1).
- Agent IDs are assigned deterministically: `R1…Rn` for RED, `B1…Bn` for BLUE, where `n = agents_per_team`.
- Each agent tracks: `agent_id`, `team`, position `(x, y)`, `active`, `carrying_flag`.

**Default YAML:** 3 agents per team (`R1–R3`, `B1–B3`).

### 3.3 Grid and Obstacles

| Parameter | Default (code) | `configs/default.yaml` |
|-----------|----------------|-------------------------|
| `width` | 11 | 11 |
| `height` | 7 | 7 |
| `obstacles` | `[]` | `(5,1)…(5,5)` — vertical wall at column 5 |

Validation: `width >= 1`, `height >= 1`; obstacle cells must lie inside the grid.

### 3.4 Bases and Flags

| Parameter | Default (code) | `configs/default.yaml` |
|-----------|----------------|-------------------------|
| `red_base` | `(1, 3)` | `(1, 3)` |
| `blue_base` | `(9, 3)` | `(9, 3)` |

- Each team owns one flag (`Flag.team` = owning team).
- Initial flag state: `FlagStatus.AT_BASE` at that team’s base coordinates.
- Bases must be inside the grid and **not** on obstacles.

A flag is **at base** or **carried**. With `drop_flag_on_tag` (§3.6b) it can also be **dropped**
(`FlagStatus.DROPPED`): lying on the cell where its carrier was tagged. `configs/default.yaml` keeps the
original rules, where a flag is only ever at base or carried.

### 3.5 Spawn Configuration

Optional explicit lists: `red_spawns`, `blue_spawns`.

- If provided, each list must contain **exactly** `agents_per_team` positions.
- Each spawn must be in bounds and not on an obstacle.

If **not** provided, the environment generates **exactly** `agents_per_team` spawn cells per team: traversable cells sorted by Manhattan distance from that team’s base (then `y`, then `x`), taking the closest cells first. Configuration fails if not enough traversable cells exist.

**Default YAML** (explicit):

- RED: `(1,2)`, `(1,3)`, `(1,4)`
- BLUE: `(9,2)`, `(9,3)`, `(9,4)`

**Spawn randomization** (`spawn_mode`, applied at every `reset()` using the environment's seeded RNG):

| `spawn_mode` | Behavior |
|--------------|----------|
| `fixed` (default) | Use the spawn list above (or the cells nearest the base). Identical every episode. |
| `shuffle` | Same cells, randomly re-assigned among the team's agents. |
| `near_base` | Distinct random cells within `spawn_radius` (Manhattan) of the base, restricted to the team's home zone (strictly closer to own base than to the enemy base). `spawn_allow_base` controls whether the flag cell itself is allowed. Ignores explicit spawn lists. |

Construction fails if `near_base` cannot find `agents_per_team` valid cells.

### 3.6 Tagging Configuration

| Parameter | Default (code) | `configs/default.yaml` |
|-----------|----------------|-------------------------|
| `tagging_enabled` | `true` | `true` |

When enabled, a flag carrier is tagged according to `tagging_rule` (see §7.5). The value is copied into `GameState.tagging_enabled` at reset.

### 3.6b Flag rules (v2)

These keys default to the **original rules**, so `configs/default.yaml` and every older config behave exactly as before.
`configs/ctf_v2.yaml` (same map as `default.yaml`) switches the new rules on.

| Parameter | Default (code and `default.yaml`) | `configs/ctf_v2.yaml` | Meaning |
|-----------|-----------------------------------|------------------------|---------|
| `carrier_slots` | `null` | `[1]` | 1-based team slots (`R1`/`B1` = slot 1) allowed to pick up and carry the enemy flag; `null` = every agent. Other agents standing on a flag do nothing. The environment has no roles, so this is keyed to the slot index (the default RL layout is slot 1 = attacker, 2 = defender, 3 = scout). |
| `tagging_rule` | `any_enemy` | `headcount` | `any_enemy`: one active enemy on the carrier's cell tags it. `headcount`: active enemies on the cell must **outnumber** the active friendly **non-carrier** agents on it. |
| `drop_flag_on_tag` | `false` | `true` | `true`: a tagged carrier's flag stays on the cell where it was tagged (`DROPPED`). `false`: it goes back to its base. The tagged carrier is eliminated either way. |
| `dropped_flag_enemy_pickup` | `true` | `true` | An eligible enemy (see `carrier_slots`) may pick a dropped flag up again. |
| `dropped_flag_return_on_touch` | `true` | `true` | An active agent of the flag's **own** team on the dropped flag's cell sends it back to its base. |

The last two keys only matter when `drop_flag_on_tag` is `true`; if it is, they must not both be `false` (a dropped flag could never move again).

Consequences worth knowing under the v2 rules:

- A lone carrier loses to one defender. Carrier + one escort beats one defender. Two defenders beat carrier + one escort. Any enemy agent counts as a tagger (a scout too), and any friendly non-carrier counts as an escort (a defender or scout too).
- A parked defender no longer makes the flag unbeatable, but a team can only score with an agent whose slot is in `carrier_slots`. With `carrier_slots: [1]` and no respawn, **once a team's slot-1 agent is tagged that team can no longer win** (the best it can reach is a draw). Use `carrier_slots: [1, 2]` (two attackers) if the team should be able to recover.

### 3.7 Episode Length

| Parameter | Default (code) | `configs/default.yaml` |
|-----------|----------------|-------------------------|
| `max_episode_steps` | 200 | 200 |
| `seed` | 42 | 42 |

If `timestep` reaches `max_episode_steps` without a win, the episode ends with `truncation=True`, `winner=None`.

### 3.8 Visibility

| Parameter | Default (code) | `configs/default.yaml` |
|-----------|----------------|-------------------------|
| `visibility_radius` | 3 | 3 |

Manhattan radius used by `ObservationMode.PARTIAL` (see §6). Copied into `GameState.visibility_radius` at reset; `observe(..., visibility_radius=r)` can override it per call.

### 3.9 Defaults and strict parsing

Defaults are declared once, in the `CTFConfig` dataclass. YAML files only override them. `load_config` / `config_from_dict` reject unknown keys and wrongly typed values (e.g. `width: "11"`), so a misspelled parameter fails loudly instead of silently using a default. `config_to_dict` returns a YAML-serializable copy for logging the exact configuration of a run.

---

## 4. Action Space

Each **active** agent selects one of five actions (`Action` enum):

| Action | Grid delta |
|--------|------------|
| `UP` | `(0, -1)` |
| `DOWN` | `(0, +1)` |
| `LEFT` | `(-1, 0)` |
| `RIGHT` | `(+1, 0)` |
| `STAY` | `(0, 0)` |

No diagonal movement.

**Movement validation (per agent, from pre-step position):**

- `STAY` is always valid for active agents.
- A move is **rejected** (agent remains in place) if the target cell is out of bounds or an obstacle.
- Rejected moves are recorded in `info["rejected_moves"]` (agent id → action name string).

**Agent–agent collision:** Movement does **not** block on other agents. Multiple agents may occupy the same cell.

**Inactive agents:** Do not require an entry in the action dict; they do not move.

**Action dict requirements for `step()`:**

- Every **active** agent must have an action.
- Unknown agent id → `KeyError`.
- Invalid action string → `ValueError` from `Action(...)`.

---

## 5. State Representation

`GameState` is the full simulator state.

| Field | Meaning |
|-------|---------|
| `grid` | `GridMap`: `width`, `height`, `obstacles` (frozen set) |
| `agents` | `dict[str, Agent]` |
| `flags` | `dict[Team, Flag]` |
| `red_base`, `blue_base` | Base coordinates |
| `timestep` | Integer step counter (incremented once per successful `step()` call) |
| `winner` | `Team` or `None` |
| `terminated` | Episode ended (win or max steps) |
| `truncation` | `True` if ended due to step limit without winner |
| `tagging_enabled` | Snapshot of tagging setting for this episode |

**Agent** (`Agent`): `agent_id`, `team`, `x`, `y`, `active`, `carrying_flag`.

**Flag** (`Flag`): `team` (owner), `status` (`AT_BASE` | `CARRIED`), `x`, `y`, `carrier_id` (when carried).

---

## 6. Observation Model

Two modes are implemented.

**`ObservationMode.GLOBAL`** returns `GlobalObservation`:

- Grid: `width`, `height`, `traversable` (height×width bool matrix), `obstacles`
- `red_base`, `blue_base`
- Full `agents` and `flags` maps
- `timestep`, `winner`, `terminated`, `truncation`, `tagging_enabled`, `visibility_radius`
- `to_dict()` for serialization

**`ObservationMode.PARTIAL`** (requires `team=`) returns `PartialTeamObservation`, the team-level fog-of-war view:

- Friendly agents (including eliminated ones) are always fully visible.
- An enemy agent is visible only if it is active and within `visibility_radius` (Manhattan) of an **active** friendly agent. Eliminated agents see nothing.
- The enemy flag is visible under the same rule, whether it is at base, carried or **dropped**; the team's own flag is always visible. A team that did not see a flag being dropped therefore does not know where it lies until someone comes within `visibility_radius` of it.
- Because the view is the union over all active teammates, whatever one agent (e.g. a scout) sees, the whole team sees.

---

## 7. Game Mechanics

### 7.1 Movement

All active agents’ moves are proposed from the **same pre-step** positions, then applied together. Inactive agents stay fixed.

### 7.2 Flag Return (dropped flags only)

Runs **before** pickups. If `dropped_flag_return_on_touch` is `true` and a flag is `DROPPED`, the first active agent (sorted ids) of the flag's own team that shares its cell sends it back: the flag becomes `AT_BASE` at its base coordinates and a `flag_return` event (`agent_id`, `flag_team`) is emitted. Standing on the cell counts, so a defender that tags a carrier and stays there returns the flag on the next step. Because returns run first, if a defender and an enemy attacker reach a dropped flag in the same step, the flag goes home and the attacker cannot pick it up.

### 7.3 Flag Pickup

After movement (and returns), an **active** agent that is **not** already carrying a flag may pick up the **enemy** flag if:

- the agent's slot may carry (`carrier_slots`; always true when it is `null`),
- the enemy flag is `AT_BASE`, or `DROPPED` with `dropped_flag_enemy_pickup` enabled, and
- the agent shares the flag's cell.

Effects: `agent.carrying_flag = True`; enemy flag becomes `CARRIED` with `carrier_id` set. A `flag_pickup` event is emitted; picking up a dropped flag adds `"from_dropped": true` to it (the key is absent otherwise, so original events are unchanged).

Agents are processed in dict iteration order (insertion order: `R1…Rn`, then `B1…Bn`). Only one pickup per flag per step is possible because after the first pickup the flag is no longer available.

### 7.4 Flag Carrying

While `CARRIED`, the flag’s `(x, y)` follows the carrier after movement (`_sync_carried_flag_positions`). The agent’s `carrying_flag` indicates it holds the **opponent’s** flag (only enemy pickup is implemented).

### 7.5 Tagging

When `tagging_enabled` is true, each active carrier is checked according to `tagging_rule`.

**`any_enemy`** (original): carriers are scanned in agent dict order; if any **opposing** active agent shares the cell, the carrier is tagged, and the first such agent in dict order is recorded as `by_agent_id`. Teammates on the same cell do **not** matter. Carriers are handled one after another, so if two carriers meet on one cell only the first one scanned is tagged (it becomes inactive and can no longer tag the other).

**`headcount`**: the carrier is tagged if the number of active enemies on its cell is **greater** than the number of active friendly agents on it that are not carrying a flag. All carriers are judged on the same snapshot, so two carriers that meet with no helpers are **both** tagged. `by_agent_id` is the lowest-id enemy on the cell.

Effects of a tag: carrier `active=False`, `carrying_flag=False`; a `tag` event (`agent_id`, `by_agent_id`, `flag_team` = the flag that was being carried — the **enemy team’s** flag identifier). With `drop_flag_on_tag` a `flag_drop` event (`agent_id`, `flag_team`, `x`, `y`) follows.

### 7.6 Flag Reset

When a carrier is tagged and `drop_flag_on_tag` is `false`, the carried enemy flag returns to **`AT_BASE`** at its **original base coordinates** (from config). When it is `true`, the flag becomes `DROPPED` at the carrier's cell instead (see §7.2 and §7.3 for what happens next).

### 7.7 Capture / Winning

After tagging, an **active** agent with `carrying_flag=True` on **its own base** cell wins:

- `winner` = agent’s team
- `terminated` = `True`
- Events: `flag_capture`, then `episode_end` with `reason="flag_capture"` and `winner` set to the team name string.

“Capture” in the win sense means **returning the enemy flag to your base**, not merely touching the enemy flag.

### 7.8 No Respawn

Tagged (inactive) agents remain inactive for the rest of the episode. No respawn mechanic exists.

---

## 8. Step Execution Order

Each call to `step()` applies this **fixed** sequence:

1. **Parse actions** — validate ids and required active-agent actions.
2. **Propose and apply movement** — simultaneous intent from pre-step positions.
3. **Synchronize carried flag positions** — update coordinates for flags already carried.
4. **Flag return** — dropped flags go home when an active agent of their own team is on them (v2 rules; no-op otherwise).
5. **Flag pickup** — enemy flag at base (or dropped, if enabled), same cell, slot allowed to carry.
6. **Tagging** — if enabled; `any_enemy` or `headcount`.
7. **Scoring / win check** — carrier on own base.
8. **Timestep increment** — `timestep += 1`.
9. **Maximum-step truncation** — if `timestep >= max_episode_steps` and not already terminated: set `truncation=True`, `terminated=True`, `winner=None`, emit `episode_end` with `reason="max_steps"`.

**Ordering implications:**

- **Return → pickup → tag → score** on the same step.
- A carrier who is tagged in phase 6 cannot score in phase 7 that same step.
- Example: pickup on the enemy base with a defender on the same cell can produce pickup then tag before any win check.

Co-location of non-carriers is allowed throughout; only tagging special-cases carriers vs opponents.

---

## 9. Configuration and Reproducibility

- **Loading:** `load_config("configs/default.yaml")` or construct `CTFConfig(...)` in code.
- **Validation:** On construction, `CTFConfig` checks grid dimensions, episode length, agent count, base/spawn bounds, bases/spawns not on obstacles, explicit spawn counts, and feasibility of auto-generated spawns.
- **Seed:** the environment owns a private `random.Random`, created from `CTFConfig.seed`. `reset(seed=…)` reseeds it; `reset()` without a seed *continues* the stream, so consecutive episodes differ (when `spawn_mode` is random) while the whole sequence stays reproducible from `config.seed`.
- **Determinism:** `step()` never consumes RNG state. Given the same spawns and actions, transitions are identical. With `spawn_mode: fixed` (the default) every episode starts identically, so repeated matches between deterministic strategies are identical too; use `shuffle` or `near_base` to get varied, seeded episodes.
- The RNG is environment-local; the global `random` module is never used.

---

## 10. Environment API

Main class: `CTFEnvironment(config: CTFConfig)`.

| Member | Behavior |
|--------|----------|
| `reset(*, seed=None)` | New episode; returns **snapshot** `GameState`. Samples spawns from the internal RNG; `seed` reseeds it first. |
| `step(actions)` | One transition; returns `(state, rewards, done, info)`. `state` is a snapshot. Raises if not reset or already terminated. |
| `observe(mode=GLOBAL, *, team=None, visibility_radius=None)` | Builds a `GlobalObservation`, or a `PartialTeamObservation` for `mode=PARTIAL` (needs `team`). Radius defaults to `config.visibility_radius`. |
| `state` | **Snapshot** copy of current state (mutating it does not affect the simulator). |
| `done` | `True` when internal `terminated` is set. |
| `config` | The `CTFConfig` instance. |

**`step(actions)`** expects `dict[str, Action | str]` keyed by agent id.

**Returns:**

- `rewards`: one float per agent id; **always `0.0` in Prototype 1**.
- `done`: same as `state.terminated` after the step.
- `info`: see §11.

Typical loop:

```python
env = CTFEnvironment(load_config("configs/default.yaml"))
state = env.reset()
while not env.done:
    actions = {aid: Action.STAY for aid in env.config.agent_ids()}  # example
    state, rewards, done, info = env.step(actions)
```

External controllers must supply actions for all **active** agents each step.

---

## 11. Event Reporting

`info` on each step contains:

### `rejected_moves`

`dict[str, str]`: agent id → rejected action name (empty dict if none).

### `events`

`list[dict]`: structured records for **this step only**.

| `type` | Fields | When |
|--------|--------|------|
| `flag_pickup` | `agent_id`, `flag_team` | Enemy flag picked up at base |
| `tag` | `agent_id`, `by_agent_id`, `flag_team` | Carrier tagged |
| `flag_capture` | `agent_id`, `flag_team` | Win condition (enemy flag returned to own base) |
| `episode_end` | `winner` (`str` or `null`), `reason` | `"flag_capture"` or `"max_steps"` |

Win steps may emit both `flag_capture` and `episode_end` in the same list.

---

## 12. Visualization

A **separate** Pygame layer lives under `visualization/` (not in `src/ctf/`):

- `visualization/renderer.py` — draws a `GameState` snapshot (grid, bases, obstacles, flags, agents).
- `visualization/demo.py` — stepped animation of the shared demo script in `scripts/demo_scenario.py`.

The visualizer:

- Uses the public API (`reset`, `step`, `state`, `info["events"]`).
- Does **not** implement movement, flags, tagging, or termination rules.
- Does **not** implement AI or pathfinding; it replays a **predefined action sequence** (same scenario as `scripts/run_demo.py`).

Launch: `uv run python visualization/demo.py` (requires `pygame` dependency).

---

## 13. Extensibility

Control loop: `observation → strategy → actions → environment → next state`. Strategy code stays outside `src/ctf/`; rewards and extra observation modes can be layered on without changing current step logic.

---

## 14. Current Scope and Exclusions

**Not implemented in Prototype 1** (may be added later only if the finalized methodology requires):

- AI / hand-crafted strategies (beyond external scripts)
- Reinforcement learning or MARL training loops
- Built-in opponent AI
- A*, Minimax, MCTS, or other planners inside the env
- Networking or multiplayer
- Sophisticated combat beyond tagging
- Dropped flags, respawn, or extra game modes
- Advanced graphics or game UI (basic Pygame demo only)
- Non-zero reward shaping or experiment runners in the core package

---

## 15. Prototype 1 Verification

Documented verification status (implementation / QA, not research results):

| Check | Status |
|-------|--------|
| Automated tests | `uv run pytest` (all passing) |
| Deterministic text demo | `scripts/run_demo.py` — RED wins, `reason=flag_capture` |
| Flag pickup and return | Covered by tests and end-to-end demo |
| Episode termination | Win and max-step truncation tested |
| Visualization | Pygame demo runs shared scenario; environment unchanged |

No performance benchmarks or coordination-strategy research claims are made at this stage.
