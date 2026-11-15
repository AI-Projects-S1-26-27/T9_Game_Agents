# Minimax opponent (alpha-beta pruning)

`src/rl/minimax.py`. A classical search opponent for evaluation: it is a fixed, non-learning team
that thinks a few moves ahead. It is selected like any other opponent (`kind: minimax`, parameters
in YAML, strict schema `MinimaxParams`) and is also a controller in the viewer
(`configs/viz/controllers.yaml`, label "Minimax (alpha-beta, depth 2)").

## What it does, step by step

1. **Reads only what its team can see** (the fog-of-war observation). Visible enemies are simulated.
   If an enemy carries *our* flag, the flag's position tells us where the carrier is, so that enemy is
   simulated even outside our view. Enemies we cannot see are ignored. It remembers where it last saw
   the enemy flag lying dropped (or where our own carrier vanished), until it sees that cell empty.
2. **Simulates with its own fast copy of the rules** (`_Sim`). `CTFEnvironment.step` copies dataclasses
   and is too slow to call thousands of times per decision. `_Sim` works on tuples of ints and follows the
   same phase order (moves, carried flags follow carriers, dropped-flag returns, pickups, tagging,
   scoring). `tests/test_rl_minimax.py` replays random games on 6 rule variants and checks that the
   copy matches the real environment after every step.
3. **Searches.** One level = one game step: our team picks a joint move (MAX), then the opponent team
   picks its worst reply (MIN), then both are applied at once. `depth` is the number of levels
   (`depth: 2` = two game steps ahead). Letting the opponent reply after seeing our move is pessimistic
   (real moves are simultaneous); it is the usual way to make a simultaneous game fit minimax.
4. **Keeps the tree small.** 5 moves x 4 agents is 625 joint moves per side, far too many. Each agent
   only tries its best `own_moves_per_agent` moves (and each visible enemy its `enemy_moves_per_agent`
   best), ranked by path distance to a target:
   our carrier goes home, an attacker goes to the enemy flag (or escorts our carrier), a defender goes to
   the threat (enemy carrier, our dropped flag, an attacker close to our flag, else our base).
   Joint moves are tried best-first, which is what makes alpha-beta cut a lot.
   `max_nodes` caps the simulated steps per decision as a hard time limit.
5. **Scores leaf positions** with a weighted sum (all weights in YAML, `MinimaxWeights`):
   our carrier's progress home and holding the flag, our nearest attacker's distance to the enemy flag,
   escorts staying near our carrier, the same terms for the enemy with a minus sign, defenders heading
   for the threat, agents alive, and enemy attackers close to our flag. A capture is +/- `win`.
   Which slots attack: those allowed by the environment's `carrier_slots`. The others defend. With the
   original rules (everyone may carry) the last slot defends.

## Alpha-beta pruning

`alpha_beta: true` skips branches that cannot change the result: once a reply already makes a move
worse than the best move found so far (alpha), the remaining replies to that move are not examined.
The chosen move and its value are identical to plain minimax; only the number of simulated steps drops.
`alpha_beta: false` runs plain minimax, which is how the report can show the saving
(`tests/test_rl_minimax.py` checks equal values and fewer nodes at depth 1 to 3).
The decision's cost is in `strategy.last_stats` (`nodes`, `value`, `seconds`, `budget_hit`).

## Choosing the depth (measured, single run, sandbox CPU)

30 episodes per cell, both sides alternating, `configs/ctf_v2.yaml` with `spawn_mode: near_base`,
opponent noise 0, `own_moves_per_agent: 2`, `enemy_moves_per_agent: 2`. Wins / losses / draws of the minimax team:

| opponent         | depth 1 (about 1 ms/decision) | depth 2 (about 7-12 ms/decision) |
|------------------|-------------------------------|----------------------------------|
| random           | 28 / 0 / 2                    | 28 / 0 / 2                       |
| stay             | 19 / 0 / 11                   | 21 / 0 / 9                       |
| scripted_leash   | 18 / 12 / 0                   | 18 / 11 / 1                      |
| scripted_parked  | 8 / 17 / 5                    | 11 / 11 / 8                      |
| scripted_chaser  | 14 / 16 / 0                   | 21 / 9 / 0                       |
| scripted_escort  | 27 / 3 / 0                    | 29 / 1 / 0                       |

Depth 2 is the default: clearly stronger than depth 1 against the chaser and the parked guard, and fast
enough for a 200-episode evaluation. Depth 3 was tried at 10 episodes per cell with a 3000-node cap: it
hit the cap, took about 30 ms per decision and was not better, so it is not the default. These are
small samples; the real numbers come from `evaluate.py` (200 episodes, Wilson intervals).
Timings were taken in a Linux sandbox, not on the Intel MacBook.

## Limits to state in the report

- Pessimistic reply model (the opponent sees our move) and only the top moves per agent are searched,
  so it is a heuristic searcher, not a perfect player.
- It ignores unseen enemies, so a surprise attacker from the fog is only handled once it is visible.
- The evaluation weights are hand-set, not tuned; they are YAML values so they can be changed and reported.
- It is slow compared with the scripted opponents (milliseconds versus microseconds per decision),
  so it is not in the default training curriculum; the training throughput cost has to be measured first.

## Parameters (`MinimaxParams`)

| key | meaning |
|-----|---------|
| `depth` | game steps looked ahead, >= 1 |
| `own_moves_per_agent` | best moves tried per own agent, 1 to 5 |
| `enemy_moves_per_agent` | best moves tried per visible enemy, 1 to 5 |
| `danger_radius` | an enemy attacker closer than this to our flag is a threat |
| `max_nodes` | cap on simulated steps per decision |
| `alpha_beta` | pruning on/off (same answer, different cost) |
| `weights.*` | the evaluation terms listed above |
