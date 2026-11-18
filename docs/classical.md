# Classical AI opponent

`src/classical_ai/` is a **verbatim copy** of the `classical_ai` branch (A* path finding, an influence map of enemy
danger, a small behaviour tree, one controller per role). It is not edited on this branch: when the classical teammate
changes it, copy the files again (`git checkout origin/classical_ai -- src/classical_ai`, then delete any `__pycache__`).
`src/rl/classical_opponent.py` (adapter) and `src/rl/classical_fixes.py` (the fixes) are the only code of ours around it.

## What the adapter adds

| Problem in the classical code | What `ClassicalOpponent` does |
|---|---|
| Default roles are positional over the agents that are still ACTIVE (first = attacker, second = defender, the rest scouts). Wrong for v2 and they shift when an agent is tagged. | Always passes a fixed `{agent id: role}` table built from `params.roles` (slot 1 first). |
| `ClassicalStrategy` keeps its coordinator (shared belief map) for as long as the object lives. | `reset()` drops it, so every episode starts with an empty belief map. |
| It does not know the `carrier_slots` rule. | Refuses a config with an attacker in a slot that cannot carry (`ConfigError`). |
| Known problems of the behaviour tree (dropped flags, headcount rule, literals). | Optional switches, see "The fixes" below. |

It is a normal opponent kind, so the same YAML entry works everywhere:

```yaml
- kind: classical
  label: classical
  params: {roles: [attacker, attacker, defender, scout]}   # strict: roles is the only key
```

* Evaluation: `python -m rl.evaluate --checkpoint <pt> --eval-config configs/rl/eval.yaml --opponents classical`
  (entry already in `configs/rl/eval.yaml`).
* Viewer: `python -m viz play --map configs/ctf_v2.yaml --checkpoint <pt> --red RL --blue Classical`
  (entry in `configs/viz/controllers.yaml`; `params.roles` drive its behaviour and are also the labels shown).
* Training: it is accepted as a curriculum opponent (it is in `KNOWN_KINDS`) but nobody has trained against it.

Cost: about 0.5 ms per team decision (measured on one sandbox CPU core); minimax depth 2 is about 10 ms (docs/minimax.md).

## The fixes (`classical_fixed`)

`src/classical_ai/` stays a verbatim copy. The known problems are fixed in `src/rl/classical_fixes.py`, which
subclasses the teammate's roles and coordinator, so re-copying the branch never conflicts. Every fix has its own
key in `params` (all optional; the defaults are the ORIGINAL behaviour, and with every key at its default the
verbatim code runs, so the old `classical` numbers stay comparable). `configs/rl/eval.yaml` and
`configs/viz/controllers.yaml` have a second entry, `classical_fixed`, that spells every key out.

| Problem in the classical code | Key | What the fix does |
|---|---|---|
| A dropped OWN flag is ignored by the defender. | `defend_dropped_flag` | The defender walks onto it (an active agent of the flag's own team on a dropped flag sends it home). |
| A dropped ENEMY flag is only found while it is in sight; out of sight the attackers walk to the enemy base. | `remember_dropped_flag` | The strategy remembers the last seen dropped enemy flag and puts it back into the observation until one of our agents sees that cell without the flag. |
| No idea about the headcount rule (a carrier is tagged only when enemies on its cell outnumber its friendly non-carriers there). | `buddy_system` | Attackers enter a guarded enemy flag cell together (the first waits next to it, non-carriers cannot be tagged); the carrier waits one step for an adjacent escort when an enemy is within `carrier_danger_radius`; an escort on the carrier's cell repeats the carrier's move. The wait is limited to `pair_wait_steps`. |
| One interceptor can never tag a carrier that has an escort. | `intercept_in_pairs` | With an enemy carrier in view, the defender and the scout wait together on the first cell of the carrier's way home that both reach in time (plain chase if there is none). |
| The belief map leaks between episodes unless `reset()` is called. | always on in the fixed strategy | A step counter that goes back means a new episode: it starts fresh. |
| Hard-coded literals: influence radius 3, penalty 5.0, probability weight 3.0, scout view radius 3 for global observations. | `influence_radius`, `enemy_penalty`, `probability_weight`; the scout radius is the observation's own | Parameters; defaults are the old literals. |
| Role assignment shifts when agents die. | (adapter) | Already fixed by the fixed role table above. |

Not fixed: the behaviour tree itself (what each role wants) is the teammate's design, and the fixes only add rules
around it. The defender sees the carrier only while it is within the visibility radius of an active agent of ours.

### Smoke measurement (not the evaluation)

Classical team against the evaluation opponents, `spawn_mode: near_base`, 20 seeds x both sides = 40 games per cell,
one sandbox CPU, seeds 0-19 (the games are deterministic, so these numbers repeat). The CLI evaluation (200 games)
is the number to quote.

| opponent | original W/D/L | fixed W/D/L |
|---|---|---|
| stay | 32/8/0 | 39/1/0 |
| scripted_leash | 32/0/8 | 38/0/2 |
| scripted_parked | 5/27/8 | 34/3/3 |
| scripted_chaser | 32/0/8 | 38/0/2 |
| scripted_escort | 27/0/13 | 29/2/9 |
| minimax depth 2 | 11/3/26 | 31/8/1 |

Why `stay` and `scripted_parked` drew so often for the ORIGINAL classical team: with `near_base` spawns one of the four
agents of a team lands on its own flag cell in 4 of 21 games (19%), and under the headcount rule such an agent tags
every LONE carrier that picks up the flag. Only two agents entering together get in. The original team drew 8 of 40
games against `stay`, the fixed one 1 of 40. (This is a reading of the numbers, not a traced cause for every game.)

The same mechanism is NOT the whole story for the RL policy's draws against `stay`. Two draws were traced (viewer
maps with `near_base`, 9.0M checkpoint, RED = RL): on `open_field` seed 3 a stationary blue agent starts on its own
flag cell; on `default` seed 2 no agent starts there, but both RL attackers pick up the flag and then walk, one after
the other and without an escort, onto the cell (7,3) where a stationary blue agent stands, and are tagged there
(1 enemy against 0 friendly non-carriers). After that the two RL agents left at home never attack. A third draw
(`open_field` seed 2) was not traced.

Cost of the fixes: the pair rules add a few path searches per decision (the whole smoke above, 480 games, ran in
about 60 s including minimax).

## What it still does NOT know

* The behaviour tree has no concept of roles beyond the fixed table and no plan for guarding the flag against a team
  that enters in pairs (the defender sits on its flag).
* Nothing is learned: it cannot adapt to an opponent.
* The fixes were written for the v2 rules (carrier slots, headcount, drop on tag); on the legacy 3-agent rules
  (`configs/default.yaml`) they were not tested.

No claim is made that this is the best the classical approach can do.
