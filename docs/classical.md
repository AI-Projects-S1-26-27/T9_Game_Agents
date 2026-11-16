# Classical AI opponent

`src/classical_ai/` is a **verbatim copy** of the `classical_ai` branch (A* path finding, an influence map of enemy
danger, a small behaviour tree, one controller per role). It is not edited on this branch: when the classical teammate
changes it, copy the files again (`git checkout origin/classical_ai -- src/classical_ai`, then delete any `__pycache__`).
`src/rl/classical_opponent.py` is the only code of ours around it.

## What the adapter adds

| Problem in the classical code | What `ClassicalOpponent` does |
|---|---|
| Default roles are positional over the agents that are still ACTIVE (first = attacker, second = defender, the rest scouts). Wrong for v2 and they shift when an agent is tagged. | Always passes a fixed `{agent id: role}` table built from `params.roles` (slot 1 first). |
| `ClassicalStrategy` keeps its coordinator (shared belief map) for as long as the object lives. | `reset()` drops it, so every episode starts with an empty belief map. |
| It does not know the `carrier_slots` rule. | Refuses a config with an attacker in a slot that cannot carry (`ConfigError`). |

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

## What it does NOT know (v2 rules) and was smoke-tested for

Smoke test (both colours, 20 games per opponent and side, near_base spawns, radius 3, wins/draws/losses of the
classical team): no exceptions. Against random 36/4/0, stay 35/5/0, leash 27/2/11, chaser 31/0/9, escort 30/0/10
(wins/draws/losses over both sides, 40 games each). Against a parked defender it wins only 3 of 40 (24 draws, 13 losses)
and against minimax depth 2 it loses more than it wins (9 wins, 9 draws, 22 losses). With 20 games per side these are
rough numbers; the CLI evaluation (200 games) is the measurement to quote.
Things the classical code does not handle, found by reading it and (for the first) probing it; they are the
classical teammate's to fix, not ours:

* `FlagStatus.DROPPED` is never mentioned. A dropped OWN flag is ignored by the defender (probed: it stays at its base
  instead of touching the flag, which would send it home). A dropped ENEMY flag is only found while it is in sight (by
  reading the code: out of sight, the attacker falls back to walking to the enemy base).
* The behaviour tree has no idea about the headcount tagging rule. In one recorded game both attackers walked into
  the enemy base together and were tagged there (frequency not measured).
* The per-episode state leak is fixed only by our `reset()`; any other caller that reuses a `ClassicalStrategy` still
  leaks the belief map between episodes.
* Hard-coded literals (influence radius 3, penalty 5.0, probability weight 3.0, scout view radius 3 for global
  observations) are not configurable.
* Role assignment by itself (without our table) shifts when agents die.

No claim is made that this is the best the classical approach can do: it is the version on the branch, with roles
fixed, on the v2 rules.
