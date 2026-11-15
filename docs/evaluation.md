# Evaluation (`python -m rl.evaluate`)

Training metrics (`metrics.jsonl`) are measured while the policy still samples actions and the opponents
are partly randomized (`opponent_noise`). They are good for watching learning, not for reporting results.
`rl.evaluate` measures the finished policy cleanly.

## What "noise-free" means

* Opponents play exactly as scripted. The eval YAML has no `opponent_noise` key at all (unknown keys are errors).
* The policy plays **argmax** actions by default (`policy_mode: argmax`, what the demo uses). `policy_mode: sampled`
  draws from the policy like training did; running both is the argmax-vs-sampled check (B2b-2 automates it).
* Spawns are randomized (`env_overrides.spawn_mode: near_base`, `fixed_spawn_fraction: 0.0`) but **seeded**: every
  episode's spawns and opponent randomness come from `Random(f"{seed}:{opponent}:{side}:{i}")`. The same config
  gives the same games, and two checkpoints evaluated with the same config face the same games, so they can be
  compared directly.
* The policy plays both sides (`sides: [red, blue]`); `episodes` (default 200) is split evenly over them.

## Running it

```bash
python -m rl.evaluate --checkpoint runs/<run>/checkpoints/latest.pt
python -m rl.evaluate --checkpoint ... --opponents random scripted_chaser --set episodes=40
python -m rl.evaluate --checkpoint ... --env-set visibility_radius=6           # change the env at eval time
python -m rl.evaluate --checkpoint ... --env-config configs/ctf_v2.yaml        # play on another env YAML
python -m rl.evaluate --checkpoint runs/colab_run1/checkpoints/latest.pt --eval-config configs/rl/eval_legacy.yaml
```

Without `--env-config` the env stored in the checkpoint is used, so an old 3-agent checkpoint is evaluated under
its own rules automatically (it needs `eval_legacy.yaml`, whose scripted opponents have three roles).
The printed table is also saved as JSON (default `outputs/eval/<run>_<checkpoint>.json`, git-ignored).

## Metrics (per opponent; rows for both sides pooled, then each side)

| Column / JSON key | Meaning |
|---|---|
| `win`, `draw`, `loss` | Share of episodes; Wilson interval at `confidence`. A draw is a timeout. |
| `episode_length` | Mean steps per episode. |
| `captures`, `pickups` | Totals and per-episode means for the policy's team. |
| `episodes_with_pickup`, `time_to_first_pickup` | Share of episodes with at least one pickup; mean step of the first one (episodes with a pickup only). |
| `pickup_to_capture` | Captures / pickups. Pickups inside one episode are not independent, so the interval is approximate. |
| `tags_inflicted`, `tags_suffered` | Per-episode means. Suffered = our flag carrier was tagged. |
| `stay_fraction` | Share of agent-steps that chose STAY, overall and per role. |
| `scout_coverage` | Share of traversable cells that were within `visibility_radius` of an active scout at some point; `null` with no scout role. |
| `action_histograms` | Per role, share of each action. |
| `heatmaps` | Per role, share of position-steps per cell (rows = y). With both sides pooled on a left-right symmetric map BLUE is flipped, so "own base on the left" for everyone (`heatmaps_team_relative: true`). One side alone: environment coordinates. Pooled sides on an asymmetric map: `null`. |
| `route`, `route_carrying` | How often `route_roles` agents ENTER the centre column(s) through the `top`, `middle` or `bottom` lane (all entries / only entries made while carrying the flag). On the default map the wall leaves a gap at the top and one at the bottom. `top_share` has a Wilson interval. |
| `defender_distance` | Mean Manhattan distance of the defender to its own base cell and to its own flag's current cell (they differ once the flag is carried or dropped). |

Rates use Wilson intervals; means use `mean +- z * standard error`. The interval level is `confidence`.

## Opponents

`configs/rl/eval.yaml` lists them: random, stay, four scripted variants (leash, parked, chaser, escort) and minimax.
Each entry is `{kind, label, params}`; `label` is the name used in reports and in `--opponents`.
The classical AI and the LLM agent are separate branches. To evaluate against one, register a factory and add an
entry with that kind to the eval YAML:

```python
from rl.evaluate import EXTERNAL_OPPONENTS
EXTERNAL_OPPONENTS["classical"] = lambda spec, env_config, rng: MyClassicalStrategy(env_config, **spec.params)
```

The factory returns any `BaseStrategy`. An unregistered, non-built-in kind is a clear `ConfigError`.

## Files

* `src/rl/eval_config.py`  strict `EvalConfig` dataclasses (every key required).
* `src/rl/eval_stats.py`   Wilson interval, normal mean interval.
* `src/rl/evaluate.py`     episode runner (`play_episode`), recorder, `summarize`, `evaluate_policy`, report, CLI.
* `configs/rl/eval.yaml`, `configs/rl/eval_legacy.yaml`
* `tests/test_rl_evaluate.py`
