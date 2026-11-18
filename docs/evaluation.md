# Evaluation (`python -m rl.evaluate`)

Training metrics (`metrics.jsonl`) are measured while the policy still samples actions and the opponents
are partly randomized (`opponent_noise`). They are good for watching learning, not for reporting results.
`rl.evaluate` measures the finished policy cleanly.

## What "noise-free" means

* Opponents play exactly as scripted. The eval YAML has no `opponent_noise` key at all (unknown keys are errors).
* The policy plays **argmax** actions by default (`policy_mode: argmax`, what the demo uses). `policy_mode: sampled`
  draws from the policy like training did; the argmax-vs-sampled check below runs both on the same games.
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

## Extra checks (held-out maps, role swap, argmax vs sampled)

Configured by the `held_out`, `role_swap` and `policy_check` sections of the eval YAML (all keys required), run after
the main table, printed below it and saved as top-level keys of the JSON (`held_out`, `role_swap`, `policy_check`;
`null` when disabled, empty or skipped with `--skip held_out role_swap policy_check`). They never change the main
table: its game seeds are untouched, so numbers measured before B2b-2 (for example the legacy baseline) are reproduced.

**Held-out maps** (`held_out.maps`). Other 11x7 layouts from `configs/maps/` that the policy never saw in training
(training uses the map of `configs/ctf_v2.yaml`, which equals `configs/maps/default.yaml`). Only `obstacles`, the two
bases and the spawn cells are taken from the map file; rules, visibility, spawn mode and episode length stay those of
the policy's environment. A map with a different width, height or `agents_per_team` is a clear error, which is why
`eval_legacy.yaml` (3 agents) lists no maps. `held_out.opponents` picks which of the top-level opponents to play
(leave minimax out: it is slow). JSON: `held_out[map][opponent][all|red|blue]`, same fields as the main table.
Each map is marked SEEN (the checkpoint's own config trained on that layout: its base map or a `maps` pool of any phase,
compared by walls and bases) or UNSEEN; the report prints the two groups separately and the JSON has
`held_out_seen: {map: bool}`. Old checkpoints have no pools, so only their base layout is seen (docs/multimap.md).

**Role-swap sensitivity** (`role_swap`). The policy plays `role_swap.episodes` games per opponent. At every decision,
for each pair `[A, B]` in `swaps`, the network is asked again with the roles of the whole team swapped (agents playing A
are shown B and the other way round; BOTH the role one-hot in the input vector and the role-specific output head
change; what the agent sees and `self_can_carry` do not). The result is the share of decisions whose argmax action
changed (Wilson interval), per pair and overall. A policy that ignores its role scores 0, which is the only thing this
number can prove.

**Report only, no pass/fail.** An earlier version had a `min_change_rate` pass line. It was removed (the key is now an
unknown-key error in the eval YAML) because it could not fail: on the v2 architecture, three freshly initialised
networks scored 71 / 93 / 96, 77 / 58 / 84 and 93 / 86 / 63 % for attacker<->defender / attacker<->scout /
defender<->scout (overall 72-85%), and the Run 1 policy at 3.75M steps scored 62 / 74 / 40% (overall 60%). Training
does not push the number up, so it is neither a floor nor a target. It shows that the role input is *used*, not that it
is used *well*. Role specialisation is judged from the behaviour numbers that the evaluation already reports: the
defender's distance to its own flag and base, scout map coverage, per-role action histograms and position heatmaps.
No thresholds are set for those either: with one trained model and no untrained reference run there is no data to
justify a number, so they stay reported values (compare them between runs and against the scripted opponents).
(Measured on `configs/rl/eval.yaml`'s role-swap settings, 20 games per opponent, seeds 0-2 for the untrained networks.)

**Argmax vs sampled** (`policy_check`). For each listed opponent the same seeded games are played once with argmax and
once with sampled actions (sampling generator reseeded with `seed`). Reported: win/draw/loss for each mode, the
difference `argmax win - sampled win` with a Newcombe interval (treats the samples as independent; the games are paired,
so it is slightly conservative) and `top_action_prob`, the mean probability of the chosen-most-likely action
(1.0 = deterministic policy). A big gap means the demo (argmax) plays differently from what training measured.

## Noise-free evaluation during training (`noise_free_eval`)

`configs/rl/default.yaml` has a `noise_free_eval` section. When `enabled`, the trainer pauses after the update that
crosses each multiple of `every_steps` environment steps, plays the CURRENT policy through the same code as the CLI on
the small eval YAML named in `config_path` (`configs/rl/eval_train.yaml`: 20 games per opponent, no minimax, the two
UNSEEN maps `center_gate` and `long_wall_south`, a short role-swap check, policy check off) and writes:

* an event record `noise_free_eval` in `metrics.jsonl` (scalars `eval/win_rate/<opponent>`, `eval/mean_win_rate`,
  `eval/min_win_rate`, `eval/stay_fraction`, `eval_held_out/...`, `eval/role_swap_changed`, `eval/seconds`) and the same
  scalars to TensorBoard. They are event records on purpose: `viz.report` and every other reader of the update curve
  skip them;
* the full report as `<run>/eval/step_<steps>.json`;
* one console line starting with `--> noise-free eval`.

`at_end` adds one evaluation after the last update (after the final checkpoint is written). `overrides` changes values
of the eval YAML (for example `{episodes: 40}`). The eval seed is fixed, so successive points use the same games.
It does not disturb training: it uses its own random generators and restores the model's train/eval mode (a test
trains the same seed with and without it and compares the weights). If an evaluation raises, the run prints the error,
logs a `noise_free_eval_failed` event and keeps training. The eval YAML is loaded when the trainer is built, so a
wrong path or a team-size mismatch stops the run before any training. The same startup check builds every eval opponent
once (on the main evaluation's environments and, for each held-out map, on that map's environment with the opponents it
will play), so an opponent that could never work (`source: league`, which an evaluation cannot play because it has no
league; a missing or incompatible checkpoint file; a wrong role count; an invalid minimax or scripted parameter; an
unreadable or wrong-sized held-out map) raises a `ConfigError` naming the opponent right away. Without it the run would
only discover the problem at the first evaluation (step 500k) and fail again at every later one. The console line of an
evaluation shows win rates only; the role-swap numbers are in `metrics.jsonl` / TensorBoard as reported values.
Cost: one evaluation with `eval_train.yaml` took about 33 s on a single sandbox CPU core with an untrained 4-agent
network (full 200-step games); it was not timed on Colab. Raise `every_steps` or lower `overrides.episodes` if it
slows a run too much.
Old checkpoints have no such section: they load with the hook off (`SECTION_COMPAT` in `rl/checkpoint.py`); a resumed
run uses the setting of the YAML you pass.

## Opponents

`configs/rl/eval.yaml` lists them: random, stay, four scripted variants (leash, parked, chaser, escort), minimax and the
classical AI (`kind: classical`, roles fixed per slot; see `docs/classical.md` for what it does and does not know)
and `classical_fixed`, the same AI with its known problems fixed (switches in `params`).
Each entry is `{kind, label, params}`; `label` is the name used in reports and in `--opponents`.
The LLM agent has no hookup (it cannot run on Colab). To evaluate against one, register a factory and add an
entry with that kind to the eval YAML:

```python
from rl.evaluate import EXTERNAL_OPPONENTS
EXTERNAL_OPPONENTS["llm"] = lambda spec, env_config, rng: MyLLMStrategy(env_config, **spec.params)
```

The factory returns any `BaseStrategy`. An unregistered, non-built-in kind is a clear `ConfigError`.

## Files

* `src/rl/eval_config.py`  strict `EvalConfig` dataclasses (every key required).
* `src/rl/eval_stats.py`   Wilson interval, normal mean interval.
* `src/rl/evaluate.py`     episode runner (`play_episode`), recorder, `summarize`, `evaluate_policy`, held-out maps,
  role-swap and argmax-vs-sampled checks (`evaluate_extras`), report, CLI.
* `src/rl/train_eval.py`   the training hook (`NoiseFreeEvaluator`, `flatten_scalars`).
* `configs/rl/eval.yaml`, `configs/rl/eval_legacy.yaml`, `configs/rl/eval_train.yaml` (the in-training one)
* `tests/test_rl_evaluate.py`, `tests/test_rl_evaluate_checks.py`, `tests/test_rl_train_eval.py`
