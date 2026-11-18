# RL hand-off for the research documentation

For the teammate who keeps the research record (`research_documentation/`, written 16 Nov 2026). That record describes
the RL work as it was at commit `6653d95` (Run 1 at 8.0M steps, full evaluation not done). Since then the repository
moved on. This file says what is outdated, what to add, where each number comes from, and what must not be claimed.
Everything below is traceable to a file in this repository; nothing here is a new measurement.

## 1. What in the current record is outdated

| Statement in the record | Now |
|---|---|
| README: RL is "implemented and trained, evaluation incomplete"; the full `configs/rl/eval.yaml` has no committed output. | The full evaluation exists: `outputs/eval/v2_run1_6.0M.json`, `v2_run2_7.5M.json`, `v2_run2_9M.json` (200 games per opponent, seed 2026), plus `v2_run1_6.0M_minimax3.json` and `v2_run1_8.0M_heldout.json`. Log: `outputs/eval/review.log`. |
| `RL_MAPPO_IMPLEMENTATION.md`: `metrics.jsonl` stops at 7,000,064 although the checkpoint is at 8,001,536. | The file was incomplete when pushed (Colab Drive had not synced). `runs/v2_run1/metrics.jsonl` now ends at step 8,001,536. |
| The checkpoint of interest is `step_0008001536.pt`. | In the author's review the 8.0M checkpoint scored 70.5% against minimax depth 2 (200 games) against 100% at 6.0M; only the 6.0M file is in the repository (`v2_run1_6.0M.json`, `v2_run1_6.0M_minimax3.json`), so the 8.0M and 5.0M-7.5M minimax numbers are NOT traceable yet (ask the user to push those JSON/log files, otherwise leave them out). Run 2 (below) is the current model. |
| "Held-out `two_corridors` vs leash: 0 wins in 10 games, too small." | Superseded: the Run 1 policy won 36.9% of 160 games on `two_corridors` (table 3). |
| "Self-play: code is present, the run did not use it." | Still true for Run 1 AND Run 2. |
| "A scout information-sharing ablation was not run." | For RL it will not be run: the team's view is the union of all active teammates' views, which is a rule of the environment (`docs/environment.md`), not a switch of the policy. |
| One map layout. | Run 2 trains on a pool of five layouts and is tested on two layouts it never saw (`docs/multimap.md`). |

## 2. What to write for RL (suggested outline of the updated RL document)

1. Implementation (keep the existing sections 2 and 3: architecture and update rule; they still match the code).
2. Environment of the RL work: four agents per team, headcount tagging, carrier slots, 200-step cap (`configs/ctf_v2.yaml`, `docs/environment.md`). **These are different rules from the three-agent record**, so no RL number is comparable with the 252 LLM pairs (see section 5).
3. Run 1: `runs/v2_run1/` (config, six `run_info*.json`, `metrics.jsonl`, `eval/`), 8.0M steps (first start plus four resumes, one `run_info*.json` each). Figure `training_curves.png` shows the win rate staying near zero for about 3M steps (a "grab the flag and freeze" start; the cause is a hypothesis only) and then rising once the last curriculum phase starts.
4. Run 2: `runs/v2_run2_multimap/`, resumed from the Run 1 checkpoint at 6.0M steps, trained to 9.0M on a pool of five maps (weights in `configs/rl/default.yaml`), six launch records. Held out for evaluation only: `center_gate`, `long_wall_south`.
5. Evaluation protocol (`configs/rl/eval.yaml`): 200 games per opponent (100 as RED, 100 as BLUE), seed 2026, argmax actions, `near_base` spawns, visibility radius 3, Wilson 95% intervals; map evaluation 160 games per map (4 scripted opponents x 40).
6. Results: tables 2 and 3 below, the draw analysis, the figures of section 4.
7. Behaviour findings and limits (section 6 of this file).
8. Classical AI on the v2 rules: `docs/classical.md` (adapter, the fixes `classical_fixed`, smoke numbers marked as smoke).

## 3. Numbers (generated from the JSON files named above; D = draws, L = losses, interval = Wilson 95%)

Table 2: win rate per opponent, 200 games each, `near_base` spawns.

| Opponent | Run 1, 6.0M | Run 2, 7.5M | Run 2, 9.0M |
|---|---|---|---|
| random | 171/200 = 85.5% [80, 90] (D 29, L 0) | 185/200 = 92.5% [88, 95] (D 15, L 0) | 186/200 = 93.0% [89, 96] (D 14, L 0) |
| stay | 155/200 = 77.5% [71, 83] (D 45, L 0) | 171/200 = 85.5% [80, 90] (D 29, L 0) | 181/200 = 90.5% [86, 94] (D 19, L 0) |
| scripted_leash | 191/200 = 95.5% [92, 98] (D 8, L 1) | 194/200 = 97.0% [94, 99] (D 6, L 0) | 196/200 = 98.0% [95, 99] (D 4, L 0) |
| scripted_parked | 169/200 = 84.5% [79, 89] (D 31, L 0) | 175/200 = 87.5% [82, 91] (D 25, L 0) | 177/200 = 88.5% [83, 92] (D 23, L 0) |
| scripted_chaser | 194/200 = 97.0% [94, 99] (D 6, L 0) | 195/200 = 97.5% [94, 99] (D 5, L 0) | 200/200 = 100.0% [98, 100] (D 0, L 0) |
| scripted_escort | 190/200 = 95.0% [91, 97] (D 10, L 0) | 195/200 = 97.5% [94, 99] (D 5, L 0) | 196/200 = 98.0% [95, 99] (D 4, L 0) |
| minimax | 200/200 = 100.0% [98, 100] (D 0, L 0) | 200/200 = 100.0% [98, 100] (D 0, L 0) | 187/200 = 93.5% [89, 96] (D 13, L 0) |
| classical | 165/200 = 82.5% [77, 87] (D 35, L 0) | 175/200 = 87.5% [82, 91] (D 25, L 0) | 174/200 = 87.0% [82, 91] (D 26, L 0) |

Table 3: win rate per map, pooled over the four scripted opponents (160 games). "Seen/UNSEEN" is about Run 2's training pool; Run 1 trained on the default layout only, so for Run 1 every map except `default` was unseen.

| Map (seen by Run 2?) | Run 1, 6.0M | Run 2, 7.5M | Run 2, 9.0M |
|---|---|---|---|
| center_gate (UNSEEN) | 8/160 = 5.0% [3, 10] | 78/160 = 48.8% [41, 56] | 81/160 = 50.6% [43, 58] |
| center_pillar (seen) | 143/160 = 89.4% [84, 93] | 154/160 = 96.2% [92, 98] | 153/160 = 95.6% [91, 98] |
| long_wall_south (UNSEEN) | 0/160 = 0.0% [0, 2] | 0/160 = 0.0% [0, 2] | 0/160 = 0.0% [0, 2] |
| open_field (seen) | 72/160 = 45.0% [37, 53] | 140/160 = 87.5% [81, 92] | 145/160 = 90.6% [85, 94] |
| pillars (seen) | 83/160 = 51.9% [44, 59] | 141/160 = 88.1% [82, 92] | 147/160 = 91.9% [87, 95] |
| two_corridors (seen) | 59/160 = 36.9% [30, 45] | 152/160 = 95.0% [90, 97] | 154/160 = 96.2% [92, 98] |

Reading (interpretation, not a new test): in-distribution scripted opponents are near saturation for both runs; the large change is the maps (Run 1 to Run 2) and the minimax column (Run 2 at 9.0M is lower than at 7.5M, 93.5% [89, 96] against 100% [98, 100]). The two UNSEEN maps are the only transfer evidence: `center_gate` about half, `long_wall_south` zero in all three evaluations (the map has only a bottom gap; in the author's review Run 1 always took the top lane, which is not a file in the repo, and the cause for Run 2 is not measured). The classical column is the ORIGINAL classical team on the v2 rules.

## 4. Figures (three or four pictures are enough)

```
uv run python scripts/make_rl_plots.py --run "Run 1=runs/v2_run1" --run "Run 2 (from 6.0M)=runs/v2_run2_multimap" \
  --eval "Run 1 (6.0M)=outputs/eval/v2_run1_6.0M.json" --eval "Run 2 (9.0M)=outputs/eval/v2_run2_9M.json" --out outputs/plots
uv run python -m viz report --run runs/v2_run1 --out outputs/plots/run1_training_report.png
uv run python -m viz report --run runs/v2_run2_multimap --out outputs/plots/run2_training_report.png
```

| Picture | Shows | Use for |
|---|---|---|
| `training_curves.png` | Rolling training win rate and entropy, both runs | Training budget; the flat start of Run 1. Training win rate is NOT evaluation performance. |
| `eval_by_opponent.png` | Table 2 as bars with Wilson intervals | Main result. |
| `eval_held_out_maps.png` | Table 3 as bars, hatched = never trained on | Transfer and its limit. |
| `training_per_map.png` | Run 2 training win rate per pool map | No pool map is stuck. |
| `run*_training_report.png` | Eight panels per run (outcomes, returns, entropy, losses) | Appendix. |

More evaluation files can be added as extra `--eval LABEL=file.json` arguments (settings in `configs/plots/rl_plots.yaml`).

## 5. Comparability: what can and cannot be compared

* RL vs classical and RL vs scripted/minimax: all on the v2 rules, same seeds, same spawn setting. Comparable with each other.
* Classical vs LLM: three-agent rules, the 252 pairs of the record. Comparable with each other.
* RL vs LLM: not comparable (different rules, different opponents). The only bridge is the classical team, which appears on both rule sets; the record should say so and should not put RL and the LLM in one table. A three-way comparison needs the LLM commander ported to the v2 rules or an RL policy trained on the three-agent rules; neither is done.
* `classical` on the v2 rules is not the same measurement as the 600 classical matches of the record (different rules, a four-agent role table).

## 6. Do not claim

* Training win rate (0.94-0.95) as performance: the evaluation tables are the performance.
* That more training helped: Run 2 at 9.0M is lower against minimax than at 7.5M (table 2, both files in the repo), and the author's review saw the same for Run 1 between 6.5M and 8.0M (files not in the repo yet). The cause is untested.
* Transfer to the four maps Run 2 trained on; only `center_gate` and `long_wall_south` are unseen.
* That the scout is useful, or that role specialisation was shown from the role-swap rate (an untrained network can also score well on it). The share of STAY decisions of all agents in the in-training evaluations of Run 2 is logged as `eval/stay_fraction` in `metrics.jsonl` (0.46 at 7.0M, 0.19 at 9.0M); a per-role figure is not in any file.
* H1 (scout sharing), H2 (mixed role team beats pure teams), H3 (attacker plus escort vs a lone parked defender): not tested. A replacement for H1 is the eval-time visibility ablation (section 7).
* A number for the LLM commander on the v2 rules, or for RL on the three-agent rules.
* The cause of the draws against `stay`: two games were traced (`docs/classical.md`), a third was not.
* One training seed per run: no seed-to-seed variance exists.

## 7. Numbers still to come from the user's evaluation runs (heavy jobs, run on the Mac)

Not in the repository yet; leave them as "pending" until the JSON files exist: minimax depth 3 and `classical`/`classical_fixed` for Run 2 at 7.5M and 9.0M (block `outputs/eval/run_final.sh`), and which checkpoint is frozen. Visibility ablation on the frozen checkpoint (radius 3 is already in its full evaluation; this adds 6 and 20, where 20 means the whole 11x7 board):

```
CK=runs/v2_run2_multimap/checkpoints/step_0009000960.pt     # the frozen checkpoint
for R in 6 20; do
  uv run python -m rl.evaluate --checkpoint $CK --eval-config configs/rl/eval.yaml \
    --opponents scripted_leash scripted_parked scripted_chaser scripted_escort \
    --skip held_out role_swap policy_check --env-set visibility_radius=$R --out outputs/eval/r2_9M_radius$R.json
done
```

(Not run by the author of this file: it is evaluation work. It evaluates the radius-3 model at larger radii; it does not retrain.)

## 8. Files to copy into the research folder

`configs/ctf_v2.yaml`, `configs/rl/default.yaml`, `configs/rl/eval.yaml`, `configs/rl/eval_train.yaml`, `configs/maps/*.yaml`,
`outputs/eval/*.json` and `review.log`, `docs/multimap.md`, `docs/classical.md`, `docs/evaluation.md`, the PNGs of section 4, and
`runs/v2_run*/config*.yaml` and `run_info*.json` (not the checkpoints). Use the record's own evidence labels (experimental result,
implementation finding, committed training log, interpretation, planned work), and mark every statement of this file's section 3
as "experimental result" with the JSON file name.

## 9. Disclosure

The RL code, evaluation, plots and this note were written with AI assistance (Claude). Each team member must still be able
to explain their part in the viva; the report needs the AI-usage disclosure section.
