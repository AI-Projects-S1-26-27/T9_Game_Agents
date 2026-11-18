# Demo checklist (live demo + viva)

Everything below was run headless in a Linux sandbox. The Pygame window itself was never opened by the author of this
file: do the check list in section 2 on your own Mac before the demo. The demo does not have to show RL winning: it
shows what the policy does, including where it fails. Losing to the classical team is a result, not a problem.

## 1. Prepare (about 5-10 minutes, Mac, repo root)

```
git pull
sh scripts/demo_prepare.sh runs/v2_run2_multimap/checkpoints/step_0009000960.pt     # use the checkpoint you freeze
```

This writes `outputs/viz/demo_maps/*.yaml` (the map files with `spawn_mode: near_base`) and one fallback GIF per demo in
`outputs/viz/demo_gifs/` (ten GIFs, seeds as in the table). Why the copies: the files in `configs/maps/` use fixed start
cells, but the policy was trained and evaluated with random start cells near the base, so the demo and the report
numbers should use the same. If the script stops, the GIFs made so far are still there. If the laptop fails, show the GIFs.

## 2. Live-window check list (do it once before the demo)

* The window is 1360x800 by default. On a smaller screen add `--set window.width=1200 --set window.height=720` to any command below.
  The font is `arial` (falls back to pygame's default if missing); check that the text is readable and no glyphs are missing.
* Keys: `space` play/pause, `left`/`right` step, `up`/`down` speed, `f` fog, `v` vision, `tab` view team, `n`/`m` select agent, `r` new episode, `return` new seed, `w` swap sides, `F1`/`F2` change the red/blue controller, `h` help, `escape` closes.
* Keep the GIF folder open in another window as the fallback.

## 3. The demos, one command each

Set the checkpoint once per terminal (use the one you freeze):

```
CK=runs/v2_run2_multimap/checkpoints/step_0009000960.pt
```

The seed and the outcome are what the sandbox produced with the 9.0M checkpoint, RED = RL, `near_base` spawns, one game each.
A draw is a 200-step timeout. If you freeze another checkpoint, run `scripts/demo_prepare.sh` again and look at the GIFs first.

| # | Command | Seed result | What to point at |
|---|---|---|---|
| 1 | `uv run python -m viz play --map outputs/viz/demo_maps/default.yaml --checkpoint $CK --red RL --blue Minimax --seed 0` | RL wins | Two attackers go in together, the defender stays home, the scout looks. |
| 2 | `uv run python -m viz play --map outputs/viz/demo_maps/default.yaml --checkpoint $CK --red RL --blue "Classical AI (A*" --seed 0` | RL wins (30 steps) | Same map against the original classical team. |
| 3 | `uv run python -m viz play --map outputs/viz/demo_maps/default.yaml --checkpoint $CK --red RL --blue "Classical AI, fixed" --seed 0` | draw | The fixed classical team (pair interception) tags both RL carriers; the RL team never attacks again. A weakness, shown openly. |
| 4 | `uv run python -m viz play --map outputs/viz/demo_maps/default.yaml --checkpoint $CK --red "Classical AI, fixed" --blue RL --seed 0` | RL (BLUE) wins | The same two teams with the sides swapped: one seed is not a win rate. |
| 5 | `uv run python -m viz play --map outputs/viz/demo_maps/default.yaml --checkpoint $CK --red RL --blue "Stay still" --seed 2` | draw | Even a team that never moves can draw: both RL attackers walk, one after the other and alone, onto the cell of a stationary agent and are tagged (headcount rule). |
| 6 | `uv run python -m viz play --map outputs/viz/demo_maps/open_field.yaml --checkpoint $CK --red RL --blue parked --seed 0` | RL wins | Open map against a defender parked on the flag (the old single-map model won 45% on this map in the evaluation, this one 90%; the map is a training map now). |
| 7 | `uv run python -m viz play --map outputs/viz/demo_maps/pillars.yaml --checkpoint $CK --red RL --blue "Classical AI (A*" --seed 0` | RL wins | A trained-on layout with obstacles. |
| 8 | `uv run python -m viz play --map outputs/viz/demo_maps/two_corridors.yaml --checkpoint $CK --red RL --blue "Classical AI (A*" --seed 0` | RL wins | Two lanes; the multi-map training pool at work. |
| 9 | `uv run python -m viz play --map outputs/viz/demo_maps/center_gate.yaml --checkpoint $CK --red RL --blue "Classical AI (A*" --seed 0` | RL wins (20 steps) | A map it never trained on. In the 160-game evaluation it wins about half, so this seed is a good one, not typical. |
| 10 | `uv run python -m viz play --map outputs/viz/demo_maps/long_wall_south.yaml --checkpoint $CK --red RL --blue "Classical AI (A*" --seed 0` | draw | The failure case: only a bottom gap, the policy heads for the top lane. 0% in the evaluation. |

Live tips: press `w` to swap sides and `return` for a new seed; `F2` changes the blue controller without restarting.
These are single games. Quote the 200-game tables in `outputs/eval/` in the report, not a demo seed.

Replay of a recording made by `demo_prepare.sh` (no model needed): `uv run python -m viz replay outputs/viz/d03_default_vs_classical_fixed.json.gz`.

## 4. Graphs (three commands, then a folder of PNGs)

```
uv run python -m viz report --run runs/v2_run1 --out outputs/plots/run1_training_report.png
uv run python -m viz report --run runs/v2_run2_multimap --out outputs/plots/run2_training_report.png
uv run python scripts/make_rl_plots.py --run "Run 1=runs/v2_run1" --run "Run 2 (from 6.0M)=runs/v2_run2_multimap" --eval "Run 1 (6.0M)=outputs/eval/v2_run1_6.0M.json" --eval "Run 2 (9.0M)=outputs/eval/v2_run2_9M.json" --out outputs/plots
```

`make_rl_plots.py` writes `training_curves.png` (win rate and entropy over steps, both runs: shows the flat start of Run 1, then
the climb, then Run 2 continuing), `training_per_map.png` (Run 2 win rate on each pool map), `eval_by_opponent.png` (Wilson intervals),
`eval_held_out_maps.png` (seen vs never-trained-on maps). Settings are in `configs/plots/rl_plots.yaml`. Add more `--eval LABEL=file.json`
arguments for the other checkpoints, minimax depth 3 and the `classical_fixed` evaluation once those files exist.
The two `viz report` pictures are dense (eight panels each); for slides use the three pictures of `make_rl_plots.py`.

Learning over training time, as a video (Run 1 has two checkpoints in the repo: 6.0M and 8.0M):
`uv run python -m viz timelapse --run runs/v2_run1 --count 3 --opponent parked --map outputs/viz/demo_maps/default.yaml --seed 0 --out outputs/viz/run1_timelapse.mp4`

## 5. Honest limits to say yourself (a viva question will find them anyway)

* One training seed per run; no league. Run 1 got worse against minimax after 6.5M steps, so more steps is not always better.
* Unseen maps: `center_gate` about half, `long_wall_south` 0%. The other four maps are training maps, so they are not transfer evidence.
* Draws against `stay`/parked teams: traced in two games so far, see `docs/classical.md` (a stationary agent on its flag cell, or a stationary agent on the carrier's path, tags a lone carrier). A third traced cause may exist.
* The scout often stands still; scout sharing is a game rule (the team view is the union of all agents' views), not a tested hypothesis.
* No LLM commander on this branch.

## 6. Viva: the multi-map design in four lines

* Why a pool of maps: the single-map policy always took the top lane and scored 0% on a map with only a bottom gap.
* Why mirror symmetry: BLUE is mirrored so one policy plays both sides; that only works if the map looks the same from both sides (`maps.py` checks it).
* Why a held-out split: `center_gate` and `long_wall_south` are never trained on, so they measure transfer; the checks in `validate_map_pools` stop a held-out map from entering the pool.
* Why one RNG draw rule: the map is drawn in `reset()` only when the pool has more than one map, so old single-map seeds reproduce exactly.
