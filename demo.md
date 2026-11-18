# Demo checklist (live demo + viva)

Everything below was run headless in a Linux sandbox. The Pygame window itself was never opened by the author of this
file: do the check list in section 2 on your own Mac before the demo.

## 1. Prepare (about 5-10 minutes, Mac, repo root)

```
git pull
sh scripts/demo_prepare.sh runs/v2_run2_multimap/checkpoints/step_0009000960.pt     # use the checkpoint you freeze
```

This writes `outputs/viz/demo_maps/*.yaml` (the map files with `spawn_mode: near_base`) and seven fallback GIFs in
`outputs/viz/demo_gifs/` (seed 0). Why the copies: the files in `configs/maps/` use fixed start cells, but the policy was
trained and evaluated with random start cells near the base, so the demo and the report numbers should use the same.
If the script stops, the GIFs made so far are still there. If the demo laptop fails, show the GIFs.

## 2. Live-window check list (do it once before the demo)

* `uv run python -m viz play --map outputs/viz/demo_maps/default.yaml --checkpoint <ckpt> --red RL --blue Minimax`
* The window is 1360x800 by default. On a smaller screen add `--set window.width=1200 --set window.height=720` to the command. The font is `arial` (falls back to pygame's default if missing); check that the text is readable and no glyphs are missing.
* Keys: `space` play/pause, `left`/`right` step, `up`/`down` speed, `f` fog, `v` vision, `tab` view team, `n`/`m` select agent, `r` new episode, `return` new seed, `w` swap sides, `F1`/`F2` change the red/blue controller, `h` help.
* Escape closes it. Have the GIFs open in another window as the fallback.

## 3. Demo order and what to say

| # | Command (map copy, blue controller) | What it shows |
|---|---|---|
| 1 | `default`, `--blue Minimax` | Roles at work: attackers pair up, defender stays home. |
| 2 | `open_field`, `--blue parked` | A map the old single-map model failed on (45% then, 90% now: seen in training). |
| 3 | `pillars`, `--blue "Classical AI (A*"` | Pair entry and carrying the flag home with an escort. |
| 4 | `center_gate`, `--blue "Classical AI (A*"` | A map it never trained on: about half the games are won. |
| 5 | `long_wall_south`, `--blue "Classical AI (A*"` | The honest failure: only a bottom gap, the policy takes the top lane, 0% in the evaluation. |

Seeds (sandbox, 9.0M checkpoint, RED = RL, `near_base`, seeds 0-5, one game each; a draw is a 200-step timeout):
default vs Minimax 6 of 6 won (27-40 steps); open_field vs parked 4 of 6; open_field vs classical 4 of 6; pillars vs classical 4 of 6;
two_corridors vs classical 5 of 6; center_gate vs classical 3 of 6; long_wall_south vs classical 0 of 6. Seed 0 won in every case except
long_wall_south. These are 6 games each, not the evaluation: quote the 200-game tables from `outputs/eval/` in the report. If you freeze another
checkpoint, rerun `scripts/demo_prepare.sh` and look at the GIFs before choosing seeds.

Do NOT show `Classical AI, fixed` as a win. In the sandbox the RL policy (RED) drew all 6 seeds against it on `default`
(its pair interception kills the RL attackers and the RL team does not attack again). The 200-game number for it comes
from `outputs/eval/*classical*.json`; use that, whatever it says.

## 4. Honest limits to say yourself (a viva question will find them anyway)

* One training seed per run; no league. Run 1 got worse against minimax after 6.5M steps, so more steps is not always better.
* Unseen maps: `center_gate` about half, `long_wall_south` 0%. The other four maps are training maps, so they are not transfer evidence.
* Draws against `stay`/parked teams: a stationary agent that spawns on its own flag cell tags every lone carrier (4 of 21 near-base spawn layouts); only two attackers entering together get in.
* The scout often stands still; scout sharing is a game rule (the team view is the union of all agents' views), not a tested hypothesis.
* No LLM commander on this branch.

## 5. Viva: the multi-map design in four lines

* Why a pool of maps: the single-map policy always took the top lane and scored 0% on a map with only a bottom gap.
* Why mirror symmetry: BLUE is mirrored so one policy plays both sides; that only works if the map looks the same from both sides (`maps.py` checks it).
* Why a held-out split: `center_gate` and `long_wall_south` are never trained on, so they measure transfer; the checks in `validate_map_pools` stop a held-out map from entering the pool.
* Why one RNG draw rule: the map is drawn in `reset()` only when the pool has more than one map, so old single-map seeds reproduce exactly.
