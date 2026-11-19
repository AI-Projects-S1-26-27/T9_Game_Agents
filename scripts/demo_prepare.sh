#!/bin/sh
# Prepares the live demo: (1) near_base copies of the map files, (2) one fallback GIF per demo of docs/demo.md.
# Usage (repo root):  sh scripts/demo_prepare.sh runs/v2_run2_multimap/checkpoints/step_0009000960.pt
# The viewer map files use spawn_mode: fixed (one start position per map); training and the evaluation used
# spawn_mode: near_base (random start cells near the base). The copies below use near_base so the demo shows the
# situation the numbers in the report are about. Output goes to outputs/viz/ (ignored by git).
# The demo table below (name, map, red, blue, seed) is the same table as in docs/demo.md.
set -e
CK="${1:?usage: sh scripts/demo_prepare.sh <checkpoint.pt>}"
MAPS=outputs/viz/demo_maps
GIFS=outputs/viz/demo_gifs
mkdir -p "$MAPS" "$GIFS"
for f in configs/maps/*.yaml; do
  sed 's/^spawn_mode: fixed/spawn_mode: near_base/' "$f" > "$MAPS/$(basename "$f")"
done
rec() {  # rec <name> <map> <red> <blue> <seed>
  uv run python -m viz record --map "$MAPS/$2.yaml" --checkpoint "$CK" --red "$3" --blue "$4" --seed "$5" \
    --out "outputs/viz/$1.json.gz" --gif "$GIFS/$1.gif"
}
rec d01_default_vs_minimax              default          RL                    "Minimax"             0
rec d02_default_vs_classical            default          RL                    "Classical AI (A*"    0
rec d03_default_vs_classical_fixed      default          RL                    "Classical AI, fixed" 0
rec d04_classical_fixed_red_vs_rl_blue  default          "Classical AI, fixed" RL                    0
rec d05_default_vs_stay_draw            default          RL                    "Stay still"          2
rec d06_open_field_vs_parked            open_field       RL                    "parked"              0
rec d07_pillars_vs_classical            pillars          RL                    "Classical AI (A*"    0
rec d08_two_corridors_vs_classical      two_corridors    RL                    "Classical AI (A*"    0
rec d09_center_gate_UNSEEN_vs_classical center_gate      RL                    "Classical AI (A*"    0
rec d10_long_wall_south_UNSEEN_fails    long_wall_south  RL                    "Classical AI (A*"    0
echo DEMO_PREPARED
