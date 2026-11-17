#!/bin/sh
# Prepares the live demo: (1) near_base copies of the map files, (2) fallback GIFs of the demo games.
# Usage (repo root):  sh scripts/demo_prepare.sh runs/v2_run2_multimap/checkpoints/step_0009000960.pt
# The viewer map files use spawn_mode: fixed (one start position per map); training and the evaluation used
# spawn_mode: near_base (random start cells near the base). The copies below use near_base so the demo shows the
# situation the numbers in the report are about. Output goes to outputs/viz/ (ignored by git).
set -e
CK="${1:?usage: sh scripts/demo_prepare.sh <checkpoint.pt>}"
SEED="${2:-0}"
MAPS=outputs/viz/demo_maps
GIFS=outputs/viz/demo_gifs
mkdir -p "$MAPS" "$GIFS"
for f in configs/maps/*.yaml; do
  sed 's/^spawn_mode: fixed/spawn_mode: near_base/' "$f" > "$MAPS/$(basename "$f")"
done
rec() {  # rec <map name> <blue controller> <gif name>
  uv run python -m viz record --map "$MAPS/$1.yaml" --checkpoint "$CK" --red RL --blue "$2" --seed "$SEED" \
    --out "outputs/viz/$3.json.gz" --gif "$GIFS/$3.gif"
}
rec default        "Minimax"        default_vs_minimax
rec default        "Classical AI (A*" default_vs_classical
rec open_field     "parked"         open_field_vs_parked
rec pillars        "Classical AI (A*" pillars_vs_classical
rec two_corridors  "Classical AI (A*" two_corridors_vs_classical
rec center_gate    "Classical AI (A*" center_gate_vs_classical_UNSEEN
rec long_wall_south "Classical AI (A*" long_wall_south_vs_classical_UNSEEN_FAILS
echo DEMO_PREPARED
