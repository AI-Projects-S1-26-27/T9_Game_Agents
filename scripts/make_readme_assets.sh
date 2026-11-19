#!/bin/sh
# Rebuilds every figure, screenshot and GIF of README.md from committed files (no training, no evaluation).
# Usage (repo root):  sh scripts/make_readme_assets.sh runs/v2_run2_multimap/checkpoints/step_0009000960.pt
# Needs the rl and viz extras. Set PYTHON to change the interpreter (default: "uv run python").
# Intermediate recordings go to outputs/viz/readme/ (ignored by git); the images go to docs/img/.
set -e
CK="${1:?usage: sh scripts/make_readme_assets.sh <checkpoint.pt>}"
PY="${PYTHON:-uv run python}"
IMG=docs/img
TMP=outputs/viz/readme
SMALL="--set export.width=960 --set export.height=565"   # GIFs: smaller window, lighter files
mkdir -p "$IMG" "$TMP/maps"

# 1. training curves, per-map curves, per-opponent and per-map evaluation bars
$PY scripts/make_rl_plots.py \
  --run "Run 1 (single map)=runs/v2_run1" --run "Run 2 (5 maps)=runs/v2_run2_multimap" --run "Run 3 (all 7 maps)=runs/v2_run3_allmaps" \
  --eval "Run 1 (6.0M)=outputs/eval/v2_run1_6.0M.json" --eval "Run 2 (9.0M)=outputs/eval/v2_run2_9M.json" --out "$IMG"

# 2. map gallery and the report's experiments
$PY scripts/make_readme_assets.py --out "$IMG"

# 3. UI screenshots and GIFs: near-base copies of the maps (what training and evaluation used), seed 0, RL = RED
for f in configs/maps/*.yaml; do
  sed 's/^spawn_mode: fixed/spawn_mode: near_base/' "$f" > "$TMP/maps/$(basename "$f")"
done
rec() {  # rec <name> <map> <blue controller> [extra args]
  name="$1"; map="$2"; blue="$3"; shift 3
  $PY -m viz record --map "$TMP/maps/$map.yaml" --checkpoint "$CK" --red RL --blue "$blue" --seed 0 --out "$TMP/$name.json.gz" "$@"
}
rec d01 default "Minimax" --gif "$IMG/demo_rl_vs_minimax.gif" $SMALL
rec d02 default "Classical AI (A*" --gif "$IMG/demo_rl_vs_classical.gif" $SMALL
rec d09 center_gate "Classical AI (A*" --gif "$IMG/demo_unseen_center_gate.gif" $SMALL
rec d10 long_wall_south "Classical AI (A*"
rec d01_full default "Minimax"
$PY -m viz export "$TMP/d01_full.json.gz" --png "$IMG/viewer_rl_vs_minimax.png" --frame 20
$PY -m viz export "$TMP/d10.json.gz" --png "$IMG/viewer_failure_long_wall_south.png" --frame 60
echo README_ASSETS_DONE
