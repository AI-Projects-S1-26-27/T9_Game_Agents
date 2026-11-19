#!/bin/sh
set -e
export PYTHONUNBUFFERED=1
T0=$(date +%s)
bar() {
  case $1 in
    0) echo "[..........] 0/3" ;;
    1) echo "[###.......] 1/3" ;;
    2) echo "[#######...] 2/3" ;;
    3) echo "[##########] 3/3" ;;
  esac
}
stamp() { echo; echo "=== $(bar $1)  $2  (elapsed $(( $(date +%s) - T0 ))s) ==="; }

stamp 0 "6.0M full eval"
uv run python -m rl.evaluate --checkpoint runs/v2_run1/checkpoints/step_0006000640.pt --eval-config configs/rl/eval.yaml --out output2/v2_run1_6.0M.json

stamp 1 "8.0M held-out maps"
uv run python -m rl.evaluate --checkpoint runs/v2_run1/checkpoints/step_0008001536.pt --eval-config configs/rl/eval.yaml --opponents random --skip role_swap policy_check --out output2/v2_run1_8.0M_heldout.json

stamp 2 "6.0M vs minimax depth 3 (100 games)"
uv run python -m rl.evaluate --checkpoint runs/v2_run1/checkpoints/step_0006000640.pt --eval-config output2/eval_minimax3.yaml --opponents minimax --skip held_out role_swap policy_check --set episodes=100 --out output2/v2_run1_6.0M_minimax3.json

stamp 3 "ALL_DONE"
