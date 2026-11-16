#!/bin/sh
set -x
uv run python -m rl.evaluate --checkpoint runs/v2_run1/checkpoints/step_0006000640.pt --eval-config configs/rl/eval.yaml --out outputs/eval/v2_run1_6.0M.json
uv run python -m rl.evaluate --checkpoint runs/v2_run1/checkpoints/step_0008001536.pt --eval-config configs/rl/eval.yaml --opponents random --skip role_swap policy_check --out outputs/eval/v2_run1_8.0M_heldout.json
uv run python -m rl.evaluate --checkpoint runs/v2_run1/checkpoints/step_0006000640.pt --eval-config outputs/eval/eval_minimax3.yaml --opponents minimax --skip held_out role_swap policy_check --set episodes=100 --out outputs/eval/v2_run1_6.0M_minimax3.json
echo ALL_DONE
