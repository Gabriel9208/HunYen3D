#!/bin/bash
# Stage-1 capacity gate: overfit the single deterministic train mesh per cell, then eval it (mesh metrics).
# Detached run:  setsid nohup bash scripts/run_grid_overfit.sh > outputs/grid_overfit_stage1.log 2>&1 &
set -u
cd /home/gabriel/dev/HunYen3D
CELLS="A_lat2048_l4_8 B_lat2048_l6_12 C_lat4096_l4_8 D_lat4096_l6_12"

for cell in $CELLS; do
  echo "############## OVERFIT TRAIN: $cell ##############  $(date)"
  uv run python main.py +experiment=grid_overfit capacity=$cell wandb.mode=disabled || { echo "TRAIN FAILED: $cell"; continue; }

  ckpt=$(ls -t outputs/grid_overfit/$cell/*/checkpoints/last.pt 2>/dev/null | head -1)
  echo "############## OVERFIT EVAL:  $cell  (ckpt=$ckpt) ##############"
  uv run python scripts/evaluate.py +experiment=grid_overfit capacity=$cell \
    +ckpt=$ckpt +shapes=1 +viz=true +name=overfit_$cell wandb.mode=disabled \
    2>&1 | grep -iE 'chamfer|f-score|normal|V-IoU|S-IoU|no zero-crossing'
done
echo "############## STAGE 1 DONE ##############  $(date)"
