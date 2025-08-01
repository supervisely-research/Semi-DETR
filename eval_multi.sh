#!/bin/bash

set -e  # Exit on first error

# Define checkpoint paths
checkpoints=(
    "save/iter_5000.pth"
    "save/iter_10000.pth"
    "iter_35000.pth"
    "iter_40000.pth"
    "iter_45000.pth"
    "iter_50000.pth"
    # "iter_55000.pth"
)

# Base path for checkpoints
base_path="output/kanal75/split_400_2500/run_1"

# Loop through each checkpoint
for checkpoint in "${checkpoints[@]}"; do
    echo "Evaluating checkpoint: $checkpoint"
    python tools/test.py \
        configs/detr_ssod/detr_ssod_dino_detr_r50_kanal75.py \
        "$base_path/$checkpoint" \
        --eval bbox \
        --work-dir output/kanal75/split_400_2500/run_1/eval
done

echo "All evaluations completed successfully!"