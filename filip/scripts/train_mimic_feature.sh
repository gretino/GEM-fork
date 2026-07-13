#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/home/qfbqt/repo/GEM-fork"

# Default values
CONFIG="filip/configs/mimic_feature_pretrain.yaml"
GPU="0"
RESUME_FROM=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG="$2"; shift ;;
        -g|--gpu) GPU="$2"; shift ;;
        -r|--resume_from) RESUME_FROM="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Running Stage 1 pretraining..."
echo "Config: $CONFIG"
echo "GPU: $GPU"

ARGS=("--config" "$CONFIG")
if [[ -n "$RESUME_FROM" ]]; then
    ARGS+=("--resume_from" "$RESUME_FROM")
fi

CUDA_VISIBLE_DEVICES=$GPU python filip/train/train_mimic_feature.py "${ARGS[@]}"
