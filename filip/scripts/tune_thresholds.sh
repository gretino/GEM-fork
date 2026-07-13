#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/home/qfbqt/repo/GEM-fork"

# Default values
CONFIG="filip/configs/ptbxl_diagnosis_adapt.yaml"
CHECKPOINT=""
GPU="0"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG="$2"; shift ;;
        -ckpt|--checkpoint) CHECKPOINT="$2"; shift ;;
        -g|--gpu) GPU="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$CHECKPOINT" ]]; then
    # Try to auto-detect best checkpoint
    EXPERIMENT_NAME=$(grep "experiment_name:" $CONFIG | awk '{print $2}')
    if [[ -z "$EXPERIMENT_NAME" ]]; then
        EXPERIMENT_NAME="ptbxl_diagnosis_adapt"
    fi
    
    CHECKPOINT="/outputs/filip/$EXPERIMENT_NAME/checkpoints/best.pt"
    if [[ ! -w "/outputs" ]]; then
        CHECKPOINT="outputs/filip/$EXPERIMENT_NAME/checkpoints/best.pt"
    fi
    
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "Error: Must provide --checkpoint path. Auto-detection failed: $CHECKPOINT not found."
        exit 1
    fi
    echo "Auto-detected checkpoint: $CHECKPOINT"
fi

echo "Running FILIP Threshold Tuning..."
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "GPU: $GPU"

CUDA_VISIBLE_DEVICES=$GPU python filip/eval/tune_thresholds.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT"
