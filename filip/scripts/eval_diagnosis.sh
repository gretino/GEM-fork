#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/home/qfbqt/repo/GEM-fork"

# Default values
CONFIG="filip/configs/ptbxl_diagnosis_adapt.yaml"
CHECKPOINT=""
SPLIT="test"
GPU="0"
THRESHOLDS_FILE=""
OUT_DIR=""
EXCLUDE_CLASSES=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG="$2"; shift ;;
        -ckpt|--checkpoint) CHECKPOINT="$2"; shift ;;
        -s|--split) SPLIT="$2"; shift ;;
        -g|--gpu) GPU="$2"; shift ;;
        -t|--thresholds_file) THRESHOLDS_FILE="$2"; shift ;;
        -o|--out_dir) OUT_DIR="$2"; shift ;;
        -e|--exclude_classes) EXCLUDE_CLASSES="$2"; shift ;;
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

echo "Running FILIP Evaluation..."
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Split: $SPLIT"
echo "GPU: $GPU"
if [[ -n "$THRESHOLDS_FILE" ]]; then
    echo "Thresholds File: $THRESHOLDS_FILE"
fi
if [[ -n "$OUT_DIR" ]]; then
    echo "Output Directory: $OUT_DIR"
fi
if [[ -n "$EXCLUDE_CLASSES" ]]; then
    echo "Exclude Classes: $EXCLUDE_CLASSES"
fi

ARGS=("--config" "$CONFIG" "--checkpoint" "$CHECKPOINT" "--split" "$SPLIT")
if [[ -n "$THRESHOLDS_FILE" ]]; then
    ARGS+=("--thresholds_file" "$THRESHOLDS_FILE")
fi
if [[ -n "$OUT_DIR" ]]; then
    ARGS+=("--out_dir" "$OUT_DIR")
fi
if [[ -n "$EXCLUDE_CLASSES" ]]; then
    ARGS+=("--exclude_classes" "$EXCLUDE_CLASSES")
fi

CUDA_VISIBLE_DEVICES=$GPU python filip/eval/evaluate_diagnosis.py "${ARGS[@]}"
