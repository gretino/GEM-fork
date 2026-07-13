#!/bin/bash
# Run inference + metric evaluation for PTB-XL superclass evaluation on LANSG_GEM model.
# Phase 1: model_ecg_resume.py  -> writes JSONL predictions to results/LANSG_GEM/ptb-test/
# Phase 2: evaluate_gem.py      -> reads JSONL, prints and saves metrics
# Usage:
#   bash scripts/run_ptbxl_super_gem.sh
#   bash scripts/run_ptbxl_super_gem.sh --gpu 1
#   bash scripts/run_ptbxl_super_gem.sh --output /custom/path/to/results
#
set -e

# Setup environment variables for protobuf and library paths
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
if [[ -n "$CONDA_PREFIX" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi


# Determine directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Handle execution from workspace root or scripts/ directory
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
    REPO_ROOT="$(dirname "$SCRIPT_DIR")"
else
    REPO_ROOT="$SCRIPT_DIR"
    SCRIPT_DIR="$REPO_ROOT/scripts"
fi

MODEL_PATH="/home/qfbqt/8TB/checkpoints/LANSG_GEM"
QUESTION_FILE="$REPO_ROOT/data/ecg_bench/ptb-test.json"
IMAGE_FOLDER="$REPO_ROOT/data/ecg_images/ptb-xl-gen"
ECG_FOLDER="$REPO_ROOT/data/ecg_timeseries/ptbxl"

SAVE_DIR=""
GPU=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --output|--save-dir) SAVE_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Set default save directory if not provided
if [[ -z "$SAVE_DIR" ]]; then
    SAVE_DIR="$REPO_ROOT/results/LANSG_GEM/ptb-test"
fi

ANSWERS_FILE="${SAVE_DIR}/step-final.jsonl"
METRICS_FILE="${SAVE_DIR}/metrics.txt"
LOG_FILE="$REPO_ROOT/logs/eval_LANSG_GEM_ptbxl.log"


mkdir -p "$SAVE_DIR" "$REPO_ROOT/logs"

echo "========================================================"
echo "  PTB-XL Superclass Evaluation on LANSG_GEM"
echo "  Model Path  : $MODEL_PATH"
echo "  Test Data   : $QUESTION_FILE"
echo "  Results Dir : $SAVE_DIR"
echo "  GPU         : $GPU"
echo "========================================================"

# ------------------------------------------------------------------
# Phase 1 — Inference
# ------------------------------------------------------------------
echo "  [Phase 1] Running inference..."
CUDA_VISIBLE_DEVICES=$GPU python "$REPO_ROOT/llava/eval/model_ecg_resume.py" \
  --model-path "$MODEL_PATH" \
  --image-folder "$IMAGE_FOLDER" \
  --question-file "$QUESTION_FILE" \
  --answers-file "$ANSWERS_FILE" \
  --conv-mode "llava_v1" \
  --ecg-folder "$ECG_FOLDER" \
  --open_clip_config "coca_ViT-B-32" \
  --temperature 0 \
  --num_beams 1 \
  --max_new_tokens 1024 \
  2>&1 | tee "$LOG_FILE"

# ------------------------------------------------------------------
# Phase 2 — Metric computation
# ------------------------------------------------------------------
echo ""
echo "  [Phase 2] Computing metrics..."
python "$SCRIPT_DIR/evaluate_gem.py" \
  --results_file "$ANSWERS_FILE" \
  --track "superclass" \
  --test_file "$QUESTION_FILE" \
  2>&1 | tee "$METRICS_FILE"

echo "========================================================"
echo "Evaluation completed!"
echo "Predictions: $ANSWERS_FILE"
echo "Metrics saved to: $METRICS_FILE"
echo "========================================================"
