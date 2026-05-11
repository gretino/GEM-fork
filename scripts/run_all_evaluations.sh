#!/bin/bash
# Run inference + metric evaluation for all 4 QLoRA experiment checkpoints.
# Phase 1: model_ecg_resume.py  -> writes JSONL predictions to /results/<run_name>/
# Phase 2: evaluate_gem.py      -> reads JSONL, prints and saves metrics
#
# Usage:
#   bash scripts/run_all_evaluations.sh
#   bash scripts/run_all_evaluations.sh --gpu 1        # choose a specific GPU (default: 0)
#
# The script stops immediately if any step fails (set -e).
set -e

# ---------------------------------------------------------------------------
# Paths — adjust these if your layout differs
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

CKPT_BASE="$REPO_ROOT/checkpoints"
DATA_DIR="$REPO_ROOT/data/gem_data"
IMAGE_FOLDER="$REPO_ROOT/data/ecg_images"
ECG_FOLDER="$REPO_ROOT/data/ecg_timeseries"
ECG_TOWER="/home/qfbqt/8TB/checkpoints/cpt_wfep_epoch_20.pt"

RESULTS_DIR="$REPO_ROOT/results"
LOG_DIR="$REPO_ROOT/logs"

# GPU to use for inference (override with --gpu N)
GPU=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ---------------------------------------------------------------------------
# Experiment definitions: (run_name, test_data_file, eval_track)
#   eval_track must be "superclass" or "subclass" (used by evaluate_gem.py)
# ---------------------------------------------------------------------------
declare -a EXPERIMENTS=(
    "superclass_reasoning_qlora   gem_test_superclass_reasoning.json   superclass"
    "superclass_no_reasoning_qlora gem_test_superclass_no_reasoning.json superclass"
    "subclass_reasoning_qlora     gem_test_subclass_reasoning.json     subclass"
    "subclass_no_reasoning_qlora  gem_test_subclass_no_reasoning.json  subclass"
)

TOTAL=${#EXPERIMENTS[@]}
echo "========================================================"
echo "  GEM QLoRA — Evaluation of $TOTAL checkpoints"
echo "  Results : $RESULTS_DIR"
echo "  GPU     : $GPU"
echo "========================================================"

for i in "${!EXPERIMENTS[@]}"; do
    read -r RUN_NAME TEST_FILE TRACK <<< "${EXPERIMENTS[$i]}"
    EXP_NUM=$((i + 1))

    CKPT_PATH="$CKPT_BASE/$RUN_NAME"
    TEST_PATH="$DATA_DIR/$TEST_FILE"
    OUT_DIR="$RESULTS_DIR/$RUN_NAME"
    ANSWERS_FILE="$OUT_DIR/predictions.jsonl"
    METRICS_FILE="$OUT_DIR/metrics.txt"
    LOG_FILE="$LOG_DIR/eval_${RUN_NAME}.log"

    mkdir -p "$OUT_DIR"

    echo ""
    echo "[$EXP_NUM/$TOTAL] ─── $RUN_NAME ───────────────────────────"
    echo "  Checkpoint : $CKPT_PATH"
    echo "  Test data  : $TEST_PATH"
    echo "  Track      : $TRACK"
    echo "  Predictions: $ANSWERS_FILE"
    echo "  Metrics    : $METRICS_FILE"
    echo "  Log        : $LOG_FILE"
    echo "  Started at : $(date)"

    # ------------------------------------------------------------------
    # Phase 1 — Inference
    # ------------------------------------------------------------------
    echo "  [Phase 1] Running inference..."
    CUDA_VISIBLE_DEVICES=$GPU python "$REPO_ROOT/llava/eval/model_ecg_resume.py" \
        --model-path "$CKPT_PATH" \
        --model-base "LANSG/GEM" \
        --image-folder "$IMAGE_FOLDER" \
        --ecg-folder "$ECG_FOLDER" \
        --question-file "$TEST_PATH" \
        --answers-file "$ANSWERS_FILE" \
        --conv-mode "llava_v1" \
        --ecg_tower "$ECG_TOWER" \
        --open_clip_config "coca_ViT-B-32" \
        --temperature 0.0 \
        --num_beams 1 \
        --max_new_tokens 512 \
        2>&1 | tee "$LOG_FILE"

    # ------------------------------------------------------------------
    # Phase 2 — Metric computation
    # ------------------------------------------------------------------
    echo "  [Phase 2] Computing metrics..."
    python "$SCRIPT_DIR/evaluate_gem.py" \
        --results_file "$ANSWERS_FILE" \
        --track "$TRACK" \
        2>&1 | tee "$METRICS_FILE"

    echo "  Finished at: $(date)"
    echo "[$EXP_NUM/$TOTAL] DONE: $RUN_NAME"
    echo "  Metrics saved to: $METRICS_FILE"
done

echo ""
echo "========================================================"
echo "All $TOTAL evaluations completed!"
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Quick summary:"
for i in "${!EXPERIMENTS[@]}"; do
    read -r RUN_NAME TEST_FILE TRACK <<< "${EXPERIMENTS[$i]}"
    METRICS_FILE="$RESULTS_DIR/$RUN_NAME/metrics.txt"
    echo ""
    echo "  ── $RUN_NAME ──"
    if [[ -f "$METRICS_FILE" ]]; then
        cat "$METRICS_FILE"
    else
        echo "  (metrics file not found)"
    fi
done
echo "========================================================"
