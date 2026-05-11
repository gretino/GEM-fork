#!/bin/bash
# Run all 4 QLoRA experiment variants sequentially.
# Each job gets its own log file under logs/.
# The script stops immediately if any experiment fails.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TRAIN_SCRIPT="$SCRIPT_DIR/train_gem_qlora.sh"
DATA_DIR="$REPO_ROOT/data/gem_data"
LOG_DIR="$REPO_ROOT/logs"

mkdir -p "$LOG_DIR"

# 4 experiment variants: (data_file, run_name)
declare -a EXPERIMENTS=(
    "gem_train_superclass_reasoning.json    superclass_reasoning"
    "gem_train_superclass_no_reasoning.json superclass_no_reasoning"
    "gem_train_subclass_reasoning.json      subclass_reasoning"
    "gem_train_subclass_no_reasoning.json   subclass_no_reasoning"
)

TOTAL=${#EXPERIMENTS[@]}
echo "========================================"
echo "Running $TOTAL QLoRA experiments"
echo "Logs will be written to: $LOG_DIR"
echo "========================================"

for i in "${!EXPERIMENTS[@]}"; do
    read -r DATA_FILE RUN_NAME <<< "${EXPERIMENTS[$i]}"
    DATA_PATH="$DATA_DIR/$DATA_FILE"
    LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
    EXP_NUM=$((i + 1))

    echo ""
    echo "[$EXP_NUM/$TOTAL] Starting experiment: $RUN_NAME"
    echo "  Data: $DATA_PATH"
    echo "  Log:  $LOG_FILE"
    echo "  Started at: $(date)"

    bash "$TRAIN_SCRIPT" "$DATA_PATH" "$RUN_NAME" > "$LOG_FILE" 2>&1

    echo "  Finished at: $(date)"
    echo "[$EXP_NUM/$TOTAL] DONE: $RUN_NAME"
done

echo ""
echo "========================================"
echo "All $TOTAL experiments completed successfully!"
echo "Checkpoints: $REPO_ROOT/checkpoints/"
echo "Logs:        $LOG_DIR/"
echo "========================================"
