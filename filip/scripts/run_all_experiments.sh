#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/home/qfbqt/repo/GEM-fork"

# Check for --resume argument
RESUME_ARG=""
if [[ "$1" == "--resume" || "$2" == "--resume" ]]; then
    RESUME_ARG="-r"
    echo "=========================================="
    echo "Resume enabled for checkpoints."
    echo "=========================================="
fi

# 1. Seed the existing 100% PTBXL-Super results to avoid retraining
echo "Seeding 100% PTB-XL Superclass results from existing baseline..."
mkdir -p outputs/filip/ptbxl_super_adapt_100/validation_tuning
mkdir -p outputs/filip/ptbxl_super_adapt_100/evaluation

SRC_DIR="outputs/filip/ptbxl_adapt_ablation_large_14"
if [ -d "$SRC_DIR" ]; then
    cp "$SRC_DIR/checkpoints/best.pt" "outputs/filip/ptbxl_super_adapt_100/best.pt" 2>/dev/null
    cp "$SRC_DIR/ptb-val-tuning/thresholds.json" "outputs/filip/ptbxl_super_adapt_100/validation_tuning/thresholds.json" 2>/dev/null
    cp "$SRC_DIR/ptb-test-tuned/metrics.txt" "outputs/filip/ptbxl_super_adapt_100/evaluation/metrics.txt" 2>/dev/null
    echo "Seeded successfully!"
else
    echo "Warning: Base PTB-XL Super directory $SRC_DIR not found. Seeding skipped."
fi

# 2. Define the tasks list: format name:config:pcts
TASKS=(
    "ptbxl_super_adapt:filip/configs/ptbxl_super_adapt.yaml:1 10"
    "ptbxl_sub_adapt:filip/configs/ptbxl_sub_adapt.yaml:1 10 100"
    "ptbxl_rhythm_adapt:filip/configs/ptbxl_rhythm_adapt.yaml:1 10 100"
    "ptbxl_form_adapt:filip/configs/ptbxl_form_adapt.yaml:1 10 100"
    "csn_adapt:filip/configs/csn_adapt_verified.yaml:1 10 100"
    "cpsc_adapt:filip/configs/cpsc_adapt_verified.yaml:1 10 100"
)

COUNTER_FILE="tasks_counter.txt"
echo 0 > "$COUNTER_FILE"

run_worker() {
    local gpu=$1
    while true; do
        # Acquire lock and read/increment counter
        exec 200>tasks_counter.lock
        flock 200
        
        idx=$(cat "$COUNTER_FILE")
        if [ "$idx" -ge "${#TASKS[@]}" ]; then
            flock -u 200
            break
        fi
        
        next_idx=$((idx + 1))
        echo "$next_idx" > "$COUNTER_FILE"
        
        flock -u 200
        
        # Parse the task details
        task_info="${TASKS[$idx]}"
        IFS=':' read -r name config pcts <<< "$task_info"
        
        echo "=================================================="
        echo "GPU $gpu is starting task: $name ($pcts)"
        echo "=================================================="
        
        # Execute the helper script
        bash filip/scripts/run_experiment_group.sh -c "$config" -n "$name" -g "$gpu" -p "$pcts" $RESUME_ARG
    done
}

# 3. Start parallel worker processes
echo "Spawning dynamic worker on GPU 0..."
run_worker 0 &
pid0=$!

echo "Spawning dynamic worker on GPU 1..."
run_worker 1 &
pid1=$!

# Wait for both worker pipelines to finish
wait $pid0
wait $pid1

# 4. Cleanup task coordination files
rm -f "$COUNTER_FILE" tasks_counter.lock

echo "All experiment groups finished execution!"

# 5. Generate metrics summary tables
echo "Generating final summary tables..."
python3 filip/scripts/generate_summary.py
