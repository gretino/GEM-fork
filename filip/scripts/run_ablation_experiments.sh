#!/bin/bash
# run_ablation_experiments.sh
# Automates the 3 ablation studies on the FILIP model across multiple GPUs

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

RESUME_ARG=""
if [[ "$1" == "-r" || "$1" == "--resume" ]]; then
    RESUME_ARG="-r"
    echo "Resume enabled for checkpoints."
fi

echo "================================================================"
echo "Starting FILIP Ablation Studies in Parallel"
echo "================================================================"

# Define the tasks list: format name:config:pcts
TASKS=(
    #"ptbxl_sub_adapt_enrich:filip/configs/ptbxl_sub_adapt_enrich.yaml:1 10 100"
    "ptbxl_sub_adapt_asl:filip/configs/ptbxl_sub_adapt_asl.yaml:1 10 100"
    #"ptbxl_sub_adapt_enrich_asl:filip/configs/ptbxl_sub_adapt_enrich_asl.yaml:1 10 100"
)

COUNTER_FILE="tasks_counter_ablation.txt"
echo 0 > "$COUNTER_FILE"

run_worker() {
    local gpu=$1
    while true; do
        # Acquire lock and read/increment counter
        exec 200>tasks_counter_ablation.lock
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

# Start parallel worker processes
echo "Spawning dynamic worker on GPU 0..."
run_worker 0 &
pid0=$!

echo "Spawning dynamic worker on GPU 1..."
run_worker 1 &
pid1=$!

# Wait for both worker pipelines to finish
wait $pid0
wait $pid1

# Cleanup task coordination files
rm -f "$COUNTER_FILE" tasks_counter_ablation.lock

echo "================================================================"
echo "All ablation studies completed successfully."
echo "================================================================"
