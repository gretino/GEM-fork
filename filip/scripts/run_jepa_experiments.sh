#!/bin/bash
# filip/scripts/run_jepa_experiments.sh
# Automates Phase 2 JEPA experiments on unused GPUs

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

RESUME_ARG=""
if [[ "$1" == "-r" || "$1" == "--resume" ]]; then
    RESUME_ARG="-r"
    echo "Resume enabled for checkpoints."
fi

# Define the tasks list: format name:config:pcts
TASKS=(
    "ptbxl_sub_adapt_jepa:filip/configs/ptbxl_sub_adapt_jepa.yaml:1 10 100"
    "ptbxl_form_adapt_jepa:filip/configs/ptbxl_form_adapt_jepa.yaml:1 10 100"
    "csn_adapt_jepa:filip/configs/csn_adapt_jepa.yaml:1 10 100"
)

COUNTER_FILE="tasks_counter_jepa.txt"
echo 0 > "$COUNTER_FILE"

# Detect unused/idle GPUs (Memory used < 1000 MiB and Utilization < 10%)
UNUSED_GPUS=()
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    for ((i=0; i<NUM_GPUS; i++)); do
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i 2>/dev/null || echo 99999)
        util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $i 2>/dev/null || echo 100)
        echo "GPU $i: Memory Used = ${mem}MiB, Utilization = ${util}%"
        if [ "$mem" -lt 1000 ] && [ "$util" -lt 10 ]; then
            UNUSED_GPUS+=($i)
        fi
    done
else
    echo "nvidia-smi not found. Defaulting to GPU 0."
    UNUSED_GPUS=(0)
fi

if [ ${#UNUSED_GPUS[@]} -eq 0 ]; then
    echo "WARNING: No completely idle/unused GPUs detected (Memory < 1GB and Utilization < 10%)."
    # Fallback to asking user or defaulting to the most free GPU
    # Let's find the GPU with the lowest memory usage
    best_gpu=0
    min_mem=999999
    if command -v nvidia-smi &> /dev/null; then
        for ((i=0; i<NUM_GPUS; i++)); do
            mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i 2>/dev/null || echo 999999)
            if [ "$mem" -lt "$min_mem" ]; then
                min_mem=$mem
                best_gpu=$i
            fi
        done
        echo "Defaulting to GPU $best_gpu (lowest memory usage: ${min_mem}MiB)"
        UNUSED_GPUS+=($best_gpu)
    else
        UNUSED_GPUS+=(0)
    fi
fi

echo "Starting experiments on GPU(s): ${UNUSED_GPUS[*]}"

run_worker() {
    local gpu=$1
    while true; do
        # Acquire lock and read/increment counter
        exec 200>tasks_counter_jepa.lock
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
        
        # Execute the helper script (this handles train -> tune -> eval)
        bash filip/scripts/run_experiment_group.sh -c "$config" -n "$name" -g "$gpu" -p "$pcts" $RESUME_ARG
    done
}

# Start worker process for each unused GPU
pids=()
for gpu in "${UNUSED_GPUS[@]}"; do
    echo "Spawning dynamic worker on GPU $gpu..."
    run_worker "$gpu" &
    pids+=($!)
done

# Wait for all worker pipelines to finish
for pid in "${pids[@]}"; do
    wait "$pid"
done

# Cleanup task coordination files
rm -f "$COUNTER_FILE" tasks_counter_jepa.lock

echo "================================================================"
echo "All JEPA Phase 2 experiments completed successfully."
echo "================================================================"
