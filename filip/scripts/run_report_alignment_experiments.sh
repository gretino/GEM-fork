#!/bin/bash
# filip/scripts/run_report_alignment_experiments.sh
# Automates downstream adaptation experiments for report alignment pretraining across multiple GPUs

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

RESUME_ARG=""
SKIP_ARG="true"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--resume) RESUME_ARG="-r"; shift ;;
        -s|--skip_completed) SKIP_ARG="$2"; shift 2 ;;
        --no_skip|--no-skip) SKIP_ARG="false"; shift ;;
        *) shift ;;
    esac
done

if [[ -n "$RESUME_ARG" ]]; then
    echo "Resume enabled for checkpoints."
fi
echo "Skip completed experiments: $SKIP_ARG"


# Define the downstream adaptation tasks list: format name:config:pcts
TASKS=(
    "ptbxl_super_report_align_adapt:filip/configs/report_alignment_adapt/ptbxl_super_adapt.yaml:1 10 100"
    "ptbxl_sub_report_align_adapt:filip/configs/report_alignment_adapt/ptbxl_sub_adapt.yaml:1 10 100"
    "ptbxl_rhythm_report_align_adapt:filip/configs/report_alignment_adapt/ptbxl_rhythm_adapt.yaml:1 10 100"
    "ptbxl_form_report_align_adapt:filip/configs/report_alignment_adapt/ptbxl_form_adapt.yaml:1 10 100"
    "cpsc_report_align_adapt:filip/configs/report_alignment_adapt/cpsc_adapt.yaml:1 10 100"
    "csn_report_align_adapt:filip/configs/report_alignment_adapt/csn_adapt.yaml:1 10 100"
)


COUNTER_FILE="tasks_counter_report_align.txt"
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
    echo "WARNING: No completely idle GPUs detected. Defaulting to GPU 0."
    UNUSED_GPUS=(0)
fi

echo "================================================================"
echo "Starting Report Alignment Downstream Experiments on GPU(s): ${UNUSED_GPUS[*]}"
echo "================================================================"

run_worker() {
    local gpu=$1
    while true; do
        # Acquire lock and read/increment counter
        exec 200>tasks_counter_report_align.lock
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
        
        # Execute the helper script (handles train 1/10/100 -> tune -> eval)
        bash filip/scripts/run_experiment_group.sh -c "$config" -n "$name" -g "$gpu" -p "$pcts" -s "$SKIP_ARG" $RESUME_ARG

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
rm -f "$COUNTER_FILE" tasks_counter_report_align.lock

echo "================================================================"
echo "All Report Alignment Downstream Experiments Completed."
echo "================================================================"

echo "Generating summary tables..."
python3 filip/scripts/generate_summary.py
