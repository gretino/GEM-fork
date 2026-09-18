#!/bin/bash
# filip/scripts/run_experiments.sh
# Generic, reusable multi-GPU runner for FILIP downstream adaptation experiments.
# Can run from a manifest in filip/configs/runs/*.yaml or a list of config files.

set -e

# Repository root setup
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

# Detect Python interpreter
if [ -x "/home/qfbqt/miniconda3/envs/gem/bin/python" ]; then
    PYTHON="/home/qfbqt/miniconda3/envs/gem/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

# Defaults
RUN_MANIFEST=""
CONFIGS=()
EXPLICIT_GPUS=""
TRAIN_PCT=""
SKIP_COMPLETED="true"
RESUME_ARG=""
DRY_RUN="false"
RUN_NAME=""

# Parse command-line arguments
POSITIONAL_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--manifest|--run_config)
            RUN_MANIFEST="$2"; shift 2 ;;
        -g|--gpus)
            EXPLICIT_GPUS="$2"; shift 2 ;;
        -p|--train_pct|--pct)
            TRAIN_PCT="$2"; shift 2 ;;
        -r|--resume)
            RESUME_ARG="--resume"; shift ;;
        -s|--skip_completed)
            SKIP_COMPLETED="$2"; shift 2 ;;
        --no_skip|--no-skip)
            SKIP_COMPLETED="false"; shift ;;
        --dry_run|--dry-run)
            DRY_RUN="true"; shift ;;
        -h|--help)
            echo "Usage: $0 [manifest_or_config.yaml...] [options]"
            echo ""
            echo "Options:"
            echo "  -m, --manifest <file.yaml>   Run manifest from filip/configs/runs/"
            echo "  -g, --gpus <0,1,...>         Comma-separated list of GPUs to use"
            echo "  -p, --train_pct <int>        Training data percentage (default: 100)"
            echo "  -r, --resume                 Resume from latest.pt if found"
            echo "  -s, --skip_completed <bool>  Skip tasks with existing metrics.txt (default: true)"
            echo "      --dry_run                Show detected GPUs and queued tasks without running"
            exit 0
            ;;
        *)
            POSITIONAL_ARGS+=("$1"); shift ;;
    esac
done

# If positional arguments exist, check if first arg is a manifest or a config list
if [ -z "$RUN_MANIFEST" ] && [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
    if [[ "${POSITIONAL_ARGS[0]}" == *.yaml || "${POSITIONAL_ARGS[0]}" == *.yml ]]; then
        # Check if it has a 'configs:' key (manifest) or 'experiment_name' (single config)
        IS_MANIFEST=$($PYTHON -c "
import yaml, sys
try:
    data = yaml.safe_load(open(sys.argv[1]))
    print('yes' if isinstance(data, dict) and 'configs' in data else 'no')
except Exception:
    print('no')
" "${POSITIONAL_ARGS[0]}")
        if [ "$IS_MANIFEST" == "yes" ]; then
            RUN_MANIFEST="${POSITIONAL_ARGS[0]}"
            POSITIONAL_ARGS=("${POSITIONAL_ARGS[@]:1}")
        fi
    fi
fi

# Load from manifest if provided
if [ -n "$RUN_MANIFEST" ]; then
    if [ ! -f "$RUN_MANIFEST" ]; then
        echo "Error: Manifest file not found: $RUN_MANIFEST"
        exit 1
    fi
    MANIFEST_NAME=$($PYTHON -c "import yaml; print(yaml.safe_load(open('$RUN_MANIFEST')).get('name', ''))")
    MANIFEST_PCT=$($PYTHON -c "import yaml; print(yaml.safe_load(open('$RUN_MANIFEST')).get('train_pct', 100))")
    MANIFEST_CONFIGS=$($PYTHON -c "import yaml; print(' '.join(yaml.safe_load(open('$RUN_MANIFEST')).get('configs', [])))")
    
    [ -z "$RUN_NAME" ] && RUN_NAME="$MANIFEST_NAME"
    [ -z "$TRAIN_PCT" ] && TRAIN_PCT="$MANIFEST_PCT"
    read -r -a CONFIGS <<< "$MANIFEST_CONFIGS"
fi

# Append any remaining positional config arguments
if [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
    CONFIGS+=("${POSITIONAL_ARGS[@]}")
fi

# Fallback defaults
[ -z "$TRAIN_PCT" ] && TRAIN_PCT="100"
[ -z "$RUN_NAME" ] && RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "Error: No config files specified. Provide a manifest (--manifest) or config paths."
    exit 1
fi

# -----------------------------------------------------------------------------
# GPU Detection Logic
# -----------------------------------------------------------------------------
GPUS=()
if [ -n "$EXPLICIT_GPUS" ]; then
    IFS=',' read -r -a GPUS <<< "$EXPLICIT_GPUS"
    echo "Using user-specified GPUs: ${GPUS[*]}"
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
    echo "Using CUDA_VISIBLE_DEVICES GPUs: ${GPUS[*]}"
elif command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Scanning $NUM_GPUS system GPUs for spare/idle devices (Memory < 1000 MiB, Util < 10%)..."
    for ((i=0; i<NUM_GPUS; i++)); do
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i 2>/dev/null || echo 99999)
        util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $i 2>/dev/null || echo 100)
        echo "  - GPU $i: Memory = ${mem}MiB, Utilization = ${util}%"
        if [ "$mem" -lt 1000 ] && [ "$util" -lt 10 ]; then
            GPUS+=($i)
        fi
    done
    if [ ${#GPUS[@]} -eq 0 ]; then
        echo "Notice: No completely idle GPUs detected. Defaulting to GPU 0."
        GPUS=(0)
    else
        echo "Found ${#GPUS[@]} spare GPU(s): ${GPUS[*]}"
    fi
else
    echo "nvidia-smi not available. Defaulting to GPU 0."
    GPUS=(0)
fi

echo "========================================================================"
echo "    FILIP EXPERIMENT RUNNER"
echo "    Run Name        : $RUN_NAME"
echo "    Train Split Pct : $TRAIN_PCT%"
echo "    Queued Configs  : ${#CONFIGS[@]}"
echo "    Active GPUs     : ${GPUS[*]}"
echo "    Skip Completed  : $SKIP_COMPLETED"
echo "========================================================================"
for i in "${!CONFIGS[@]}"; do
    echo "  $((i+1)). ${CONFIGS[$i]}"
done
echo "========================================================================"

if [ "$DRY_RUN" = "true" ]; then
    echo "Dry run complete. Exiting without launching training."
    exit 0
fi

# -----------------------------------------------------------------------------
# Dynamic Work Queue Execution
# -----------------------------------------------------------------------------
COUNTER_FILE=".tasks_counter_${RUN_NAME}_$$.txt"
LOCK_FILE=".tasks_counter_${RUN_NAME}_$$.lock"
echo 0 > "$COUNTER_FILE"

cleanup() {
    rm -f "$COUNTER_FILE" "$LOCK_FILE"
}
trap cleanup EXIT INT TERM

run_single_task() {
    local gpu=$1
    local cfg=$2

    if [ ! -f "$cfg" ]; then
        echo "[GPU $gpu] Error: Config file not found at $cfg"
        return 1
    fi

    local exp_name
    exp_name=$($PYTHON -c "import yaml; print(yaml.safe_load(open('$cfg'))['experiment_name'])")

    local out_dir
    if [[ "$exp_name" =~ _[0-9]+$ ]]; then
        out_dir="outputs/filip/${exp_name}"
    else
        out_dir="outputs/filip/${exp_name}_${TRAIN_PCT}"
    fi

    local best_ckpt="${out_dir}/best.pt"
    local eval_dir="${out_dir}/evaluation"
    local thresholds_file="${eval_dir}/tuned_thresholds.json"

    echo "========================================================================"
    echo "[GPU $gpu] STARTING TASK: $exp_name"
    echo "  Config    : $cfg"
    echo "  Output Dir: $out_dir"
    echo "========================================================================"

    # Check if already completed
    if [ "$SKIP_COMPLETED" = "true" ] && [ -f "${eval_dir}/metrics.txt" ] && [ -s "${eval_dir}/metrics.txt" ]; then
        echo "[GPU $gpu] Task $exp_name is already completed (${eval_dir}/metrics.txt exists). Skipping."
        return 0
    fi

    # 1. Training (Stage 2 adaptation)
    echo "[GPU $gpu] 1. Training downstream adaptation: $exp_name..."
    local train_args=("--config" "$cfg" "--train_pct" "$TRAIN_PCT" "--out_dir" "$out_dir")
    if [ -n "$RESUME_ARG" ]; then
        train_args+=("$RESUME_ARG")
    fi
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON filip/train/train_ptbxl_adapt.py "${train_args[@]}"

    if [ ! -f "$best_ckpt" ]; then
        echo "[GPU $gpu] Error: Checkpoint missing at $best_ckpt after training."
        return 1
    fi

    # 2. Threshold tuning
    echo "[GPU $gpu] 2. Tuning decision thresholds on validation split..."
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON filip/eval/tune_thresholds.py \
        --config "$cfg" \
        --checkpoint "$best_ckpt" \
        --out_dir "$eval_dir"

    # 3. Test Evaluation
    echo "[GPU $gpu] 3. Evaluating test set metrics with tuned thresholds..."
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON filip/eval/evaluate_diagnosis.py \
        --config "$cfg" \
        --checkpoint "$best_ckpt" \
        --thresholds_file "$thresholds_file" \
        --out_dir "$eval_dir"

    echo "[GPU $gpu] COMPLETED: $exp_name successfully!"
}

run_worker() {
    local gpu=$1
    while true; do
        exec 200>"$LOCK_FILE"
        flock 200

        local idx
        idx=$(cat "$COUNTER_FILE")
        if [ "$idx" -ge "${#CONFIGS[@]}" ]; then
            flock -u 200
            break
        fi

        local next_idx=$((idx + 1))
        echo "$next_idx" > "$COUNTER_FILE"
        flock -u 200

        local cfg="${CONFIGS[$idx]}"
        run_single_task "$gpu" "$cfg"
    done
}

# Spawn worker per GPU
WORKER_PIDS=()
for gpu_id in "${GPUS[@]}"; do
    echo "Spawning dynamic worker on GPU $gpu_id..."
    run_worker "$gpu_id" &
    WORKER_PIDS+=($!)
done

# Wait for all workers to complete
for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid"
done

echo "========================================================================"
echo "    ALL QUEUED TASKS COMPLETED"
echo "========================================================================"

# -----------------------------------------------------------------------------
# Summary Report Generation
# -----------------------------------------------------------------------------
REPORT_CFGS_STR="${CONFIGS[*]}"
export REPORT_CFGS_STR
export RUN_NAME_EXPORT="$RUN_NAME"
export TRAIN_PCT_EXPORT="$TRAIN_PCT"

$PYTHON - << 'EOF'
import os
import yaml

def parse_metrics(filepath):
    if not os.path.exists(filepath):
        return None
    metrics = {}
    with open(filepath, 'r') as f:
        for line in f:
            for k in ['Macro AUC', 'Micro AUC', 'Macro F1', 'Micro F1', 'Accuracy', 'Hamming Loss']:
                if line.startswith(k + ':'):
                    metrics[k] = line.split(':')[-1].strip()
    return metrics

run_name = os.environ.get("RUN_NAME_EXPORT", "experiments")
train_pct = os.environ.get("TRAIN_PCT_EXPORT", "100")
cfgs = [c for c in os.environ.get("REPORT_CFGS_STR", "").split() if c]

metric_keys = ['Macro AUC', 'Macro F1', 'Micro F1', 'Accuracy', 'Hamming Loss']

baseline_dir = "outputs/filip/vit_large_ptbxl_sub_report_align_adapt_100"
baseline_m = parse_metrics(os.path.join(baseline_dir, "evaluation", "metrics.txt"))

raw_baseline_dir = "outputs/filip/vit_large_ptbxl_sub_report_align_adapt_raw_clip_100"
raw_baseline_m = parse_metrics(os.path.join(raw_baseline_dir, "evaluation", "metrics.txt"))

header = f"{'Model / Experiment':<50} | " + " | ".join([f"{k:<11}" for k in metric_keys])
sep = "-" * len(header)

print("\n" + "=" * len(header))
print(f"             EXPERIMENT EVALUATION SUMMARY REPORT ({run_name})")
print("=" * len(header))
print(header)
print(sep)

md_lines = [
    f"# Experiment Evaluation Summary Report: {run_name}\n",
    "| Model / Experiment | " + " | ".join(metric_keys) + " |",
    "|:---|:" + ":|:".join(["---" for _ in metric_keys]) + ":|"
]

if baseline_m:
    row = f"{'MIMIC-Pretrained Full Fine-Tune (Existing Baseline)':<50} | " + " | ".join([f"{baseline_m.get(k, '-'):<11}" for k in metric_keys])
    print(row)
    md_lines.append(f"| **MIMIC-Pretrained Full Fine-Tune (Baseline)** | " + " | ".join([baseline_m.get(k, '-') for k in metric_keys]) + " |")

if raw_baseline_m:
    row = f"{'Raw CLIP Full Fine-Tune (Existing Baseline)':<50} | " + " | ".join([f"{raw_baseline_m.get(k, '-'):<11}" for k in metric_keys])
    print(row)
    md_lines.append(f"| **Raw CLIP Full Fine-Tune (Baseline)** | " + " | ".join([raw_baseline_m.get(k, '-') for k in metric_keys]) + " |")

for cfg in cfgs:
    try:
        data = yaml.safe_load(open(cfg))
        exp_name = data.get('experiment_name', os.path.splitext(os.path.basename(cfg))[0])
    except Exception:
        exp_name = os.path.splitext(os.path.basename(cfg))[0]

    if exp_name.endswith(f"_{train_pct}"):
        out_dir = os.path.join("outputs", "filip", exp_name)
    else:
        out_dir = os.path.join("outputs", "filip", f"{exp_name}_{train_pct}")

    m_path = os.path.join(out_dir, "evaluation", "metrics.txt")
    m = parse_metrics(m_path)
    display_name = os.path.basename(out_dir)
    if m:
        row = f"{display_name:<50} | " + " | ".join([f"{m.get(k, '-'):<11}" for k in metric_keys])
        print(row)
        md_lines.append(f"| **{display_name}** | " + " | ".join([m.get(k, '-') for k in metric_keys]) + " |")
    else:
        row = f"{display_name:<50} | (metrics not found)"
        print(row)
        md_lines.append(f"| **{display_name}** | " + " | ".join(["-" for _ in metric_keys]) + " |")

print("=" * len(header) + "\n")

report_path = f"outputs/filip/{run_name}_summary_report.md"
os.makedirs("outputs/filip", exist_ok=True)
with open(report_path, "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"Summary report written to: {report_path}\n")
EOF
