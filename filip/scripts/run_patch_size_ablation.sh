#!/bin/bash
set -e

# Change to repository root
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

# Detect Python interpreter
if [ -x "/home/qfbqt/miniconda3/envs/gem/bin/python" ]; then
    PYTHON="/home/qfbqt/miniconda3/envs/gem/bin/python"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    PYTHON="python3"
fi

# Detect GPUs to use (default: 0 and 1, or parse CUDA_VISIBLE_DEVICES)
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    GPUS=(0 1)
fi

echo "========================================================================"
echo "    FILIP Patch Size Ablation (Patch 16 vs 32 + 4 Registers)"
echo "    Full Pipeline: Stage 1 (MIMIC Report Pretrain) -> Stage 2 (PTB-XL Sub)"
echo "    Queue-based execution across GPUs: ${GPUS[*]}"
echo "    Python: $PYTHON"
echo "========================================================================"

# Task definitions: format name:stage1_config:stage1_ckpt:stage2_config:out_dir
TASKS=(
    "ablation_size_16:filip/configs/ablation_patch_size/mimic_pretrain_patch16_register4.yaml:outputs/filip/mimic_report_alignment_vit_base_patch16_register4/checkpoints/best.pt:filip/configs/ablation_patch_size/ablation_size_16.yaml:outputs/filip/ablation_size_16"
    "ablation_size_32:filip/configs/ablation_patch_size/mimic_pretrain_patch32_register4.yaml:outputs/filip/mimic_report_alignment_vit_base_patch32_register4/checkpoints/best.pt:filip/configs/ablation_patch_size/ablation_size_32.yaml:outputs/filip/ablation_size_32"
)

COUNTER_FILE="tasks_counter_patch_size.txt"
LOCK_FILE="tasks_counter_patch_size.lock"
echo 0 > "$COUNTER_FILE"

run_single_pipeline() {
    local gpu=$1
    local name=$2
    local stage1_cfg=$3
    local stage1_ckpt=$4
    local stage2_cfg=$5
    local out_dir=$6

    local eval_dir="${out_dir}/evaluation"
    local thresholds_file="${eval_dir}/thresholds.json"
    local best_ckpt="${out_dir}/best.pt"

    echo "========================================================================"
    echo "[GPU $gpu] STARTING TASK: $name"
    echo "  Stage 1 Config: $stage1_cfg"
    echo "  Stage 2 Config: $stage2_cfg"
    echo "  Output Directory: $out_dir"
    echo "========================================================================"

    # 1. Stage 1 Pretraining on MIMIC report alignment (if not already completed)
    if [ -f "$stage1_ckpt" ]; then
        echo "[GPU $gpu] Stage 1 checkpoint found at $stage1_ckpt. Skipping pretraining."
    else
        echo "[GPU $gpu] Running Stage 1 MIMIC Report Alignment Pretraining (8 epochs)..."
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON filip/train/train_mimic_feature.py --config "$stage1_cfg"
    fi

    if [ ! -f "$stage1_ckpt" ]; then
        echo "[GPU $gpu] Error: Stage 1 checkpoint missing at $stage1_ckpt after pretraining."
        return 1
    fi

    # 2. Stage 2 Downstream Adaptation on PTB-XL Subclass 100%
    echo "[GPU $gpu] Running Stage 2 Downstream Adaptation ($name)..."
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON filip/train/train_ptbxl_adapt.py \
        --config "$stage2_cfg" \
        --train_pct 100 \
        --out_dir "$out_dir"

    # 3. Decision threshold tuning on validation split
    echo "[GPU $gpu] Tuning decision thresholds on validation split..."
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON filip/eval/tune_thresholds.py \
        --config "$stage2_cfg" \
        --checkpoint "$best_ckpt" \
        --out_dir "$eval_dir"

    # 4. Evaluation on test split
    echo "[GPU $gpu] Evaluating test split with tuned thresholds..."
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON filip/eval/evaluate_diagnosis.py \
        --config "$stage2_cfg" \
        --checkpoint "$best_ckpt" \
        --thresholds_file "$thresholds_file" \
        --out_dir "$eval_dir"

    echo "[GPU $gpu] COMPLETED TASK: $name successfully!"
}

run_worker() {
    local gpu=$1
    while true; do
        exec 200>"$LOCK_FILE"
        flock 200

        idx=$(cat "$COUNTER_FILE")
        if [ "$idx" -ge "${#TASKS[@]}" ]; then
            flock -u 200
            break
        fi

        next_idx=$((idx + 1))
        echo "$next_idx" > "$COUNTER_FILE"

        flock -u 200

        task_info="${TASKS[$idx]}"
        IFS=':' read -r name stage1_cfg stage1_ckpt stage2_cfg out_dir <<< "$task_info"

        run_single_pipeline "$gpu" "$name" "$stage1_cfg" "$stage1_ckpt" "$stage2_cfg" "$out_dir"
    done
}

WORKER_PIDS=()
for gpu_id in "${GPUS[@]}"; do
    echo "Spawning dynamic queue worker on GPU $gpu_id..."
    run_worker "$gpu_id" &
    WORKER_PIDS+=($!)
done

# Wait for all workers in the queue to complete
for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid"
done

# Cleanup coordination files
rm -f "$COUNTER_FILE" "$LOCK_FILE"

echo "========================================================================"
echo "    GENERATING ABLATION SUMMARY REPORT"
echo "========================================================================"

REPORT_DIRS=("outputs/filip/ablation_size_16" "outputs/filip/ablation_size_32")
export REPORT_DIRS_STR="${REPORT_DIRS[*]}"

$PYTHON - << 'EOF'
import os
import sys

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

reference_models = [
    ("MIMIC-Pretrained ViT-Large (Best)", "outputs/filip/vit_large_ptbxl_sub_report_align_adapt_100/evaluation/metrics.txt"),
    ("Raw CLIP ViT-Large (Patch 14)", "outputs/filip/vit_large_ptbxl_sub_report_align_adapt_raw_clip_100/evaluation/metrics.txt"),
]

eval_dirs = [d for d in os.environ.get("REPORT_DIRS_STR", "").split() if d]
metric_keys = ['Macro AUC', 'Macro F1', 'Micro F1', 'Accuracy', 'Hamming Loss']

print("\n" + "=" * 96)
print("       PATCH SIZE ABLATION (WITH MIMIC PRETRAINING + 4 REGISTERS)")
print("=" * 96)
header = f"{'Model / Experiment':<45} | " + " | ".join([f"{k:<11}" for k in metric_keys])
print(header)
print("-" * 96)

md_lines = [
    "# Patch Size Ablation Evaluation Summary Report\n",
    "Stage 1 MIMIC Report Alignment Pretraining + 4 Registers -> Downstream PTB-XL Subclass (100%)\n",
    "| Model / Experiment | " + " | ".join(metric_keys) + " |",
    "|:---|:" + ":|:".join(["---" for _ in metric_keys]) + ":|"
]

for label, path in reference_models:
    m = parse_metrics(path)
    if m:
        row = f"{label:<45} | " + " | ".join([f"{m.get(k, '-'):<11}" for k in metric_keys])
        print(row)
        md_lines.append(f"| {label} | " + " | ".join([m.get(k, '-') for k in metric_keys]) + " |")

print("-" * 96)

for ed in eval_dirs:
    m_path = os.path.join(ed, "evaluation", "metrics.txt")
    m = parse_metrics(m_path)
    name = os.path.basename(ed)
    if m:
        row = f"{name:<45} | " + " | ".join([f"{m.get(k, '-'):<11}" for k in metric_keys])
        print(row)
        md_lines.append(f"| **{name}** | " + " | ".join([m.get(k, '-') for k in metric_keys]) + " |")
    else:
        row = f"{name:<45} | (metrics.txt not yet generated)"
        print(row)
        md_lines.append(f"| **{name}** | " + " | ".join(["-" for _ in metric_keys]) + " |")

print("=" * 96 + "\n")

report_md_path = "outputs/filip/ablation_patch_size_summary.md"
os.makedirs("outputs/filip", exist_ok=True)
with open(report_md_path, "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"Summary report written to: {report_md_path}")
EOF

echo "========================================================================"
echo "    ALL ABLATION EXPERIMENTS COMPLETED SUCCESSFULLY"
echo "========================================================================"
