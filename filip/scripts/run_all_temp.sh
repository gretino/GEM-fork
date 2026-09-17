#!/bin/bash
set -e

export PYTHONPATH=.
PYTHON="/home/qfbqt/miniconda3/envs/gem/bin/python"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

echo "========================================================================"
echo "    FILIP Downstream Ablation Pipeline (Train -> Tune -> Test -> Report)"
echo "    Using GPU: $GPU"
echo "========================================================================"

# Default config(s) if none provided via command-line arguments
if [ "$#" -gt 0 ]; then
    CONFIGS=("$@")
else
    CONFIGS=(
      "filip/configs/report_alignment_adapt_vit_large/ptbxl_sub_adapt_raw_clip.yaml"
    )
fi

echo "Queued Configs (${#CONFIGS[@]}):"
for cfg in "${CONFIGS[@]}"; do
    echo "  - $cfg"
done
echo "========================================================================"

REPORT_DIRS=()

for cfg in "${CONFIGS[@]}"; do
    if [ ! -f "$cfg" ]; then
        echo "Error: Config file not found at $cfg"
        exit 1
    fi

    EXP_NAME=$($PYTHON -c "import yaml; print(yaml.safe_load(open('$cfg'))['experiment_name'])")
    if [[ "$EXP_NAME" =~ _[0-9]+$ ]]; then
        OUT_DIR="outputs/filip/${EXP_NAME}"
    else
        OUT_DIR="outputs/filip/${EXP_NAME}_100"
    fi
    CHECKPOINT="${OUT_DIR}/best.pt"
    EVAL_DIR="${OUT_DIR}/evaluation"
    THRESHOLDS_FILE="${EVAL_DIR}/tuned_thresholds.json"
    REPORT_DIRS+=("$OUT_DIR")

    echo "------------------------------------------------------------------------"
    echo "1. Training Downstream Model: $EXP_NAME"
    echo "   Config: $cfg"
    echo "   Output Dir: $OUT_DIR"
    echo "------------------------------------------------------------------------"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON filip/train/train_ptbxl_adapt.py \
        --config "$cfg" \
        --train_pct 100 \
        --out_dir "$OUT_DIR"

    echo "------------------------------------------------------------------------"
    echo "2. Tuning Decision Thresholds on Validation Split"
    echo "   Checkpoint: $CHECKPOINT"
    echo "   Evaluation Dir: $EVAL_DIR"
    echo "------------------------------------------------------------------------"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON filip/eval/tune_thresholds.py \
        --config "$cfg" \
        --checkpoint "$CHECKPOINT" \
        --out_dir "$EVAL_DIR"

    echo "------------------------------------------------------------------------"
    echo "3. Evaluating Test Set Metrics with Tuned Thresholds"
    echo "   Thresholds File: $THRESHOLDS_FILE"
    echo "------------------------------------------------------------------------"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON filip/eval/evaluate_diagnosis.py \
        --config "$cfg" \
        --checkpoint "$CHECKPOINT" \
        --thresholds_file "$THRESHOLDS_FILE" \
        --out_dir "$EVAL_DIR"

    echo "Completed $EXP_NAME successfully!"
done

echo "========================================================================"
echo "4. Generating Ablation Metrics Report"
echo "========================================================================"

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

baseline_dir = "outputs/filip/vit_large_ptbxl_sub_report_align_adapt_100"
baseline_metrics_path = os.path.join(baseline_dir, "evaluation", "metrics.txt")
baseline_metrics = parse_metrics(baseline_metrics_path)

eval_dirs = [d for d in os.environ.get("REPORT_DIRS_STR", "").split() if d]
if not eval_dirs:
    eval_dirs = ["outputs/filip/vit_large_ptbxl_sub_report_align_adapt_raw_clip_100"]

metric_keys = ['Macro AUC', 'Macro F1', 'Micro F1', 'Accuracy', 'Hamming Loss']

print("\n" + "=" * 90)
print("              DOWNSTREAM ABLATION EVALUATION SUMMARY REPORT")
print("=" * 90)
header = f"{'Model / Experiment':<45} | " + " | ".join([f"{k:<11}" for k in metric_keys])
print(header)
print("-" * 90)

md_lines = [
    "# Downstream Ablation Evaluation Summary Report\n",
    "| Model / Experiment | " + " | ".join(metric_keys) + " |",
    "|:---|:" + ":|:".join(["---" for _ in metric_keys]) + ":|"
]

if baseline_metrics:
    b_row = f"{'MIMIC-Pretrained Baseline (Existing Best)':<45} | " + " | ".join([f"{baseline_metrics.get(k, '-'):<11}" for k in metric_keys])
    print(b_row)
    md_lines.append(f"| **MIMIC-Pretrained Baseline (Existing Best)** | " + " | ".join([baseline_metrics.get(k, '-') for k in metric_keys]) + " |")

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

print("=" * 90 + "\n")

report_md_path = "outputs/filip/ablation_metrics_report_temp.md"
os.makedirs("outputs/filip", exist_ok=True)
with open(report_md_path, "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"Summary report written to: {report_md_path}")
EOF

echo "========================================================================"
echo "    PIPELINE COMPLETED SUCCESSFULLY"
echo "========================================================================"
