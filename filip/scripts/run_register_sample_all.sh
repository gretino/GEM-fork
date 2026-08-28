#!/bin/bash
set -e

export PYTHONPATH=.
PYTHON="/home/qfbqt/miniconda3/envs/gem/bin/python"

echo "========================================================================"
echo "    ViT Register Token Retraining Experiment Pipeline"
echo "========================================================================"

STAGE1_CFG="filip/configs/register_experiment/mimic_report_alignment_vit_large_register4.yaml"
STAGE1_CKPT="outputs/filip/mimic_report_alignment_vit_large_register4_sample/checkpoints/best.pt"

if [ -f "$STAGE1_CKPT" ]; then
    echo "Stage 1 register checkpoint found at $STAGE1_CKPT. Proceeding to Stage 2!"
else
    echo "========================================================================"
    echo "Stage 1: Pretraining ViT-Large + 4 Register Tokens on MIMIC-IV-ECG (8 Epochs)..."
    echo "========================================================================"
    $PYTHON filip/train/train_mimic_feature.py --config "$STAGE1_CFG"
fi

echo "========================================================================"
echo "Stage 2: Downstream Adaptation with 4 Register Tokens (10 Epochs)"
echo "========================================================================"

CONFIGS=(
  "filip/configs/register_experiment/ptbxl_sub_adapt_register4.yaml"
  "filip/configs/register_experiment/csn_adapt_register4.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    EXP_NAME=$($PYTHON -c "import yaml; print(yaml.safe_load(open('$cfg'))['experiment_name'])")
    OUT_DIR="outputs/filip/${EXP_NAME}"
    CHECKPOINT="${OUT_DIR}/best.pt"
    EVAL_DIR="${OUT_DIR}/evaluation"
    THRESHOLDS_FILE="${EVAL_DIR}/tuned_thresholds.json"

    echo "------------------------------------------------------------------------"
    echo "Running Adaptation: $EXP_NAME"
    echo "------------------------------------------------------------------------"
    
    # 1. Train downstream adaptation model
    $PYTHON filip/train/train_ptbxl_adapt.py --config "$cfg" --train_pct 100 --out_dir "$OUT_DIR"
    
    # 2. Tune decision thresholds on validation split
    $PYTHON filip/eval/tune_thresholds.py --config "$cfg" --checkpoint "$CHECKPOINT" --out_dir "$EVAL_DIR"
    
    # 3. Evaluate test set performance using tuned thresholds
    $PYTHON filip/eval/evaluate_diagnosis.py --config "$cfg" --checkpoint "$CHECKPOINT" --thresholds_file "$THRESHOLDS_FILE" --out_dir "$EVAL_DIR"

    echo "Completed $EXP_NAME successfully!"
done

echo "========================================================================"
echo "    REGISTER TOKEN EXPERIMENT PIPELINE COMPLETED SUCCESSFULLY"
echo "========================================================================"
