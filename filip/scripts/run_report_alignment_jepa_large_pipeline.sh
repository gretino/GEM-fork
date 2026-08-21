#!/bin/bash
set -e

export PYTHONPATH=.
PYTHON="/home/qfbqt/miniconda3/envs/gem/bin/python"

# GPU List: Default to available GPUs (e.g. "0,1") or use CUDA_VISIBLE_DEVICES
IFSS_SAVE=$IFS
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=$IFSS_SAVE

NUM_GPUS=${#GPUS[@]}
GPU_CSV=$(IFS=','; echo "${GPUS[*]}")
echo "========================================================================"
echo "Multi-GPU Execution Enabled: Using ${NUM_GPUS} GPUs [${GPU_CSV}]"
echo "========================================================================"

STAGE1_DIR="outputs/filip/report_alignment_jepa_large_pretrain/checkpoints"
STAGE1_CKPT="${STAGE1_DIR}/best.pt"

if [[ "$SKIP_STAGE1" == "1" ]] || [[ -f "$STAGE1_CKPT" ]]; then
    echo "========================================================================"
    echo "Stage 1 checkpoint found at $STAGE1_CKPT."
    echo "Skipping Stage 1 pretraining and proceeding directly to Stage 2!"
    echo "========================================================================"
else
    echo "========================================================================"
    echo "Stage 1: Pretraining ViT-Large Backbone with Report Alignment + JEPA Loss..."
    echo "========================================================================"
    CUDA_VISIBLE_DEVICES=${GPUS[0]} $PYTHON filip/train/train_mimic_feature.py --config filip/configs/report_alignment_jepa_large_pretrain.yaml
    
    echo "Cleaning up intermediate checkpoints in Stage 1..."
    find "$STAGE1_DIR" -type f -name "*.pt" ! -name "best.pt" -delete 2>/dev/null || true
fi

echo "========================================================================"
echo "Stage 2: Parallel ViT-Large + JEPA Adaptation across 6 Benchmark Datasets..."
echo "========================================================================"

CONFIGS=(
  "filip/configs/report_alignment_adapt_jepa_large/ptbxl_super_adapt.yaml"
  "filip/configs/report_alignment_adapt_jepa_large/ptbxl_sub_adapt.yaml"
  "filip/configs/report_alignment_adapt_jepa_large/ptbxl_rhythm_adapt.yaml"
  "filip/configs/report_alignment_adapt_jepa_large/ptbxl_form_adapt.yaml"
  "filip/configs/report_alignment_adapt_jepa_large/cpsc_adapt.yaml"
  "filip/configs/report_alignment_adapt_jepa_large/csn_adapt.yaml"
)

PERCENTAGES=(1 10 100)

run_task() {
    local cfg="$1"
    local pct="$2"
    local gpu="$3"

    EXP_NAME=$($PYTHON -c "import yaml; print(yaml.safe_load(open('$cfg'))['experiment_name'])")
    OUT_DIR="outputs/filip/${EXP_NAME}_${pct}"
    CHECKPOINT="${OUT_DIR}/best.pt"
    EVAL_DIR="${OUT_DIR}/evaluation"
    THRESHOLDS_FILE="${EVAL_DIR}/tuned_thresholds.json"

    # Skip if already completed
    if [ -f "${EVAL_DIR}/metrics.txt" ] && [ -s "${EVAL_DIR}/metrics.txt" ]; then
        echo "[GPU $gpu] $EXP_NAME ($pct%) already completed (${EVAL_DIR}/metrics.txt exists). Skipping..."
        # Clean up any residual non-best checkpoints
        find "$OUT_DIR" -maxdepth 1 -type f -name "*.pt" ! -name "best.pt" -delete 2>/dev/null || true
        return 0
    fi

    echo "[GPU $gpu] Launching $EXP_NAME ($pct%)..."
    
    # 1. Train downstream adaptation model
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON filip/train/train_ptbxl_adapt.py --config "$cfg" --train_pct "$pct" --out_dir "$OUT_DIR"
    
    # 2. Tune decision thresholds on validation split
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON filip/eval/tune_thresholds.py --config "$cfg" --checkpoint "$CHECKPOINT" --out_dir "$EVAL_DIR"
    
    # 3. Evaluate test set performance using tuned thresholds
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON filip/eval/evaluate_diagnosis.py --config "$cfg" --checkpoint "$CHECKPOINT" --thresholds_file "$THRESHOLDS_FILE" --out_dir "$EVAL_DIR"

    # 4. Remove all saved checkpoints other than best.pt
    echo "[GPU $gpu] Removing extra checkpoints in $OUT_DIR (keeping best.pt)..."
    find "$OUT_DIR" -maxdepth 1 -type f -name "*.pt" ! -name "best.pt" -delete 2>/dev/null || true

    echo "[GPU $gpu] Completed $EXP_NAME ($pct%) successfully!"
}

# Dynamic VRAM-Aware GPU Allocation Queue
declare -A GPU_PIDS

find_free_gpu() {
    local min_free_mb=18000 # Minimum 18GB free VRAM required
    while true; do
        best_gpu=""
        max_free=0
        for gpu in "${GPUS[@]}"; do
            pid="${GPU_PIDS[$gpu]}"
            if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
                # Query actual free memory on this GPU via nvidia-smi
                free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d '[:space:]')
                if [[ -n "$free_mem" ]] && [ "$free_mem" -ge "$min_free_mb" ]; then
                    if [ "$free_mem" -gt "$max_free" ]; then
                        max_free=$free_mem
                        best_gpu=$gpu
                    fi
                fi
            fi
        done
        
        if [ -n "$best_gpu" ]; then
            echo "$best_gpu"
            return 0
        fi
        
        sleep 5
    done
}

for cfg in "${CONFIGS[@]}"; do
  for pct in "${PERCENTAGES[@]}"; do
    target_gpu=$(find_free_gpu)
    
    run_task "$cfg" "$pct" "$target_gpu" &
    GPU_PIDS[$target_gpu]=$!
  done
done

# Wait for all background tasks to finish
wait

echo "========================================================================"
echo "Stage 3: Post-Training Checkpoint Cleanup & Comparative Summary Generation..."
echo "========================================================================"

find "outputs/filip/report_alignment_jepa_large_pretrain" -type f -name "*.pt" ! -name "best.pt" -delete 2>/dev/null || true

for cfg in "${CONFIGS[@]}"; do
  EXP_NAME=$($PYTHON -c "import yaml; print(yaml.safe_load(open('$cfg'))['experiment_name'])")
  for pct in "${PERCENTAGES[@]}"; do
    OUT_DIR="outputs/filip/${EXP_NAME}_${pct}"
    if [ -d "$OUT_DIR" ]; then
      find "$OUT_DIR" -maxdepth 1 -type f -name "*.pt" ! -name "best.pt" -delete 2>/dev/null || true
    fi
  done
done

$PYTHON filip/scripts/generate_summary.py

echo "========================================================================"
echo "Pipeline report_alignment_jepa_large completed successfully!"
echo "========================================================================"
