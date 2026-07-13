#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/home/qfbqt/repo/GEM-fork"
export PYTHONUNBUFFERED=1

# Default values
CONFIG=""
NAME=""
GPU="0"
PCTS=()
RESUME=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG="$2"; shift ;;
        -n|--name) NAME="$2"; shift ;;
        -g|--gpu) GPU="$2"; shift ;;
        -p|--pcts) IFS=' ' read -r -a PCTS <<< "$2"; shift ;;
        -r|--resume) RESUME=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$CONFIG" || -z "$NAME" ]]; then
    echo "Usage: $0 -c <config> -n <name> -g <gpu> -p <pcts_list> [-r/--resume]"
    exit 1
fi

echo "=========================================="
echo "Starting experiment group: $NAME"
echo "Config: $CONFIG"
echo "GPU: $GPU"
echo "Percentages to run: ${PCTS[*]}"
echo "Resume enabled: $RESUME"
echo "=========================================="

for pct in "${PCTS[@]}"; do
    out_dir="outputs/filip/${NAME}_${pct}"
    echo "------------------------------------------"
    echo "Running percentage: $pct%"
    echo "Output directory: $out_dir"
    echo "------------------------------------------"
    
    # 1. Training
    resume_arg=""
    if [ "$RESUME" = true ] && [ -f "${out_dir}/latest.pt" ]; then
        resume_arg="--resume_from ${out_dir}/latest.pt"
        echo "Found existing checkpoint. Resuming from: ${out_dir}/latest.pt"
    fi
    
    echo "[Step 1/3] Training model with $pct% data..."
    CUDA_VISIBLE_DEVICES=$GPU python3 filip/train/train_ptbxl_adapt.py \
        --config "$CONFIG" \
        --train_pct "$pct" \
        --out_dir "$out_dir" \
        $resume_arg
        
    if [ $? -ne 0 ]; then
        echo "Error: Training failed for $pct%"
        exit 1
    fi
    
    # 2. Threshold Tuning
    echo "[Step 2/3] Tuning decision thresholds on validation split..."
    CUDA_VISIBLE_DEVICES=$GPU python3 filip/eval/tune_thresholds.py \
        --config "$CONFIG" \
        --checkpoint "${out_dir}/best.pt" \
        --out_dir "${out_dir}/validation_tuning"
        
    if [ $? -ne 0 ]; then
        echo "Error: Threshold tuning failed for $pct%"
        exit 1
    fi
    
    # 3. Final Evaluation
    echo "[Step 3/3] Evaluating model on test split..."
    CUDA_VISIBLE_DEVICES=$GPU python3 filip/eval/evaluate_diagnosis.py \
        --config "$CONFIG" \
        --checkpoint "${out_dir}/best.pt" \
        --thresholds_file "${out_dir}/validation_tuning/thresholds.json" \
        --out_dir "${out_dir}/evaluation"
        
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed for $pct%"
        exit 1
    fi
    
    echo "Finished percentage $pct% successfully!"
done

echo "Finished experiment group $NAME successfully!"
