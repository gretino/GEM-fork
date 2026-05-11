#!/bin/bash

# distributed training configurations
GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT="1234"
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

# your huggingface configurations
#export HF_HOME=""

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <data_path> <run_name>"
    exit 1
fi

data_path="$1"
BASE_RUN_NAME="$2"

echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# QLoRA configuration: 4-bit base model with LoRA adapters.
# llava/train/train.py defaults lora_r to 64; use a smaller rank for QLoRA memory efficiency.
LORA_R=32
LORA_ALPHA=64

LLM_VERSION="LANSG/GEM"
version=llava_v1

image_folder="./data/ecg_images"
ecg_folder="./data/ecg_timeseries"
ecg_tower="/home/qfbqt/8TB/checkpoints/cpt_wfep_epoch_20.pt"

num_epochs=1
GRAD_ACC_STEP=16
BATCH_PER_GPU=1
TOTAL_BATCH_SIZE=$(($WORLD_SIZE * $BATCH_PER_GPU))

torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --node_rank $NODE_RANK \
    --master_port $MASTER_PORT \
    --nnodes $NNODES \
    ./llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --model_name_or_path ${LLM_VERSION} \
    --version ${version} \
    --data_path ${data_path} \
    --ecg_folder ${ecg_folder} \
    --ecg_tower ${ecg_tower} \
    --open_clip_config coca_ViT-B-32 \
    --image_folder $image_folder \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio ori \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/${BASE_RUN_NAME}_qlora \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size $BATCH_PER_GPU \
    --per_device_eval_batch_size $BATCH_PER_GPU \
    --gradient_accumulation_steps $GRAD_ACC_STEP \
    --save_strategy "steps" \
    --save_steps 0.2 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb
