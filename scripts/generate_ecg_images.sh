#!/bin/bash
set -e

# Generate PTB-XL ECG images
cd /home/qfbqt/repo/GEM-fork/gem_generation/ecg-image-generator

python gen_ecg_images_from_data_batch.py \
    -i /home/qfbqt/datasets/ptb_xl_1.0.3/records500 \
    -o /home/qfbqt/8TB/datasets/ptb-xl-gen \
    --num_threads 16 --num_columns 2 -se 0

# Create symlink
mkdir -p /home/qfbqt/repo/GEM-fork/data/ecg_images
ln -sfn /home/qfbqt/8TB/datasets/ptb-xl-gen /home/qfbqt/repo/GEM-fork/data/ecg_images/ptb-xl-gen
