#!/bin/bash
set -e

# Generate MIMIC-IV-ECG images
# The raw dataset is located at ~/8TB/blmcg/datasets/physionet.org.5/files/mimic-iv-ecg/1.0/
# The output will be saved to our local datasets folder, matching the dataset.py expectation.

INPUT_DIR="/home/qfbqt/8TB/blmcg/datasets/physionet.org.5/files/mimic-iv-ecg/1.0/files"
OUTPUT_DIR="/home/qfbqt/8TB/datasets/mimic-iv-ecg/images"

echo "Generating MIMIC-IV-ECG images from $INPUT_DIR to $OUTPUT_DIR..."

cd /home/qfbqt/repo/GEM-fork/gem_generation/ecg-image-generator

python gen_ecg_images_from_data_batch.py \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    --num_threads 16 \
    --num_columns 1 \
    -se 0

echo "Done generating MIMIC-IV-ECG images!"
