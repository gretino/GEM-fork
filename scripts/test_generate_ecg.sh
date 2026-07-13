#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: $0 <record_number>"
    echo "Example: $0 18016"
    exit 1
fi

# Ensure 5 digit zero-padding
RECORD=$(printf "%05d" "$1")
# Calculate the subdirectory (e.g., 18016 -> 18000)
SUBDIR=$(printf "%05d" $((10#$RECORD / 1000 * 1000)))

DAT_FILE="/home/qfbqt/datasets/ptb_xl_1.0.3/records500/${SUBDIR}/${RECORD}_hr.dat"
HEA_FILE="/home/qfbqt/datasets/ptb_xl_1.0.3/records500/${SUBDIR}/${RECORD}_hr.hea"

if [ ! -f "$DAT_FILE" ]; then
    echo "Error: Dat file not found: $DAT_FILE"
    exit 1
fi

cd /home/qfbqt/repo/GEM-fork/gem_generation/ecg-image-generator
OUT_DIR="./test_output_manual"
mkdir -p "$OUT_DIR"

echo "Generating ECG image for record $RECORD..."
conda run -n gem python gen_ecg_image_from_data.py \
    -i "$DAT_FILE" \
    -hea "$HEA_FILE" \
    -o "$OUT_DIR" \
    -se 0 \
    --num_columns 1 \
    -st 0

echo "Done! The generated image is saved in:"
realpath "$OUT_DIR"
