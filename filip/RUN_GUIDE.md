# Running the FILIP Experiment

This guide explains how to prepare the data and run the newly implemented FILIP experiment architecture.

## 0. Data Processing Pipeline (MIMIC-IV-ECG)

Before training, the raw MIMIC-IV-ECG database must be processed into formatted JSON labels and pristine `.png` images.

The full pipeline consists of the following sequence:

```bash
# Navigate to the repo root
cd /home/qfbqt/repo/GEM-fork

# 1. Build the master list from the raw datasets (usually only needs to be run once)
python filip/data_processing/build_master_list.py

# 2. Resample the dataset to your desired size (e.g. 1,000 records). 
# This shuffles the records and splits them 8:1:1 into train/val/test JSONs.
python filip/data_processing/sample_splits.py --num_items 1000

# 3. Extract clinical features and diagnoses for the newly sampled records
python filip/data_processing/extract_clinical_features.py
python filip/data_processing/extract_measurements.py

# 4. Render the target `.dat` signals into 1x12 pristine format images.
# This script automatically reads the JSON files built in step 2 and generates exactly those images.
python filip/scripts/generate_mimic_subset.py
```d

> [!TIP]
> If you ever want to increase the dataset size to 10,000, simply run the sequence above starting from **Step 2** and pass `--num_items 10000` to the `sample_splits.py` script.


## 1. Modifying Configuration for Trial Runs vs Full Runs

Both stages are controlled completely by their respective YAML files in `/filip/configs/`. You don't need to change the python code to adjust run settings.

If you want to do a **trial run** (to ensure the model fits in VRAM and no bugs exist), lower the `epochs` and `batch_size`:

1. Open `filip/configs/mimic_feature_pretrain.yaml`.
2. Edit the `training` section:
   ```yaml
   training:
     batch_size: 4       # small batch for trial
     learning_rate: 1.0e-4
     epochs: 1           # 1 epoch for testing
     mixed_precision: true
   ```
3. Run the script: `bash filip/scripts/train_mimic_feature.sh`

For a **full run**, simply change `batch_size` back to your GPU's capacity (e.g., 32 or 64) and set `epochs` to your desired length (e.g., 10).

## 2. Running for a Subset of Data (Percentage / Count)

If you want to quickly iterate over just a small subset of the dataset (e.g., the first 100 items), you can use the `--max_batches` limit trick natively supported by PyTorch loops. But an easier way without editing code is simply truncating the JSON records file during testing.

Alternatively, if you'd prefer to edit the script, you can open `filip/train/train_mimic_feature.py` and modify the loop:

```python
        for i, batch in enumerate(pbar):
            if i >= 50:  # Break after 50 batches for a subset run!
                break
```
We deliberately kept the training loops lightweight and standard so you can easily inject early stops like this.

## 3. Running in the Background (Recommended for Full Runs)

Because the full dataset is large, you should use `nohup` to run the bash scripts so they continue even if your SSH terminal disconnects:

```bash
mkdir -p filip/logs
nohup bash filip/scripts/train_mimic_feature.sh > filip/logs/stage1.log 2>&1 &
```

You can view the progress anytime with:
```bash
tail -f filip/logs/stage1.log
```

## 4. Stage 2 (PTB-XL Adaptation)

Once Stage 1 finishes, it will save a `best.pt` checkpoint to `outputs/filip/[experiment_name]/checkpoints/best.pt`.

Before training Stage 2, you must prepare the PTB-XL dataset images and split files:

```bash
# 1. Render all PTB-XL ECG waveforms as flat 1-column PNG images (takes some time)
bash scripts/generate_ecg_images.sh

# 2. Run the PTB-XL preprocessing script to align the images with metadata and build splits
PYTHONPATH=. python filip/data_processing/prepare_ptbxl.py
```

The Stage 2 config (`filip/configs/ptbxl_diagnosis_adapt.yaml`) is already wired to look for the Stage 1 pretrained checkpoint.
To start Stage 2 training in the background, run:

```bash
nohup bash filip/scripts/train_ptbxl_adapt.sh > filip/logs/stage2.log 2>&1 &
```

# 1. Train on CSN folds
bash filip/scripts/train_ptbxl_adapt.sh -c filip/configs/csn_diagnosis_adapt.yaml -g <gpu_id>

# 2. Tune decision thresholds on CSN validation fold
bash filip/scripts/tune_thresholds.sh -c filip/configs/csn_diagnosis_adapt.yaml -g <gpu_id>

# 3. Evaluate model on CSN test fold using the tuned thresholds
bash filip/scripts/eval_diagnosis.sh \
  -c filip/configs/csn_diagnosis_adapt.yaml \
  -ckpt outputs/filip/csn_adapt/checkpoints/best.pt \
  -t outputs/filip/csn_adapt/ptb-val-tuning/thresholds.json \
  -o outputs/filip/csn_adapt/test-tuned \
  -g <gpu_id>

# 1. Generate CSN images in 1x12 layout
python gem_generation/ecg-image-generator/gen_ecg_images_from_data_batch.py \
  -i /home/qfbqt/8TB/datasets/physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords \
  -o /home/qfbqt/8TB/datasets/csn_new \
  --num_columns 1 \
  --num_threads 16

python filip/data_processing/prepare_csn.py

# 2. Generate CPSC 2018 images in 1x12 layout
rm -rf /home/qfbqt/repo/GEM-fork/data/cpsc2018/images
python gem_generation/ecg-image-generator/gen_ecg_images_from_data_batch.py \
  -i /home/qfbqt/8TB/datasets/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018 \
  -o /home/qfbqt/repo/GEM-fork/data/cpsc2018/images \
  --num_columns 1 \
  --num_threads 16

python filip/data_processing/prepare_cpsc2018.py

# 3. Generate Georgia images in 1x12 layout
rm -rf /home/qfbqt/repo/GEM-fork/data/georgia/images
python gem_generation/ecg-image-generator/gen_ecg_images_from_data_batch.py \
  -i /home/qfbqt/8TB/datasets/physionet.org/files/challenge-2020/1.0.2/training/georgia \
  -o /home/qfbqt/repo/GEM-fork/data/georgia/images \
  --num_columns 1 \
  --num_threads 16

python filip/data_processing/prepare_georgia.py

# Train
bash filip/scripts/train_ptbxl_adapt.sh -c filip/configs/cpsc_diagnosis_adapt.yaml -g <gpu_id>
# Tune Thresholds
bash filip/scripts/tune_thresholds.sh -c filip/configs/cpsc_diagnosis_adapt.yaml -g <gpu_id>
# Evaluate
bash filip/scripts/eval_diagnosis.sh \
  -c filip/configs/cpsc_diagnosis_adapt.yaml \
  -ckpt outputs/filip/cpsc_adapt/checkpoints/best.pt \
  -t outputs/filip/cpsc_adapt/ptb-val-tuning/thresholds.json \
  -o outputs/filip/cpsc_adapt/test-tuned \
  -g 1

# Train
bash filip/scripts/train_ptbxl_adapt.sh -c filip/configs/georgia_diagnosis_adapt.yaml -g <gpu_id>
# Tune Thresholds
bash filip/scripts/tune_thresholds.sh -c filip/configs/georgia_diagnosis_adapt.yaml -g <gpu_id>
# Evaluate
bash filip/scripts/eval_diagnosis.sh \
  -c filip/configs/georgia_diagnosis_adapt.yaml \
  -ckpt outputs/filip/georgia_adapt/checkpoints/best.pt \
  -t outputs/filip/georgia_adapt/ptb-val-tuning/thresholds.json \
  -o outputs/filip/georgia_adapt/test-tuned \
  -g 1
