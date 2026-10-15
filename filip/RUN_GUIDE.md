# FILIP Experiment Execution Guide

This guide provides instructions for data preparation, training, downstream adaptation, evaluation, and visualization for the **FILIP-style ECG alignment experiment pipeline**.

---

## Architecture Overview

The FILIP architecture consists of a two-stage training scheme:
1. **Stage 1 (MIMIC-IV-ECG Pretraining)**:
   - **Feature-Alignment**: Aligns ECG image patches with intermediate ECG clinical feature labels.
   - **Report-Alignment**: Aligns ECG image patches with raw cardiology report tokens (ViT-Base or ViT-Large backbone).
2. **Stage 2 (Downstream Adaptation & Evaluation)**:
   - Adapts the pretrained vision encoder representations across **6 verified benchmark datasets** under varying data split percentages (1%, 10%, 100%).
   - Preserves feature-level interpretability via feature-consistency or prompt-based text adaptation.
   - Uses validation-set threshold tuning to optimize class-specific decision boundaries before final test evaluation.

For dataset split counts and class distributions, see [docs/filip_dataset_splits_summary.md](file:///home/qfbqt/repo/GEM-fork/docs/filip_dataset_splits_summary.md).

---

## 0. Data Preparation Pipeline

### 0.1 Stage 1 Pretraining Data (MIMIC-IV-ECG)

Before running Stage 1 pretraining, process raw MIMIC-IV-ECG waveforms into formatted JSON metadata and 12-lead ECG images:

```bash
# Navigate to repository root
cd /home/qfbqt/repo/GEM-fork

# 1. Build master record list from raw files (typically run once)
python filip/data_processing/build_master_list.py

# 2. Sample records into 8:1:1 train/val/test splits (e.g. 10,000 records)
python filip/data_processing/sample_splits.py --num_items 10000

# 3. Extract clinical features and machine measurements
python filip/data_processing/extract_clinical_features.py
python filip/data_processing/extract_measurements.py

# 4. Render ECG signals into 1x12 layout images
python filip/scripts/generate_mimic_subset.py
```

### 0.2 Stage 2 Downstream Benchmark Data (6 Verified Tasks)

The benchmark suite evaluates across 6 downstream tasks:
* **PTB-XL Superclass** (`ptbxl_super` - 5 classes)
* **PTB-XL Subclass** (`ptbxl_sub` - 23 classes)
* **PTB-XL Rhythm** (`ptbxl_rhythm` - 12 classes)
* **PTB-XL Form** (`ptbxl_form` - 19 classes)
* **CPSC 2018 / ICBEB** (`cpsc2018` - 9 classes)
* **Chapman-Shaoxing-Ningbo** (`csn` - 38 classes)

#### Step 1: Render Waveform Images (if not already generated)
```bash
# PTB-XL images (1x12 layout)
bash scripts/generate_ecg_images.sh

# CPSC 2018 images (1x12 layout)
python gem_generation/ecg-image-generator/gen_ecg_images_from_data_batch.py \
  -i /home/qfbqt/8TB/datasets/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018 \
  -o data/cpsc2018/images \
  --num_columns 1 \
  --num_threads 16

# CSN images (1x12 layout)
python gem_generation/ecg-image-generator/gen_ecg_images_from_data_batch.py \
  -i /home/qfbqt/8TB/datasets/physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords \
  -o /home/qfbqt/8TB/datasets/csn_new \
  --num_columns 1 \
  --num_threads 16
```

#### Step 2: Generate Verified Split Metadata
Run the unified preparation script to create split JSONs and symlinks for all 6 tasks under `data/`:
```bash
python filip/data_processing/prepare_verified_splits.py
```
This produces:
* `data/ptbxl_super_class/{train,val,test}_records.json` & `diagnoses.json`
* `data/ptbxl_sub_class/{train,val,test}_records.json` & `diagnoses.json`
* `data/ptbxl_rhythm/{train,val,test}_records.json` & `diagnoses.json`
* `data/ptbxl_form/{train,val,test}_records.json` & `diagnoses.json`
* `data/cpsc2018_verified/{train,val,test}_records.json` & `diagnoses.json`
* `data/csn_verified/{train,val,test}_records.json` & `diagnoses.json`

---

## 1. Stage 1: MIMIC-IV-ECG Pretraining

Stage 1 configurations are located in `filip/configs/`.

### Report-Alignment Pretraining (ViT-Large, Recommended)
```bash
bash filip/scripts/train_mimic_feature.sh \
  -c filip/configs/mimic_report_alignment_vit_large_pretrain.yaml \
  -g 0
```

### Feature-Alignment Pretraining (Base)
```bash
bash filip/scripts/train_mimic_feature.sh \
  -c filip/configs/mimic_feature_pretrain.yaml \
  -g 0
```

### Background Execution with Logging
```bash
mkdir -p filip/logs
nohup bash filip/scripts/train_mimic_feature.sh \
  -c filip/configs/mimic_report_alignment_vit_large_pretrain.yaml \
  -g 0 \
  > filip/logs/stage1_pretrain.log 2>&1 &

# Monitor progress
tail -f filip/logs/stage1_pretrain.log
```

---

## 2. Stage 2: Downstream Adaptation & Evaluation

Downstream evaluation follows a **3-step lifecycle** for every task:
1. **Train** downstream adaptation model on training split (`train_ptbxl_adapt.py`).
2. **Tune** class-specific decision thresholds on validation split (`tune_thresholds.py`).
3. **Evaluate** model on test split using tuned thresholds (`evaluate_diagnosis.py`).

There are multiple ways to execute Stage 2 experiments depending on your workflow.

---

### Method A: Generic Multi-GPU Runner (Recommended)

[`filip/scripts/run_experiments.sh`](file:///home/qfbqt/repo/GEM-fork/filip/scripts/run_experiments.sh) is a multi-GPU queue runner. It distributes tasks across idle or specified GPUs, executes all 3 lifecycle steps per task, and generates an aggregated Markdown summary report.

#### 1. Running a Manifest from `filip/configs/runs/*.yaml`
Manifests define a group of configs and data fractions to run:
```bash
# Run baseline experiments on GPUs 0, 1, 2, 3 across all splits (1%, 10%, 100%)
bash filip/scripts/run_experiments.sh \
  -m filip/configs/runs/raw_clip_baseline.yaml \
  -g 0,1,2,3

# Run ablation baseline manifest
bash filip/scripts/run_experiments.sh \
  -m filip/configs/runs/ablation_baseline.yaml \
  -g 0,1
```

#### 2. Running Individual Configs with the Generic Runner
You can also pass individual config files directly:
```bash
# Run a single config for 100% data on GPU 0
bash filip/scripts/run_experiments.sh \
  filip/configs/report_alignment_adapt_vit_large/ptbxl_sub_adapt.yaml \
  -g 0 \
  -p 100

# Run multiple configs across 1%, 10%, and 100% splits
bash filip/scripts/run_experiments.sh \
  filip/configs/report_alignment_adapt_vit_large/ptbxl_super_adapt.yaml \
  filip/configs/report_alignment_adapt_vit_large/ptbxl_sub_adapt.yaml \
  -g 0,1 \
  -p 1,10,100
```

#### Key Command-Line Options for `run_experiments.sh`:
| Option | Description | Default |
| :--- | :--- | :--- |
| `-m, --manifest <file>` | Path to run manifest YAML under `filip/configs/runs/` | None |
| `-g, --gpus <list>` | Comma-separated GPU IDs (e.g. `0,1,2,3`) | Auto-detects idle GPUs |
| `-p, --train_pct <list>` | Comma-separated data split percentages (e.g. `1,10,100`) | Manifest setting or `100` |
| `-r, --resume` | Automatically resume from `latest.pt` if available | False |
| `-s, --skip_completed <bool>`| Skip tasks where `evaluation/metrics.txt` already exists | `true` |
| `--dry_run` | Print queued tasks and detected GPUs without running | False |

#### Automated Output Report:
Upon completion of all tasks, `run_experiments.sh` writes a summary table to:
`outputs/filip/{run_name}_summary_report.md`

---

### Method B: Experiment Group Runner (Single GPU, Multiple Splits)

[`filip/scripts/run_experiment_group.sh`](file:///home/qfbqt/repo/GEM-fork/filip/scripts/run_experiment_group.sh) is ideal when you want to run a single configuration sequentially across multiple data percentages on a dedicated GPU:

```bash
bash filip/scripts/run_experiment_group.sh \
  -c filip/configs/report_alignment_adapt_vit_large/ptbxl_sub_adapt.yaml \
  -n vit_large_ptbxl_sub \
  -g 0 \
  -p "1 10 100"
```

---

### Method C: End-to-End Pipeline Scripts

For standardized end-to-end benchmarking, dedicated pipeline scripts automate Stage 1 + Stage 2 execution:

* **ViT-Large Full Pipeline** (`filip/scripts/run_vit_large_pipeline.sh`):
  Checks for Stage 1 checkpoint (trains if missing), then runs Stage 2 adaptation across all 6 benchmark datasets at 1%, 10%, and 100% data splits in parallel across available GPUs:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 bash filip/scripts/run_vit_large_pipeline.sh
  ```

* **JEPA Pretraining + Adaptation Pipeline** (`filip/scripts/run_report_alignment_jepa_large_pipeline.sh`):
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 bash filip/scripts/run_report_alignment_jepa_large_pipeline.sh
  ```

* **Patch Size Ablation Pipeline (16 vs 32)** (`filip/scripts/run_patch_size_ablation.sh`):
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 bash filip/scripts/run_patch_size_ablation.sh
  ```

* **Baseline Ablations (Frozen vs Unfreeze Last 2 Layers)** (`filip/scripts/run_ablation_experiments.sh`):
  ```bash
  bash filip/scripts/run_ablation_experiments.sh
  ```

---

### Method D: Manual Single-Task Execution (Debugging / Custom Runs)

If you need to manually step through training, threshold tuning, and evaluation for a single task:

```bash
CONFIG="filip/configs/report_alignment_adapt_vit_large/ptbxl_sub_adapt.yaml"
OUT_DIR="outputs/filip/vit_large_ptbxl_sub_100"
GPU=0

# Step 1: Train downstream adaptation
CUDA_VISIBLE_DEVICES=$GPU python filip/train/train_ptbxl_adapt.py \
  --config "$CONFIG" \
  --train_pct 100 \
  --out_dir "$OUT_DIR"

# Step 2: Tune decision thresholds on validation split
CUDA_VISIBLE_DEVICES=$GPU python filip/eval/tune_thresholds.py \
  --config "$CONFIG" \
  --checkpoint "${OUT_DIR}/best.pt" \
  --out_dir "${OUT_DIR}/evaluation"

# Step 3: Evaluate test split metrics using tuned thresholds
CUDA_VISIBLE_DEVICES=$GPU python filip/eval/evaluate_diagnosis.py \
  --config "$CONFIG" \
  --checkpoint "${OUT_DIR}/best.pt" \
  --thresholds_file "${OUT_DIR}/evaluation/tuned_thresholds.json" \
  --out_dir "${OUT_DIR}/evaluation"
```

Alternatively, use the single-stage wrapper scripts:
```bash
# Train
bash filip/scripts/train_ptbxl_adapt.sh -c "$CONFIG" -g "$GPU"

# Tune thresholds
bash filip/scripts/tune_thresholds.sh -c "$CONFIG" -ckpt "${OUT_DIR}/best.pt" -g "$GPU"

# Evaluate
bash filip/scripts/eval_diagnosis.sh \
  -c "$CONFIG" \
  -ckpt "${OUT_DIR}/best.pt" \
  -t "${OUT_DIR}/evaluation/tuned_thresholds.json" \
  -o "${OUT_DIR}/evaluation" \
  -g "$GPU"
```

---

## 3. Visualization and Interpretability Analysis

### Alignment Visualizations
To generate patch-feature alignment heatmap overlays on sample ECG images:
```bash
bash filip/scripts/visualize_alignment.sh
```

### Interactive Notebooks
* **Text & Patch Alignment Explorer** (`filip/eval/text_alignment_visualizer.ipynb`):
  Loads a Stage 1 report checkpoint or Stage 2 model, scores ECGs against prompt text/diagnoses, and visualizes token-level patch alignment heatmaps.
* **Interactive Diagnostic Visualizer** (`filip/eval/interactive_visualizer.ipynb`):
  Provides interactive inspection of ECG lead segments and attention maps.

### Patch Masking Interpretability Evaluation
To run the minimal patch-masking sanity check (verifying whether masking top-aligned patches degrades class logits more than random masking), consult:
[filip/patch_masking_eval_task_minimal.md](file:///home/qfbqt/repo/GEM-fork/filip/patch_masking_eval_task_minimal.md)
