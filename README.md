# GEM & FILIP-ECG: Fine-Grained Multimodal ECG Alignment & Grounded Understanding

This repository contains the codebase for grounded ECG understanding and fine-grained patch-level alignment models:
1. **FILIP-ECG Pipeline (`filip/`)**: A fine-grained token- and patch-level alignment architecture for 12-lead ECG waveforms and rendered images. Stage 1 pretrains vision backbones on MIMIC-IV-ECG (via intermediate clinical features or raw cardiology reports), and Stage 2 adapts representations across a comprehensive 6-task downstream diagnostic benchmark suite.
2. **GEM Multimodal Foundation Model (`llava/`, `ecg_coca/`)**: An ECG-focused multimodal large language model (MLLM) combining 12-lead ECG waveforms, rendered paper ECG images, and text instructions.

---

## Quick Navigation

* **Experiment Execution Guide**: [filip/RUN_GUIDE.md](filip/RUN_GUIDE.md) (Complete instructions for data preparation, pretraining, multi-GPU adaptation runners, threshold tuning, and evaluation)
* **Dataset Splits & Benchmarks**: [docs/filip_dataset_splits_summary.md](docs/filip_dataset_splits_summary.md) (Sample counts and diagnostic class breakdowns for all 6 downstream datasets)
* **FILIP Architecture & Specification**: [FILIP-experiment.md](FILIP-experiment.md)
* **Repository Architecture Summary**: [docs/REPOSITORY_SUMMARY.md](docs/REPOSITORY_SUMMARY.md)

---

## Quickstart: Running Experiments

### 1. Data Setup
Prepare the 6 downstream evaluation benchmark splits (PTB-XL Superclass, Subclass, Rhythm, Form, CPSC 2018, and CSN):
```bash
python filip/data_processing/prepare_verified_splits.py
```

### 2. Stage 1 Pretraining (MIMIC-IV-ECG)
Pretrain the vision encoder (ViT-Large report alignment):
```bash
bash filip/scripts/train_mimic_feature.sh \
  -c filip/configs/mimic_report_alignment_vit_large_pretrain.yaml \
  -g 0
```

### 3. Stage 2 Downstream Adaptation & Evaluation
Use the generic multi-GPU runner with a run manifest (automatically executes training, validation threshold tuning, test evaluation, and Markdown report generation):
```bash
bash filip/scripts/run_experiments.sh \
  -m filip/configs/runs/raw_clip_baseline.yaml \
  -g 0,1,2,3
```

For more details on running individual configs, running split fractions (`-p 1,10,100`), or using interactive visualizers, see [filip/RUN_GUIDE.md](filip/RUN_GUIDE.md).