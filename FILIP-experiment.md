# Project Overview
We are experimenting on a new training scheme inspired by the FILIP model, which tries to align intermediate diagnosis of ECG to image patches instead of raw images. We want to use it to increase the interpretability of the ECG diagnosis and potentially improve the model performance.

For this project, we want to avoid changing the current codebase, so if certain parts of the codebase doesn't work, please write a new script variation but do not change the original code. This is important.
Any new code should be placed in /filip/ directory while maintaining the directory structures.

# FILIP-Style ECG Image Patch-to-Feature Alignment: Implementation Plan

## 1. Final design decision

Implement a FILIP-inspired ECG image model as a separate experiment path under `/filip/`.

The model should learn:

```text
ECG image patches -> intermediate ECG feature items -> diagnosis
```

Use the preprocessed data already available in the repository. Read `FILIP-data.md` before implementing the dataset loader. Treat `FILIP-data.md` as the source of truth for data paths, schemas, label names, splits, and expected fields.

This implementation is a new experiment branch. Keep the current GEM implementation runnable as the baseline.

## 2. Required experiment stages

Implement two training stages.

### Stage 1: MIMIC feature-alignment pretraining

Train the ECG image encoder and feature-alignment head on MIMIC-derived intermediate ECG feature labels.

The input is:

```text
ECG image
```

The supervised target is:

```text
intermediate ECG feature labels
```

The model learns:

```text
image patch tokens -> feature logits
```

Save this checkpoint as the FILIP-style ECG image foundation model.

### Stage 2: PTB-XL diagnosis adaptation

Load the Stage 1 checkpoint.

Adapt the model on PTB-XL diagnosis labels using a small adaptation training setup.

The input is:

```text
ECG image
```

The supervised target is:

```text
PTB-XL diagnosis labels
```

The model should still compute intermediate feature logits and patch-feature similarity maps during PTB-XL adaptation.

The PTB-XL adaptation should train:

```text
image encoder adapters / LoRA modules
feature-alignment projection if enabled by config
diagnosis head
```

Use diagnosis loss as the main PTB-XL loss.

Add a feature-consistency loss to preserve the MIMIC-trained feature behavior during PTB-XL adaptation.

## 3. Directory structure

Create the following structure under `/filip/`:

```text
/filip/
  configs/
    mimic_feature_pretrain.yaml
    ptbxl_diagnosis_adapt.yaml

  data/
    dataset.py
    collator.py
    vocab.py

  model/
    vision_encoder.py
    feature_alignment.py
    filip_ecg_model.py
    losses.py

  train/
    train_mimic_feature.py
    train_ptbxl_adapt.py

  eval/
    evaluate_features.py
    evaluate_diagnosis.py
    visualize_patch_alignment.py

  scripts/
    train_mimic_feature.sh
    train_ptbxl_adapt.sh
    eval_features.sh
    eval_diagnosis.sh
    visualize_alignment.sh

  utils/
    metrics.py
    checkpoint.py
    logging.py
    seed.py
```

Preserve the original repository structure outside `/filip/`.

## 4. Data loader requirements

Implement all new data loading in:

```text
/filip/data/dataset.py
/filip/data/collator.py
/filip/data/vocab.py
```

Use the preprocessed data described in `FILIP-data.md`.

The dataset loader should support samples with any subset of the following labels:

```text
image
feature_targets
feature_mask
feature_confidence
diagnosis_targets
diagnosis_mask
```

Use this label convention:

```text
1  = present
0  = explicitly absent
-1 = unknown / ignored
```

The collator should produce tensors:

```python
batch["images"]              # [B, C, H, W]
batch["feature_targets"]     # [B, F], optional
batch["feature_mask"]        # [B, F], optional
batch["feature_confidence"]  # [B, F], optional
batch["diagnosis_targets"]   # [B, C], optional
batch["diagnosis_mask"]      # [B, C], optional
batch["sample_ids"]          # list[str]
```

Use `feature_mask` and `diagnosis_mask` to decide which losses are active for each batch.

## 5. Model architecture

Implement the model in:

```text
/filip/model/filip_ecg_model.py
```

The model should contain:

```text
vision_encoder
feature_alignment_head
diagnosis_head
```

The forward pass should produce:

```python
{
    "patch_features": patch_features,
    "feature_logits": feature_logits,
    "diagnosis_logits": diagnosis_logits,
    "patch_feature_similarity": patch_feature_similarity
}
```

Expected shapes:

```text
patch_features:            [B, P, H]
feature_logits:            [B, F]
diagnosis_logits:          [B, C]
patch_feature_similarity:  [B, P, F]
```

Where:

```text
B = batch size
P = number of image patches
H = hidden size
F = number of intermediate ECG feature items
C = number of diagnosis classes
```

## 6. Vision encoder

Implement the vision encoder wrapper in:

```text
/filip/model/vision_encoder.py
```

Use an image encoder that returns patch-level tokens.

The encoder output must be patch tokens, not only a pooled global embedding.

Required output:

```python
patch_features = vision_encoder(images)
```

Shape:

```text
[B, P, H]
```

Support loading pretrained weights from the existing GEM-compatible vision tower if available, but keep the implementation inside `/filip/`.

## 7. Feature-alignment head

Implement the FILIP-style patch-feature alignment module in:

```text
/filip/model/feature_alignment.py
```

The module should compute similarity between each ECG image patch and each ECG feature item.

Required computation:

```python
V = image_patch_projection(patch_features)   # [B, P, A]
Q = feature_embedding.weight                 # [F, A]

V = normalize(V)
Q = normalize(Q)

similarity = einsum("bpa,fa->bpf", V, Q) * scale
```

Then pool over patches to get feature logits:

```python
feature_logits = topk_mean(similarity, dim=patch_dim)
```

Default pooling:

```text
top-k mean
```

Default `topk`:

```text
8
```

The feature-alignment head should return:

```python
feature_logits
patch_feature_similarity
```

## 8. Diagnosis head

Implement the diagnosis head inside:

```text
/filip/model/filip_ecg_model.py
```

Use feature logits as the main diagnosis input.

Default diagnosis path:

```python
diagnosis_logits = diagnosis_head(feature_logits)
```

Support an optional config mode that concatenates pooled image embedding with feature logits:

```python
diagnosis_logits = diagnosis_head(concat([pooled_image_embedding, feature_logits]))
```

Use the feature-only path as the default method.

## 9. Losses

Implement losses in:

```text
/filip/model/losses.py
```

### 9.1 MIMIC feature loss

Use binary cross entropy with logits.

Apply `feature_mask`.

Apply `feature_confidence` when available.

Formula:

```python
raw_loss = BCEWithLogits(feature_logits, feature_targets)
weighted_loss = raw_loss * feature_confidence
feature_loss = weighted_loss[feature_mask].sum() / feature_confidence[feature_mask].sum().clamp_min(1.0)
```

When `feature_confidence` is unavailable, use confidence value `1.0` for valid labels.

### 9.2 PTB-XL diagnosis loss

Use binary cross entropy with logits.

Apply `diagnosis_mask`.

```python
diagnosis_loss = BCEWithLogits(diagnosis_logits, diagnosis_targets)
diagnosis_loss = diagnosis_loss[diagnosis_mask].mean()
```

### 9.3 Feature-consistency loss for PTB-XL adaptation

During Stage 2, keep a frozen copy of the Stage 1 model or cache frozen feature logits.

Compute:

```python
current_feature_probs = sigmoid(current_feature_logits)
frozen_feature_probs = sigmoid(frozen_feature_logits)

feature_consistency_loss = MSE(current_feature_probs, frozen_feature_probs)
```

Use this loss during PTB-XL adaptation:

```python
total_loss = diagnosis_loss + lambda_consistency * feature_consistency_loss
```

Default:

```text
lambda_consistency = 0.1
```

## 10. Stage 1 training script

Implement:

```text
/filip/train/train_mimic_feature.py
/filip/scripts/train_mimic_feature.sh
```

This script should:

1. Load the MIMIC feature-alignment training split described in `FILIP-data.md`.
2. Build the ECG image encoder.
3. Build the feature-alignment head.
4. Train using feature loss.
5. Log feature metrics.
6. Save the Stage 1 checkpoint.

Required outputs:

```text
/outputs/filip/mimic_feature_pretrain/checkpoints/
/outputs/filip/mimic_feature_pretrain/logs/
/outputs/filip/mimic_feature_pretrain/config.yaml
```

## 11. Stage 2 adaptation script

Implement:

```text
/filip/train/train_ptbxl_adapt.py
/filip/scripts/train_ptbxl_adapt.sh
```

This script should:

1. Load the Stage 1 MIMIC feature-pretrained checkpoint.
2. Load the PTB-XL diagnosis training split described in `FILIP-data.md`.
3. Enable small adaptation modules such as LoRA/adapters when configured.
4. Keep feature-logit computation active.
5. Compute diagnosis loss.
6. Compute feature-consistency loss.
7. Save the adapted PTB-XL checkpoint.

Required outputs:

```text
/outputs/filip/ptbxl_diagnosis_adapt/checkpoints/
/outputs/filip/ptbxl_diagnosis_adapt/logs/
/outputs/filip/ptbxl_diagnosis_adapt/config.yaml
```

## 12. Configuration files

Create:

```text
/filip/configs/mimic_feature_pretrain.yaml
/filip/configs/ptbxl_diagnosis_adapt.yaml
```

The MIMIC config should include:

```yaml
dataset_name: mimic
task: feature_pretrain
data_config: FILIP-data.md

model:
  vision_encoder: clip_vit
  use_feature_alignment: true
  feature_pooling: topk
  feature_topk: 8
  feature_align_dim: 256

loss:
  feature_loss_weight: 1.0

training:
  batch_size: 32
  learning_rate: 1.0e-4
  epochs: 10
  mixed_precision: true
```

The PTB-XL config should include:

```yaml
dataset_name: ptbxl
task: diagnosis_adapt
data_config: FILIP-data.md
stage1_checkpoint: /outputs/filip/mimic_feature_pretrain/checkpoints/best.pt

model:
  use_feature_alignment: true
  diagnosis_from_features: true
  use_adapters: true
  adapter_type: lora

loss:
  diagnosis_loss_weight: 1.0
  feature_consistency_weight: 0.1

training:
  batch_size: 32
  learning_rate: 5.0e-5
  epochs: 10
  mixed_precision: true
```

Adjust paths and hyperparameters according to the actual data specification in `FILIP-data.md`.

## 13. Evaluation

Implement:

```text
/filip/eval/evaluate_features.py
/filip/eval/evaluate_diagnosis.py
```

### 13.1 Feature evaluation

For MIMIC feature-alignment evaluation, report:

```text
feature macro AUC
feature micro AUC
feature macro F1
feature micro F1
per-feature AUC
per-feature F1
```

Use only valid labels according to `feature_mask`.

### 13.2 Diagnosis evaluation

For PTB-XL diagnosis evaluation, report:

```text
diagnosis macro AUC
diagnosis micro AUC
diagnosis macro F1
diagnosis micro F1
hamming loss
subset accuracy
```

Use only valid labels according to `diagnosis_mask`.

Save metrics to:

```text
/outputs/filip/{experiment_name}/metrics.json
/outputs/filip/{experiment_name}/per_class_metrics.csv
```

## 14. Patch-feature visualization

Implement:

```text
/filip/eval/visualize_patch_alignment.py
/filip/scripts/visualize_alignment.sh
```

The visualization script should:

1. Load a trained checkpoint.
2. Load selected ECG image samples.
3. Compute `patch_feature_similarity`.
4. Select target features.
5. Overlay high-similarity patches on the ECG image.
6. Save visualization images.

Required output path:

```text
/outputs/filip/{experiment_name}/alignment_visualizations/
```

For each sample, save:

```text
{sample_id}_{feature_id}.png
```

Also save a JSON sidecar:

```json
{
  "sample_id": "...",
  "feature_id": "...",
  "top_patch_indices": [...],
  "top_patch_scores": [...]
}
```

## 15. Logging

Log the following during Stage 1:

```text
loss/feature
feature/macro_auc
feature/micro_auc
feature/macro_f1
feature/micro_f1
feature/valid_label_ratio
```

Log the following during Stage 2:

```text
loss/diagnosis
loss/feature_consistency
loss/total
diagnosis/macro_auc
diagnosis/micro_auc
diagnosis/macro_f1
diagnosis/micro_f1
```

Log the learning rate, batch size, checkpoint path, and config path for every run.

## 16. Checkpoint requirements

Save checkpoints with the following fields:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "config": ...,
    "feature_vocab": ...,
    "diagnosis_vocab": ...,
    "epoch": ...,
    "global_step": ...,
    "best_metric": ...
}
```

Stage 2 checkpoints should record the Stage 1 checkpoint path used for initialization.

## 17. Required experiment outputs

After implementation, produce these runnable commands:

```bash
bash /filip/scripts/train_mimic_feature.sh
bash /filip/scripts/train_ptbxl_adapt.sh
bash /filip/scripts/eval_features.sh
bash /filip/scripts/eval_diagnosis.sh
bash /filip/scripts/visualize_alignment.sh
```

Each script should use config files under:

```text
/filip/configs/
```

Each script should write outputs under:

```text
/outputs/filip/
```

## 18. Success criteria

The implementation is complete when the following are true:

```text
1. The original GEM baseline code still runs from the existing scripts.
2. All FILIP adaptation code lives under /filip/.
3. The MIMIC feature-pretraining script runs from start to checkpoint save.
4. The PTB-XL adaptation script loads the MIMIC checkpoint and trains diagnosis prediction.
5. The model outputs feature logits and diagnosis logits.
6. The model outputs patch-feature similarity maps.
7. Feature evaluation works on MIMIC validation/test splits.
8. Diagnosis evaluation works on PTB-XL validation/test splits.
9. Patch-feature visualization saves ECG image overlays.
10. All data loading follows the schema documented in FILIP-data.md.
```

## 19. Final implementation summary

Build a separate FILIP-style ECG image model under `/filip/`.

Use MIMIC to train image patch-to-feature alignment.

Use PTB-XL to adapt the model for diagnosis prediction.

Keep feature logits and patch-feature similarity active during PTB-XL adaptation so the model natively supports image-level feature alignment and interpretability.

Keep the original GEM code unchanged so it remains available as the baseline.
