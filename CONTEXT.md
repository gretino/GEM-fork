# FILIP Experiment Context

Domain terminology and resolved design decisions for the FILIP-style ECG alignment model experiment.

## Language

**Unified Data Schema**:
A shared JSON format (similar to MIMIC's `diagnoses.json` and `features.json`) that maps string `study_id` to binary target arrays. Both MIMIC and PTB-XL datasets will be processed into this format using offline scripts before training.
_Avoid_: PTB-XL native format, MIMIC native format

**Feature Alignment Stage (Stage 1)**:
Pretraining the model on MIMIC-IV-ECG using `feature_targets` (20 intermediate features) to learn patch-to-feature similarity.

**Diagnosis Adaptation Stage (Stage 2)**:
Adapting the pretrained model on PTB-XL using `diagnosis_targets` (specifically, the 5 diagnostic superclasses: NORM, MI, HYP, CD, STTC), applying a feature-consistency loss to maintain the patch-level interpretability learned in Stage 1.

**Feature Alignment Logit Scale**:
A learnable scalar parameter initialized to 14.28 (1/0.07) and clipped to prevent gradient explosion. It scales the cosine similarity before pooling to produce feature logits.

**Standalone Vision Encoder**:
A clean wrapper around `transformers.CLIPVisionModel` implemented inside `/filip/model/vision_encoder.py`. It explicitly avoids depending on the GEM `llava` codebase to keep the experiment isolated.

## Relationships

- **Unified Data Schema** applies to both **Feature Alignment Stage** and **Diagnosis Adaptation Stage**.

## Flagged ambiguities

- *PTB-XL Dataset Format*: `FILIP-experiment.md` assumed PTB-XL was already preprocessed to the correct format. Resolved: PTB-XL will be processed by a separate offline script to match the **Unified Data Schema** and its images will be regenerated. The dataloader can safely assume the unified format.
