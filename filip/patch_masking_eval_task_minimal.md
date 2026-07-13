# Minimal Patch-Masking Interpretability Evaluation for ECG-FILIP

## Goal

Implement a small patch-masking sanity test for the ECG-FILIP model.

The purpose is to check whether the model's feature-level patch heatmaps are actually connected to its feature predictions. If the model highlights certain ECG image patches for a feature such as ST depression, T-wave inversion, bundle branch block, or atrial fibrillation, then masking those highlighted patches should reduce the predicted probability for that feature more than masking random patches.

This is an evaluation-only task. Do not modify the training pipeline unless a small non-invasive model-output hook is needed to expose patch-feature scores.

## Background

The current training pipeline has two stages:

1. Stage 1 trains the ECG-FILIP model on MIMIC-style ECG images with intermediate feature labels.
2. Stage 2 adapts the model to PTB-XL diagnosis prediction.

Recent experiments showed that ViT-L/14 and finer visual tokenization significantly improve PTB-XL performance. Direct ViT-L/14 training on PTB-XL performs worse than MIMIC Stage 1 followed by PTB-XL adaptation, so Stage 1 pretraining appears useful. However, changing the Stage 2 feature-consistency loss does not strongly change diagnosis metrics.

Because the main value of ECG-FILIP is interpretability, the next step is not more diagnosis ablation. The next step is to directly test whether the learned feature-to-patch heatmaps behave meaningfully.

## Main Evaluation Question

For a selected ECG feature and image:

- Run the model normally and record the feature probability.
- Use the feature-specific patch heatmap to find the most relevant patches.
- Mask those top-scoring patches.
- Run the model again and record the new feature probability.
- Compare the probability drop against masking the same number of random patches.

Expected result:

- Top-score patch masking should reduce the selected feature probability more than random patch masking.

If this holds consistently, it supports the claim that the heatmap is model-grounded rather than only visually decorative.

## Keep the First Version Minimal

Do only the following in the first implementation:

- Use one trained checkpoint.
- Use one dataset split.
- Evaluate feature predictions, not final diagnosis predictions.
- Use one mask ratio: 10 percent of image patches.
- Use two masking conditions only: top-k patches and random-k patches.
- Use known positive feature labels when available.
- Save CSV summaries and a small set of visual examples.

Do not implement the following in the first version:

- Multiple mask ratios.
- Bottom-k masking.
- Insertion curves.
- Diagnosis-level masking.
- Clinical localization scoring.
- Extra training runs.
- New loss functions.

These can be added later only if the minimal test shows a useful signal.

## Required Inputs

The script should accept:

- Path to a trained ECG-FILIP checkpoint.
- Path or config name for the evaluation dataset split.
- Model configuration needed to restore the image encoder and feature branch.
- Feature vocabulary or feature-name mapping.
- Image size and ViT patch size.
- Output directory.

Prefer evaluating on the MIMIC validation or test split first, because MIMIC has intermediate feature labels. PTB-XL can be evaluated later only as a prediction-based sanity check if feature labels are unavailable.

## Required Model Outputs

The model evaluation path needs to return:

- Feature logits for intermediate ECG features.
- Feature probabilities after sigmoid.
- Feature-specific patch scores or patch-feature similarity scores.

The patch-feature score tensor must allow the script to rank image patches for a specific feature.

If the current forward pass does not return patch-feature scores, add an evaluation-only option that returns them. Keep normal training behavior unchanged.

## Sample Selection

For the first version, evaluate only known positive feature labels.

Use image-feature pairs where:

- The feature label is present.
- The label value is `1` if the dataset uses `1 = present`, `0 = explicitly absent`, and `-1 = unknown`.

Do not treat unknown labels as negatives. Do not evaluate every feature for every image. Most image-feature pairs are irrelevant and will make the result noisy.

If the number of positive examples is very large, allow a max-sample limit per feature so the evaluation finishes quickly.

## Masking Method

Use image-space patch masking.

For each selected image-feature pair:

1. Rank the patches using that feature's patch scores.
2. Select the top 10 percent of patches.
3. Replace those patch regions with a background-like value or local mean value.
4. Re-run the model and measure the feature probability drop.
5. Repeat with random patches of the same count.

Use the same masking method for top-k and random-k conditions.

Avoid black square masks unless the ECG image background is naturally black. For standard ECG images, use a white/background-like value or local mean replacement. The masking should remove signal information without creating a strong artificial artifact.

Apply masking consistently with the preprocessing pipeline. Prefer masking before normalization if the dataset pipeline makes that practical.

## Random Baseline

Random masking is the only required control in the first version.

For each image-feature pair, run multiple random trials and average them. Use 10 random trials by default. More trials can be used later if the result is noisy.

The important comparison is:

- Probability drop after top-k masking.
- Average probability drop after random-k masking.

A meaningful heatmap should produce a positive top-random gap.

## Metrics

For each evaluated image-feature pair, compute:

- Sample ID.
- Feature name.
- Baseline feature probability.
- Feature probability after top-k masking.
- Mean feature probability after random-k masking.
- Standard deviation of random-masked feature probability.
- Absolute top-k probability drop.
- Absolute random-k probability drop.
- Top-random gap.
- Normalized top-k probability drop.
- Normalized random-k probability drop.

The main metric is:

```text
Top-random gap = top-k probability drop - random-k probability drop
```

A positive gap means the heatmap-selected patches affect the feature prediction more than random patches.

## Output Files

Write results to an output directory such as:

```text
outputs/masking_eval_minimal/
```

Required files:

- `pair_level_results.csv`
- `summary_by_feature.csv`
- `summary_overall.csv`
- `examples/`

## `pair_level_results.csv`

This file should contain one row per evaluated image-feature pair.

Include:

- Sample ID.
- Feature name.
- Baseline feature probability.
- Top-k masked probability.
- Random-k masked probability mean.
- Random-k masked probability standard deviation.
- Top-k probability drop.
- Random-k probability drop.
- Top-random gap.
- Normalized top-k drop.
- Normalized random-k drop.

## `summary_by_feature.csv`

This file should aggregate results by feature.

Include:

- Feature name.
- Number of evaluated samples.
- Mean baseline probability.
- Mean top-k probability drop.
- Mean random-k probability drop.
- Mean top-random gap.
- Mean normalized top-k drop.
- Mean normalized random-k drop.

## `summary_overall.csv`

This file should aggregate across all evaluated image-feature pairs.

Include:

- Number of evaluated image-feature pairs.
- Number of evaluated features.
- Mean top-k probability drop.
- Mean random-k probability drop.
- Mean top-random gap.
- Mean normalized top-k drop.
- Mean normalized random-k drop.

## Visual Examples

Save a small number of qualitative examples, for example 20 total examples.

Each example should include:

- Original ECG image.
- Feature heatmap overlay.
- Top-k masked image.
- Random-k masked image.
- Baseline feature probability.
- Top-k masked feature probability.
- Random-k masked feature probability.

Select examples from known positive feature labels with reasonably high baseline feature probability.

## Success Criteria

The first version is successful if it can answer this question:

```text
Do feature heatmap-selected patches reduce the corresponding feature probability more than random patches?
```

A useful result should show:

- Mean top-k probability drop greater than mean random-k probability drop.
- Positive top-random gap overall.
- Positive top-random gap for at least some clinically meaningful features.
- Visual examples where the highlighted and masked regions are plausible.

If the gap is small or inconsistent, do not immediately add more ablations. First inspect whether the patch-feature scores, patch indexing, image preprocessing, and masking value are implemented correctly.

## Implementation Priority

The coding agent should prioritize:

1. Correct extraction of feature logits and patch-feature scores.
2. Correct mapping from patch index to image region.
3. Correct masking in the same image space used by the model.
4. Reliable CSV output.
5. A small number of visual examples.

Keep this implementation small and easy to debug. The goal is a first sanity check, not a full interpretability benchmark.
