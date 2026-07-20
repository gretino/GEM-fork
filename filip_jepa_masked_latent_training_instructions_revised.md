# Implementation Instructions: Replace Morphology Regression with Masked Latent Prediction

## 1. Required Change

Remove the seven-metric morphology regression task from the FILIP pretraining variant, including its regression head, targets, normalization, missing-value masks, loss, and logging.

Replace it with a JEPA-style masked latent prediction task while keeping the existing FILIP image-patch alignment objective unchanged.

Use:

```text
L_total = L_FILIP + 0.3 * L_JEPA
```

Initialize from the best checkpoint trained with the original FILIP objective, not from the checkpoint further trained with morphology regression.

## 2. JEPA Components

Use the existing trainable ViT-L/14 image encoder as the context encoder.

Create:

- an EMA target encoder with the same architecture and initial weights as the context encoder;
- a lightweight two-layer Transformer predictor for masked patch representations.

The target encoder must process the complete unmasked image, receive no gradients, remain outside the optimizer, and update after each optimizer step using:

```text
target = 0.996 * target + 0.004 * context
```

The context encoder must process a masked image. Replace masked patch embeddings with a learned mask token before the Transformer blocks so the encoder cannot access their original content.

The predictor must use the contextualized image tokens, mask tokens, and positional embeddings to predict one latent vector for each masked patch location. Do not provide diagnosis labels, MIMIC classes, or morphology metrics to the predictor.

## 3. Masking Procedure

Perform masking on the ViT patch grid using contiguous rectangular blocks.

Never mask:

- the top patch row;
- the bottom patch row;
- the leftmost patch column.

These excluded areas may remain visible as context.

For each image:

- sample four rectangular target blocks;
- use widths of 2–4 patch columns;
- use heights of 1–2 patch rows;
- mask approximately 25–35% of eligible patches in total;
- resample blocks when more than 25% of a new block overlaps an existing block;
- keep at least 50% of eligible patches visible;
- generate a new mask for every image on every training iteration.

Apply the mask only to the context-encoder input. The EMA target encoder must always receive the complete unmasked image.

## 4. Latent Targets and Loss

Use the normalized final-layer spatial patch representations from the EMA target encoder as targets. Exclude the class token and other non-spatial tokens.

Gather target and predicted representations only at masked patch locations.

Use continuous latent prediction without quantization:

```text
prediction = normalize(predicted_masked_latents)
target = stop_gradient(normalize(target_masked_latents))
L_JEPA = SmoothL1(prediction, target)
```

Average the loss across latent dimensions, masked patches, and batch samples.

Log `L_FILIP`, `L_JEPA`, `0.3 * L_JEPA`, and `L_total` separately.

Use an unmasked context-encoder pass for the FILIP objective and a masked context-encoder pass for the JEPA objective. The JEPA predictor and EMA target encoder must not participate in FILIP similarity computation.

## 5. Regular Prediction Dropout

Add standard dropout to the regular downstream classification path:

```text
encoded_features
    -> Dropout(p=0.1)
    -> classifier
    -> logits
```

Apply dropout only during training. Do not drop complete image patches, modify the diagnosis-to-patch similarity maps, or change the existing ViT attention, MLP-dropout, or stochastic-depth settings.

## 6. Saved State

Include the EMA target encoder and latent predictor in pretraining checkpoints so training can resume correctly. They are pretraining-only components and are not required when loading the final image encoder for downstream tasks.
