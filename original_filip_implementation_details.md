# FILIP Implementation Details Summary

Source paper: **FILIP: Fine-grained Interactive Language-Image Pre-training**, ICLR 2022.

## 1. Core Implementation Idea

FILIP keeps the efficient **dual-stream CLIP-style architecture**, but replaces the usual global image-text cosine similarity with a **fine-grained token-level late-interaction similarity**.

The key implementation change is in the contrastive loss:

- Encode image and text independently.
- Keep token-level image patch features and token-level text features.
- Project both modalities into the same embedding space.
- L2-normalize token embeddings.
- Compute token-wise image-text similarity.
- For each visual token, take the most similar textual token.
- For each textual token, take the most similar visual token.
- Average these token-wise maximum similarities to produce image-to-text and text-to-image logits.
- Use the resulting directional logits in a symmetric contrastive loss.

This means FILIP does **not** add cross-attention, self-attention over concatenated image-text tokens, object detectors, region proposals, or extra localization supervision. The fine-grained alignment emerges from the contrastive objective itself.

## 2. Model Architecture

FILIP uses separate image and text encoders.

### 2.1 Image Encoder

The image encoder follows CLIP-style ViT design.

The paper trains two variants:

| Model | Input resolution | Image encoder layers | Width | Heads |
|---|---:|---:|---:|---:|
| FILIPbase | 224 x 224 | 12 | 768 | 12 |
| FILIPlarge | 224 x 224 | 24 | 1024 | 16 |

Implementation notes:

- Images are resized to `224 x 224` during pre-training.
- The image is patchified by the ViT backbone.
- Patch-level visual token outputs are used for FILIP late interaction.
- The paper describes a ViT input with a `[CLS]` token, but later says `[CLS]` is removed for linear probing. For implementation, do not rely on a single global `[CLS]` vector for the FILIP objective. The important representation is the set of visual patch tokens.

### 2.2 Text Encoder

The text encoder follows CLIP's modified decoder-only Transformer design.

| Model | Text encoder layers | Width | Heads |
|---|---:|---:|---:|
| FILIPbase | 12 | 512 | 8 |
| FILIPlarge | 12 | 768 | 12 |

Text implementation details:

- Use lower-cased BPE tokenization.
- Vocabulary size: `49,408`.
- Each text sequence starts with `[BOS]` and ends with `[EOS]`.
- Maximum text length: `77` tokens.
- Padded text tokens must be masked out during late-interaction similarity computation.

### 2.3 Projection Head

Both image and text token representations are linearly projected into the shared multimodal space.

| Model | Shared embedding dimension |
|---|---:|
| FILIPbase | 256 |
| FILIPlarge | 256 |

The paper reduces the projection dimension to `256` mainly for efficiency. After projection, token features are separately L2-normalized.

## 3. Fine-Grained Contrastive Loss

Given a mini-batch of `b` paired samples:

```text
{(image_1, text_1), ..., (image_b, text_b)}
```

The matching image-text pair is the positive pair. Other samples in the same batch are treated as in-batch negatives.

### 3.1 Token Feature Shapes

For image `i` and text `j`:

```text
V_i = image token features with shape [n_img, d]
T_j = text token features with shape [n_txt, d]
```

where:

- `n_img` is the number of visual tokens.
- `n_txt` is the number of non-padded textual tokens.
- `d = 256` in the paper's main implementation.

Compute token-wise similarity:

```text
S_ij = V_i @ T_j.T
```

Shape:

```text
S_ij: [n_img, n_txt]
```

### 3.2 Image-to-Text Similarity

For image-to-text similarity, each image token chooses its most similar text token:

```text
s_I(i, j) = mean_over_image_tokens(max_over_text_tokens(S_ij))
```

Equivalent implementation:

```python
score_i2t = S_ij.max(dim=1).values.mean()
```

### 3.3 Text-to-Image Similarity

For text-to-image similarity, each text token chooses its most similar image token:

```text
s_T(i, j) = mean_over_text_tokens(max_over_image_tokens(S_ij))
```

Equivalent implementation:

```python
score_t2i = S_ij.max(dim=0).values.mean()
```

### 3.4 Directional Similarity Matrices

For a batch, compute two different logit matrices:

```text
logits_i2t[i, j] = s_I(i, j)
logits_t2i[i, j] = s_T(i, j)
```

Important: `s_I(i, j)` and `s_T(i, j)` are not necessarily equal, because they average over different token axes.

### 3.5 Symmetric Contrastive Objective

Use symmetric image-to-text and text-to-image contrastive losses.

For image-to-text:

```text
image i should match text i over all candidate texts j in the batch
```

For text-to-image:

```text
text i should match image i over all candidate images j in the batch
```

The final loss is the average of both directions.

Implementation sketch:

```python
labels = torch.arange(batch_size, device=device)

loss_i2t = cross_entropy(logits_i2t / temperature, labels)
loss_t2i = cross_entropy(logits_t2i.T / temperature, labels)
loss = 0.5 * (loss_i2t + loss_t2i)
```

The paper uses a learnable contrastive temperature initialized to `0.07`.

## 4. Differences from ColBERT-Style Late Interaction

FILIP is inspired by ColBERT-style late interaction, but changes several important details:

1. **Remove padded textual tokens** from similarity calculation.
2. **Average** token-wise maximum similarities instead of summing them.
3. Optimize the score using **symmetric contrastive learning**, not pairwise document ranking loss.

These changes are important. The paper reports that using the original ColBERT-style setting causes worse zero-shot ImageNet accuracy and poorer word-patch alignment, often aligning object patches to padded tokens instead of class-name tokens.

## 5. Distributed Training Efficiency

The naive late-interaction loss is expensive because it requires token-wise similarity between every image and every text in the batch.

FILIP uses three efficiency tricks:

### 5.1 Reduce Projection Dimension

Use token embedding dimension `256` instead of CLIP-style `512` or `768`.

### 5.2 Use FP16 Token Features for Communication and Similarity

Before distributed node communication, convert the last-layer image and text token features from FP32 to FP16.

The matrix multiplication for late interaction is also performed in reduced precision.

### 5.3 Keep Only Top 25% Representative Tokens

Before node communication, each local worker selects the top `25%` tokens with the highest token-wise maximum similarity scores.

The paper's final efficient setting is:

| Loss | Embedding dim | Precision | Token ratio |
|---|---:|---:|---:|
| FILIP late interaction | 256 | FP16 | 25% |

The paper reports that this setting has training time and memory close to the original CLIP loss while keeping most of the performance gain.

## 6. Pre-Training Setup

### 6.1 Common Hyperparameters

| Hyperparameter | Value |
|---|---:|
| Vocabulary size | 49,408 |
| Initial temperature | 0.07 |
| LAMB beta1 | 0.9 |
| LAMB beta2 | 0.999 |
| LAMB epsilon | 1e-4 |
| Warm-up iterations | 3,000 |
| Training epochs | 30 |

Other training details:

- Optimizer: LAMB.
- Scheduler: cosine learning rate schedule.
- Warmup: linear warmup.
- Mixed precision: used.
- Gradient checkpointing: used.
- Weight decay is applied to most parameters.
- Weight decay is not applied to bias, layer normalization, token embedding, positional embedding, or contrastive temperature.

### 6.2 Learning Rate Scaling

The paper uses a square-root batch-size scaling rule:

```text
peak_lr = base_lr * sqrt(total_batch_size / 512)
```

### 6.3 Model- and Dataset-Specific Hyperparameters

| Model | Dataset | Total batch size | Base LR | Weight decay |
|---|---|---:|---:|---:|
| FILIPbase | YFCC100M | 1024 x 8 | 6e-3 | 3e-2 |
| FILIPbase | FILIP340M | 320 x 128 | 2e-3 | 3e-3 |
| FILIPlarge | FILIP340M | 160 x 192 | 1.5e-3 | 3e-3 |

### 6.4 Training Scale

The paper reports the following training scale:

| Model | Hardware scale | Training time |
|---|---:|---:|
| FILIPbase | 128 cards | about 9 days |
| FILIPlarge | 192 cards | about 24 days |

Training was mainly conducted on Nvidia V100 GPUs and Ascend cards.

## 7. Pre-Training Data Construction

FILIP uses a large image-text corpus called FILIP300M plus public datasets.

Datasets used:

| Dataset | Approximate size used |
|---|---:|
| CC3M | 3M |
| CC12M | 10M |
| YFCC100M filtered subset | 26M |
| FILIP300M | 300M |
| Total | about 340M |

Filtering rules mentioned in the paper:

- Remove images whose shorter side is less than `200` pixels.
- Remove images with aspect ratio larger than `3`.
- Keep only English texts.
- Remove meaningless texts such as file-name-like captions.
- Remove image-text pairs where the text is repeated more than `10` times.
- Replace person names in text with a special `<person>` token.
- Remove text containing sensitive words.

## 8. Image and Text Augmentation

FILIP applies augmentation to both images and text during pre-training.

### 8.1 Image Augmentation

Use AutoAugment for image augmentation.

### 8.2 Text Augmentation

Use back-translation.

Procedure:

1. Translate original English text into German.
2. Translate it back into English.
3. Translate original English text into Russian.
4. Translate it back into English.
5. Store three candidate texts per image:
   - original text
   - German back-translated text
   - Russian back-translated text
6. During batch construction, randomly sample one of the three text candidates.

## 9. Prompt Templates and Prompt Ensemble

For downstream zero-shot classification, FILIP uses prompt templates.

The general template is:

```text
[prefix] {label}, [category description]. [suffix].
```

Where:

- `prefix` is an in-context phrase such as `a photo of a`.
- `label` is the class label.
- `category description` gives dataset-specific type information, such as `a type of pet`.
- `suffix` is an extra sentence. The paper finds that suffixes containing `it`, such as `I like it.`, can improve performance.

For visualization, the paper uses a single prompt:

```text
a photo of a {label}.
```

### 9.1 Prompt Ensemble Implementation

Unlike CLIP, FILIP should not ensemble prompts by averaging text embeddings, because each prompt has different token-level representations.

Instead, compute the FILIP late-interaction similarity for each prompt independently, then average the final similarities:

```text
score(image, class) = mean(score(image, prompt_1), ..., score(image, prompt_C))
```

Implementation sketch:

```python
scores = []
for prompt in prompts_for_class:
    text_tokens = text_encoder(tokenize(prompt))
    score = filip_similarity_i2t(image_tokens, text_tokens)
    scores.append(score)

class_score = torch.stack(scores).mean()
```

## 10. Image-Text Retrieval Fine-Tuning

Retrieval is evaluated on Flickr30K and MSCOCO.

Dataset setup:

| Dataset | Test set | Fine-tuning set |
|---|---:|---:|
| Flickr30K | 1K test set | 30K training set |
| MSCOCO | 5K test set | 113K training set |

Fine-tuning details:

| Hyperparameter | Value |
|---|---:|
| Image size | 392 x 392 |
| Training epochs | 3 |
| Optimizer | LAMB |
| Batch size | 5120 |
| Base LR | 2e-4 |
| Weight decay | 3e-4 |

Implementation details:

- Use image-text similarity for ranking.
- Fine-tune using the contrastive loss.
- Since each image may have multiple positive captions, assign each positive caption probability `1 / num_positives` in the target distribution.
- Use prompts during evaluation.

## 11. Linear Probe Setup

For image classification linear probing:

- Freeze the whole backbone.
- Train only a final linear classifier.
- Since FILIP focuses on patch tokens, use mean pooling over visual tokens to form a global image representation.

For most datasets:

- Use scikit-learn logistic regression.
- Use L-BFGS.
- Maximum iterations: `1000`.

For ImageNet:

| Hyperparameter | Value |
|---|---:|
| Image size | 224 x 224 |
| Training epochs | 90 |
| Optimizer | SGD |
| Batch size | 4096 |
| Base LR | 0.1 |
| Weight decay | 0 |

Additional ImageNet linear-probe details:

- Add BatchNorm before the linear classifier.
- Use random resized crop.
- Use horizontal flipping.
- Use cosine learning rate schedule.
- Use 10 warmup epochs.

## 12. Word-Patch Alignment Visualization

The paper visualizes learned alignment using token-wise similarity.

Procedure:

1. Tokenize the prompt, usually `a photo of a {label}.`.
2. Encode the image into visual patch tokens.
3. Encode the prompt into textual tokens.
4. Compute the token-wise similarity matrix between image patches and text tokens.
5. For each image patch, find the textual token with maximum similarity.
6. Display the selected text-token index at the center of each image patch.
7. Highlight patches whose selected token corresponds to the class-label token.

This visualization is simple and directly follows from the late-interaction matrix. No bounding-box labels are used.

The paper also shows Grad-CAM heatmaps. Their Grad-CAM visualization is based on average self-attention maps over image patches classified to target textual tokens in the last image-encoder layer, averaged over attention heads.

## 13. Inference Behavior

FILIP keeps the main efficiency advantage of dual-stream models:

- Image features can be precomputed offline.
- Text features can be precomputed offline.
- Retrieval still uses matrix multiplication and ranking.
- No cross-attention needs to be run at inference.

The paper reports retrieval inference time close to CLIP after applying the efficiency tricks: reduced embedding dimension and FP16 token features.

## 14. Minimal Implementation Checklist

For a practical reimplementation, implement the following components first:

1. CLIP-style dual encoder.
2. Image encoder output as patch-token sequence.
3. Text encoder output as token sequence with padding mask.
4. Linear projection to 256 dimensions for both modalities.
5. L2 normalization for token embeddings.
6. Token-wise similarity matrix for every image-text pair in batch.
7. Image-to-text score: mean over image-token max over text tokens.
8. Text-to-image score: mean over text-token max over image tokens.
9. Padded text token masking before max operation.
10. Symmetric contrastive loss with learnable temperature initialized to 0.07.
11. Prompt ensemble by averaging final similarities, not text embeddings.
12. Optional distributed-efficiency path: FP16 token features, 256-d projection, and top-25% token selection.

## 15. Important Pitfalls

- Do not compute contrastive logits from only global CLS/EOS embeddings if implementing FILIP. That becomes CLIP-like behavior.
- Do not include padded text tokens in the max operation.
- Do not sum token-wise maximum similarities; average them.
- Do not assume image-to-text and text-to-image scores are identical.
- Do not average prompt embeddings for prompt ensemble; average final FILIP similarities.
- Do not add cross-attention unless intentionally building a different model. FILIP's claim is that fine-grained interaction comes from the loss, not from fusion layers.
- Start without the top-25% token selection if implementing on a small setup. Add it only after the basic late-interaction loss works.
d