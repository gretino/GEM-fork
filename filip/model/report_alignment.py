import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReportAlignmentHead(nn.Module):
    """Project patches and tokens and score every image/report pair in a batch."""

    def __init__(self, image_hidden_size, text_hidden_size, align_dim=256, topk=8):
        super().__init__()
        self.topk = topk
        self.image_projection = nn.Linear(image_hidden_size, align_dim, bias=False)
        self.text_projection = nn.Linear(text_hidden_size, align_dim, bias=False)
        # Log-parameterized scale initialized to log(1/0.07) ≈ 2.659
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    @property
    def scale(self):
        """Effective similarity scale tensor clamped up to 100.0."""
        return self.logit_scale.exp().clamp(max=100.0)

    def _score_all(self, patch_features, token_features, token_mask, use_all_patch_pooling=True):
        """Return directional logits (i2t_logits, t2i_logits) and unpooled similarities."""
        image = F.normalize(self.image_projection(patch_features), dim=-1)
        text = F.normalize(self.text_projection(token_features), dim=-1)
        scale = self.scale

        # Cosine similarity matrix: [B, C, P, T]
        similarities = torch.einsum("bpa,cta->bcpt", image, text)
        valid_tokens = token_mask.bool()

        # 1. Image-to-Text: Every image patch selects its best valid text token.
        masked = similarities.masked_fill(~valid_tokens[None, :, None, :], float("-inf"))
        patch_scores = masked.max(dim=-1).values
        patch_scores = torch.where(torch.isfinite(patch_scores), patch_scores, torch.zeros_like(patch_scores))

        if use_all_patch_pooling:
            i2t_score = patch_scores.mean(dim=-1)
        else:
            k = min(self.topk, patch_scores.shape[-1])
            i2t_score = patch_scores.topk(k, dim=-1).values.mean(dim=-1)

        # 2. Text-to-Image: Every valid text token selects its best image patch.
        t2i_token_scores = similarities.max(dim=-2).values
        weights = valid_tokens.to(t2i_token_scores.dtype)[None, :, :]
        t2i_score = (t2i_token_scores * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1)

        i2t_logits = i2t_score * scale
        t2i_logits = t2i_score * scale

        return i2t_logits, t2i_logits, similarities

    def forward(self, patch_features, token_features, token_mask, use_all_patch_pooling=True):
        """
        Args:
            patch_features: ``[B, P, H_i]`` image patch representations.
            token_features: ``[B, T, H_t]`` contextual report tokens.
            token_mask: ``[B, T]``; true only for report content tokens.

        Returns:
            Tuple of ``(i2t_logits, t2i_logits)`` pairwise image/report logit matrices ``[B, B]``
            and positive-pair patch to token similarities ``[B, P, T]``.
        """
        i2t_logits, t2i_logits, similarities = self._score_all(
            patch_features, token_features, token_mask, use_all_patch_pooling=use_all_patch_pooling
        )
        positive_similarity = similarities.diagonal(dim1=0, dim2=1).permute(2, 0, 1).contiguous()
        return (i2t_logits, t2i_logits), positive_similarity

    def score_prompts(self, patch_features, token_features, token_mask):
        """Score ``B`` ECGs against ``C`` prompts without assuming paired batches.

        Returns combined logits shaped ``[B, C]`` and token-resolved maps shaped
        ``[B, C, P, T]`` for diagnosis likelihoods and visualization.
        """
        i2t_logits, t2i_logits, similarities = self._score_all(patch_features, token_features, token_mask)
        combined_logits = 0.5 * (i2t_logits + t2i_logits)
        return combined_logits, similarities

