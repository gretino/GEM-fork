"""FILIP late-interaction alignment between image patches and report tokens."""

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
        self.scale = nn.Parameter(torch.ones([]) * 14.28)

    def _score_all(self, patch_features, token_features, token_mask):
        """Return pooled logits and unpooled similarities for all image/text pairs."""
        image = F.normalize(self.image_projection(patch_features), dim=-1)
        text = F.normalize(self.text_projection(token_features), dim=-1)
        scale = torch.clamp(self.scale, min=0.0, max=100.0)

        similarities = torch.einsum("bpa,cta->bcpt", image, text) * scale
        valid_tokens = token_mask.bool()
        token_scores = similarities.max(dim=2).values
        token_weights = valid_tokens.to(token_scores.dtype).unsqueeze(0)
        token_score = (token_scores * token_weights).sum(dim=-1) / token_weights.sum(dim=-1).clamp_min(1)

        masked = similarities.masked_fill(~valid_tokens[None, :, None, :], float("-inf"))
        patch_scores = masked.max(dim=-1).values
        patch_scores = torch.where(torch.isfinite(patch_scores), patch_scores, torch.zeros_like(patch_scores))
        k = min(self.topk, patch_scores.shape[-1])
        patch_score = patch_scores.topk(k, dim=-1).values.mean(dim=-1)
        return 0.5 * (token_score + patch_score), similarities

    def forward(self, patch_features, token_features, token_mask):
        """
        Args:
            patch_features: ``[B, P, H_i]`` image patch representations.
            token_features: ``[B, T, H_t]`` contextual report tokens.
            token_mask: ``[B, T]``; true only for report content tokens.

        Returns:
            Pairwise image/report logits ``[B, B]`` and positive-pair patch to
            token similarities ``[B, P, T]``.
        """
        logits, similarities = self._score_all(patch_features, token_features, token_mask)
        positive_similarity = similarities.diagonal(dim1=0, dim2=1).permute(2, 0, 1).contiguous()
        return logits, positive_similarity

    def score_prompts(self, patch_features, token_features, token_mask):
        """Score ``B`` ECGs against ``C`` prompts without assuming paired batches.

        Returns logits shaped ``[B, C]`` and token-resolved maps shaped
        ``[B, C, P, T]`` for diagnosis likelihoods and visualization.
        """
        return self._score_all(patch_features, token_features, token_mask)
