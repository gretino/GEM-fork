# filip/model/feature_alignment.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAlignmentHead(nn.Module):
    def __init__(self, hidden_size, num_features, align_dim=256, topk=8):
        super().__init__()
        self.num_features = num_features
        self.align_dim = align_dim
        self.topk = topk
        
        # Project image patches to alignment dimension
        self.image_patch_projection = nn.Linear(hidden_size, align_dim, bias=False)
        
        # Learnable feature embeddings
        self.feature_embedding = nn.Embedding(num_features, align_dim)
        
        # Learnable scale for cosine similarity (init to 14.28 ≈ 1/0.07)
        self.scale = nn.Parameter(torch.ones([]) * 14.28)
        
    def forward(self, patch_features):
        """
        patch_features: [B, P, H]
        Returns:
            feature_logits: [B, F]
            patch_feature_similarity: [B, P, F]
        """
        # [B, P, A]
        V = self.image_patch_projection(patch_features)
        
        # [F, A]
        Q = self.feature_embedding.weight
        
        # Normalize
        V = F.normalize(V, dim=-1)
        Q = F.normalize(Q, dim=-1)
        
        # Cosine similarity: einsum("bpa,fa->bpf", V, Q)
        # Shape: [B, P, F]
        similarity = torch.einsum("bpa,fa->bpf", V, Q)
        
        # Apply learnable scale, clamped to prevent extreme values
        # Clamping scale up to 100 as per CLIP
        scale = torch.clamp(self.scale, max=100.0)
        scaled_similarity = similarity * scale
        
        # Top-K Mean pooling over patches
        # [B, topk, F]
        topk_sim, _ = torch.topk(scaled_similarity, self.topk, dim=1)
        
        # [B, F]
        feature_logits = topk_sim.mean(dim=1)
        
        return feature_logits, scaled_similarity
