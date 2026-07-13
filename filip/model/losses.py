# filip/model/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def feature_loss(feature_logits, feature_targets, feature_mask, feature_confidence=None):
    """
    Computes BCEWithLogits loss with masking.
    feature_logits: [B, F]
    feature_targets: [B, F]
    feature_mask: [B, F]
    feature_confidence: [B, F] optional
    """
    raw_loss = F.binary_cross_entropy_with_logits(feature_logits, feature_targets, reduction='none')
    
    if feature_confidence is not None:
        weighted_loss = raw_loss * feature_confidence
    else:
        weighted_loss = raw_loss
        feature_confidence = torch.ones_like(feature_mask)
        
    # Mask out invalid targets
    masked_loss = weighted_loss * feature_mask
    masked_conf = feature_confidence * feature_mask
    
    loss = masked_loss.sum() / masked_conf.sum().clamp_min(1.0)
    return loss

def diagnosis_loss(diagnosis_logits, diagnosis_targets, diagnosis_mask):
    """
    Computes BCEWithLogits loss with masking for diagnosis.
    """
    raw_loss = F.binary_cross_entropy_with_logits(diagnosis_logits, diagnosis_targets, reduction='none')
    masked_loss = raw_loss * diagnosis_mask
    
    # Average over valid items
    loss = masked_loss.sum() / diagnosis_mask.sum().clamp_min(1.0)
    return loss

def feature_consistency_loss(current_feature_logits, frozen_feature_logits):
    """
    Computes MSE between probs to maintain consistency with Stage 1.
    """
    current_probs = torch.sigmoid(current_feature_logits)
    frozen_probs = torch.sigmoid(frozen_feature_logits)
    return F.mse_loss(current_probs, frozen_probs)
