# filip/model/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def report_alignment_loss(report_logits):
    """Symmetric in-batch contrastive loss for matched ECG/report pairs."""
    if report_logits.ndim != 2 or report_logits.shape[0] != report_logits.shape[1]:
        raise ValueError("report_logits must be a square [batch, batch] tensor")
    targets = torch.arange(report_logits.shape[0], device=report_logits.device)
    image_to_text = F.cross_entropy(report_logits, targets)
    text_to_image = F.cross_entropy(report_logits.transpose(0, 1), targets)
    return 0.5 * (image_to_text + text_to_image)

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

def asymmetric_loss(diagnosis_logits, diagnosis_targets, diagnosis_mask, gamma_pos=0.0, gamma_neg=2.0, clip=0.05, eps=1e-8):
    """
    Computes Asymmetric Loss for multilabel classification.
    """
    probs = torch.sigmoid(diagnosis_logits)
    
    probs_pos = probs
    probs_neg = 1 - probs
    
    # Asymmetric Clipping for negative samples
    if clip > 0:
        probs_neg = (probs_neg + clip).clamp(max=1.0)
        
    los_pos = diagnosis_targets * torch.log(probs_pos.clamp(min=eps))
    los_neg = (1 - diagnosis_targets) * torch.log(probs_neg.clamp(min=eps))
    loss = los_pos + los_neg
    
    # Asymmetric Focusing
    pt0 = probs_pos * diagnosis_targets
    pt1 = probs_neg * (1 - diagnosis_targets)
    pt = pt0 + pt1
    
    one_sided_gamma = gamma_pos * diagnosis_targets + gamma_neg * (1 - diagnosis_targets)
    one_sided_w = torch.pow(1 - pt, one_sided_gamma)
    
    raw_loss = - (loss * one_sided_w)
    
    masked_loss = raw_loss * diagnosis_mask
    return masked_loss.sum() / diagnosis_mask.sum().clamp_min(1.0)

def masked_mse_loss(preds, targets, mask):
    """
    Computes MSE loss only on valid (masked) elements.
    preds: [B, N]
    targets: [B, N]
    mask: [B, N]
    """
    raw_loss = F.mse_loss(preds, targets, reduction='none')
    masked_loss = raw_loss * mask
    
    return masked_loss.sum() / mask.sum().clamp_min(1.0)
