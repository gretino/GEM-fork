# filip/eval/visualize_filip_loss_diagnostic.py

import os
import math
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from filip.data.dataset import ECGImageDataset
from filip.data.collator import ecg_collate_fn
from filip.model.filip_ecg_model import FILIPECGModel
from filip.model.losses import report_alignment_loss
from filip.train.train_mimic_feature import tokenize_reports, compute_report_metrics, ExpandToSquare


def main():
    config_path = "filip/configs/mimic_report_alignment_pretrain.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['model']['use_report_alignment'] = True
    config['model']['use_feature_alignment'] = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== FILIP Loss Static Diagnostic Run on device: {device} ===")

    data_root = "/home/qfbqt/8TB/datasets/mimic-iv-ecg/"
    if not os.path.exists(data_root):
        data_root = "data/mimic-iv-ecg"

    image_size = config.get('model', {}).get('image_size', 224)
    transform = transforms.Compose([
        ExpandToSquare(background_color=(255, 255, 255)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    dataset = ECGImageDataset(data_root=data_root, split='train', dataset_name='mimic', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=ecg_collate_fn)

    # Grab static batch of 16 samples
    batch = next(iter(dataloader))
    images = batch['images'].to(device)
    report_texts = batch['report_texts']
    B = len(report_texts)

    print(f"Loaded static batch of {B} ECG-report samples.")
    print(f"Report 0: '{report_texts[0][:90]}...'")
    print(f"Report 1: '{report_texts[1][:90]}...'")

    # Load Model & Tokenizer
    model = FILIPECGModel(config).to(device)
    text_model_name = config.get('model', {}).get('text_encoder', config.get('model', {}).get('vision_encoder'))
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    text_batch = tokenize_reports(tokenizer, report_texts, config, device)

    # Model Forward Pass
    model.train()
    model.zero_grad()

    patch_features = model.vision_encoder(images)  # [B, P, H_i]
    text_outputs = model.text_encoder(
        input_ids=text_batch['input_ids'],
        attention_mask=text_batch['attention_mask']
    )
    token_features = text_outputs.last_hidden_state  # [B, T, H_t]
    content_mask = text_batch['content_mask']        # [B, T]

    head = model.report_alignment_head
    image_proj = F.normalize(head.image_projection(patch_features), dim=-1)  # [B, P, A]
    text_proj = F.normalize(head.text_projection(token_features), dim=-1)    # [B, T, A]
    scale = head.scale.item()

    # Raw 4D Cosine Similarity Tensor [B, B, P, T]
    similarities = torch.einsum("bpa,cta->bcpt", image_proj, text_proj)
    valid_tokens = content_mask.bool()  # [B, T]

    # Image-to-Text Unscaled Patch Scores [B, B, P]
    masked = similarities.masked_fill(~valid_tokens[None, :, None, :], float("-inf"))
    patch_scores = masked.max(dim=-1).values  # [B, B, P]
    patch_scores = torch.where(torch.isfinite(patch_scores), patch_scores, torch.zeros_like(patch_scores))
    i2t_unscaled = patch_scores.mean(dim=-1)  # [B, B]

    # Text-to-Image Unscaled Token Scores [B, B, T]
    t2i_token_scores = similarities.max(dim=-2).values  # [B, B, T]
    weights = valid_tokens.to(t2i_token_scores.dtype)[None, :, :]  # [1, B, T]
    t2i_unscaled = (t2i_token_scores * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1)  # [B, B]

    i2t_logits = i2t_unscaled * scale
    t2i_logits = t2i_unscaled * scale

    loss = report_alignment_loss((i2t_logits, t2i_logits))
    loss.backward()

    img_grad_norm = head.image_projection.weight.grad.norm().item() if head.image_projection.weight.grad is not None else 0.0
    txt_grad_norm = head.text_projection.weight.grad.norm().item() if head.text_projection.weight.grad is not None else 0.0
    scale_grad = head.logit_scale.grad.item() if head.logit_scale.grad is not None else 0.0

    eye = torch.eye(B, dtype=torch.bool, device=device)
    pos_i2t_unscaled = i2t_unscaled.diagonal().detach().cpu().numpy()
    neg_i2t_unscaled = i2t_unscaled[~eye].detach().cpu().numpy()

    pos_t2i_unscaled = t2i_unscaled.diagonal().detach().cpu().numpy()
    neg_t2i_unscaled = t2i_unscaled[~eye].detach().cpu().numpy()

    i2t_logits_np = i2t_logits.detach().cpu().numpy()
    pos_logits = i2t_logits.diagonal().detach().cpu().numpy()
    neg_logits = i2t_logits[~eye].detach().cpu().numpy()

    print("\n" + "=" * 70)
    print("      FILIP LOSS STATIC DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"Calculated FILIP Loss        : {loss.item():.4f} (Expected uniform random ln({B}) ≈ {math.log(B):.4f})")
    print(f"Effective Similarity Scale   : {scale:.4f}")
    print(f"Raw Logit Scale Parameter    : {head.logit_scale.item():.4f} (Grad: {scale_grad:.6f})")
    print(f"Image Projection Grad Norm   : {img_grad_norm:.6f}")
    print(f"Text Projection Grad Norm    : {txt_grad_norm:.6f}")

    print("\n--- UNSCALED COSINE SIMILARITY ANALYSIS ---")
    print(f"Positive Pairs (I->T) Mean   : {pos_i2t_unscaled.mean():.4f} ± {pos_i2t_unscaled.std():.4f} [Min: {pos_i2t_unscaled.min():.4f}, Max: {pos_i2t_unscaled.max():.4f}]")
    print(f"Negative Pairs (I->T) Mean   : {neg_i2t_unscaled.mean():.4f} ± {neg_i2t_unscaled.std():.4f} [Min: {neg_i2t_unscaled.min():.4f}, Max: {neg_i2t_unscaled.max():.4f}]")
    print(f"Unscaled Diagonal Gap        : {pos_i2t_unscaled.mean() - neg_i2t_unscaled.mean():.4f}")

    print(f"Positive Pairs (T->I) Mean   : {pos_t2i_unscaled.mean():.4f} ± {pos_t2i_unscaled.std():.4f}")
    print(f"Negative Pairs (T->I) Mean   : {neg_t2i_unscaled.mean():.4f} ± {neg_t2i_unscaled.std():.4f}")

    print("\n--- SCALED LOGITS ANALYSIS ---")
    print(f"Positive Logits Mean         : {pos_logits.mean():.4f} ± {pos_logits.std():.4f}")
    print(f"Negative Logits Mean         : {neg_logits.mean():.4f} ± {neg_logits.std():.4f}")
    print(f"Scaled Diagonal Gap          : {pos_logits.mean() - neg_logits.mean():.4f}")

    # Create output directory for visualization plots
    out_dir = "outputs/filip_diagnostics"
    os.makedirs(out_dir, exist_ok=True)

    # Plot 1: Logits Heatmap Matrix [B x B]
    plt.figure(figsize=(10, 8))
    sns.heatmap(i2t_logits_np, annot=True, fmt=".1f", cmap="viridis", cbar=True)
    plt.title(f"Pairwise Image-to-Text Logits Matrix (B={B})")
    plt.xlabel("Report Text Index")
    plt.ylabel("ECG Image Index")
    plt.tight_layout()
    heatmap_path = os.path.join(out_dir, "pairwise_logits_heatmap.png")
    plt.savefig(heatmap_path, dpi=200)
    plt.close()
    print(f"\nSaved pairwise logits heatmap to: {heatmap_path}")

    # Plot 2: Distribution of Unscaled Positive vs Negative Similarities
    plt.figure(figsize=(9, 5))
    plt.hist(pos_i2t_unscaled, bins=15, alpha=0.6, color='blue', label='Positive Pairs (Diagonal)')
    plt.hist(neg_i2t_unscaled, bins=25, alpha=0.5, color='red', label='Negative Pairs (Off-Diagonal)')
    plt.axvline(pos_i2t_unscaled.mean(), color='blue', linestyle='dashed', linewidth=2, label=f'Pos Mean ({pos_i2t_unscaled.mean():.3f})')
    plt.axvline(neg_i2t_unscaled.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Neg Mean ({neg_i2t_unscaled.mean():.3f})')
    plt.title("Distribution of Raw Unscaled Cosine Similarities (I->T)")
    plt.xlabel("Unscaled Cosine Similarity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(out_dir, "pos_vs_neg_similarity_hist.png")
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"Saved similarity distribution plot to: {hist_path}")

    # Plot 3: Sample 0 Unpooled Patch-Token Alignment Heatmap [P x T]
    sample0_sim = similarities[0, 0].detach().cpu().numpy()  # [P, T]
    valid_T0 = content_mask[0].sum().item()
    sample0_sim_valid = sample0_sim[:, :valid_T0]

    plt.figure(figsize=(12, 6))
    sns.heatmap(sample0_sim_valid.T, cmap="coolwarm", cbar=True)
    plt.title(f"Sample 0 Patch-Token Cosine Similarities (P={sample0_sim_valid.shape[0]} patches, T={valid_T0} tokens)")
    plt.xlabel("ECG Patch Index (1 to 49)")
    plt.ylabel("Report Token Index")
    plt.tight_layout()
    sample_path = os.path.join(out_dir, "sample0_patch_token_matrix.png")
    plt.savefig(sample_path, dpi=200)
    plt.close()
    print(f"Saved Sample 0 patch-token matrix to: {sample_path}")

    # Save summary report text file
    summary_path = os.path.join(out_dir, "diagnostic_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"FILIP Loss Static Diagnostic Summary\n")
        f.write(f"===================================\n")
        f.write(f"Batch Size                 : {B}\n")
        f.write(f"Loss                       : {loss.item():.4f}\n")
        f.write(f"Effective Scale            : {scale:.4f}\n")
        f.write(f"Positive Sim Mean (I->T)   : {pos_i2t_unscaled.mean():.4f}\n")
        f.write(f"Negative Sim Mean (I->T)   : {neg_i2t_unscaled.mean():.4f}\n")
        f.write(f"Unscaled Diagonal Gap      : {pos_i2t_unscaled.mean() - neg_i2t_unscaled.mean():.4f}\n")
        f.write(f"Scaled Diagonal Gap        : {pos_logits.mean() - neg_logits.mean():.4f}\n")
        f.write(f"Image Proj Grad Norm       : {img_grad_norm:.6f}\n")
        f.write(f"Text Proj Grad Norm        : {txt_grad_norm:.6f}\n")

    print(f"Saved text diagnostic summary to: {summary_path}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
