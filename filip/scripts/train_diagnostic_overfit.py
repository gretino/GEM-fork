# filip/scripts/train_diagnostic_overfit.py

import os
import math
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from filip.data.dataset import ECGImageDataset
from filip.data.collator import ecg_collate_fn
from filip.model.filip_ecg_model import FILIPECGModel
from filip.model.losses import report_alignment_loss
from filip.train.train_mimic_feature import tokenize_reports, compute_report_metrics, ExpandToSquare


def parse_args():
    parser = argparse.ArgumentParser(description="FILIP Single-Batch Diagnostic Overfitting Test")
    parser.add_argument('--config', type=str, default='filip/configs/mimic_report_alignment_pretrain.yaml')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--proj_lr', type=float, default=1.0e-3)
    parser.add_argument('--encoder_lr', type=float, default=1.0e-5)
    parser.add_argument('--unfreeze_encoders', action='store_true', help="Unfreeze vision and text encoders during overfit test")
    parser.add_argument('--unfreeze_vision_only', action='store_true', help="Unfreeze vision encoder only")
    return parser.parse_args()


def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Force report alignment mode
    config['model']['use_report_alignment'] = True
    config['model']['use_feature_alignment'] = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Diagnostic Overfit Test on device: {device}")

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
    dataloader = DataLoader(dataset, batch_size=args.batch_size * 4, shuffle=True, collate_fn=ecg_collate_fn)

    # Select 1 fixed batch of N unique report texts
    fixed_batch = None
    for batch in dataloader:
        reports = batch['report_texts']
        # Filter for unique reports
        unique_indices = []
        seen = set()
        for idx, r in enumerate(reports):
            r_norm = r.strip().lower()
            if r_norm and r_norm not in seen:
                seen.add(r_norm)
                unique_indices.append(idx)
            if len(unique_indices) == args.batch_size:
                break
        
        if len(unique_indices) == args.batch_size:
            fixed_batch = {
                'images': batch['images'][unique_indices],
                'sample_ids': [batch['sample_ids'][i] for i in unique_indices],
                'report_texts': [batch['report_texts'][i] for i in unique_indices]
            }
            break

    if fixed_batch is None:
        raise RuntimeError(f"Could not find {args.batch_size} unique reports in dataset samples!")

    print(f"Selected fixed batch of {args.batch_size} unique image-report pairs.")
    print("Sample report snippet 0:", fixed_batch['report_texts'][0][:80])
    print("Sample report snippet 1:", fixed_batch['report_texts'][1][:80])

    # Model Setup
    model = FILIPECGModel(config).to(device)
    from transformers import AutoTokenizer
    text_model_name = config.get('model', {}).get('text_encoder', config.get('model', {}).get('vision_encoder'))
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    # Configure freezing / parameter groups
    if not args.unfreeze_encoders and not args.unfreeze_vision_only:
        print("Mode: Freeze Vision & Text Encoders (train projections & scale only)")
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        for p in model.text_encoder.parameters():
            p.requires_grad = False
    elif args.unfreeze_vision_only:
        print("Mode: Unfreeze Vision Encoder, freeze Text Encoder")
        for p in model.text_encoder.parameters():
            p.requires_grad = False
    else:
        print("Mode: Unfreeze Both Vision & Text Encoders")

    proj_params = []
    encoder_params = []
    no_decay_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "logit_scale" in name or "bias" in name:
            no_decay_params.append(p)
        elif "vision_encoder" in name or "text_encoder" in name:
            encoder_params.append(p)
        else:
            proj_params.append(p)

    param_groups = [
        {'params': proj_params, 'lr': args.proj_lr, 'weight_decay': 1e-2},
        {'params': no_decay_params, 'lr': args.proj_lr, 'weight_decay': 0.0},
    ]
    if encoder_params:
        param_groups.append({'params': encoder_params, 'lr': args.encoder_lr, 'weight_decay': 1e-2})

    optimizer = torch.optim.AdamW(param_groups)

    images = fixed_batch['images'].to(device)
    text_batch = tokenize_reports(tokenizer, fixed_batch['report_texts'], config, device)

    print("\nStarting Diagnostic Overfitting Optimization...")
    print(f"{'Step':<6} | {'Loss':<8} | {'I2T Top1':<9} | {'T2I Top1':<9} | {'Scale':<7} | {'Diag Gap':<9} | {'Proj Grad':<9}")
    print("-" * 75)

    model.train()
    initial_loss = None
    final_loss = None

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        outputs = model(images, **text_batch)
        loss = report_alignment_loss(outputs['report_logits'])
        loss.backward()

        # Measure grad norm of image projection
        grad_norm = 0.0
        if model.report_alignment_head.image_projection.weight.grad is not None:
            grad_norm = model.report_alignment_head.image_projection.weight.grad.norm().item()

        optimizer.step()

        metrics = compute_report_metrics(outputs['report_logits'])
        scale = model.report_alignment_head.scale.item()

        if step == 1:
            initial_loss = loss.item()

        if step % 10 == 0 or step == 1 or step == args.steps:
            final_loss = loss.item()
            print(
                f"{step:<6} | {loss.item():<8.4f} | {metrics.get('i2t_top1', 0.0):<9.3f} | "
                f"{metrics.get('t2i_top1', 0.0):<9.3f} | {scale:<7.2f} | "
                f"{metrics.get('diag_gap', 0.0):<9.4f} | {grad_norm:<9.4f}"
            )

    print("-" * 75)
    print(f"Initial Loss: {initial_loss:.4f} (Expected uniform random ln(32) ≈ 3.4657)")
    print(f"Final Loss  : {final_loss:.4f}")
    
    if final_loss < 1.0 and metrics.get('i2t_top1', 0.0) > 0.8:
        print("OVERFIT DIAGNOSTIC RESULT: SUCCESS! Single-batch model mechanics can overfit and learn alignment.")
    elif final_loss < initial_loss - 0.5:
        print("OVERFIT DIAGNOSTIC RESULT: PARTIAL PROGRESS. Loss is decreasing but not fully converging.")
    else:
        print("OVERFIT DIAGNOSTIC RESULT: FAILED TO OVERFIT! Architecture/Projections cannot differentiate single batch.")

if __name__ == "__main__":
    main()
