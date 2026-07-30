# filip/train/train_mimic_feature.py

import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision import transforms

from filip.data.dataset import ECGImageDataset
from filip.data.collator import ecg_collate_fn
from filip.model.filip_ecg_model import FILIPECGModel
from filip.model.losses import feature_loss, report_alignment_loss
import numpy as np
import torch.nn.functional as F
import random

from PIL import Image


def tokenize_reports(tokenizer, reports, config, device):
    """Tokenize raw reports and distinguish content from padding/special tokens."""
    encoded = tokenizer(
        reports,
        padding=True,
        truncation=True,
        max_length=config.get('model', {}).get('text_max_length', 77),
        return_special_tokens_mask=True,
        return_tensors='pt',
    )
    content_mask = encoded['attention_mask'].bool() & ~encoded.pop('special_tokens_mask').bool()
    if not content_mask.any(dim=1).all():
        raise ValueError("Every MIMIC sample must contain non-empty report text")
    return {
        'input_ids': encoded['input_ids'].to(device),
        'attention_mask': encoded['attention_mask'].to(device),
        'content_mask': content_mask.to(device),
    }

def generate_batch_mask(batch_size, H_grid, W_grid, device):
    masks = []
    for _ in range(batch_size):
        mask = torch.zeros((H_grid, W_grid), dtype=torch.bool)
        for _ in range(4):
            best_r, best_c, best_h, best_w = None, None, None, None
            min_overlap_ratio = 1.0
            
            for _ in range(20):
                h = random.choice([1, 2])
                w = random.choice([2, 3, 4])
                
                r_start = random.randint(1, H_grid - 1 - h)
                c_start = random.randint(1, W_grid - w)
                
                block_mask = mask[r_start:r_start+h, c_start:c_start+w]
                overlap_ratio = block_mask.float().mean().item()
                
                if overlap_ratio <= 0.25:
                    best_r, best_c, best_h, best_w = r_start, c_start, h, w
                    break
                elif overlap_ratio < min_overlap_ratio:
                    min_overlap_ratio = overlap_ratio
                    best_r, best_c, best_h, best_w = r_start, c_start, h, w
            
            if best_r is not None:
                mask[best_r:best_r+best_h, best_c:best_c+best_w] = True
                
        masks.append(mask.flatten())
    return torch.stack(masks).to(device)

class ExpandToSquare(object):
    def __init__(self, background_color=(255, 255, 255)):
        self.background_color = background_color

    def __call__(self, img):
        width, height = img.size
        if width == height:
            return img
        elif width > height:
            result = Image.new(img.mode, (width, width), self.background_color)
            result.paste(img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(img.mode, (height, height), self.background_color)
            result.paste(img, ((height - width) // 2, 0))
            return result

def train_mimic():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='filip/configs/mimic_feature_pretrain.yaml')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    report_pretraining = config.get('model', {}).get('use_report_alignment', False)
    dataset = ECGImageDataset(data_root=data_root, split='train', dataset_name='mimic', transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=ecg_collate_fn, num_workers=4, drop_last=report_pretraining)
    
    val_dataset = ECGImageDataset(data_root=data_root, split='val', dataset_name='mimic', transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=ecg_collate_fn, num_workers=4, drop_last=report_pretraining)
    
    model = FILIPECGModel(config).to(device)
    tokenizer = None
    if model.use_report_alignment:
        from transformers import AutoTokenizer
        text_model_name = config.get('model', {}).get(
            'text_encoder', config.get('model', {}).get('vision_encoder')
        )
        tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    epochs = config['training']['epochs']
    
    experiment_name = config.get('experiment_name', 'mimic_feature_pretrain')
    out_dir = f"/outputs/filip/{experiment_name}/checkpoints"
    if out_dir.startswith("/outputs") and not (os.path.exists("/outputs") and os.access("/outputs", os.W_OK)):
        out_dir = out_dir.lstrip("/")
    os.makedirs(out_dir, exist_ok=True)
    
    use_wandb = False
    report_to = config.get('training', {}).get('report_to', 'wandb')
    if report_to == 'wandb':
        try:
            import wandb
            wandb.init(
                project="filip-ecg",
                name=experiment_name,
                config={
                    "learning_rate": config['training']['learning_rate'],
                    "batch_size": config['training']['batch_size'],
                    "epochs": config['training']['epochs'],
                    "config_path": args.config,
                    "model_name": config.get('model', {}).get('vision_encoder', 'openai/clip-vit-base-patch32')
                }
            )
            use_wandb = True
        except Exception as e:
            print(f"Warning: Could not initialize Wandb ({e}). Logging to console only.")
            use_wandb = False
            
    stage1_target = "Raw Report" if model.use_report_alignment else "Feature"
    print(f"Starting Stage 1: MIMIC {stage1_target} Pretraining")
    global_step = 0
    best_metric = -float('inf')
    start_epoch = 0
    
    resume_path = args.resume_from or config.get('resume_from_checkpoint')
    if resume_path:
        if resume_path.startswith("/outputs") and not (os.path.exists("/outputs") and os.access("/outputs", os.W_OK)):
            resume_path = resume_path.lstrip("/")
        if os.path.exists(resume_path):
            print(f"Resuming training from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            
            # Load model state dict with shape filtering
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            for k, v in ckpt['model_state_dict'].items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        print(f"Skipping key '{k}' due to shape mismatch: checkpoint {v.shape} vs model {model_state_dict[k].shape}")
            model.load_state_dict(filtered_state_dict, strict=False)
            
            # Load optimizer state dict
            if 'optimizer_state_dict' in ckpt and ckpt['optimizer_state_dict'] is not None:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as e:
                    print(f"Warning: Could not load optimizer state dict: {e}")
                    
            start_epoch = ckpt.get('epoch', 0)
            global_step = ckpt.get('global_step', 0)
            best_metric = ckpt.get('best_metric', -float('inf'))
        else:
            print(f"Warning: Checkpoint not found at {resume_path}, starting from scratch!")
            
    image_size = config.get('model', {}).get('image_size', 224)
    patch_size = config.get('model', {}).get('patch_size', 14)
    H_grid = image_size // patch_size
    W_grid = image_size // patch_size
    filip_weight = config.get('loss', {}).get(
        'report_alignment_weight' if model.use_report_alignment else 'feature_loss_weight', 1.0
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch['images'].to(device)
            optimizer.zero_grad()
            
            # FILIP Pass: Unmasked context encoder pass
            if model.use_report_alignment:
                text_batch = tokenize_reports(tokenizer, batch['report_texts'], config, device)
                outputs = model(images, **text_batch)
                L_FILIP = report_alignment_loss(outputs['report_logits'])
            else:
                feature_targets = batch['feature_targets'].to(device)
                feature_mask = batch['feature_mask'].to(device)
                feature_confidence = batch['feature_confidence'].to(device)
                outputs = model(images)
                L_FILIP = feature_loss(outputs['feature_logits'], feature_targets, feature_mask, feature_confidence)
            
            if model.use_jepa:
                mask = generate_batch_mask(images.shape[0], H_grid, W_grid, device)
                prediction, target = model.forward_jepa(images, mask)
                
                prediction_norm = F.normalize(prediction, dim=-1)
                target_norm = F.normalize(target, dim=-1).detach()
                
                # Smooth L1 loss only at masked locations
                L_JEPA = F.smooth_l1_loss(prediction_norm[mask], target_norm[mask], reduction='none').sum(dim=-1).mean()
                jepa_weight = config.get('model', {}).get('jepa_loss_weight', 0.3)
                loss = filip_weight * L_FILIP + jepa_weight * L_JEPA
            else:
                loss = filip_weight * L_FILIP
                
            loss.backward()
            optimizer.step()
            
            if model.use_jepa:
                model.update_target_encoder()
                
            total_loss += loss.item()
            global_step += 1
            
            if model.use_jepa:
                pbar.set_postfix({'loss': loss.item(), 'filip': L_FILIP.item(), 'jepa': L_JEPA.item()})
            else:
                pbar.set_postfix({'loss': loss.item()})
            
            if use_wandb:
                log_dict = {
                    "loss/total_step": loss.item(),
                    "loss/filip_step": L_FILIP.item(),
                }
                if model.use_jepa:
                    log_dict.update({
                        "loss/jepa_step": L_JEPA.item(),
                        "loss/weighted_jepa_step": (jepa_weight * L_JEPA).item()
                    })
                wandb.log(log_dict, step=global_step)
            
        avg_train_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_filip_loss = 0
        val_jepa_loss = 0
        all_preds, all_targets, all_masks = [], [], []
        
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch['images'].to(device)
                if model.use_report_alignment:
                    text_batch = tokenize_reports(tokenizer, batch['report_texts'], config, device)
                    outputs = model(images, **text_batch)
                    L_FILIP = report_alignment_loss(outputs['report_logits'])
                else:
                    feature_targets = batch['feature_targets'].to(device)
                    feature_mask = batch['feature_mask'].to(device)
                    feature_confidence = batch['feature_confidence'].to(device)
                    outputs = model(images)
                    L_FILIP = feature_loss(outputs['feature_logits'], feature_targets, feature_mask, feature_confidence)
                
                if model.use_jepa:
                    mask = generate_batch_mask(images.shape[0], H_grid, W_grid, device)
                    prediction, target = model.forward_jepa(images, mask)
                    
                    prediction_norm = F.normalize(prediction, dim=-1)
                    target_norm = F.normalize(target, dim=-1)
                    
                    L_JEPA = F.smooth_l1_loss(prediction_norm[mask], target_norm[mask], reduction='none').sum(dim=-1).mean()
                    jepa_weight = config.get('model', {}).get('jepa_loss_weight', 0.3)
                    loss = filip_weight * L_FILIP + jepa_weight * L_JEPA
                    
                    val_jepa_loss += L_JEPA.item()
                else:
                    loss = filip_weight * L_FILIP
                    
                val_filip_loss += L_FILIP.item()
                val_loss += loss.item()
                
                if not model.use_report_alignment:
                    all_preds.append(outputs['feature_logits'].cpu())
                    all_targets.append(feature_targets.cpu())
                    all_masks.append(feature_mask.cpu())
                
        val_loss /= len(val_dataloader)
        val_filip_loss /= len(val_dataloader)
        if model.use_jepa:
            val_jepa_loss /= len(val_dataloader)
            
        if model.use_report_alignment:
            current_metric = -val_filip_loss
            print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f} | Val Report FILIP Loss: {val_filip_loss:.4f}")
        else:
            all_preds = torch.cat(all_preds, dim=0).numpy()
            all_targets = torch.cat(all_targets, dim=0).numpy()
            all_masks = torch.cat(all_masks, dim=0).numpy()
            from filip.utils.metrics import compute_multilabel_metrics
            metrics = compute_multilabel_metrics(all_targets, all_preds, all_masks)
            current_metric = metrics['macro_auc']
            print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f} | Val FILIP Loss: {val_filip_loss:.4f} | Macro AUC: {metrics['macro_auc']:.4f} | Micro AUC: {metrics['micro_auc']:.4f}")
        
        if use_wandb:
            log_dict = {
                "loss/total": val_loss,
                "loss/filip": val_filip_loss,
                "epoch": epoch + 1
            }
            if not model.use_report_alignment:
                log_dict.update({
                    "feature/macro_auc": metrics["macro_auc"],
                    "feature/micro_auc": metrics["micro_auc"],
                    "feature/macro_f1": metrics["macro_f1"],
                    "feature/micro_f1": metrics["micro_f1"],
                    "feature/valid_label_ratio": metrics["valid_label_ratio"],
                })
            if model.use_jepa:
                log_dict["loss/jepa"] = val_jepa_loss
            wandb.log(log_dict, step=global_step)
            
        # Checkpoint saving
        is_best = False
        if current_metric > best_metric:
            best_metric = current_metric
            is_best = True
            
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "config": config,
            "feature_vocab": dataset.feature_vocab,
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_metric": best_metric,
            "alignment_target": "raw_report" if model.use_report_alignment else "feature_labels",
        }
        
        # Save best.pt whenever it is best
        if is_best:
            torch.save(checkpoint, os.path.join(out_dir, "best.pt"))
            metric_name = "negative report loss" if model.use_report_alignment else "macro AUC"
            print(f"Saved new best model checkpoint ({metric_name}: {best_metric:.4f})")
            
        # Save epoch checkpoint only every 2 epochs
        if (epoch + 1) % 2 == 0:
            ckpt_path = os.path.join(out_dir, f"epoch_{epoch+1}.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
            
            # Keep only the most recent 3 epoch checkpoints
            epoch_ckpts = sorted(
                [f for f in os.listdir(out_dir) if f.startswith("epoch_") and f.endswith(".pt")],
                key=lambda x: int(x.split("_")[1].split(".")[0])
            )
            while len(epoch_ckpts) > 3:
                oldest = epoch_ckpts.pop(0)
                try:
                    os.remove(os.path.join(out_dir, oldest))
                    print(f"Removed old checkpoint: {oldest}")
                except Exception as e:
                    print(f"Error removing old checkpoint {oldest}: {e}")
                    
    if use_wandb:
        wandb.finish()
        
if __name__ == "__main__":
    train_mimic()
