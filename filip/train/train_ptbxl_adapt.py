# filip/train/train_ptbxl_adapt.py

import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

from torchvision import transforms

from filip.data.dataset import ECGImageDataset
from filip.data.collator import ecg_collate_fn
from filip.model.filip_ecg_model import FILIPECGModel
from filip.model.losses import diagnosis_loss, feature_consistency_loss

from PIL import Image

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

def train_ptbxl():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='filip/configs/ptbxl_diagnosis_adapt.yaml')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--train_pct', type=float, default=100.0, help='Percentage of training data to use')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory for checkpoints')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_root = config.get('data_root', "/home/qfbqt/8TB/datasets/ptb-xl/")
    if not os.path.exists(data_root):
        data_root = "data/ptb-xl"
        
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
    
    dataset = ECGImageDataset(data_root=data_root, split='train', dataset_name='ptbxl', transform=transform)
    if args.train_pct < 100.0:
        import random
        rng = random.Random(42)
        num_records = max(1, int(len(dataset.records) * (args.train_pct / 100.0)))
        dataset.records = rng.sample(dataset.records, num_records)
        print(f"Subsampled training dataset to {args.train_pct}%: using {len(dataset.records)} records")
        
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=ecg_collate_fn, num_workers=4)
    
    val_dataset = ECGImageDataset(data_root=data_root, split='val', dataset_name='ptbxl', transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=ecg_collate_fn, num_workers=4)
    
    model = FILIPECGModel(config).to(device)
    stage1_ckpt_path = config.get('stage1_checkpoint')
    if stage1_ckpt_path:
        if stage1_ckpt_path.startswith("/outputs") and not (os.path.exists("/outputs") and os.access("/outputs", os.W_OK)):
            stage1_ckpt_path = stage1_ckpt_path.lstrip("/")
        if os.path.exists(stage1_ckpt_path):
            print(f"Loading Stage 1 checkpoint from {stage1_ckpt_path}")
            ckpt = torch.load(stage1_ckpt_path, map_location=device)
            state_dict = ckpt['model_state_dict']
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        print(f"Skipping key '{k}' due to shape mismatch: checkpoint {v.shape} vs model {model_state_dict[k].shape}")
            model.load_state_dict(filtered_state_dict, strict=False)
        else:
            print(f"Warning: Stage 1 checkpoint not found at {stage1_ckpt_path}, starting from scratch!")
    else:
        print("Warning: Stage 1 checkpoint not specified, starting from scratch!")
        
    frozen_model = copy.deepcopy(model)
    frozen_model.eval()
    for param in frozen_model.parameters():
        param.requires_grad = False
        
    freeze_backbone = config.get('training', {}).get('freeze_backbone', False)
    if freeze_backbone:
        print("Freezing all parameters except diagnosis_head (Linear Probe mode)...")
        for name, param in model.named_parameters():
            if "diagnosis_head" not in name:
                param.requires_grad = False
                
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['learning_rate'])
    epochs = config['training']['epochs']
    
    lambda_consistency = config['loss'].get('feature_consistency_weight', 0.1)
    
    experiment_name = config.get('experiment_name', 'ptbxl_diagnosis_adapt')
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = f"/outputs/filip/{experiment_name}/checkpoints"
        if out_dir.startswith("/outputs") and not (os.path.exists("/outputs") and os.access("/outputs", os.W_OK)):
            out_dir = out_dir.lstrip("/")
    os.makedirs(out_dir, exist_ok=True)
    
    use_wandb = False
    report_to = config.get('training', {}).get('report_to', 'wandb')
    if report_to == 'wandb':
        try:
            import wandb
            run_name = f"{experiment_name}_{int(args.train_pct)}" if args.train_pct.is_integer() else f"{experiment_name}_{args.train_pct}"
            wandb.init(
                project="filip-ecg",
                name=run_name,
                config={
                    "learning_rate": config['training']['learning_rate'],
                    "batch_size": config['training']['batch_size'],
                    "epochs": config['training']['epochs'],
                    "config_path": args.config,
                    "stage1_checkpoint": stage1_ckpt_path,
                    "model_name": config.get('model', {}).get('vision_encoder', 'openai/clip-vit-base-patch32'),
                    "train_pct": args.train_pct
                }
            )
            use_wandb = True
        except Exception as e:
            print(f"Warning: Could not initialize Wandb ({e}). Logging to console only.")
            use_wandb = False
            
    print("Starting Stage 2: PTB-XL Diagnosis Adaptation")
    global_step = 0
    best_macro_auc = -1.0
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
            best_macro_auc = ckpt.get('best_metric', -1.0)
        else:
            print(f"Warning: Checkpoint not found at {resume_path}, starting from scratch!")
            
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch['images'].to(device)
            diagnosis_targets = batch['diagnosis_targets'].to(device)
            diagnosis_mask = batch['diagnosis_mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            with torch.no_grad():
                frozen_outputs = frozen_model(images)
            
            loss_diag = diagnosis_loss(outputs['diagnosis_logits'], diagnosis_targets, diagnosis_mask)
            loss_consist = feature_consistency_loss(outputs['feature_logits'], frozen_outputs['feature_logits'])
            
            loss = loss_diag + lambda_consistency * loss_consist
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': loss.item(), 'diag': loss_diag.item(), 'consist': loss_consist.item()})
            
            if use_wandb:
                wandb.log({
                    "loss/total_step": loss.item(),
                    "loss/diagnosis_step": loss_diag.item(),
                    "loss/feature_consistency_step": loss_consist.item()
                }, step=global_step)
            
        avg_train_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_loss_diag = 0
        val_loss_consist = 0
        all_preds = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch['images'].to(device)
                diagnosis_targets = batch['diagnosis_targets'].to(device)
                diagnosis_mask = batch['diagnosis_mask'].to(device)
                
                outputs = model(images)
                frozen_outputs = frozen_model(images)
                
                loss_diag = diagnosis_loss(outputs['diagnosis_logits'], diagnosis_targets, diagnosis_mask)
                loss_consist = feature_consistency_loss(outputs['feature_logits'], frozen_outputs['feature_logits'])
                loss = loss_diag + lambda_consistency * loss_consist
                
                val_loss += loss.item()
                val_loss_diag += loss_diag.item()
                val_loss_consist += loss_consist.item()
                
                all_preds.append(outputs['diagnosis_logits'].cpu())
                all_targets.append(diagnosis_targets.cpu())
                all_masks.append(diagnosis_mask.cpu())
                
        val_loss /= len(val_dataloader)
        val_loss_diag /= len(val_dataloader)
        val_loss_consist /= len(val_dataloader)
        
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_masks = torch.cat(all_masks, dim=0).numpy()
        
        from filip.utils.metrics import compute_diagnosis_metrics
        metrics = compute_diagnosis_metrics(all_targets, all_preds, all_masks)
        
        print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f} | Diag Loss: {val_loss_diag:.4f} | Consist Loss: {val_loss_consist:.4f} | Macro AUC: {metrics['macro_auc']:.4f} | Micro AUC: {metrics['micro_auc']:.4f}")
        
        if use_wandb:
            wandb.log({
                "loss/total": val_loss,
                "loss/diagnosis": val_loss_diag,
                "loss/feature_consistency": val_loss_consist,
                "diagnosis/macro_auc": metrics["macro_auc"],
                "diagnosis/micro_auc": metrics["micro_auc"],
                "diagnosis/macro_f1": metrics["macro_f1"],
                "diagnosis/micro_f1": metrics["micro_f1"],
                "epoch": epoch + 1
            }, step=global_step)
            
        # Checkpoint saving
        is_best = False
        if metrics["macro_auc"] > best_macro_auc:
            best_macro_auc = metrics["macro_auc"]
            is_best = True
            
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "config": config,
            "diagnosis_vocab": dataset.diagnosis_vocab,
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_metric": best_macro_auc,
            "stage1_checkpoint_path": stage1_ckpt_path
        }
        
        # Save best.pt whenever it is best
        if is_best:
            torch.save(checkpoint, os.path.join(out_dir, "best.pt"))
            print(f"Saved new best model checkpoint (Val Macro AUC: {best_macro_auc:.4f})")
            
        # Save latest.pt for resuming
        latest_path = os.path.join(out_dir, "latest.pt")
        torch.save(checkpoint, latest_path)
        print(f"Saved latest checkpoint: {latest_path}")
            
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    train_ptbxl()
