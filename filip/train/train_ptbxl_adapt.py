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
from filip.model.losses import diagnosis_loss, asymmetric_loss

from PIL import Image


DEFAULT_DIAGNOSIS_TEXT = {
    "NORM": "a normal electrocardiogram",
    "MI": "an electrocardiogram showing myocardial infarction",
    "HYP": "an electrocardiogram showing cardiac hypertrophy",
    "CD": "an electrocardiogram showing a cardiac conduction disturbance",
    "STTC": "an electrocardiogram showing an ST segment or T wave abnormality",
}


def build_diagnosis_prompts(diagnosis_list, config):
    """Build one report-like prompt per downstream label, preserving label order."""
    configured = config.get('text_diagnosis', {}).get('class_prompts', {})
    template = config.get('text_diagnosis', {}).get(
        'default_template', 'an electrocardiogram showing {label}'
    )
    return [configured.get(label, DEFAULT_DIAGNOSIS_TEXT.get(label, template.format(label=label)))
            for label in diagnosis_list]


def tokenize_prompts(tokenizer, prompts, config, device):
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=config.get('model', {}).get('text_max_length', 77),
        return_special_tokens_mask=True,
        return_tensors='pt',
    )
    content_mask = encoded['attention_mask'].bool() & ~encoded.pop('special_tokens_mask').bool()
    if not content_mask.any(dim=1).all():
        raise ValueError("Every diagnosis prompt must contain at least one content token")
    return {
        'input_ids': encoded['input_ids'].to(device),
        'attention_mask': encoded['attention_mask'].to(device),
        'content_mask': content_mask.to(device),
    }

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
    parser.add_argument('--train_pct', '--data_ratio', type=float, default=100.0, dest='train_pct', help='Percentage of training data to use')
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
    
    dataset_name = config.get('dataset_name', 'ptbxl')
    train_split = config.get('training', {}).get('train_split', 'train')
    dataset = ECGImageDataset(data_root=data_root, split=train_split, dataset_name=dataset_name, transform=transform)
    if args.train_pct < 100.0:
        import random
        rng = random.Random(42)
        num_records = max(1, int(len(dataset.records) * (args.train_pct / 100.0)))
        dataset.records = rng.sample(dataset.records, num_records)
        print(f"Subsampled training dataset to {args.train_pct}%: using {len(dataset.records)} records")
        
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=ecg_collate_fn, num_workers=4)
    
    val_dataset = ECGImageDataset(data_root=data_root, split='val', dataset_name=dataset_name, transform=transform)

    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=ecg_collate_fn, num_workers=4)
    
    num_ds_classes = len(dataset.diagnosis_list)
    config['model']['num_classes'] = num_ds_classes
    config['model']['num_diagnosis'] = num_ds_classes
    
    model = FILIPECGModel(config).to(device)
    diagnosis_mode = config.get('model', {}).get('diagnosis_mode', 'class_head')
    use_text_diagnosis = diagnosis_mode == 'text_prompts'
    if use_text_diagnosis and not model.use_report_alignment:
        raise ValueError("diagnosis_mode=text_prompts requires use_report_alignment=true")
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

    prompt_inputs = None
    diagnosis_prompts = None
    if use_text_diagnosis:
        from transformers import AutoTokenizer

        diagnosis_prompts = build_diagnosis_prompts(dataset.diagnosis_list, config)
        tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder'])
        prompt_inputs = tokenize_prompts(tokenizer, diagnosis_prompts, config, device)
        if config.get('text_diagnosis', {}).get('freeze_text_encoder', True):
            for parameter in model.text_encoder.parameters():
                parameter.requires_grad = False
        if config.get('text_diagnosis', {}).get('freeze_text_projection', True):
            for parameter in model.report_alignment_head.text_projection.parameters():
                parameter.requires_grad = False
        # This head is intentionally retained for class-head compatibility but
        # is not part of text-prompt adaptation.
        for parameter in model.diagnosis_head.parameters():
            parameter.requires_grad = False
        
    freeze_backbone = config.get('training', {}).get('freeze_backbone', False)
    unfreeze_last_n = config.get('training', {}).get('unfreeze_last_n_layers', None)
    
    if freeze_backbone and use_text_diagnosis:
        print("Freezing the vision encoder; adapting only the report alignment projection...")
        for parameter in model.vision_encoder.parameters():
            parameter.requires_grad = False
    elif freeze_backbone:
        print("Freezing all parameters except diagnosis_head (Linear Probe mode)...")
        for name, param in model.named_parameters():
            if "diagnosis_head" not in name:
                param.requires_grad = False
    elif unfreeze_last_n is not None and unfreeze_last_n > 0:
        if hasattr(model, 'vision_encoder') and hasattr(model.vision_encoder.encoder, 'vision_model') and hasattr(model.vision_encoder.encoder.vision_model, 'encoder'):
            layers = model.vision_encoder.encoder.vision_model.encoder.layers
            total_layers = len(layers)
            cutoff_layer = max(0, total_layers - unfreeze_last_n)
            print(f"Partial Freezing: Freezing ViT layers 0 to {cutoff_layer - 1} out of {total_layers} (unfreezing last {unfreeze_last_n} layers + heads)...")
            
            # Freeze embeddings and pre_layrnorm
            for param in model.vision_encoder.encoder.vision_model.embeddings.parameters():
                param.requires_grad = False
            if hasattr(model.vision_encoder.encoder.vision_model, 'pre_layrnorm') and model.vision_encoder.encoder.vision_model.pre_layrnorm is not None:
                for param in model.vision_encoder.encoder.vision_model.pre_layrnorm.parameters():
                    param.requires_grad = False
                    
            # Freeze early ViT layers
            for i in range(cutoff_layer):
                for param in layers[i].parameters():
                    param.requires_grad = False
        else:
            print(f"Warning: Could not identify transformer layer structure for unfreeze_last_n_layers={unfreeze_last_n}.")
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {trainable_params:,} trainable out of {total_params:,} total ({trainable_params/total_params*100:.2f}%).")
    
    backbone_lr = float(config.get('training', {}).get('backbone_lr', config.get('training', {}).get('learning_rate', 3.0e-5)))
    if 'head_lr' in config.get('training', {}):
        head_lr = float(config['training']['head_lr'])
    else:
        head_lr_scale = float(config.get('training', {}).get('head_lr_scale', 10.0))
        head_lr = backbone_lr * head_lr_scale
        
    weight_decay = float(config.get('training', {}).get('weight_decay', 0.01))
    
    head_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "diagnosis_head" in name:
                head_params.append(param)
            else:
                other_params.append(param)
                
    if head_params and head_lr != backbone_lr:
        print(f"Explicit Differential LRs: Backbone LR = {backbone_lr:.2e}, Diagnosis Head LR = {head_lr:.2e}")
        param_groups = [
            {'params': other_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr}
        ]
    else:
        print(f"Uniform Learning Rate: LR = {backbone_lr:.2e}")
        param_groups = [{'params': other_params + head_params, 'lr': backbone_lr}]
        
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    epochs = config['training']['epochs']
    
    experiment_name = config.get('experiment_name', 'ptbxl_diagnosis_adapt')
    pct_str = f"{int(args.train_pct)}" if args.train_pct.is_integer() else f"{args.train_pct}"
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = f"outputs/filip/{experiment_name}_{pct_str}"
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
                    "backbone_lr": backbone_lr,
                    "head_lr": head_lr,
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
            
    mode_name = "Text-Prompt" if use_text_diagnosis else "Class-Head"
    dataset_display = config.get('dataset_name', 'ptbxl').upper()
    print(f"Starting Stage 2: {dataset_display} {mode_name} Diagnosis Adaptation")

    global_step = 0
    best_macro_auc = -1.0
    start_epoch = 0
    
    resume_path = args.resume_from or config.get('resume_from_checkpoint')
    if not resume_path and os.path.exists(os.path.join(out_dir, "latest.pt")):
        resume_path = os.path.join(out_dir, "latest.pt")
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
            if use_text_diagnosis:
                outputs = model.forward_text_prompts(images, **prompt_inputs)
            else:
                outputs = model(images)
            
            use_asl = config.get('loss', {}).get('use_asl', False)
            if use_asl:
                loss = asymmetric_loss(outputs['diagnosis_logits'], diagnosis_targets, diagnosis_mask)
            else:
                loss = diagnosis_loss(outputs['diagnosis_logits'], diagnosis_targets, diagnosis_mask)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': loss.item()})
            
            if use_wandb:
                wandb.log({
                    "loss/total_step": loss.item()
                }, step=global_step)
            
        avg_train_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch['images'].to(device)
                diagnosis_targets = batch['diagnosis_targets'].to(device)
                diagnosis_mask = batch['diagnosis_mask'].to(device)
                
                if use_text_diagnosis:
                    outputs = model.forward_text_prompts(images, **prompt_inputs)
                else:
                    outputs = model(images)
                
                use_asl = config.get('loss', {}).get('use_asl', False)
                if use_asl:
                    loss = asymmetric_loss(outputs['diagnosis_logits'], diagnosis_targets, diagnosis_mask)
                else:
                    loss = diagnosis_loss(outputs['diagnosis_logits'], diagnosis_targets, diagnosis_mask)
                    
                val_loss += loss.item()
                
                all_preds.append(outputs['diagnosis_logits'].cpu())
                all_targets.append(diagnosis_targets.cpu())
                all_masks.append(diagnosis_mask.cpu())
                
        val_loss /= len(val_dataloader)
        
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_masks = torch.cat(all_masks, dim=0).numpy()
        
        from filip.utils.metrics import compute_diagnosis_metrics
        metrics = compute_diagnosis_metrics(all_targets, all_preds, all_masks)
        
        print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f} | Macro AUC: {metrics['macro_auc']:.4f} | Micro AUC: {metrics['micro_auc']:.4f}")
        
        if use_wandb:
            wandb.log({
                "loss/total": val_loss,
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
            "stage1_checkpoint_path": stage1_ckpt_path,
            "diagnosis_mode": diagnosis_mode,
            "diagnosis_prompts": diagnosis_prompts,
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
