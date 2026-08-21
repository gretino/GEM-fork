import os
import yaml
import torch
import argparse
import numpy as np
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from torchvision import transforms
from PIL import Image

from filip.data.dataset import ECGImageDataset
from filip.data.collator import ecg_collate_fn
from filip.model.filip_ecg_model import FILIPECGModel

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

def get_predictions(dataloader, model, device):
    all_preds = []
    all_targets = []
    all_masks = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Tuning Evaluation"):
            images = batch['images'].to(device)
            diagnosis_targets = batch['diagnosis_targets'].to(device)
            diagnosis_mask = batch['diagnosis_mask'].to(device)

            outputs = model(images)
            
            all_preds.append(outputs['diagnosis_logits'].cpu())
            all_targets.append(diagnosis_targets.cpu())
            all_masks.append(diagnosis_mask.cpu())

    return torch.cat(all_preds, dim=0).numpy(), \
           torch.cat(all_targets, dim=0).numpy(), \
           torch.cat(all_masks, dim=0).numpy()

def tune_thresholds():
    parser = argparse.ArgumentParser(description="Tune FILIP PTB-XL Diagnosis Model Thresholds")
    parser.add_argument('--config', type=str, default='filip/configs/ptbxl_diagnosis_adapt.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--out_dir', type=str, default=None, help='Directory to save threshold results')
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

    print("Initializing model...")
    model = FILIPECGModel(config).to(device)

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in ckpt['model_state_dict'].items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"Skipping key '{k}' due to shape mismatch: checkpoint {v.shape} vs model {model_state_dict[k].shape}")
    model.load_state_dict(filtered_state_dict, strict=False)
    
    model.eval()

    dataset_name = config.get('dataset_name', 'ptbxl')
    print(f"Loading val dataset from {data_root} for tuning...")
    val_dataset = ECGImageDataset(data_root=data_root, split='val', dataset_name=dataset_name, transform=transform)

    val_dataloader = DataLoader(val_dataset, batch_size=config.get('training', {}).get('batch_size', 32), shuffle=False, collate_fn=ecg_collate_fn, num_workers=4)
    
    val_preds, val_targets, val_masks = get_predictions(val_dataloader, model, device)
    
    print("Tuning thresholds on val set...")
    val_probs = 1.0 / (1.0 + np.exp(-val_preds))
    best_thresholds = np.full(val_targets.shape[1], 0.5)
    
    for c in range(val_targets.shape[1]):
        valid_idx = np.where(val_masks[:, c] == 1.0)[0]
        if len(valid_idx) == 0:
            continue
            
        y_t = val_targets[valid_idx, c]
        y_p_prob = val_probs[valid_idx, c]
        
        best_f1 = -1.0
        best_th = 0.5
        for th in np.arange(0.01, 1.00, 0.01):
            y_p = (y_p_prob >= th).astype(np.float32)
            f1 = f1_score(y_t, y_p, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        best_thresholds[c] = best_th
        
    print(f"Optimal thresholds per class: {best_thresholds}")

    th_dict = {val_dataset.diagnosis_list[i]: float(best_thresholds[i]) for i in range(len(best_thresholds))}

    # Determine out_dir
    if args.out_dir is None:
        experiment_name = config.get('experiment_name', 'ptbxl_diagnosis_adapt')
        out_dir = f"/outputs/filip/{experiment_name}/ptb-val-tuning"
        if out_dir.startswith("/outputs") and not (os.path.exists("/outputs") and os.access("/outputs", os.W_OK)):
            out_dir = out_dir.lstrip("/")
    else:
        out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "thresholds.json")
    
    with open(out_path, 'w') as f:
        json.dump(th_dict, f, indent=4)
        
    print(f"Tuned thresholds saved to {out_path}")

if __name__ == "__main__":
    tune_thresholds()
