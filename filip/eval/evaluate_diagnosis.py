import os
import yaml
import torch
import argparse
import numpy as np
import json
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision import transforms
from PIL import Image

from filip.data.dataset import ECGImageDataset
from filip.data.collator import ecg_collate_fn
from filip.model.filip_ecg_model import FILIPECGModel
from filip.utils.metrics import compute_diagnosis_metrics

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
        for batch in tqdm(dataloader, desc="Evaluation"):
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

def evaluate_diagnosis():
    parser = argparse.ArgumentParser(description="Evaluate FILIP PTB-XL Diagnosis Model")
    parser.add_argument('--config', type=str, default='filip/configs/ptbxl_diagnosis_adapt.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate on (e.g., test)')
    parser.add_argument('--thresholds_file', type=str, default=None, help='Path to tuned thresholds JSON file. If not provided, uses 0.5 for all classes.')
    parser.add_argument('--out_dir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--exclude_classes', type=str, default=None, help='Comma-separated list of classes to exclude from evaluation')
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
    print(f"Loading {args.split} dataset from {data_root} for evaluation...")
    dataset = ECGImageDataset(data_root=data_root, split=args.split, dataset_name=dataset_name, transform=transform)

    dataloader = DataLoader(dataset, batch_size=config.get('training', {}).get('batch_size', 32), shuffle=False, collate_fn=ecg_collate_fn, num_workers=4)

    best_thresholds = np.full(len(dataset.diagnosis_list), 0.5)
    if args.thresholds_file and os.path.exists(args.thresholds_file):
        print(f"Loading custom thresholds from {args.thresholds_file}...")
        with open(args.thresholds_file, 'r') as f:
            th_dict = json.load(f)
        for i, class_name in enumerate(dataset.diagnosis_list):
            if class_name in th_dict:
                best_thresholds[i] = th_dict[class_name]
        print(f"Using thresholds: {best_thresholds}")
    else:
        print("Using default threshold of 0.5 for all classes.")

    test_preds, test_targets, test_masks = get_predictions(dataloader, model, device)

    # Filter out excluded classes if specified
    class_names = dataset.diagnosis_list
    if args.exclude_classes:
        exclude_list = [c.strip() for c in args.exclude_classes.split(',') if c.strip()]
        print(f"Excluding classes: {exclude_list}")
        keep_indices = [i for i, c in enumerate(class_names) if c not in exclude_list]
        
        test_preds = test_preds[:, keep_indices]
        test_targets = test_targets[:, keep_indices]
        test_masks = test_masks[:, keep_indices]
        best_thresholds = best_thresholds[keep_indices]
        class_names = [class_names[i] for i in keep_indices]

    metrics = compute_diagnosis_metrics(test_targets, test_preds, test_masks, threshold=best_thresholds)

    # Format the output string
    out_text = f"Evaluating checkpoint {args.checkpoint} on Track: superclass\n"
    out_text += f"  Loaded {len(dataset)} ground-truth labels from {args.split} split\n"
    out_text += f"  Total samples evaluated: {len(test_preds)}\n"
    out_text += f"  Label space size: {len(class_names)}\n"
    out_text += f"Macro F1: {metrics['macro_f1']*100:.2f}\n"
    out_text += f"Micro F1: {metrics['micro_f1']*100:.2f}\n"
    out_text += f"Macro AUC: {metrics['macro_auc']*100:.2f}\n"
    out_text += f"Hamming Loss: {metrics['hamming_loss']*100:.2f}\n"
    out_text += f"Accuracy: {metrics['subset_accuracy']*100:.2f}\n"

    y_prob = 1.0 / (1.0 + np.exp(-test_preds))
    y_pred = (y_prob >= best_thresholds).astype(np.float32)
    
    avg_pred_per_sample = np.sum(y_pred * test_masks) / len(y_pred)
    avg_true_per_sample = np.sum(test_targets * test_masks) / len(test_targets)
    
    out_text += f"\nAverage predicted labels per sample: {avg_pred_per_sample:.4f}\n"
    out_text += f"Average ground-truth labels per sample: {avg_true_per_sample:.4f}\n\n"
    
    out_text += "Per-class Metrics:\n"
    out_text += f"{'Class':<10} | {'Threshold':<10} | {'Pred Pos Rate':<15} | {'True Pos Rate':<15} | {'F1':<10} | {'AUC':<10}\n"
    out_text += "-" * 85 + "\n"
    
    for i, class_name in enumerate(class_names):
        valid_count = np.sum(test_masks[:, i])
        if valid_count > 0:
            pred_rate = np.sum(y_pred[:, i] * test_masks[:, i]) / valid_count
            true_rate = np.sum(test_targets[:, i] * test_masks[:, i]) / valid_count
        else:
            pred_rate = 0.0
            true_rate = 0.0
            
        f1 = metrics['class_f1s'][i]
        auc = metrics['class_aucs'][i]
        th = best_thresholds[i]
        
        out_text += f"{class_name:<10} | {th:<10.2f} | {pred_rate:<15.4f} | {true_rate:<15.4f} | {f1*100:<10.2f} | {auc*100:<10.2f}\n"

    print("\n--- Evaluation Results ---")
    print(out_text)

    # Determine out_dir
    if args.out_dir is None:
        experiment_name = config.get('experiment_name', 'ptbxl_diagnosis_adapt')
        # If thresholds were loaded, maybe save to a tuned folder
        folder_suffix = "-tuned" if args.thresholds_file else ""
        out_dir = f"/outputs/filip/{experiment_name}/ptb-{args.split}{folder_suffix}"
        if out_dir.startswith("/outputs") and not (os.path.exists("/outputs") and os.access("/outputs", os.W_OK)):
            out_dir = out_dir.lstrip("/")
    else:
        out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.txt")
    
    with open(out_path, 'w') as f:
        f.write(out_text)
        
    print(f"Metrics saved to {out_path}")

if __name__ == "__main__":
    evaluate_diagnosis()
