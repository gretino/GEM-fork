# filip/eval/evaluate_filip_zero_shot.py

import os
import yaml
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoTokenizer

from filip.data.dataset import ECGImageDataset
from filip.data.collator import ecg_collate_fn
from filip.model.filip_ecg_model import FILIPECGModel
from filip.train.train_mimic_feature import ExpandToSquare
from filip.utils.metrics import compute_diagnosis_metrics

# Standard clinical prompt templates for PTB-XL superclasses
PTBXL_SUPERCLASS_PROMPTS = {
    'NORM': ["normal electrocardiogram", "normal sinus rhythm ecg", "no diagnostic abnormalities"],
    'MI': ["myocardial infarction", "inferior or anterior infarct", "st elevation myocardial infarction"],
    'STTC': ["st t wave changes", "st depression or t wave inversion", "nonspecific st-t wave abnormality"],
    'CD': ["conduction disturbance", "bundle branch block or intraventricular block", "left or right bundle branch block"],
    'HYP': ["ventricular hypertrophy", "left ventricular hypertrophy", "enlarged heart chambers"]
}

def evaluate_zero_shot():
    parser = argparse.ArgumentParser(description="Zero-Shot FILIP Evaluation on PTB-XL")
    parser.add_argument('--config', type=str, default='filip/configs/mimic_report_alignment_pretrain.yaml', help='Path to pretraining config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to FILIP report alignment checkpoint')
    parser.add_argument('--data_root', type=str, default='/home/qfbqt/8TB/datasets/ptb-xl/', help='Path to PTB-XL dataset')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate on')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== FILIP Zero-Shot Prompt Evaluation on PTB-XL (Split: {args.split}) ===")

    data_root = args.data_root
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

    # Load Model & Tokenizer
    model = FILIPECGModel(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    text_model_name = config.get('model', {}).get('text_encoder', 'openai/clip-vit-base-patch32')
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    # Encode class prompt templates
    class_names = list(PTBXL_SUPERCLASS_PROMPTS.keys()) # ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    prompt_texts = [PTBXL_SUPERCLASS_PROMPTS[c][0] for c in class_names] # Representative prompt per class

    text_inputs = tokenizer(
        prompt_texts,
        padding='max_length',
        max_length=config.get('model', {}).get('text_max_length', 77),
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        text_outputs = model.text_encoder(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask']
        )
        token_features = text_outputs.last_hidden_state  # [C, T, H_t]
        content_mask = (text_inputs['input_ids'] != tokenizer.pad_token_id)  # [C, T]

    # Evaluate images in dataloader
    dataset = ECGImageDataset(data_root=data_root, split=args.split, dataset_name='ptbxl', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=ecg_collate_fn, num_workers=4)

    all_logits = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Zero-Shot Scoring"):
            images = batch['images'].to(device)
            patch_features = model.vision_encoder(images) # [B, P, H_i]

            # Compute FILIP score_prompts between B images and C class prompts
            logits, _ = model.report_alignment_head.score_prompts(patch_features, token_features, content_mask) # [B, C]
            
            all_logits.append(logits.cpu())
            all_targets.append(batch['diagnosis_targets'].cpu())
            all_masks.append(batch['diagnosis_mask'].cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()  # [N, C]
    all_targets = torch.cat(all_targets, dim=0).numpy() # [N, C]
    all_masks = torch.cat(all_masks, dim=0).numpy()     # [N, C]

    # Apply Softmax across candidate class prompts for probability calibration
    from scipy.special import softmax
    all_probs = softmax(all_logits, axis=-1)

    # Compute AUROC & F1 metrics
    metrics = compute_diagnosis_metrics(all_targets, all_probs, all_masks)



    print("\n" + "=" * 70)
    print("      FILIP ZERO-SHOT PTB-XL EVALUATION RESULTS (NO HEAD / NO TRAINING)")
    print("=" * 70)
    print(f"Macro AUROC : {metrics['macro_auc']*100:.2f}%")
    print(f"Macro F1    : {metrics['macro_f1']*100:.2f}%")
    print(f"Micro F1    : {metrics['micro_f1']*100:.2f}%")
    print("-" * 70)
    for i, name in enumerate(class_names):
        print(f"Class: {name:<6} | AUROC: {metrics['class_aucs'][i]*100:.2f}% | F1: {metrics['class_f1s'][i]*100:.2f}%")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    evaluate_zero_shot()
