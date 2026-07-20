# filip/data/dataset.py

import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from filip.data.vocab import get_feature_vocab, get_diagnosis_vocab

class ECGImageDataset(Dataset):
    def __init__(self, data_root, split='train', dataset_name='mimic', transform=None):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        
        # Load records list
        records_path = os.path.join(data_root, f"{split}_records.json")
        if os.path.exists(records_path):
            with open(records_path, 'r') as f:
                self.records = json.load(f)
        else:
            print(f"Warning: {records_path} not found.")
            self.records = []
            
        self.feature_vocab, self.feature_list = get_feature_vocab()
        self.diagnosis_vocab, self.diagnosis_list = get_diagnosis_vocab()
        
        # Load targets based on dataset_name
        self.targets = {}
        self.morphology = {}
        if self.dataset_name == 'mimic':
            target_path = os.path.join(data_root, "features.json")
            if os.path.exists(target_path):
                with open(target_path, 'r') as f:
                    self.targets = json.load(f)
            morphology_path = os.path.join(data_root, "diagnoses.json")
            if os.path.exists(morphology_path):
                with open(morphology_path, 'r') as f:
                    self.morphology = json.load(f)
        elif self.dataset_name == 'ptbxl':
            target_path = os.path.join(data_root, "diagnoses.json")
            if os.path.exists(target_path):
                with open(target_path, 'r') as f:
                    self.targets = json.load(f)
                if self.targets:
                    first_key = list(self.targets.keys())[0]
                    self.diagnosis_list = list(self.targets[first_key].keys())
                    self.diagnosis_vocab = {diag: i for i, diag in enumerate(self.diagnosis_list)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        study_id = str(record['study_id'])
        
        # Build image path: e.g., "images/123456-0.png" or "images/123456.png"
        image_path = os.path.join(self.data_root, "images", f"{study_id}-0.png")
        if not os.path.exists(image_path):
            image_path = os.path.join(self.data_root, "images", f"{study_id}.png")
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Fallback dummy image if missing (useful for testing when data is being regenerated)
            image = Image.new('RGB', (224, 224), (255, 255, 255))
            
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        sample = {
            "images": image,
            "sample_ids": study_id,
            "feature_targets": None,
            "feature_mask": None,
            "diagnosis_targets": None,
            "diagnosis_mask": None,
            "morphology_targets": None,
            "morphology_mask": None
        }
        
        # Load targets if available
        if study_id in self.targets:
            target_dict = self.targets[study_id]
            if self.dataset_name == 'mimic':
                # Features
                feat_array = torch.zeros(len(self.feature_list), dtype=torch.float32)
                for i, feat in enumerate(self.feature_list):
                    if feat in target_dict:
                        feat_array[i] = float(target_dict[feat])
                    else:
                        feat_array[i] = -1.0 # Unknown
                
                # Mask: 1 if valid label (0 or 1), 0 if unknown (-1)
                feat_mask = (feat_array != -1.0).float()
                # Clamp -1 to 0 for binary targets (loss uses mask anyway)
                feat_array = torch.clamp(feat_array, min=0.0)
                
                sample["feature_targets"] = feat_array
                sample["feature_mask"] = feat_mask
                
            elif self.dataset_name == 'ptbxl':
                # Diagnoses
                diag_array = torch.zeros(len(self.diagnosis_list), dtype=torch.float32)
                for i, diag in enumerate(self.diagnosis_list):
                    if diag in target_dict:
                        diag_array[i] = float(target_dict[diag])
                    else:
                        diag_array[i] = -1.0
                        
                diag_mask = (diag_array != -1.0).float()
                diag_array = torch.clamp(diag_array, min=0.0)
                
                sample["diagnosis_targets"] = diag_array
                sample["diagnosis_mask"] = diag_mask
                
        # Load morphology targets for mimic if available
        if self.dataset_name == 'mimic':
            morph_array = torch.zeros(7, dtype=torch.float32)
            morph_mask = torch.zeros(7, dtype=torch.float32)
            
            if study_id in self.morphology:
                morphology_dict = self.morphology[study_id]
                morph_metrics = ['pr_interval', 'qrs_duration', 'qt_interval', 'rr_interval', 'p_axis', 'qrs_axis', 't_axis']
                bounds = {
                    'pr_interval': (0, 500),
                    'qrs_duration': (0, 300),
                    'qt_interval': (0, 800),
                    'rr_interval': (200, 3000),
                    'p_axis': (-180, 180),
                    'qrs_axis': (-180, 180),
                    't_axis': (-180, 180)
                }
                stats = {
                    'pr_interval': {'mean': 164.7816, 'std': 35.2563},
                    'qrs_duration': {'mean': 101.6943, 'std': 24.3214},
                    'qt_interval': {'mean': 400.9349, 'std': 50.2524},
                    'rr_interval': {'mean': 817.6569, 'std': 199.6918},
                    'p_axis': {'mean': 44.8669, 'std': 30.6746},
                    'qrs_axis': {'mean': 9.6230, 'std': 46.1662},
                    't_axis': {'mean': 41.4568, 'std': 59.9372}
                }
                
                for i, m in enumerate(morph_metrics):
                    val = morphology_dict.get(m)
                    if val is not None:
                        min_v, max_v = bounds[m]
                        if min_v <= val <= max_v:
                            std_val = (val - stats[m]['mean']) / stats[m]['std']
                            morph_array[i] = float(std_val)
                            morph_mask[i] = 1.0
                            
            sample["morphology_targets"] = morph_array
            sample["morphology_mask"] = morph_mask
                
        return sample
