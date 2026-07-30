# filip/data/collator.py

import torch

def ecg_collate_fn(batch):
    images = torch.stack([item["images"] for item in batch])
    sample_ids = [item["sample_ids"] for item in batch]
    
    collated = {
        "images": images,
        "sample_ids": sample_ids,
        "report_texts": [item.get("report_text", "") for item in batch],
    }
    
    # Feature Targets
    if batch[0].get("feature_targets") is not None:
        collated["feature_targets"] = torch.stack([item["feature_targets"] for item in batch])
        collated["feature_mask"] = torch.stack([item["feature_mask"] for item in batch])
        collated["feature_confidence"] = torch.ones_like(collated["feature_mask"])

    # Diagnosis Targets
    if batch[0].get("diagnosis_targets") is not None:
        collated["diagnosis_targets"] = torch.stack([item["diagnosis_targets"] for item in batch])
        collated["diagnosis_mask"] = torch.stack([item["diagnosis_mask"] for item in batch])
        
    # Morphology Targets
    if batch[0].get("morphology_targets") is not None:
        collated["morphology_targets"] = torch.stack([item["morphology_targets"] for item in batch])
        collated["morphology_mask"] = torch.stack([item["morphology_mask"] for item in batch])
    
    return collated
