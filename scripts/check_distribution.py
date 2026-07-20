import json
from collections import defaultdict

data_dir = 'data/ptbxl_sub_class'
splits = ['train', 'val', 'test']

with open(f'{data_dir}/diagnoses.json', 'r') as f:
    diagnoses = json.load(f)

for split in splits:
    with open(f'{data_dir}/{split}_records.json', 'r') as f:
        records = json.load(f)
    
    class_counts = defaultdict(int)
    total_samples = len(records)
    
    for record in records:
        study_id = record['study_id']
        labels = diagnoses.get(study_id, {})
        for label, val in labels.items():
            if val > 0:
                class_counts[label] += 1
                
    print(f"--- {split.upper()} SPLIT ({total_samples} samples) ---")
    for label, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{label}: {count} ({count/total_samples*100:.2f}%)")
    print("\n")
