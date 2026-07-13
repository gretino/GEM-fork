import os
import json
import ast
import random
import pandas as pd
from tqdm import tqdm

def main():
    print("Loading PTB-XL database and SCP statements...")
    db_path = '/home/qfbqt/datasets/ptb_xl_1.0.3/ptbxl_database_translated.csv'
    scp_path = '/home/qfbqt/datasets/ptb_xl_1.0.3/scp_statements.csv'
    gen_dir = '/home/qfbqt/8TB/datasets/ptb-xl-gen/'
    split_dir = '/home/qfbqt/8TB/datasets/ptbxl_split/'
    
    output_dir = '/home/qfbqt/8TB/datasets/ptb-xl/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Symlink images folder
    images_symlink = os.path.join(output_dir, 'images')
    if not os.path.exists(images_symlink):
        print(f"Creating symlink: {images_symlink} -> {gen_dir}")
        os.symlink(gen_dir, images_symlink)
        
    df_db = pd.read_csv(db_path)
    df_scp = pd.read_csv(scp_path, index_col=0)
    
    # Create mapping from SCP code to Superclass
    scp_to_superclass = {}
    for scp, row in df_scp.iterrows():
        if row['diagnostic'] == 1.0 and pd.notna(row['diagnostic_class']):
            scp_to_superclass[scp] = row['diagnostic_class']
            
    PTBXL_SUPERCLASSES = ["NORM", "MI", "HYP", "CD", "STTC"]
    
    print("Scanning available images in ptb-xl-gen...")
    matched_records = []
    
    for idx, row in tqdm(df_db.iterrows(), total=len(df_db)):
        ecg_id = int(row['ecg_id'])
        basename = os.path.basename(row['filename_hr'])
        img_name = f"{basename}-0.png"
        
        # Check if the image exists in ptb-xl-gen
        if os.path.exists(os.path.join(gen_dir, img_name)):
            # Determine split from ptbxl_split directories if possible
            assigned_split = None
            for s in ['train', 'val', 'test']:
                if os.path.exists(os.path.join(split_dir, s, f"{ecg_id}.png")):
                    assigned_split = s
                    break
            
            matched_records.append({
                "ecg_id": ecg_id,
                "study_id": basename,
                "patient_id": int(row['patient_id']) if not pd.isna(row['patient_id']) else ecg_id,
                "ecg_time": str(row['recording_date']) if not pd.isna(row['recording_date']) else "",
                "path": str(row['filename_hr']),
                "report_text": str(row['report']) if not pd.isna(row['report']) else "",
                "scp_codes": row['scp_codes'],
                "assigned_split": assigned_split
            })
            
    print(f"Matched {len(matched_records)} records with actual generated images.")
    
    # Check if we have 0 train or val files
    splits = {
        'train': [r for r in matched_records if r['assigned_split'] == 'train'],
        'val': [r for r in matched_records if r['assigned_split'] == 'val'],
        'test': [r for r in matched_records if r['assigned_split'] == 'test']
    }
    
    print(f"Initial split counts: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    # If train or val is empty, dynamically split the matched records (using seed 42)
    if len(splits['train']) == 0 or len(splits['val']) == 0:
        print("Warning: Some splits are empty. Dynamically splitting matched records (80/10/10) to prevent empty data splits...")
        random.seed(42)
        random.shuffle(matched_records)
        n = len(matched_records)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        
        splits['train'] = matched_records[:n_train]
        splits['val'] = matched_records[n_train:n_train+n_val]
        splits['test'] = matched_records[n_train+n_val:]
        print(f"New dynamic split counts: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
        
    # 2. Write split json files
    for s in ['train', 'val', 'test']:
        records_list = []
        for r in splits[s]:
            records_list.append({
                "subject_id": r['patient_id'],
                "study_id": r['study_id'],
                "file_name": r['study_id'],
                "ecg_time": r['ecg_time'],
                "path": r['path'],
                "report_text": r['report_text']
            })
        out_path = os.path.join(output_dir, f"{s}_records.json")
        with open(out_path, 'w') as f:
            json.dump(records_list, f, indent=4)
        print(f"Saved {len(records_list)} records to {out_path}")
        
    # 3. Build diagnoses.json
    diagnoses_dict = {}
    for r in matched_records:
        basename = r['study_id']
        scp_dict = ast.literal_eval(r['scp_codes'])
        
        diag_dict = {}
        for superclass in PTBXL_SUPERCLASSES:
            has_superclass = 0.0
            for subclass in scp_dict.keys():
                if subclass in scp_to_superclass and scp_to_superclass[subclass] == superclass:
                    has_superclass = 1.0
                    break
            diag_dict[superclass] = has_superclass
            
        diagnoses_dict[basename] = diag_dict
        
    diag_path = os.path.join(output_dir, "diagnoses.json")
    with open(diag_path, 'w') as f:
        json.dump(diagnoses_dict, f, indent=4)
    print(f"Saved diagnoses for {len(diagnoses_dict)} records to {diag_path}")
    
    # 4. Create symlink in data/ptb-xl
    repo_symlink = '/home/qfbqt/repo/GEM-fork/data/ptb-xl'
    if os.path.exists(repo_symlink):
        if os.path.islink(repo_symlink):
            os.unlink(repo_symlink)
        elif os.path.isdir(repo_symlink) and len(os.listdir(repo_symlink)) == 0:
            os.rmdir(repo_symlink)
            
    if not os.path.exists(repo_symlink):
        print(f"Creating symlink: {repo_symlink} -> {output_dir}")
        os.symlink(output_dir, repo_symlink)
        
    print("PTB-XL preprocessing completed successfully!")

if __name__ == "__main__":
    main()
