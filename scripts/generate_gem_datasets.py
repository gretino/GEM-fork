import json
import os
import random
import ast
import pandas as pd
from tqdm import tqdm

import argparse

def generate_gem_datasets():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of samples to process (for debugging)")
    args = parser.parse_args()

    print("Loading PTB-XL database and SCP statements...")
    db_path = '/home/qfbqt/datasets/ptb_xl_1.0.3/ptbxl_database.csv'
    scp_path = '/home/qfbqt/datasets/ptb_xl_1.0.3/scp_statements.csv'
    
    df_db = pd.read_csv(db_path, index_col='ecg_id')
    df_scp = pd.read_csv(scp_path, index_col=0)
    
    # Create mapping from SCP code to Superclass
    scp_to_superclass = {}
    for scp, row in df_scp.iterrows():
        if row['diagnostic'] == 1.0 and pd.notna(row['diagnostic_class']):
            scp_to_superclass[scp] = row['diagnostic_class']
            
    # Load tags for reasoning
    tags_path = '/home/qfbqt/repo/ExplanableECG/scp2tag.json'
    english_tags = []
    if os.path.exists(tags_path):
        with open(tags_path, 'r') as f:
            all_tags_dict = json.load(f)
            english_tags = list(all_tags_dict.values())
    else:
        print(f"Tags file not found at {tags_path}. Will use fallback reasoning.")
    
    ecg_ids = list(df_db.index)
    random.seed(42)
    random.shuffle(ecg_ids)
    
    if args.limit > 0:
        ecg_ids = ecg_ids[:args.limit]
    
    output_dir = '/home/qfbqt/repo/GEM-fork/data/gem_data'
    os.makedirs(output_dir, exist_ok=True)
    
    out_super_r = os.path.join(output_dir, 'gem_train_superclass_reasoning.json')
    out_super_nr = os.path.join(output_dir, 'gem_train_superclass_no_reasoning.json')
    out_sub_r = os.path.join(output_dir, 'gem_train_subclass_reasoning.json')
    out_sub_nr = os.path.join(output_dir, 'gem_train_subclass_no_reasoning.json')
    
    data_super_r, data_super_nr, data_sub_r, data_sub_nr = [], [], [], []
    
    print(f"Processing {len(ecg_ids)} records...")
    for ecg_id in tqdm(ecg_ids):
        row = df_db.loc[ecg_id]
        
        # Get patient metadata
        age = row['age'] if pd.notna(row['age']) else "N/A"
        sex = "Male" if row['sex'] == 0 else ("Female" if row['sex'] == 1 else "N/A") 
        height = row['height'] if pd.notna(row['height']) else "N/A"
        weight = row['weight'] if pd.notna(row['weight']) else "N/A"
        
        # Extract subclasses
        scp_dict = ast.literal_eval(row['scp_codes'])
        subclasses = list(scp_dict.keys())
        random.shuffle(subclasses)
        subclass_ans = ", ".join(subclasses)
        
        # Extract superclasses
        superclasses = set()
        for scp in subclasses:
            if scp in scp_to_superclass:
                superclasses.add(scp_to_superclass[scp])
        superclasses = list(superclasses)
        random.shuffle(superclasses)
        superclass_ans = ", ".join(superclasses)
        
        # Paths
        image_path = f"ptb-xl-gen/{row['filename_hr']}-0.png"
        ecg_path = f"ptbxl/{row['filename_hr']}"
        
        # Build Reasoning text
        reasoning_points = []
        for scp in subclasses:
            if scp in df_scp.index and pd.notna(df_scp.loc[scp, 'description']):
                reasoning_points.append(str(df_scp.loc[scp, 'description']))
        if reasoning_points:
            reasoning = f"The ECG shows features indicative of {', '.join(reasoning_points)}."
        else:
            reasoning = "The ECG shows no specific abnormal waveform features."
            
        # Prompts
        meta_str = f"Age: {age}, Sex: {sex}, Height: {height}, Weight: {weight}."
        prompt_superclass = f"<image>\n{meta_str} Provide the diagnostic superclasses (e.g., NORM, MI, STTC, CD, HYP)."
        prompt_subclass = f"<image>\n{meta_str} Provide the specific SCP diagnostic subclasses."
        
        # Create records
        base_record = {
            "id": str(ecg_id),
            "image": image_path,
            "ecg": ecg_path
        }
        
        # 1. Superclass Reasoning
        rec = base_record.copy()
        rec["conversations"] = [
            {"from": "human", "value": prompt_superclass},
            {"from": "gpt", "value": f"<reasoning>{reasoning}</reasoning>\n<answer>{superclass_ans}</answer>"}
        ]
        data_super_r.append(rec)
        
        # 2. Superclass No-Reasoning
        rec = base_record.copy()
        rec["conversations"] = [
            {"from": "human", "value": prompt_superclass},
            {"from": "gpt", "value": f"<answer>{superclass_ans}</answer>"}
        ]
        data_super_nr.append(rec)
        
        # 3. Subclass Reasoning
        rec = base_record.copy()
        rec["conversations"] = [
            {"from": "human", "value": prompt_subclass},
            {"from": "gpt", "value": f"<reasoning>{reasoning}</reasoning>\n<answer>{subclass_ans}</answer>"}
        ]
        data_sub_r.append(rec)
        
        # 4. Subclass No-Reasoning
        rec = base_record.copy()
        rec["conversations"] = [
            {"from": "human", "value": prompt_subclass},
            {"from": "gpt", "value": f"<answer>{subclass_ans}</answer>"}
        ]
        data_sub_nr.append(rec)

    print("Writing files...")
    
    def write_split(data, filename):
        if args.limit > 0:
            split_idx = int(len(data) * 0.8)
            train_data = data[:split_idx]
            test_data = data[split_idx:]
            test_file = filename.replace("gem_train", "gem_test")
            with open(filename, 'w') as f: json.dump(train_data, f, indent=2)
            with open(test_file, 'w') as f: json.dump(test_data, f, indent=2)
        else:
            with open(filename, 'w') as f: json.dump(data, f, indent=2)

    write_split(data_super_r, out_super_r)
    write_split(data_super_nr, out_super_nr)
    write_split(data_sub_r, out_sub_r)
    write_split(data_sub_nr, out_sub_nr)
    
    print("Done generating GEM SFT datasets.")

if __name__ == "__main__":
    generate_gem_datasets()
