import os
import json
import pandas as pd

def main():
    print("Preparing CPSC 2018 dataset...")
    
    raw_dir = '/home/qfbqt/8TB/datasets/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018'
    output_dir = '/home/qfbqt/repo/GEM-fork/data/cpsc2018/'
    images_dir = os.path.join(output_dir, 'images')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 9 CPSC 2018 classes
    CPSC_CLASSES = ['NORM', 'AFIB', '1AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    
    SNOMED_TO_CLASS = {
        '426783006': 'NORM',
        '164889003': 'AFIB',
        '270492004': '1AVB',
        '164909002': 'LBBB',
        '59118001': 'RBBB',
        '284470004': 'PAC',
        '164884008': 'PVC',
        '429622005': 'STD',
        '164931005': 'STE'
    }
    
    records_dx = {}
    records_list = []
    
    print("Scanning header files for SNOMED codes...")
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.endswith('.hea'):
                path = os.path.join(root, file)
                rec_id = file.split('.')[0]
                
                # Check if generated image exists
                # The generator saves files as <rec_id>-0.png or <rec_id>.png
                # We check for either to be safe
                img_name = f"{rec_id}-0.png"
                img_path = os.path.join(images_dir, img_name)
                if not os.path.exists(img_path):
                    img_name = f"{rec_id}.png"
                    img_path = os.path.join(images_dir, img_name)
                    
                if not os.path.exists(img_path):
                    # We still include the records even if they aren't generated yet
                    # but we output a warning. During training/eval, missing images
                    # will fallback to dummy white images, but we want to know.
                    img_name = f"{rec_id}-0.png"
                    
                try:
                    with open(path, 'r') as f:
                        for line in f:
                            if line.startswith('# Dx:'):
                                dx_line = line.strip().split(': ')[1]
                                codes = [c.strip() for c in dx_line.split(',')]
                                records_dx[rec_id] = codes
                                
                                records_list.append({
                                    "subject_id": rec_id,
                                    "study_id": rec_id,
                                    "file_name": rec_id,
                                    "ecg_time": "",
                                    "path": f"images/{img_name}",
                                    "report_text": ""
                                })
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    
    print(f"Found {len(records_list)} total records.")
    
    # Process diagnoses mapping
    diagnoses_dict = {}
    for rec_id, codes in records_dx.items():
        diag_dict = {c: 0.0 for c in CPSC_CLASSES}
        has_abnormal = False
        for code in codes:
            if code in SNOMED_TO_CLASS:
                cls = SNOMED_TO_CLASS[code]
                if cls != 'NORM':
                    diag_dict[cls] = 1.0
                    has_abnormal = True
                    
        # Apply normal class logic
        if '426783006' in codes and not has_abnormal:
            diag_dict['NORM'] = 1.0
            
        diagnoses_dict[rec_id] = diag_dict
        
    # Split 80% train, 10% val, 10% test
    df_records = pd.DataFrame(records_list)
    shuffled_df = df_records.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    n = len(shuffled_df)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    train_df = shuffled_df.iloc[:n_train]
    val_df = shuffled_df.iloc[n_train:n_train + n_val]
    test_df = shuffled_df.iloc[n_train + n_val:]
    
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    for s, split_df in splits.items():
        out_path = os.path.join(output_dir, f"{s}_records.json")
        records = split_df.to_dict(orient="records")
        with open(out_path, 'w') as f:
            json.dump(records, f, indent=4)
        print(f"Saved {len(records)} records to {out_path}")
        
    diag_path = os.path.join(output_dir, "diagnoses.json")
    with open(diag_path, 'w') as f:
        json.dump(diagnoses_dict, f, indent=4)
    print(f"Saved diagnoses to {diag_path}")
    print("CPSC 2018 dataset preparation completed successfully!")

if __name__ == "__main__":
    main()
