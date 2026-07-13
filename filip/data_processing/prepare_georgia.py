import os
import json
import pandas as pd

def main():
    print("Preparing Georgia dataset...")
    
    raw_dir = '/home/qfbqt/8TB/datasets/physionet.org/files/challenge-2020/1.0.2/training/georgia'
    output_dir = '/home/qfbqt/repo/GEM-fork/data/georgia/'
    images_dir = os.path.join(output_dir, 'images')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 6 classes (including rhythm-only abnormal class)
    GEORGIA_CLASSES = ['NORM', 'MI', 'HYP', 'CD', 'STTC', 'RHYTHM']
    
    normal_codes = {
        '426783006', '426177001', '427084000', '427393009',
        '195126007', '164921003', '426664006', '426627000',
        '413444003', '63593006'
    }
    
    cd_codes = {
        '270492004', '59118001', '164909002', '698252002',
        '233917008', '195042002', '27885002', '426434006',
        '445118002', '6374002'
    }
    
    sttc_codes = {
        '428750005', '164934002', '164930006', '59931005',
        '164931005', '429622005', '713426002', '253352002',
        '266249003', '713427006', '445211001', '55930002'
    }
    
    hyp_codes = {
        '164873001', '89792004'
    }
    
    mi_codes = {
        '164865005', '164917005'
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
                img_name = f"{rec_id}-0.png"
                img_path = os.path.join(images_dir, img_name)
                if not os.path.exists(img_path):
                    img_name = f"{rec_id}.png"
                    img_path = os.path.join(images_dir, img_name)
                    
                if not os.path.exists(img_path):
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
        diag_dict = {c: 0.0 for c in GEORGIA_CLASSES}
        superclasses = set()
        
        # Check conduction block
        if any(c in cd_codes for c in codes):
            superclasses.add('CD')
        # Check STTC
        if any(c in sttc_codes for c in codes):
            superclasses.add('STTC')
        # Check hypertrophy
        if any(c in hyp_codes for c in codes):
            superclasses.add('HYP')
        # Check MI
        if any(c in mi_codes for c in codes):
            superclasses.add('MI')
            
        # Check normal
        is_normal = False
        if any(c in normal_codes for c in codes):
            if not any(c not in normal_codes for c in codes):
                is_normal = True
                
        if is_normal:
            superclasses.add('NORM')
            
        # Check if RHYTHM
        if not superclasses:
            superclasses.add('RHYTHM')
            
        for c in superclasses:
            diag_dict[c] = 1.0
            
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
    print("Georgia dataset preparation completed successfully!")

if __name__ == "__main__":
    main()
