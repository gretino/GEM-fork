import os
import json
import ast
import pandas as pd

def main():
    print("Preparing Chapman-Shaoxing (CSN) dataset...")
    
    db_path = '/home/qfbqt/8TB/datasets/csn/csn_database.csv'
    scp_path = '/home/qfbqt/datasets/ptb_xl_1.0.3/scp_statements.csv'
    images_dir = '/home/qfbqt/8TB/datasets/csn_new/'
    
    output_dir = '/home/qfbqt/repo/GEM-fork/data/csn/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Symlink images folder
    images_symlink = os.path.join(output_dir, 'images')
    if os.path.islink(images_symlink):
        os.unlink(images_symlink)
    if not os.path.exists(images_symlink):
        print(f"Creating symlink: {images_symlink} -> {images_dir}")
        os.symlink(images_dir, images_symlink)
        
    df_db = pd.read_csv(db_path)
    df_scp = pd.read_csv(scp_path, index_col=0)
    
    # Create mapping from SCP code to Superclass
    scp_to_superclass = {}
    for scp, row in df_scp.iterrows():
        if row['diagnostic'] == 1.0 and pd.notna(row['diagnostic_class']):
            scp_to_superclass[scp] = row['diagnostic_class']
            
    abnormal_keys = set(df_scp[df_scp['diagnostic'] == 1.0].index) | {'AFIB', 'AFLT', 'SARRH', 'PAC', 'PVC', 'SVTAC'}
    
    PTBXL_SUPERCLASSES = ["NORM", "MI", "HYP", "CD", "STTC", "RHYTHM"]
    
    diagnoses_dict = {}
    records_list = []
    
    print("Processing labels and metadata...")
    for idx, row in df_db.iterrows():
        ecg_id = str(row['ecg_id'])
        image_name = f"{ecg_id}-0.png"
        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            image_name = f"{ecg_id}.png"
            image_path = os.path.join(images_dir, image_name)
        
        # We only keep records for which the image file exists
        if os.path.exists(image_path):
            scp_dict = ast.literal_eval(row['scp_codes'])
            
            # Map to 5 superclasses
            diag_dict = {superclass: 0.0 for superclass in PTBXL_SUPERCLASSES}
            superclasses = set()
            for subclass in scp_dict.keys():
                if subclass in scp_to_superclass:
                    superclasses.add(scp_to_superclass[subclass])
            
            # Handle NORM fallback: SR present with no abnormal features
            if 'SR' in scp_dict:
                if not any(k in abnormal_keys for k in scp_dict.keys() if k != 'SR'):
                    superclasses.add('NORM')
                    
            if not superclasses:
                superclasses.add('RHYTHM')
                    
            for superclass in superclasses:
                diag_dict[superclass] = 1.0
                
            diagnoses_dict[ecg_id] = diag_dict
            
            records_list.append({
                "subject_id": ecg_id,
                "study_id": ecg_id,
                "file_name": ecg_id,
                "ecg_time": "",
                "path": f"images/{image_name}",
                "report_text": ""
            })
            
    print(f"Processed {len(records_list)} valid records.")
    
    # Shuffle and split 80% train, 10% val, 10% test
    # Use deterministic random state
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
    print("CSN dataset preparation completed successfully!")

if __name__ == "__main__":
    main()
