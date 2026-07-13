import os
import json
import ast
import argparse
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Prepare PTB-XL splits by fold.")
    parser.add_argument("--align_test_with_bench", type=str, default="true",
                        help="Align test split with ptb-test.json (true/false)")
    parser.add_argument("--bench_test_path", type=str, default="data/ecg_bench/ptb-test.json",
                        help="Path to ptb-test.json benchmark file")
    parser.add_argument("--output_dir", type=str, default="data/ptb-xl-folds/",
                        help="Output directory for split JSON files")
    args = parser.parse_args()

    align_test = args.align_test_with_bench.lower() == "true"
    bench_path = args.bench_test_path
    output_dir = args.output_dir

    print(f"Align test with benchmark: {align_test}")
    if align_test:
        print(f"Benchmark test path: {bench_path}")
    print(f"Output directory: {output_dir}")

    db_path = '/home/qfbqt/datasets/ptb_xl_1.0.3/ptbxl_database_translated.csv'
    db_orig_path = '/home/qfbqt/datasets/ptb_xl_1.0.3/ptbxl_database.csv'
    scp_path = '/home/qfbqt/datasets/ptb_xl_1.0.3/scp_statements.csv'
    gen_dir = '/home/qfbqt/8TB/datasets/ptb-xl-gen/'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Symlink images folder
    images_symlink = os.path.join(output_dir, 'images')
    if not os.path.exists(images_symlink):
        print(f"Creating symlink: {images_symlink} -> {gen_dir}")
        os.symlink(gen_dir, images_symlink)
        
    df_db = pd.read_csv(db_path)
    df_db_orig = pd.read_csv(db_orig_path)
    df_scp = pd.read_csv(scp_path, index_col=0)
    
    # Create mapping from SCP code to Superclass
    scp_to_superclass = {}
    for scp, row in df_scp.iterrows():
        if row['diagnostic'] == 1.0 and pd.notna(row['diagnostic_class']):
            scp_to_superclass[scp] = row['diagnostic_class']
            
    PTBXL_SUPERCLASSES = ["NORM", "MI", "HYP", "CD", "STTC"]
    
    # Load benchmark IDs if needed
    bench_ids = set()
    if align_test:
        if os.path.exists(bench_path):
            with open(bench_path, "r") as f:
                bench_data = json.load(f)
            for item in bench_data:
                item_id_str = item["id"]
                digits = "".join([c for c in item_id_str if c.isdigit()])
                if digits:
                    bench_ids.add(int(digits))
            print(f"Loaded {len(bench_ids)} benchmark IDs from {bench_path}")
        else:
            print(f"Warning: Benchmark file not found at {bench_path}. Disabling alignment.")
            align_test = False

    print("Scanning available images in ptb-xl-gen...")
    matched_records = []
    processed_ids = set()
    
    for idx, row in tqdm(df_db.iterrows(), total=len(df_db)):
        ecg_id = int(row['ecg_id'])
        basename = os.path.basename(row['filename_hr'])
        img_name = f"{basename}-0.png"
        
        # Check if the image exists in ptb-xl-gen
        if os.path.exists(os.path.join(gen_dir, img_name)):
            fold = int(row['strat_fold'])
            assigned_split = None
            
            if align_test:
                if ecg_id in bench_ids:
                    assigned_split = 'test'
                elif 1 <= fold <= 8:
                    assigned_split = 'train'
                elif fold == 9:
                    assigned_split = 'val'
            else:
                if 1 <= fold <= 8:
                    assigned_split = 'train'
                elif fold == 9:
                    assigned_split = 'val'
                elif fold == 10:
                    assigned_split = 'test'
            
            if assigned_split is not None:
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
                processed_ids.add(ecg_id)

    # 4. Handle fallback for benchmark IDs not present in translated CSV
    if align_test:
        missing_ids_in_translated = bench_ids - processed_ids
        if missing_ids_in_translated:
            print(f"IDs in benchmark but missing from translated DB: {missing_ids_in_translated}")
            # Try to fetch from original untranslated DB
            df_fallback = df_db_orig[df_db_orig['ecg_id'].isin(missing_ids_in_translated)]
            for idx, row in df_fallback.iterrows():
                ecg_id = int(row['ecg_id'])
                basename = os.path.basename(row['filename_hr'])
                img_name = f"{basename}-0.png"
                if os.path.exists(os.path.join(gen_dir, img_name)):
                    matched_records.append({
                        "ecg_id": ecg_id,
                        "study_id": basename,
                        "patient_id": int(row['patient_id']) if not pd.isna(row['patient_id']) else ecg_id,
                        "ecg_time": str(row['recording_date']) if not pd.isna(row['recording_date']) else "",
                        "path": str(row['filename_hr']),
                        "report_text": str(row['report']) if not pd.isna(row['report']) else "",
                        "scp_codes": row['scp_codes'],
                        "assigned_split": 'test'
                    })
                    processed_ids.add(ecg_id)
                    print(f"Successfully added fallback record for missing ID: {ecg_id}")

    print(f"Matched {len(matched_records)} records with actual generated images.")
    
    splits = {
        'train': [r for r in matched_records if r['assigned_split'] == 'train'],
        'val': [r for r in matched_records if r['assigned_split'] == 'val'],
        'test': [r for r in matched_records if r['assigned_split'] == 'test']
    }
    
    print(f"Split counts: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    # 5. Write split json files
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
        
    # 6. Build diagnoses.json
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
    print("PTB-XL fold preprocessing completed successfully!")

if __name__ == "__main__":
    main()
