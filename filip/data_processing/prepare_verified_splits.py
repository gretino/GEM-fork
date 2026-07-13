import os
import json
import pandas as pd
import numpy as np

def setup_symlink(target_dir, link_name):
    if os.path.islink(link_name):
        os.unlink(link_name)
    elif os.path.exists(link_name):
        # If it's a directory but not a link, don't delete it unless it's empty
        print(f"Warning: {link_name} exists and is not a symlink. Skipping symlink creation.")
        return
    print(f"Creating symlink: {link_name} -> {target_dir}")
    os.symlink(target_dir, link_name)

def process_ptbxl_task(task_name, classes):
    print(f"Processing PTB-XL task: {task_name}...")
    verified_dir = f"data/verified_splits/ptbxl/{task_name}"
    output_dir = f"data/ptbxl_{task_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Symlink images
    setup_symlink("/home/qfbqt/8TB/datasets/ptb-xl-gen/", os.path.join(output_dir, "images"))
    
    diagnoses_dict = {}
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(verified_dir, f"ptbxl_{task_name}_{split}.csv")
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} does not exist.")
            continue
            
        df = pd.read_csv(csv_path)
        records_list = []
        for idx, row in df.iterrows():
            study_id = os.path.basename(row['filename_hr'])
            pat_id = row['patient_id']
            subject_id = str(int(pat_id)) if pd.notna(pat_id) else study_id
            
            # Record dictionary
            records_list.append({
                "subject_id": subject_id,
                "study_id": study_id,
                "file_name": study_id,
                "ecg_time": "",
                "path": f"images/{study_id}-0.png",
                "report_text": ""
            })
            
            # Diagnosis labels
            diag_dict = {}
            for cls in classes:
                diag_dict[cls] = float(row[cls]) if cls in row else 0.0
            diagnoses_dict[study_id] = diag_dict
            
        out_path = os.path.join(output_dir, f"{split}_records.json")
        with open(out_path, "w") as f:
            json.dump(records_list, f, indent=4)
        print(f"  Saved {len(records_list)} records to {out_path}")
        
    diag_path = os.path.join(output_dir, "diagnoses.json")
    with open(diag_path, "w") as f:
        json.dump(diagnoses_dict, f, indent=4)
    print(f"  Saved diagnoses to {diag_path}")

def process_cpsc():
    print("Processing CPSC 2018 (ICBEB)...")
    verified_dir = "data/verified_splits/icbeb"
    output_dir = "data/cpsc2018_verified"
    os.makedirs(output_dir, exist_ok=True)
    
    # Symlink images
    setup_symlink("/home/qfbqt/repo/GEM-fork/data/cpsc2018/images", os.path.join(output_dir, "images"))
    
    # Direct class names in the verified split
    classes = ['AFIB', 'VPC', 'NORM', '1AVB', 'CRBBB', 'STE', 'PAC', 'CLBBB', 'STD']
    
    diagnoses_dict = {}
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(verified_dir, f"icbeb_{split}.csv")
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} does not exist.")
            continue
            
        df = pd.read_csv(csv_path)
        records_list = []
        for idx, row in df.iterrows():
            study_id = str(row['filename'])
            pat_id = row['patient_id']
            subject_id = str(int(pat_id)) if pd.notna(pat_id) else study_id
            
            records_list.append({
                "subject_id": subject_id,
                "study_id": study_id,
                "file_name": study_id,
                "ecg_time": "",
                "path": f"images/{study_id}-0.png",
                "report_text": ""
            })
            
            diag_dict = {}
            for cls in classes:
                diag_dict[cls] = float(row[cls]) if cls in row else 0.0
            diagnoses_dict[study_id] = diag_dict
            
        out_path = os.path.join(output_dir, f"{split}_records.json")
        with open(out_path, "w") as f:
            json.dump(records_list, f, indent=4)
        print(f"  Saved {len(records_list)} records to {out_path}")
        
    diag_path = os.path.join(output_dir, "diagnoses.json")
    with open(diag_path, "w") as f:
        json.dump(diagnoses_dict, f, indent=4)
    print(f"  Saved diagnoses to {diag_path}")

def process_csn():
    print("Processing CSN (Chapman)...")
    verified_dir = "data/verified_splits/chapman"
    output_dir = "data/csn_verified"
    os.makedirs(output_dir, exist_ok=True)
    
    # Symlink images
    setup_symlink("/home/qfbqt/8TB/datasets/csn_new/", os.path.join(output_dir, "images"))
    
    # 38 direct classes
    classes = [
        'AQW', 'UW', 'SR', 'WPW', '2AVB', 'AT', 'VB', 'ARS', 'STTC', 'SA', 'STE',
        'VPB', 'TWO', 'STTU', 'ALS', 'APB', '2AVB1', 'PRIE', 'CCR', 'CR', 'AF',
        'AVB', 'QTIE', 'LBBB', 'VEB', 'SVT', 'RBBB', '1AVB', 'STDD', 'MI', 'AFIB',
        'TWC', 'PWC', 'ERV', 'RVH', 'LVH', 'ST', 'JEB'
    ]
    
    diagnoses_dict = {}
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(verified_dir, f"chapman_{split}.csv")
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} does not exist.")
            continue
            
        df = pd.read_csv(csv_path)
        records_list = []
        for idx, row in df.iterrows():
            ecg_id = os.path.basename(row['ecg_path']).split('.')[0]
            
            records_list.append({
                "subject_id": ecg_id,
                "study_id": ecg_id,
                "file_name": ecg_id,
                "ecg_time": "",
                "path": f"images/{ecg_id}-0.png",
                "report_text": ""
            })
            
            diag_dict = {}
            for cls in classes:
                diag_dict[cls] = float(row[cls]) if cls in row else 0.0
            diagnoses_dict[ecg_id] = diag_dict
            
        out_path = os.path.join(output_dir, f"{split}_records.json")
        with open(out_path, "w") as f:
            json.dump(records_list, f, indent=4)
        print(f"  Saved {len(records_list)} records to {out_path}")
        
    diag_path = os.path.join(output_dir, "diagnoses.json")
    with open(diag_path, "w") as f:
        json.dump(diagnoses_dict, f, indent=4)
    print(f"  Saved diagnoses to {diag_path}")

def main():
    # Classes lists matching the verified CSVs
    super_classes = ["NORM", "MI", "HYP", "CD", "STTC"] # Keep PTBXL vocab order
    
    sub_classes = [
        'AMI', 'LAFB/LPFB', 'LVH', 'STTC', 'IMI', 'SEHYP', 'CRBBB', 'WPW', 'LAO/LAE',
        'NORM', 'ISC', 'AVB', 'RAO/RAE', 'LMI', 'ISCI', 'ISCA', 'NST', 'CLBBB',
        'ILBBB', 'IRBBB', 'PMI', 'RVH', 'IVCD'
    ]
    
    rhythm_classes = [
        'SVARR', 'BIGU', 'STACH', 'SARRH', 'SBRAD', 'TRIGU', 'AFIB', 'SR', 'SVTAC',
        'PSVT', 'AFLT', 'PACE'
    ]
    
    form_classes = [
        'PAC', 'DIG', 'HVOLT', 'STD', 'LPR', 'QWAVE', 'VCLVH', 'NDT', 'TAB', 'LOWT',
        'ABQRS', 'LNGQT', 'PRC(S)', 'NT', 'STE', 'NST', 'PVC', 'INVT', 'LVOLT'
    ]
    
    # Process PTBXL tasks
    process_ptbxl_task("super_class", super_classes)
    process_ptbxl_task("sub_class", sub_classes)
    process_ptbxl_task("rhythm", rhythm_classes)
    process_ptbxl_task("form", form_classes)
    
    # Process CPSC
    process_cpsc()
    
    # Process CSN
    process_csn()
    
    print("All verified splits preprocessed successfully!")

if __name__ == "__main__":
    main()
