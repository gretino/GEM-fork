import os
import argparse
import json
import pandas as pd
import numpy as np

def extract_measurements(dataset_dir, output_dir):
    # Load all target records from splits
    all_target_records = []
    splits = ["train", "val", "test"]
    for split in splits:
        split_path = os.path.join(output_dir, f"{split}_records.json")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found at {split_path}")
        with open(split_path, "r") as f:
            all_target_records.extend(json.load(f))

    # Get set of targeted study_ids
    target_study_ids = {int(r["study_id"]) for r in all_target_records}
    print(f"Total target records to extract measurements for: {len(target_study_ids)}")

    # Load machine_measurements.csv
    measurements_csv_path = os.path.join(dataset_dir, "machine_measurements.csv")
    if not os.path.exists(measurements_csv_path):
        raise FileNotFoundError(f"machine_measurements.csv not found at {measurements_csv_path}")

    print(f"Loading measurements from {measurements_csv_path}...")
    # Read the full csv
    df = pd.read_csv(measurements_csv_path)

    # Filter to only targeted study_ids
    df_filtered = df[df["study_id"].isin(target_study_ids)].copy()
    print(f"Filtered to {len(df_filtered)} matching rows in measurements database.")

    # Helper function to convert numpy type to native python type or None
    def clean_val(val, val_type=float):
        if pd.isna(val):
            return None
        try:
            return val_type(val)
        except:
            return str(val)

    diagnoses = {}
    for _, row in df_filtered.iterrows():
        study_id = str(row["study_id"])
        
        # Extract direct fields
        rr = clean_val(row.get("rr_interval"), int)
        p_onset = clean_val(row.get("p_onset"), int)
        p_end = clean_val(row.get("p_end"), int)
        qrs_onset = clean_val(row.get("qrs_onset"), int)
        qrs_end = clean_val(row.get("qrs_end"), int)
        t_end = clean_val(row.get("t_end"), int)
        
        p_axis = clean_val(row.get("p_axis"), int)
        qrs_axis = clean_val(row.get("qrs_axis"), int)
        t_axis = clean_val(row.get("t_axis"), int)

        # Compute intervals
        heart_rate = None
        if rr is not None and rr > 0:
            heart_rate = 60000.0 / rr

        pr_interval = None
        if qrs_onset is not None and p_onset is not None:
            pr_interval = qrs_onset - p_onset

        qrs_duration = None
        if qrs_end is not None and qrs_onset is not None:
            qrs_duration = qrs_end - qrs_onset

        qt_interval = None
        if t_end is not None and qrs_onset is not None:
            qt_interval = t_end - qrs_onset

        qtc_interval = None
        if qt_interval is not None and rr is not None and rr > 0:
            qtc_interval = qt_interval / np.sqrt(rr / 1000.0)

        # Concatenate report columns
        report_cols = [f"report_{j}" for j in range(18)]
        report_parts = []
        for col in report_cols:
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                report_parts.append(val.strip())
        report_text = " ".join(report_parts)

        # Clean report text
        report_text = (report_text.lower()
                       .replace("---", "")
                       .replace("***", "")
                       .replace(" - age undetermined", ""))
        # Normalize spaces
        report_text = " ".join(report_text.split())

        diagnoses[study_id] = {
            "subject_id": clean_val(row.get("subject_id"), int),
            "study_id": int(study_id),
            "cart_id": clean_val(row.get("cart_id"), int),
            "ecg_time": clean_val(row.get("ecg_time"), str),
            "bandwidth": clean_val(row.get("bandwidth"), str),
            "filtering": clean_val(row.get("filtering"), str),
            "rr_interval": rr,
            "p_onset": p_onset,
            "p_end": p_end,
            "qrs_onset": qrs_onset,
            "qrs_end": qrs_end,
            "t_end": t_end,
            "p_axis": p_axis,
            "qrs_axis": qrs_axis,
            "t_axis": t_axis,
            "heart_rate": heart_rate,
            "pr_interval": pr_interval,
            "qrs_duration": qrs_duration,
            "qt_interval": qt_interval,
            "qtc_interval": qtc_interval,
            "report_text": report_text
        }

    output_path = os.path.join(output_dir, "diagnoses.json")
    with open(output_path, "w") as f:
        json.dump(diagnoses, f, indent=4)

    print(f"Successfully extracted and saved measurements for {len(diagnoses)} records to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract machine measurements and diagnoses.")
    parser.add_argument("--dataset_dir", type=str, default="/home/qfbqt/8TB/blmcg/datasets/physionet.org.5/files/mimic-iv-ecg/1.0/",
                        help="Path to the raw MIMIC-IV-ECG dataset directory.")
    parser.add_argument("--output_dir", type=str, default="/home/qfbqt/repo/GEM-fork/data/mimic-iv-ecg/",
                        help="Path to output directory containing record splits.")
    
    args = parser.parse_args()
    extract_measurements(args.dataset_dir, args.output_dir)
