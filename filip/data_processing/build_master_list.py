import os
import argparse
import pandas as pd
import numpy as np

def build_master_list(dataset_dirs, output_dir):
    all_valid_dfs = []
    
    for dataset_dir in dataset_dirs:
        print(f"\nProcessing directory: {dataset_dir}")
        print(f"Reading record_list.csv from {dataset_dir}...")
        record_list_path = os.path.join(dataset_dir, "record_list.csv")
        if not os.path.exists(record_list_path):
            raise FileNotFoundError(f"record_list.csv not found at {record_list_path}")
        record_list = pd.read_csv(record_list_path)

        print(f"Reading machine_measurements.csv from {dataset_dir}...")
        measurements_path = os.path.join(dataset_dir, "machine_measurements.csv")
        if not os.path.exists(measurements_path):
            raise FileNotFoundError(f"machine_measurements.csv not found at {measurements_path}")
        
        # Optimize memory usage by loading only required columns
        cols_to_use = ["study_id"] + [f"report_{j}" for j in range(18)]
        database = pd.read_csv(measurements_path, usecols=cols_to_use)

        bad_reports = [
            "--- Warning: Data quality may affect interpretation ---",
            "--- Recording unsuitable for analysis - please repeat ---",
            "Analysis error",
            "conduction defect",
            "*** report made without knowing patient's sex ***",
            "--- Suspect arm lead reversal",
            "--- Possible measurement error ---",
            "--- Pediatric criteria used ---",
            "--- Suspect limb lead reversal",
            "-------------------- Pediatric ECG interpretation --------------------",
            "Lead(s) unsuitable for analysis:",
            "LEAD(S) UNSUITABLE FOR ANALYSIS:",
            "PACER DETECTION SUSPENDED DUE TO EXTERNAL NOISE-REVIEW ADVISED",
            "Pacer detection suspended due to external noise-REVIEW ADVISED"
        ]

        print("Merging record list with measurements...")
        # Deduplicate database by study_id in case of duplicates
        database = database.drop_duplicates(subset=["study_id"])
        merged = record_list.merge(database, on="study_id", how="inner")

        print("Filtering bad reports and concatenating text...")
        report_cols = [f"report_{j}" for j in range(18)]
        
        # Function to clean individual cells in a vectorized-friendly way
        def clean_cell(val):
            if not isinstance(val, str):
                return ""
            for bad in bad_reports:
                if bad in val:
                    return ""
            return val

        # Clean report columns
        for col in report_cols:
            merged[col] = merged[col].apply(clean_cell)

        # Concatenate clean report cells
        # We join with space and strip extra spaces
        merged["report_text"] = merged[report_cols].agg(lambda row: " ".join([val for val in row if val != ""]), axis=1)

        # Filter out records that ended up with empty reports
        valid_df = merged[merged["report_text"].str.strip() != ""].copy()

        # Basic cleanups on report_text
        valid_df["report_text"] = (valid_df["report_text"].str.lower()
                                   .str.replace("---", "", regex=False)
                                   .str.replace("***", "", regex=False)
                                   .str.replace(" - age undetermined", "", regex=False))

        # Keep only the columns we need
        final_cols = ["subject_id", "study_id", "file_name", "ecg_time", "path", "report_text"]
        valid_df = valid_df[final_cols]
        
        # Add source directory
        valid_df["source_dir"] = dataset_dir
        all_valid_dfs.append(valid_df)

    print("\nConcatenating all datasets...")
    final_df = pd.concat(all_valid_dfs, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["study_id"])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_records.csv")
    final_df.to_csv(output_path, index=False)
    print(f"Successfully created master list with {len(final_df)} valid records at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create clean master list of all MIMIC-IV-ECG records.")
    parser.add_argument("--dataset_dirs", type=str, nargs="+",
                        default=[
                            "/home/qfbqt/8TB/blmcg/datasets/physionet.org.5/files/mimic-iv-ecg/1.0/",
                            "/home/qfbqt/8TB/blmcg/datasets/physionet.org/files/mimic-iv-ecg/1.0/"
                        ],
                        help="Paths to raw MIMIC-IV-ECG dataset directories.")
    parser.add_argument("--output_dir", type=str, default="/home/qfbqt/repo/GEM-fork/data/mimic-iv-ecg/",
                        help="Path to output directory for saving all_records.csv.")
    
    args = parser.parse_args()
    build_master_list(args.dataset_dirs, args.output_dir)
