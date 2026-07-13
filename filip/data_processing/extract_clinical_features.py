import os
import sys
import argparse
import json
import pandas as pd

# Add the challenge labeler directory to python path
LABELER_DIR = "/home/qfbqt/8TB/blmcg/project/ecg-ksteer/ecg-fm/labeler"
if LABELER_DIR not in sys.path:
    sys.path.append(LABELER_DIR)

from preprocess import preprocess_texts
from pattern_labeler import PatternLabeler, PatternLabelerConfig

def extract_features(output_dir):
    # Load the targeted split files
    all_target_records = []
    splits = ["train", "val", "test"]
    for split in splits:
        split_path = os.path.join(output_dir, f"{split}_records.json")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found at {split_path}")
        with open(split_path, "r") as f:
            all_target_records.extend(json.load(f))

    # Convert to DataFrame
    df = pd.DataFrame(all_target_records)
    print(f"Total target records to extract features for: {len(df)}")

    # Set up texts series for the PatternLabeler
    # Use study_id as index to easily map labels back
    texts_series = pd.Series(df["report_text"].values, index=df["study_id"].astype(str))
    
    # CRITICAL: Clear the index name to ensure the MultiIndex output in PatternLabeler
    # uses default naming (level_0, level_1, etc.) under pandas 2.x.
    texts_series.index.name = None

    print("Pre-processing reports text...")
    preprocessed_texts = preprocess_texts(texts_series)
    
    # Double check index name is cleared after preprocessing
    preprocessed_texts.index.name = None

    print("Loading PatternLabeler configuration...")
    labeler_config_dir = "/home/qfbqt/8TB/blmcg/project/ecg-ksteer/ecg-fm/data/mimic_iv_ecg-org/labeler/"
    config = PatternLabelerConfig.from_json(labeler_config_dir, progress=True)
    labeler = PatternLabeler(config)

    print("Running PatternLabeler...")
    res = labeler(preprocessed_texts)
    labels_flat = res.labels_flat

    print("Mapping labels to the 20 target features...")
    # The index of labels_flat corresponds to the study_id (index of preprocessed_texts)
    labels_flat.index = labels_flat.index.astype(str)
    
    # The 20 target features
    target_features = [
        "sinus_rhythm",
        "atrial_fibrillation",
        "tachycardia",
        "bradycardia",
        "st_elevation",
        "st_depression",
        "st_elevation_in_inferior_leads",
        "st_elevation_in_anterior_leads",
        "t_wave_inversion",
        "pathological_q_wave",
        "wide_qrs",
        "left_axis_deviation",
        "right_axis_deviation",
        "left_bundle_branch_block",
        "right_bundle_branch_block",
        "first_degree_av_block",
        "prolonged_qt",
        "left_ventricular_hypertrophy",
        "low_voltage_qrs",
        "poor_r_wave_progression"
    ]

    def map_labels(matched_names):
        feat_dict = {f: 0 for f in target_features}
        matched_lower = {n.lower() for n in matched_names}
        
        if any("sinus rhythm" in n for n in matched_lower):
            feat_dict["sinus_rhythm"] = 1
            
        if any("atrial fibrillation" in n for n in matched_lower):
            feat_dict["atrial_fibrillation"] = 1
            
        if any("tachycardia" in n for n in matched_lower):
            feat_dict["tachycardia"] = 1
            
        if any("bradycardia" in n for n in matched_lower):
            feat_dict["bradycardia"] = 1
            
        if any("st elevation" in n for n in matched_lower):
            feat_dict["st_elevation"] = 1
            
        if any("st depression" in n or "junctional depression" in n for n in matched_lower):
            feat_dict["st_depression"] = 1
            
        if any("st elevation" in n and "inferior" in n for n in matched_lower):
            feat_dict["st_elevation_in_inferior_leads"] = 1
            feat_dict["st_elevation"] = 1
            
        if any("st elevation" in n and "anterior" in n for n in matched_lower):
            feat_dict["st_elevation_in_anterior_leads"] = 1
            feat_dict["st_elevation"] = 1
            
        if any("t wave inversion" in n or "t-wave inversion" in n for n in matched_lower):
            feat_dict["t_wave_inversion"] = 1
            
        if any("pathological q wave" in n or "q wave abnormality" in n or "q-wave abnormality" in n for n in matched_lower):
            feat_dict["pathological_q_wave"] = 1
            
        if any("wide qrs" in n or "qrs prolongation" in n or "prolonged qrs" in n for n in matched_lower):
            feat_dict["wide_qrs"] = 1
            
        if any("left axis deviation" in n or "lad" == n for n in matched_lower):
            feat_dict["left_axis_deviation"] = 1
            
        if any("right axis deviation" in n or "rad" == n for n in matched_lower):
            feat_dict["right_axis_deviation"] = 1
            
        if any("left bundle branch block" in n or "lbbb" in n for n in matched_lower):
            feat_dict["left_bundle_branch_block"] = 1
            
        if any("right bundle branch block" in n or "rbbb" in n for n in matched_lower):
            feat_dict["right_bundle_branch_block"] = 1
            
        if any("first degree av block" in n or "1st degree av block" in n or "first degree atrioventricular block" in n or "1st degree atrioventricular block" in n or "borderline 1st degree a-v block" in n or "first degree a-v block" in n for n in matched_lower):
            feat_dict["first_degree_av_block"] = 1
            
        if any("prolonged qt" in n or "qt prolongation" in n or "long qt" in n or "qt interval prolongation" in n for n in matched_lower):
            feat_dict["prolonged_qt"] = 1
            
        if any("left ventricular hypertrophy" in n or "lvh" in n for n in matched_lower):
            feat_dict["left_ventricular_hypertrophy"] = 1
            
        if any("low qrs voltage" in n or "low voltage qrs" in n or "low qrs voltages" in n for n in matched_lower):
            feat_dict["low_voltage_qrs"] = 1
            
        if any("poor r wave progression" in n or "poor r-wave progression" in n or "abnormal r wave progression" in n or "abnormal r-wave progression" in n for n in matched_lower):
            feat_dict["poor_r_wave_progression"] = 1
            
        return feat_dict

    # Map features for each study_id
    output_features = {}
    
    # Iterate over all study_ids in the target df to ensure every study has a record in features.json
    for study_id in df["study_id"].astype(str):
        if study_id in labels_flat.index:
            matched_names = labels_flat.loc[study_id]
            if isinstance(matched_names, pd.DataFrame):
                matched_names = matched_names["name"].tolist()
            elif isinstance(matched_names, pd.Series):
                matched_names = [matched_names["name"]]
            else:
                matched_names = [matched_names]
            
            output_features[study_id] = map_labels(matched_names)
        else:
            # No labels matched, all features default to 0
            output_features[study_id] = {f: 0 for f in target_features}

    # Save to features.json
    features_json_path = os.path.join(output_dir, "features.json")
    with open(features_json_path, "w") as f:
        json.dump(output_features, f, indent=4)
        
    print(f"Successfully extracted and saved features for {len(output_features)} records to {features_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract clinical features using challenge PatternLabeler.")
    parser.add_argument("--output_dir", type=str, default="/home/qfbqt/repo/GEM-fork/data/mimic-iv-ecg/",
                        help="Path to directory containing record split JSON files.")
    
    args = parser.parse_args()
    extract_features(args.output_dir)
