# filip/data/vocab.py

MIMIC_FEATURES = [
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

PTBXL_SUPERCLASSES = [
    "NORM",
    "MI",
    "HYP",
    "CD",
    "STTC"
]

def get_feature_vocab():
    return {feat: i for i, feat in enumerate(MIMIC_FEATURES)}, MIMIC_FEATURES

def get_diagnosis_vocab():
    return {diag: i for i, diag in enumerate(PTBXL_SUPERCLASSES)}, PTBXL_SUPERCLASSES
