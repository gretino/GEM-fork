# FILIP Experiment Processed Dataset Structure

This document outlines the directory structure, file schemas, and output parameters of the processed MIMIC-IV-ECG dataset for the FILIP model training experiment.

---

## Directory Layout

All processed outputs are located at `/home/qfbqt/8TB/datasets/mimic-iv-ecg/` (symlinked from `data/mimic-iv-ecg/` in the repository root).

```
data/mimic-iv-ecg/  --> symlink to /home/qfbqt/8TB/datasets/mimic-iv-ecg/
├── all_records.csv
├── train_records.json
├── val_records.json
├── test_records.json
├── features.json
├── diagnoses.json
└── images/
    ├── 40002897-0.png
    ├── 40002897.hea
    └── ...
```

---

## File Schema Specifications

### 1. Partition Splits (`train_records.json`, `val_records.json`, `test_records.json`)

These files contain lists of dictionaries representing the selected studies in each split (split ratio 8:1:1, total 1,000 records).

**Keys:**
* `subject_id` (integer): Patient identifier.
* `study_id` (integer): ECG study identifier.
* `file_name` (integer): Base file name (identical to `study_id`).
* `ecg_time` (string): Date and time of the ECG.
* `path` (string): Relative path from the dataset root (e.g., `files/p1176/p11767260/s44574451/44574451`).
* `report_text` (string): Normalized, whitespace-cleaned concatenated cardiology report.

**Example:**
```json
[
    {
        "subject_id": 11767260,
        "study_id": 44574451,
        "file_name": 44574451,
        "ecg_time": "2166-09-28 19:37:00",
        "path": "files/p1176/p11767260/s44574451/44574451",
        "report_text": "sinus rhythm with bigeminal pvcs prolonged qt interval..."
    }
]
```

---

### 2. Clinical Features (`features.json`)

Contains a JSON object mapping each target `study_id` to its 20 binary feature labels. A value of `1` indicates presence; `0` indicates absence.

**Target Features:**
1. `sinus_rhythm`
2. `atrial_fibrillation`
3. `tachycardia`
4. `bradycardia`
5. `st_elevation`
6. `st_depression`
7. `st_elevation_in_inferior_leads`
8. `st_elevation_in_anterior_leads`
9. `t_wave_inversion`
10. `pathological_q_wave`
11. `wide_qrs`
12. `left_axis_deviation`
13. `right_axis_deviation`
14. `left_bundle_branch_block`
15. `right_bundle_branch_block`
16. `first_degree_av_block`
17. `prolonged_qt`
18. `left_ventricular_hypertrophy`
19. `low_voltage_qrs`
20. `poor_r_wave_progression`

**Example:**
```json
{
    "44574451": {
        "sinus_rhythm": 1,
        "atrial_fibrillation": 0,
        "tachycardia": 0,
        "bradycardia": 0,
        "st_elevation": 0,
        "st_depression": 0,
        "st_elevation_in_inferior_leads": 0,
        "st_elevation_in_anterior_leads": 0,
        "t_wave_inversion": 1,
        "pathological_q_wave": 0,
        "wide_qrs": 0,
        "left_axis_deviation": 0,
        "right_axis_deviation": 0,
        "left_bundle_branch_block": 0,
        "right_bundle_branch_block": 0,
        "first_degree_av_block": 0,
        "prolonged_qt": 1,
        "left_ventricular_hypertrophy": 0,
        "low_voltage_qrs": 0,
        "poor_r_wave_progression": 0
    }
}
```

---

### 3. Diagnoses and Machine Measurements (`diagnoses.json`)

Maps each target `study_id` to its direct and calculated machine measurements, axis angles, metadata, and normalized report text.

**Fields:**
* **Metadata**: `subject_id`, `study_id`, `cart_id`, `ecg_time`, `bandwidth`, `filtering`
* **Direct Measurements**: `rr_interval` (ms), `p_onset` (ms), `p_end` (ms), `qrs_onset` (ms), `qrs_end` (ms), `t_end` (ms)
* **Direct Axis Measurements**: `p_axis` (degrees), `qrs_axis` (degrees), `t_axis` (degrees)
* **Calculated Physiological Intervals**:
  * `heart_rate` = 60000 / rr_interval (beats per minute)
  * `pr_interval` = qrs_onset - p_onset (ms)
  * `qrs_duration` = qrs_end - qrs_onset (ms)
  * `qt_interval` = t_end - qrs_onset (ms)
  * `qtc_interval` = qt_interval / sqrt(rr_interval / 1000.0) (ms, corrected via Bazett's formula)
* **Report**: `report_text` (concatenated machine-generated cardiology report string)

**Example:**
```json
{
    "44574451": {
        "subject_id": 11767260,
        "study_id": 44574451,
        "cart_id": 6924910,
        "ecg_time": "2166-09-28 19:37:00",
        "bandwidth": "0.005-150 Hz",
        "filtering": "60 Hz notch Baseline filter",
        "rr_interval": 1120,
        "p_onset": 38,
        "p_end": 178,
        "qrs_onset": 194,
        "qrs_end": 290,
        "t_end": 740,
        "p_axis": 55,
        "qrs_axis": -15,
        "t_axis": 8,
        "heart_rate": 53.57142857142857,
        "pr_interval": 156,
        "qrs_duration": 96,
        "qt_interval": 546,
        "qtc_interval": 515.9263513330149,
        "report_text": "sinus rhythm with bigeminal pvcs prolonged qt interval..."
    }
}
```

---

### 4. Rendered ECG Images (`images/` directory)

Contains generated PNG images of the 12-lead ECG waveforms in a 4x3 layout (4 columns, 3 rows of lead signals).
* Filename convention: `{study_id}-0.png`.
* Associated header files (`{study_id}.hea`) are also stored alongside the images.
