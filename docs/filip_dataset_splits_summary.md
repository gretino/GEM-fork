# FILIP Tasks Dataset Splits Summary

This document summarizes the sample distributions across the training, validation, and testing splits for the 6 downstream evaluation datasets in the FILIP experiment pipeline, as well as the Stage 1 pretraining dataset (MIMIC-IV-ECG).

---

## 1. Overview Table: 6 Downstream Evaluation Datasets

| Dataset / Task | Identifier | Data Path | Classes | Train Samples | Val Samples | Test Samples | Total Samples | Split Ratio (Tr / Val / Te) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PTB-XL Superclass** | `ptbxl_super` | `data/ptbxl_super_class/` | 5 | 17,084 | 2,146 | 2,158 | **21,388** | 79.9% / 10.0% / 10.1% |
| **PTB-XL Subclass** | `ptbxl_sub` | `data/ptbxl_sub_class/` | 23 | 17,084 | 2,146 | 2,158 | **21,388** | 79.9% / 10.0% / 10.1% |
| **PTB-XL Rhythm** | `ptbxl_rhythm` | `data/ptbxl_rhythm/` | 12 | 16,832 | 2,100 | 2,098 | **21,030** | 80.0% / 10.0% / 10.0% |
| **PTB-XL Form** | `ptbxl_form` | `data/ptbxl_form/` | 19 | 7,197 | 901 | 880 | **8,978** | 80.2% / 10.0% / 9.8% |
| **CPSC 2018** | `cpsc2018` | `data/cpsc2018_verified/` | 9 | 4,942 | 551 | 1,374 | **6,867** | 72.0% / 8.0% / 20.0% |
| **Chapman-Shaoxing-Ningbo** | `csn` | `data/csn_verified/` | 38 | 16,545 | 1,860 | 4,620 | **23,025** | 71.9% / 8.1% / 20.1% |
| **Total (All 6 Datasets)** | — | — | — | **79,684** | **9,704** | **13,288** | **102,676** | — |

---

## 2. Detailed Dataset Specifications

### 1. PTB-XL Superclass (`ptbxl_super`)
- **Data Root:** `data/ptbxl_super_class/`
- **Total Records:** 21,388 (Train: 17,084, Val: 2,146, Test: 2,158)
- **Number of Diagnostic Classes:** 5
- **Classes:**
  - `NORM` (Normal ECG)
  - `MI` (Myocardial Infarction)
  - `HYP` (Hypertrophy)
  - `CD` (Conduction Disturbance)
  - `STTC` (ST/T Changes)

### 2. PTB-XL Subclass (`ptbxl_sub`)
- **Data Root:** `data/ptbxl_sub_class/`
- **Total Records:** 21,388 (Train: 17,084, Val: 2,146, Test: 2,158)
- **Number of Diagnostic Classes:** 23
- **Classes:**
  - `AMI`, `LAFB/LPFB`, `LVH`, `STTC`, `IMI`, `SEHYP`, `CRBBB`, `WPW`, `LAO/LAE`, `NORM`, `ISC`, `AVB`, `RAO/RAE`, `LMI`, `ISCI`, `ISCA`, `NST`, `CLBBB`, `ILBBB`, `IRBBB`, `PMI`, `RVH`, `IVCD`

### 3. PTB-XL Rhythm (`ptbxl_rhythm`)
- **Data Root:** `data/ptbxl_rhythm/`
- **Total Records:** 21,030 (Train: 16,832, Val: 2,100, Test: 2,098)
- **Number of Diagnostic Classes:** 12
- **Classes:**
  - `SVARR`, `BIGU`, `STACH`, `SARRH`, `SBRAD`, `TRIGU`, `AFIB`, `SR`, `SVTAC`, `PSVT`, `AFLT`, `PACE`

### 4. PTB-XL Form (`ptbxl_form`)
- **Data Root:** `data/ptbxl_form/`
- **Total Records:** 8,978 (Train: 7,197, Val: 901, Test: 880)
- **Number of Diagnostic Classes:** 19
- **Note:** Excludes records that lack specific morphological/form diagnostic annotations.
- **Classes:**
  - `PAC`, `DIG`, `HVOLT`, `STD`, `LPR`, `QWAVE`, `VCLVH`, `NDT`, `TAB`, `LOWT`, `ABQRS`, `LNGQT`, `PRC(S)`, `NT`, `STE`, `NST`, `PVC`, `INVT`, `LVOLT`

### 5. CPSC 2018 (`cpsc2018` / ICBEB)
- **Data Root:** `data/cpsc2018_verified/`
- **Total Records:** 6,867 (Train: 4,942, Val: 551, Test: 1,374)
- **Number of Diagnostic Classes:** 9
- **Classes:**
  - `AFIB` (Atrial Fibrillation)
  - `1AVB` (First-degree AV Block)
  - `CLBBB` (Complete Left Bundle Branch Block)
  - `CRBBB` (Complete Right Bundle Branch Block)
  - `PAC` (Premature Atrial Contraction)
  - `VPC` (Premature Ventricular Contraction)
  - `STD` (ST-segment Depression)
  - `STE` (ST-segment Elevation)
  - `NORM` (Normal)

### 6. Chapman-Shaoxing-Ningbo (`csn`)
- **Data Root:** `data/csn_verified/`
- **Total Records:** 23,025 (Train: 16,545, Val: 1,860, Test: 4,620)
- **Number of Diagnostic Classes:** 38
- **Classes:**
  - `AQW`, `UW`, `SR`, `WPW`, `2AVB`, `AT`, `VB`, `ARS`, `STTC`, `SA`, `STE`, `VPB`, `TWO`, `STTU`, `ALS`, `APB`, `2AVB1`, `PRIE`, `CCR`, `CR`, `AF`, `AVB`, `QTIE`, `LBBB`, `VEB`, `SVT`, `RBBB`, `1AVB`, `STDD`, `MI`, `AFIB`, `TWC`, `PWC`, `ERV`, `RVH`, `LVH`, `ST`, `JEB`

---

## 3. Pretraining Dataset: MIMIC-IV-ECG

Used for Stage 1 feature and report alignment pretraining.

| Dataset | Identifier | Data Path | Train Samples | Val Samples | Test Samples | Total Records |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **MIMIC-IV-ECG** | `mimic` | `data/mimic-iv-ecg/` | 240,000 | 30,000 | 30,000 | **300,000** |

---

## 4. File Schema & Preprocessing References
- Data generation & split script: `filip/data_processing/prepare_verified_splits.py`
- In each data directory, partitions are defined by:
  - `train_records.json`
  - `val_records.json`
  - `test_records.json`
  - `diagnoses.json`
