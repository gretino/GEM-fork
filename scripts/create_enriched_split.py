import os
import sys
import json
import math
import random
import numpy as np
import wfdb
import shutil
import glob
from collections import defaultdict
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import multiprocessing as mp

# Add renderer to path
IMAGE_GEN_DIR = "/home/qfbqt/repo/GEM-fork/gem_generation/ecg-image-generator"
if IMAGE_GEN_DIR not in sys.path:
    sys.path.append(IMAGE_GEN_DIR)

from gen_ecg_image_from_data import run_single_file

class ImageGenArgs:
    def __init__(self, **kwargs):
        self.input_directory = ""
        self.output_directory = ""
        self.seed = -1
        self.num_leads = "twelve"
        self.max_num_images = -1
        self.config_file = "config.yaml"
        self.resolution = 200
        self.pad_inches = 0
        self.print_header = False
        self.num_columns = 1
        self.full_mode = "II"
        self.mask_unplotted_samples = False
        self.add_qr_code = False
        self.link = ""
        self.num_words = 5
        self.x_offset = 30
        self.y_offset = 30
        self.handwriting_size_factor = 0.2
        self.crease_angle = 90
        self.num_creases_vertically = 10
        self.num_creases_horizontally = 10
        self.rotate = 0
        self.noise = 50
        self.crop = 0.01
        self.temperature = 40000
        self.random_resolution = False
        self.random_padding = False
        self.random_grid_color = False
        self.standard_grid_color = 5
        self.calibration_pulse = 1.0
        self.random_grid_present = 1.0
        self.random_print_header = 0.0
        self.random_bw = 0.0
        self.remove_lead_names = True
        self.lead_name_bbox = False
        self.store_config = 0
        self.deterministic_offset = False
        self.deterministic_num_words = False
        self.deterministic_hw_size = False
        self.deterministic_angle = False
        self.deterministic_vertical = False
        self.deterministic_horizontal = False
        self.deterministic_rot = False
        self.deterministic_noise = False
        self.deterministic_crop = False
        self.deterministic_temp = False
        self.fully_random = False
        self.hw_text = False
        self.wrinkles = False
        self.augment = False
        self.lead_bbox = False
        
        for k, v in kwargs.items():
            setattr(self, k, v)

def get_snr_noise(signal, snr_db, smooth_sigma=2.0):
    p_sig = np.var(signal)
    if p_sig == 0:
        return np.zeros_like(signal)
    
    # Target noise power
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    
    # Generate standard white noise
    noise = np.random.normal(0, 1.0, size=signal.shape)
    
    # Smooth the noise to make it less fine-grained/spiky
    noise = gaussian_filter1d(noise, sigma=smooth_sigma)
    
    # Rescale the smoothed noise to match the target SNR power
    noise_var = np.var(noise)
    if noise_var > 0:
        noise = noise * np.sqrt(p_noise / noise_var)
        
    return noise

def process_record(args_tuple):
    record, labels, median_freq, class_counts, raw_ptbxl_dir, aug_wfdb_dir, original_cwd, images_dir = args_tuple
    
    new_records = []
    new_diagnoses = {}
    
    study_id = record['study_id']
    
    # Calculate max multiplier
    max_multiplier = 0.0
    for label, val in labels.items():
        if val > 0:
            freq = class_counts.get(label, 0)
            if freq < median_freq and freq > 0:
                multiplier = min(3.0, math.sqrt(median_freq / freq))
                max_multiplier = max(max_multiplier, multiplier)
                
    # Calculate extra copies to create
    if max_multiplier > 1.0:
        extra_weight = max_multiplier - 1.0
        num_copies = int(math.floor(extra_weight))
        if random.random() < (extra_weight - num_copies):
            num_copies += 1
            
        for c_idx in range(num_copies):
            new_study_id = f"{study_id}_enriched_{c_idx}"
            
            # Load waveform
            num_str = study_id.split('_')[0]
            num_int = int(num_str)
            subdir = f"{(num_int // 1000) * 1000:05d}"
            
            raw_path_100 = os.path.join(raw_ptbxl_dir, "records100", subdir, study_id)
            raw_path_500 = os.path.join(raw_ptbxl_dir, "records500", subdir, study_id)
            
            raw_path = None
            if os.path.exists(raw_path_100 + ".dat"):
                raw_path = raw_path_100
            elif os.path.exists(raw_path_500 + ".dat"):
                raw_path = raw_path_500
                
            if not raw_path:
                print(f"Warning: Raw waveform not found for {study_id}")
                continue
                
            # Read signal
            signals, fields = wfdb.rdsamp(raw_path)
            
            # Add noise with a lower SNR (10-20 dB) to make it more visible
            snr_db = random.uniform(15.0, 25.0)
            noisy_signals = np.zeros_like(signals)
            for lead_idx in range(signals.shape[1]):
                noise = get_snr_noise(signals[:, lead_idx], snr_db)
                noisy_signals[:, lead_idx] = signals[:, lead_idx] + noise
                
            # Write augmented WFDB
            aug_path_prefix = os.path.join(aug_wfdb_dir, new_study_id)
            fmt_list = ['16'] * signals.shape[1]
            wfdb.wrsamp(new_study_id, fs=fields['fs'], units=fields['units'],
                        sig_name=fields['sig_name'], p_signal=noisy_signals,
                        fmt=fmt_list, write_dir=aug_wfdb_dir)
                        
            # Change directory to IMAGE_GEN_DIR to avoid missing Fonts/config
            os.chdir(IMAGE_GEN_DIR)
            
            # Generate image
            args = ImageGenArgs(
                input_directory=aug_wfdb_dir,
                output_directory=aug_wfdb_dir,
                input_file=os.path.join(aug_wfdb_dir, new_study_id + ".dat"),
                header_file=os.path.join(aug_wfdb_dir, new_study_id + ".hea"),
                start_index=-1,
                encoding=new_study_id
            )
            try:
                run_single_file(args)
                
                # Move image to final directory
                gen_img_path = os.path.join(aug_wfdb_dir, new_study_id + "-0.png")
                final_img_path = os.path.join(original_cwd, images_dir, f"{new_study_id}-0.png")
                
                if os.path.exists(gen_img_path):
                    shutil.move(gen_img_path, final_img_path)
                    
                    # Add new record
                    new_rec = record.copy()
                    new_rec['study_id'] = new_study_id
                    new_rec['file_name'] = new_study_id
                    new_rec['path'] = f"images/{new_study_id}-0.png"
                    new_rec['is_repeated'] = True
                    new_records.append(new_rec)
                    
                    # Also duplicate diagnosis for new study_id
                    new_diagnoses[new_study_id] = labels.copy()
            except Exception as e:
                print(f"Failed generating image for {new_study_id}: {e}")
            finally:
                # Always revert cwd back
                os.chdir(original_cwd)
                
    return record, new_records, new_diagnoses

def main():
    random.seed(42)
    np.random.seed(42)
    
    data_dir = 'data/ptbxl_sub_class'
    raw_ptbxl_dir = os.path.expanduser('~/datasets/ptb_xl_1.0.3/')
    if not os.path.exists(raw_ptbxl_dir):
        # try without trailing slash
        pass
        
    with open(f'{data_dir}/diagnoses.json', 'r') as f:
        diagnoses = json.load(f)
        
    with open(f'{data_dir}/train_records.json', 'r') as f:
        train_records = json.load(f)
        
    # Calculate frequencies
    class_counts = defaultdict(int)
    for record in train_records:
        study_id = record['study_id']
        labels = diagnoses.get(study_id, {})
        for label, val in labels.items():
            if val > 0:
                class_counts[label] += 1
                
    # Filter positive classes to find median
    positive_freqs = [count for count in class_counts.values() if count > 0]
    median_freq = np.median(positive_freqs)
    print(f"Median positive label frequency: {median_freq}")
    
    enriched_records = []
    
    # Setup temporary directory for augmented signals
    aug_wfdb_dir = os.path.abspath(os.path.join(data_dir, 'augmented_wfdb'))
    os.makedirs(aug_wfdb_dir, exist_ok=True)
    images_dir = os.path.abspath(os.path.join(data_dir, 'images'))
    os.makedirs(images_dir, exist_ok=True)
    
    # Save original working directory
    original_cwd = os.getcwd()
    
    tasks = []
    class_counts_dict = dict(class_counts)
    for record in train_records:
        labels = diagnoses.get(record['study_id'], {})
        tasks.append((record, labels, median_freq, class_counts_dict, raw_ptbxl_dir, aug_wfdb_dir, original_cwd, images_dir))
        
    num_workers = mp.cpu_count()
    print(f"Starting multiprocessing with {num_workers} workers...")
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_record, tasks), total=len(tasks), desc="Processing train records"))
        
    for original_rec, new_recs, new_diags in results:
        enriched_records.append(original_rec)
        enriched_records.extend(new_recs)
        diagnoses.update(new_diags)
        
                    
    # Save enriched records and updated diagnoses
    with open(f'{data_dir}/train_enriched_records.json', 'w') as f:
        json.dump(enriched_records, f, indent=4)
        
    with open(f'{data_dir}/diagnoses.json', 'w') as f:
        json.dump(diagnoses, f, indent=4)
        
    print(f"Original train size: {len(train_records)}")
    print(f"Enriched train size: {len(enriched_records)}")
    print(f"Generated {len(enriched_records) - len(train_records)} enriched records (~{((len(enriched_records) - len(train_records))/len(enriched_records))*100:.2f}% of total).")

if __name__ == "__main__":
    main()
