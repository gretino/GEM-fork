import os
import sys
import argparse
import json
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

# We add the original image generator directory to sys.path
IMAGE_GEN_DIR = "/home/qfbqt/repo/GEM-fork/gem_generation/ecg-image-generator"
if IMAGE_GEN_DIR not in sys.path:
    sys.path.append(IMAGE_GEN_DIR)

# Change working directory so all relative paths (Fonts/, config.yaml) resolve correctly
os.chdir(IMAGE_GEN_DIR)

from gen_ecg_image_from_data import run_single_file

class ImageGenArgs:
    def __init__(self, **kwargs):
        # Default parameters from get_parser in gen_ecg_images_from_data_multi.py
        self.input_directory = ""
        self.output_directory = ""
        self.seed = -1
        self.num_leads = "twelve"
        self.max_num_images = -1
        self.config_file = "config.yaml"
        self.resolution = 200
        self.pad_inches = 0
        self.print_header = False
        self.num_columns = -1
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
        
        # Override with custom kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

def process_single_record(record, dataset_dir, images_dir):
    path_rel = record["path"]
    study_id = str(record["study_id"])
    
    # Skip if image already exists
    out_file = os.path.join(images_dir, f"{study_id}-0.png")
    if os.path.exists(out_file):
        return True, study_id

    # Input files
    input_file = os.path.join(dataset_dir, path_rel + ".dat")
    header_file = os.path.join(dataset_dir, path_rel + ".hea")
    
    if not os.path.exists(input_file) or not os.path.exists(header_file):
        return False, f"Missing files for {study_id}"

    args = ImageGenArgs(
        input_directory=dataset_dir,
        output_directory=images_dir,
        input_file=input_file,
        header_file=header_file,
        start_index=-1,
        encoding=study_id
    )
    
    try:
        run_single_file(args)
        return True, study_id
    except Exception as e:
        return False, f"Failed for {study_id}: {str(e)}"

def generate_images(dataset_dir, output_dir, images_dir, num_workers):
    # Load targeted records from split JSON files
    all_target_records = []
    splits = ["train", "val", "test"]
    for split in splits:
        split_path = os.path.join(output_dir, f"{split}_records.json")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found at {split_path}")
        with open(split_path, "r") as f:
            all_target_records.extend(json.load(f))

    print(f"Total target records to generate images for: {len(all_target_records)}")
    os.makedirs(images_dir, exist_ok=True)

    # Parallel processing using Pool
    process_func = partial(process_single_record, dataset_dir=dataset_dir, images_dir=images_dir)
    
    success_count = 0
    failure_messages = []
    
    print(f"Starting parallel ECG image generation with {num_workers} workers...")
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(all_target_records), desc="Generating Images") as pbar:
            for success, msg in pool.imap_unordered(process_func, all_target_records):
                if success:
                    success_count += 1
                else:
                    failure_messages.append(msg)
                pbar.update(1)

    print(f"Image generation complete: {success_count} / {len(all_target_records)} succeeded.")
    if failure_messages:
        print(f"First 10 failures:")
        for msg in failure_messages[:10]:
            print(" -", msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel ECG image generation for the FILIP experiment.")
    parser.add_argument("--dataset_dir", type=str, default="/home/qfbqt/8TB/blmcg/datasets/physionet.org.5/files/mimic-iv-ecg/1.0/",
                        help="Path to the raw MIMIC-IV-ECG dataset directory.")
    parser.add_argument("--output_dir", type=str, default="/home/qfbqt/repo/GEM-fork/data/mimic-iv-ecg/",
                        help="Path to output directory containing record splits.")
    parser.add_argument("--images_dir", type=str, default="/home/qfbqt/repo/GEM-fork/data/mimic-iv-ecg/images/",
                        help="Path to target directory for generated PNG images.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help="Number of workers to use for parallel processing.")
    
    args = parser.parse_args()
    generate_images(args.dataset_dir, args.output_dir, args.images_dir, args.num_workers)
