import os
import json
import subprocess
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_record(record, raw_data_root, output_dir, ecg_gen_dir):
    # Skip if image already exists
    study_id = record.get('study_id', os.path.basename(record['path']))
    img_path = os.path.join(output_dir, f"{study_id}-0.png")
    if os.path.exists(img_path):
        return True

    hea_path = os.path.join(raw_data_root, f"{record['path']}.hea")
    dat_path = os.path.join(raw_data_root, f"{record['path']}.dat")
    
    if not os.path.exists(hea_path) or not os.path.exists(dat_path):
        return False
        
    cmd = [
        "python", "gen_ecg_image_from_data.py",
        "-i", dat_path,
        "-hea", hea_path,
        "-o", output_dir,
        "--num_columns", "1",
        "-se", "0",
        "-st", "-1"
    ]
    
    # Suppress stdout to avoid console spam, let stderr print if errors occur
    subprocess.run(cmd, cwd=ecg_gen_dir, check=True, stdout=subprocess.DEVNULL)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int, default=16, help="Number of concurrent threads to run.")
    args = parser.parse_args()

    dataset_root = "/home/qfbqt/8TB/datasets/mimic-iv-ecg"
    raw_data_root = "/home/qfbqt/8TB/blmcg/datasets/physionet.org.5/files/mimic-iv-ecg/1.0"
    output_dir = os.path.join(dataset_root, "images")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Gather all target records
    splits = ["train", "val", "test"]
    records = []
    for split in splits:
        with open(os.path.join(dataset_root, f"{split}_records.json"), "r") as f:
            records.extend(json.load(f))
            
    print(f"Loaded {len(records)} records from splits.")

    # 2. Run the image generator in parallel
    ecg_gen_dir = "/home/qfbqt/repo/GEM-fork/gem_generation/ecg-image-generator"
    print(f"Generating images into {output_dir} using {args.num_threads} threads...")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_record, record, raw_data_root, output_dir, ecg_gen_dir) 
            for record in records
        ]
        
        # Process results as they complete with a progress bar
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                if future.result():
                    success_count += 1
            except subprocess.CalledProcessError as e:
                # The subprocess stderr will have already been printed to the console
                pass
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                
    print(f"\nDone! Successfully generated {success_count} images.")

if __name__ == "__main__":
    main()
