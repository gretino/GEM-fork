import os
import argparse
import pandas as pd
import json

def sample_splits(output_dir, num_items, seed):
    all_records_path = os.path.join(output_dir, "all_records.csv")
    if not os.path.exists(all_records_path):
        raise FileNotFoundError(f"all_records.csv not found at {all_records_path}")

    print(f"Loading master list from {all_records_path}...")
    df = pd.read_csv(all_records_path)

    print(f"Shuffling dataset with seed {seed}...")
    shuffled_df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"Sampling {num_items} records...")
    if len(shuffled_df) < num_items:
        raise ValueError(f"Requested {num_items} items, but only {len(shuffled_df)} are available.")
    
    sampled_df = shuffled_df.iloc[:num_items].copy()

    # Split 8:1:1
    train_size = int(num_items * 0.8)
    val_size = int(num_items * 0.1)
    test_size = num_items - train_size - val_size

    train_df = sampled_df.iloc[:train_size]
    val_df = sampled_df.iloc[train_size:train_size + val_size]
    test_df = sampled_df.iloc[train_size + val_size:]

    print(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Convert to list of dicts and save to JSON
    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    for name, split_df in splits.items():
        records = split_df.to_dict(orient="records")
        output_path = os.path.join(output_dir, f"{name}_records.json")
        with open(output_path, "w") as f:
            json.dump(records, f, indent=4)
        print(f"Saved {name} split to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample and split clean master list into train/val/test splits.")
    parser.add_argument("--output_dir", type=str, default="/home/qfbqt/repo/GEM-fork/data/mimic-iv-ecg/",
                        help="Path to output directory containing all_records.csv.")
    parser.add_argument("--num_items", type=str, default="1000",
                        help="Total number of items to sample across all splits.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling.")

    args = parser.parse_args()
    
    # Handle potentially string-formatted num_items
    num_items = int(args.num_items)
    sample_splits(args.output_dir, num_items, args.seed)
