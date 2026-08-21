# filip/scripts/analyze_report_duplicates.py

import os
import json
import torch
from collections import Counter
from torch.utils.data import DataLoader
from filip.data.dataset import ECGImageDataset


def normalize_text(text):
    return " ".join(text.strip().lower().split())


def main():
    data_root = "/home/qfbqt/8TB/datasets/mimic-iv-ecg/"
    if not os.path.exists(data_root):
        data_root = "data/mimic-iv-ecg"

    print("Loading MIMIC-IV ECG dataset records...")
    dataset = ECGImageDataset(data_root=data_root, split='train', dataset_name='mimic')
    total_samples = len(dataset)
    print(f"Total samples in training set: {total_samples}")

    raw_reports = []
    norm_reports = []

    for i in range(total_samples):
        rec = dataset.records[i]
        r = rec.get("report_text", "")
        raw_reports.append(r)
        norm_reports.append(normalize_text(r))

    raw_counts = Counter(raw_reports)
    norm_counts = Counter(norm_reports)

    unique_raw = len(raw_counts)
    unique_norm = len(norm_counts)

    raw_dup_samples = sum(count for count in raw_counts.values() if count > 1)
    norm_dup_samples = sum(count for count in norm_counts.values() if count > 1)

    top_10_norm = norm_counts.most_common(10)

    print("\n--- GLOBAL DATASET DUPLICATION STATISTICS ---")
    print(f"Total samples:                 {total_samples}")
    print(f"Unique raw reports:            {unique_raw} ({unique_raw/total_samples*100:.2f}%)")
    print(f"Unique normalized reports:     {unique_norm} ({unique_norm/total_samples*100:.2f}%)")
    print(f"Samples in raw duplicate sets: {raw_dup_samples} ({raw_dup_samples/total_samples*100:.2f}%)")
    print(f"Samples in norm duplicate sets:{norm_dup_samples} ({norm_dup_samples/total_samples*100:.2f}%)")
    print(f"Most frequent report count:    {norm_counts.most_common(1)[0][1]}")

    print("\nTop 5 Most Frequent Normalized Reports:")
    for text, count in top_10_norm[:5]:
        print(f"  [{count} times] ({count/total_samples*100:.2f}%): '{text[:80]}...'")

    from filip.data.collator import ecg_collate_fn
    for bsize in [32, 64]:
        dataloader = DataLoader(dataset, batch_size=bsize, shuffle=True, collate_fn=ecg_collate_fn)

        dup_batches = 0
        total_batches = len(dataloader)
        max_dups_in_batch = 0
        sum_dups_in_batch = 0

        for batch in dataloader:
            b_reports = [normalize_text(r) for r in batch['report_texts']]

            b_counts = Counter(b_reports)
            num_unique = len(b_counts)
            dups = bsize - num_unique
            if dups > 0:
                dup_batches += 1
            max_dups_in_batch = max(max_dups_in_batch, dups)
            sum_dups_in_batch += dups

        avg_dups = sum_dups_in_batch / total_batches
        print(f"\n--- BATCH-LEVEL STATISTICS (Batch Size = {bsize}) ---")
        print(f"Batches containing duplicates: {dup_batches}/{total_batches} ({dup_batches/total_batches*100:.2f}%)")
        print(f"Average duplicates per batch:  {avg_dups:.2f} / {bsize} ({avg_dups/bsize*100:.2f}%)")
        print(f"Max duplicates in a single batch: {max_dups_in_batch} / {bsize}")

if __name__ == "__main__":
    main()
