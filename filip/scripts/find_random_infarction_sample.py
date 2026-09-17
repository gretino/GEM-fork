#!/usr/bin/env python3
"""
Temporary script to find a random item in the MIMIC dataset whose parsed text
contains "infarction", and print out the entire parsed text used in FILIP alignment.
"""

import os
import sys
import json
import random
import argparse

# Add repository root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def find_mimic_data_root(custom_path=None):
    if custom_path and os.path.exists(custom_path):
        return custom_path
    candidates = [
        "/home/qfbqt/8TB/datasets/mimic-iv-ecg/",
        os.path.join(PROJECT_ROOT, "data", "mimic-iv-ecg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find MIMIC dataset root in candidate paths: {candidates}")


def load_split_records(data_root, split):
    records_file = os.path.join(data_root, f"{split}_records.json")
    if not os.path.exists(records_file):
        raise FileNotFoundError(f"Records file not found: {records_file}")
    with open(records_file, "r") as f:
        records = json.load(f)
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Find a random MIMIC sample containing 'infarction' and display its parsed report text."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to MIMIC dataset root directory (default: auto-detected)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="Dataset split to search in (default: train)",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default="infarction",
        help="Keyword to filter parsed text by (default: 'infarction')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic sampling",
    )
    parser.add_argument(
        "--show_tokens",
        action="store_true",
        default=True,
        help="Also show CLIP tokenization details used during FILIP alignment (default: True)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    data_root = find_mimic_data_root(args.data_root)
    print(f"Dataset root: {data_root}")

    # Load records according to split
    splits_to_load = ["train", "val", "test"] if args.split == "all" else [args.split]
    all_matching = []

    for s in splits_to_load:
        records = load_split_records(data_root, s)
        print(f"Loaded {len(records)} records from {s}_records.json")
        for rec in records:
            report_text = rec.get("report_text", "")
            if args.keyword.lower() in report_text.lower():
                all_matching.append((s, rec))

    print(f"\nTotal records matching '{args.keyword}': {len(all_matching)}")
    if not all_matching:
        print(f"No records found containing '{args.keyword}'.")
        return

    # Pick a random item
    split, selected_item = random.choice(all_matching)
    study_id = selected_item.get("study_id")
    subject_id = selected_item.get("subject_id")
    report_text = selected_item.get("report_text", "")

    print("\n" + "=" * 80)
    print(f"RANDOM SELECTED SAMPLE (Split: {split})")
    print("=" * 80)
    print(f"Study ID   : {study_id}")
    print(f"Subject ID : {subject_id}")
    print(f"ECG Time   : {selected_item.get('ecg_time', 'N/A')}")
    print(f"Image Path : images/{study_id}-0.png")
    print("-" * 80)
    print("ENTIRE PARSED TEXT FOR FILIP ALIGNMENT:")
    print("-" * 80)
    print(report_text)
    print("-" * 80)

    # Optional CLIP tokenization details
    if args.show_tokens:
        try:
            from transformers import AutoTokenizer

            tokenizer_name = "openai/clip-vit-base-patch32"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            encoded = tokenizer(
                [report_text],
                padding=True,
                truncation=True,
                max_length=77,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            content_mask = (
                encoded["attention_mask"].bool()
                & ~encoded.pop("special_tokens_mask").bool()
            )
            tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
            content_tokens = [
                tok for tok, mask in zip(tokens, content_mask[0]) if mask
            ]

            print(f"\nCLIP Tokenization ({tokenizer_name}, max_length=77):")
            print(f"Total tokens in sequence (including BOS/EOS): {len(tokens)}")
            print(f"Active content tokens for FILIP alignment : {len(content_tokens)}")
            print(f"Content tokens: {content_tokens}")
        except Exception as e:
            print(f"\nNote: Could not run tokenizer ({e}).")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
