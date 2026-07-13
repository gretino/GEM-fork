"""
evaluate_gem.py — compute classification metrics for GEM QLoRA experiments.

The original evaluate_gem.py expected a ``golden`` key inside each prediction
JSONL record, but the inference script (model_ecg_resume.py) never writes that
field.  This version looks up ground-truth labels from the companion test JSON
file (``--test_file``) by matching ``question_id`` (predictions) ↔ ``id``
(test data).  The argument is optional for backward compatibility: if
``--test_file`` is omitted the script falls back to the ``golden`` field in
the JSONL, reproducing the original behaviour (which will still give zeros if
that field is absent).

Usage
-----
    python scripts/evaluate_gem.py \\
        --results_file results/subclass_no_reasoning_qlora/predictions.jsonl \\
        --track subclass \\
        --test_file  data/gem_data/gem_test_subclass_no_reasoning.json

Tracks
------
  superclass  Labels are one or more of {NORM, MI, STTC, CD, HYP}.
              Both prediction and ground-truth are extracted from inside
              <answer>…</answer> tags and split on comma/whitespace.
  subclass    Labels are free-form SCP codes, comma-separated inside
              <answer>…</answer>.
"""

import os
import json
import argparse
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer


# ---------------------------------------------------------------------------
# Label parsing helpers
# ---------------------------------------------------------------------------

def _strip_answer_tag(text: str) -> str:
    """Return the content inside <answer>…</answer>, or the whole text."""
    if "<answer>" in text:
        text = text.split("<answer>")[-1]
    if "</answer>" in text:
        text = text.split("</answer>")[0]
    return text.strip()


def _parse_labels(text: str, label_space=None) -> list:
    """
    Parse a comma- or semicolon-separated label string into a list of clean label strings.

    For superclass tracks we intersect with the known label space so that
    free-text pollution is ignored.  For subclass tracks every token is accepted.
    """
    text = _strip_answer_tag(text)
    text = text.replace(";", ",")
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    if label_space is not None:
        tokens = [t for t in tokens if t in label_space]
    return tokens



# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_pred, y_true, classes=None):
    mlb = MultiLabelBinarizer(classes=classes)
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    hl = hamming_loss(y_true_bin, y_pred_bin)

    f1_scores = f1_score(y_true_bin, y_pred_bin, average=None, zero_division=0)
    macro_f1 = np.mean(f1_scores)
    micro_f1 = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)

    auc_scores = []
    for i in range(y_true_bin.shape[1]):
        if y_true_bin[:, i].sum() == 0:
            auc_scores.append(np.nan)
            continue
        try:
            auc = roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i])
        except ValueError:
            auc = np.nan
        auc_scores.append(auc)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        macro_auc = np.nanmean(auc_scores)

    accuracy = accuracy_score(y_true_bin, y_pred_bin)

    return {
        "Macro F1": macro_f1 * 100,
        "Micro F1": micro_f1 * 100,
        "Macro AUC": macro_auc * 100,
        "Hamming Loss": hl * 100,
        "Accuracy": accuracy * 100,
    }


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------

def evaluate(results_file: str, track: str, test_file: str | None = None):
    print(f"Evaluating {results_file} on Track: {track}")

    # ------------------------------------------------------------------
    # Build ground-truth lookup from the test JSON (preferred path)
    # ------------------------------------------------------------------
    gt_lookup: dict[str, str] = {}
    if test_file:
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        for item in test_data:
            item_id = str(item["id"])
            # Ground truth is the second conversation turn (gpt response)
            gt_raw = item["conversations"][1]["value"]
            gt_lookup[item_id] = gt_raw
        print(f"  Loaded {len(gt_lookup)} ground-truth labels from {test_file}")
    else:
        print(
            "  WARNING: --test_file not provided.  Falling back to 'golden' field "
            "in the JSONL (may give zeros if that field is absent)."
        )

    # ------------------------------------------------------------------
    # Known label space per track
    # ------------------------------------------------------------------
    if track == "superclass":
        label_space = ["NORM", "MI", "STTC", "CD", "HYP"]
    else:
        label_space = None  # determined dynamically from data

    # ------------------------------------------------------------------
    # Load predictions and pair with ground truth
    # ------------------------------------------------------------------
    predict_list = []
    golden_list = []
    missing_gt = 0

    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = str(item.get("question_id", ""))

            # ---- Predicted labels ----
            response = item.get("text", item.get("response", ""))
            pred = _parse_labels(response, label_space)

            # ---- Ground-truth labels ----
            if gt_lookup:
                gt_raw = gt_lookup.get(qid)
                if gt_raw is None:
                    missing_gt += 1
                    golden = []
                else:
                    golden = _parse_labels(gt_raw, label_space)
            else:
                # Fallback: use field embedded in JSONL
                golden_raw = item.get("golden", [])
                if isinstance(golden_raw, str):
                    golden = _parse_labels(golden_raw, label_space)
                elif isinstance(golden_raw, list):
                    golden = golden_raw
                else:
                    golden = []

            predict_list.append(pred)
            golden_list.append(golden)

    if missing_gt:
        print(f"  WARNING: {missing_gt} predictions had no matching ground-truth entry.")

    # ------------------------------------------------------------------
    # Build unified class list
    # ------------------------------------------------------------------
    all_classes: set = set()
    if label_space:
        all_classes.update(label_space)
    else:
        for p in predict_list:
            all_classes.update(p)
        for g in golden_list:
            all_classes.update(g)
    all_classes_list = sorted(all_classes)

    print(f"  Total samples evaluated: {len(predict_list)}")
    print(f"  Label space size: {len(all_classes_list)}")

    # ------------------------------------------------------------------
    # Compute and print metrics
    # ------------------------------------------------------------------
    metrics = compute_metrics(predict_list, golden_list, classes=all_classes_list)
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate GEM QLoRA predictions against ground-truth labels."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the JSONL predictions file.",
    )
    parser.add_argument(
        "--track",
        type=str,
        required=True,
        choices=["superclass", "subclass"],
        help="Evaluation track.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help=(
            "Path to the GEM test JSON file used during inference.  "
            "When provided, ground-truth labels are read from here rather "
            "than from the 'golden' field in the JSONL."
        ),
    )
    args = parser.parse_args()
    evaluate(args.results_file, args.track, args.test_file)
