import os
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer

def compute_metrics(y_pred, y_true, classes=None):
    mlb = MultiLabelBinarizer(classes=classes)
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    
    hl = hamming_loss(y_true_bin, y_pred_bin)
    
    f1_scores = f1_score(y_true_bin, y_pred_bin, average=None)
    macro_f1 = np.mean(f1_scores)
    micro_f1 = f1_score(y_true_bin, y_pred_bin, average='micro')
    
    auc_scores = []
    for i in range(y_true_bin.shape[1]):
        try:
            auc = roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i])
        except ValueError:
            auc = np.nan
        auc_scores.append(auc)
    macro_auc = np.nanmean(auc_scores)
    
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    
    return {
        "Macro F1": macro_f1 * 100,
        "Micro F1": micro_f1 * 100,
        "Macro AUC": macro_auc * 100,
        "Hamming Loss": hl * 100,
        "Accuracy": accuracy * 100
    }

def evaluate(results_file, track):
    print(f"Evaluating {results_file} on Track: {track}")
    
    predict_list = []
    golden_list = []
    
    if track == "superclass":
        label_space = ["NORM", "MI", "STTC", "CD", "HYP"]
    else:
        label_space = None 
        
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            
            response = item.get("text", item.get("response", ""))
            if "<answer>" in response:
                response = response.split("<answer>")[-1].replace("</answer>", "").strip()
                
            golden = item.get("golden", [])
            
            if track == "superclass":
                predict = [label for label in label_space if label in response]
            else:
                predict = [l.strip() for l in response.split(",")]
                
            predict_list.append(predict)
            golden_list.append(golden)
            
    all_classes = set()
    if label_space:
        all_classes.update(label_space)
    else:
        for p in predict_list: all_classes.update(p)
        for g in golden_list: all_classes.update(g)
    all_classes = list(all_classes)
            
    metrics = compute_metrics(predict_list, golden_list, classes=all_classes)
    
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True, help="Path to the JSONL results file")
    parser.add_argument("--track", type=str, required=True, choices=["superclass", "subclass"], help="Evaluation track")
    args = parser.parse_args()
    
    evaluate(args.results_file, args.track)
