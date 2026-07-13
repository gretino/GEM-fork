import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss, accuracy_score

def compute_multilabel_metrics(y_true, y_pred_logits, mask, threshold=0.5):
    """
    y_true: [N, F] numpy array of binary labels (0 or 1)
    y_pred_logits: [N, F] numpy array of model outputs (logits)
    mask: [N, F] numpy array of masks (1 for valid, 0 for ignore)
    """
    # Apply sigmoid to logits to get probabilities
    y_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))
    y_pred = (y_prob >= threshold).astype(np.float32)
    
    # Calculate valid label ratio
    valid_label_ratio = float(np.mean(mask))
    
    # Filter using mask: for each class, only compute metrics on valid entries
    num_classes = y_true.shape[1]
    
    class_aucs = []
    class_f1s = []
    
    all_y_true_flat = []
    all_y_prob_flat = []
    all_y_pred_flat = []
    
    for c in range(num_classes):
        valid_idx = np.where(mask[:, c] == 1.0)[0]
        if len(valid_idx) == 0:
            continue
        
        y_t = y_true[valid_idx, c]
        y_p_prob = y_prob[valid_idx, c]
        y_p = y_pred[valid_idx, c]
        
        # We need both classes to be present for ROC AUC
        if len(np.unique(y_t)) > 1:
            auc = roc_auc_score(y_t, y_p_prob)
            class_aucs.append(auc)
        else:
            class_aucs.append(np.nan)
            
        f1 = f1_score(y_t, y_p, zero_division=0)
        class_f1s.append(f1)
        
        all_y_true_flat.extend(y_t)
        all_y_prob_flat.extend(y_p_prob)
        all_y_pred_flat.extend(y_p)
        
    # Macro metrics: average of class metrics (ignoring NaNs)
    macro_auc = float(np.nanmean(class_aucs)) if not np.all(np.isnan(class_aucs)) else 0.5
    macro_f1 = float(np.nanmean(class_f1s)) if len(class_f1s) > 0 else 0.0
    
    # Micro metrics: flat computation over all valid predictions
    if len(all_y_true_flat) > 0:
        all_y_true_flat = np.array(all_y_true_flat)
        all_y_prob_flat = np.array(all_y_prob_flat)
        all_y_pred_flat = np.array(all_y_pred_flat)
        
        micro_auc = float(roc_auc_score(all_y_true_flat, all_y_prob_flat)) if len(np.unique(all_y_true_flat)) > 1 else 0.5
        micro_f1 = float(f1_score(all_y_true_flat, all_y_pred_flat, zero_division=0))
    else:
        micro_auc = 0.5
        micro_f1 = 0.0
        
    metrics = {
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "valid_label_ratio": valid_label_ratio,
        "class_aucs": class_aucs,
        "class_f1s": class_f1s
    }
    
    return metrics

def compute_diagnosis_metrics(y_true, y_pred_logits, mask, threshold=0.5):
    """
    For Stage 2 PTB-XL evaluation.
    y_true: [N, C] numpy array of labels
    y_pred_logits: [N, C] numpy array of logits
    mask: [N, C] numpy array of masks
    """
    base_metrics = compute_multilabel_metrics(y_true, y_pred_logits, mask, threshold)
    
    y_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))
    y_pred = (y_prob >= threshold).astype(np.float32)
    
    # Subset accuracy (exact match) and Hamming loss are defined on complete predictions.
    # Where mask is 0, we can fill with target label to not penalize.
    y_pred_masked = np.copy(y_pred)
    # Align predictions with true targets where masked out, to ignore masked indices in global accuracy
    y_pred_masked[mask == 0.0] = y_true[mask == 0.0]
    
    sub_acc = float(accuracy_score(y_true, y_pred_masked))
    h_loss = float(hamming_loss(y_true, y_pred_masked))
    
    metrics = {
        "macro_auc": base_metrics["macro_auc"],
        "micro_auc": base_metrics["micro_auc"],
        "macro_f1": base_metrics["macro_f1"],
        "micro_f1": base_metrics["micro_f1"],
        "hamming_loss": h_loss,
        "subset_accuracy": sub_acc,
        "class_aucs": base_metrics["class_aucs"],
        "class_f1s": base_metrics["class_f1s"]
    }
    
    return metrics
