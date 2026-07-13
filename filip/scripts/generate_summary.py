import os
import re

def parse_metrics(filepath):
    if not os.path.exists(filepath):
        return "-"
    macro_f1, macro_auc, accuracy = "-", "-", "-"
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("Macro F1:"):
                    macro_f1 = line.split(":")[-1].strip()
                elif line.startswith("Macro AUC:"):
                    macro_auc = line.split(":")[-1].strip()
                elif line.startswith("Accuracy:"):
                    accuracy = line.split(":")[-1].strip()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return {"F1": macro_f1, "AUC": macro_auc, "Accuracy": accuracy}

def build_markdown_table(metric_name, data):
    # data is a dict: {experiment_display_name: {pct: value}}
    header = f"## {metric_name} (%)\n\n"
    header += "| Experiment | 1% | 10% | 100% |\n"
    header += "|------------|----|-----|------|\n"
    
    rows = []
    for exp_name, pct_data in data.items():
        val_1 = pct_data.get(1, "-")
        val_10 = pct_data.get(10, "-")
        val_100 = pct_data.get(100, "-")
        rows.append(f"| {exp_name} | {val_1} | {val_10} | {val_100} |")
        
    return header + "\n".join(rows) + "\n\n"

def main():
    experiments = [
        ("PTB-XL Superclass", "ptbxl_super_adapt"),
        ("PTB-XL Subclass", "ptbxl_sub_adapt"),
        ("PTB-XL Rhythm", "ptbxl_rhythm_adapt"),
        ("PTB-XL Form", "ptbxl_form_adapt"),
        ("CSN (Chapman)", "csn_adapt"),
        ("CPSC2018 (ICBEB)", "cpsc_adapt")
    ]
    
    percentages = [1, 10, 100]
    
    results = {
        "Macro F1": {},
        "Macro AUC": {},
        "Accuracy": {}
    }
    
    for display_name, folder_name in experiments:
        results["Macro F1"][display_name] = {}
        results["Macro AUC"][display_name] = {}
        results["Accuracy"][display_name] = {}
        
        for pct in percentages:
            # Check standard path outputs/filip/xxx_adapt_[pct]/evaluation/metrics.txt
            filepath = f"outputs/filip/{folder_name}_{pct}/evaluation/metrics.txt"
            metrics = parse_metrics(filepath)
            
            if metrics == "-":
                results["Macro F1"][display_name][pct] = "-"
                results["Macro AUC"][display_name][pct] = "-"
                results["Accuracy"][display_name][pct] = "-"
            else:
                results["Macro F1"][display_name][pct] = metrics["F1"]
                results["Macro AUC"][display_name][pct] = metrics["AUC"]
                results["Accuracy"][display_name][pct] = metrics["Accuracy"]

    # Build report content
    report_content = "# Evaluation Summary Report (Verified Splits)\n\n"
    report_content += build_markdown_table("Macro F1", results["Macro F1"])
    report_content += build_markdown_table("Macro AUC", results["Macro AUC"])
    report_content += build_markdown_table("Accuracy", results["Accuracy"])
    
    output_path = "outputs/filip/evaluation_summary.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_content)
        
    print("==========================================")
    print("Summary Tables Generated successfully!")
    print(f"Saved to: {output_path}")
    print("==========================================")
    print(report_content)

if __name__ == "__main__":
    main()
