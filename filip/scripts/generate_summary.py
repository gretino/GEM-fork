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

def build_markdown_table(metric_name, tasks_data, model_variants):
    # tasks_data is a dict: {task_display_name: {model_name: {pct: value}}}
    header = f"## {metric_name} (%)\n\n"
    col_names = [mv[0] for mv in model_variants]
    header += "| Dataset / Task | Data Ratio | " + " | ".join(col_names) + " |\n"
    header += "|----------------|------------|" + "|".join(["-----------------" for _ in model_variants]) + "|\n"
    
    rows = []
    for task_name, models_dict in tasks_data.items():
        for pct in [1, 10, 100]:
            vals = [models_dict.get(m_display, {}).get(pct, "-") for m_display, _ in model_variants]
            rows.append(f"| {task_name} | {pct}% | " + " | ".join(vals) + " |")
        rows.append("|--- |--- |" + "|".join(["---" for _ in model_variants]) + "|")
        
    return header + "\n".join(rows) + "\n\n"

def main():
    tasks = [
        ("PTB-XL Superclass (5)", "ptbxl_super"),
        ("PTB-XL Subclass (23)", "ptbxl_sub"),
        ("PTB-XL Rhythm (12)", "ptbxl_rhythm"),
        ("PTB-XL Form (19)", "ptbxl_form"),
        ("CPSC2018 ICBEB (9)", "cpsc"),
        ("CSN Chapman (12)", "csn"),
    ]
    
    model_variants = [
        ("JEPA (Baseline)", lambda t: f"{t}_adapt_jepa"),
        ("FILIP (ViT-Base)", lambda t: f"{t}_report_align_adapt"),
        ("FILIP (ViT-Large)", lambda t: f"vit_large_{t}_report_align_adapt"),
        ("FILIP (ViT-Large + JEPA)", lambda t: f"vit_large_jepa_{t}_report_align_adapt"),
    ]

    percentages = [1, 10, 100]
    
    results = {
        "Macro F1": {},
        "Macro AUC": {},
        "Accuracy": {}
    }
    
    for task_display, task_key in tasks:
        results["Macro F1"][task_display] = {}
        results["Macro AUC"][task_display] = {}
        results["Accuracy"][task_display] = {}
        
        for model_display, name_fn in model_variants:
            folder_base = name_fn(task_key)
            results["Macro F1"][task_display][model_display] = {}
            results["Macro AUC"][task_display][model_display] = {}
            results["Accuracy"][task_display][model_display] = {}
            
            for pct in percentages:
                filepath = f"outputs/filip/{folder_base}_{pct}/evaluation/metrics.txt"
                metrics = parse_metrics(filepath)
                
                if metrics == "-":
                    results["Macro F1"][task_display][model_display][pct] = "-"
                    results["Macro AUC"][task_display][model_display][pct] = "-"
                    results["Accuracy"][task_display][model_display][pct] = "-"
                else:
                    results["Macro F1"][task_display][model_display][pct] = metrics["F1"]
                    results["Macro AUC"][task_display][model_display][pct] = metrics["AUC"]
                    results["Accuracy"][task_display][model_display][pct] = metrics["Accuracy"]

    # Build report content
    variant_names = ", ".join([f"**{m[0]}**" for m in model_variants])
    report_content = f"# Comparative Benchmark Summary Report\n\n"
    report_content += f"Comparing: {variant_names} across 1%, 10%, and 100% data ratios.\n\n"
    report_content += build_markdown_table("Macro AUC", results["Macro AUC"], model_variants)
    report_content += build_markdown_table("Macro F1", results["Macro F1"], model_variants)
    report_content += build_markdown_table("Accuracy", results["Accuracy"], model_variants)
    
    output_path = "outputs/filip/evaluation_summary.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_content)
        
    print("========================================================================")
    print("Comparative Summary Tables Generated successfully!")
    print(f"Saved to: {output_path}")
    print("========================================================================")
    print(report_content)

if __name__ == "__main__":
    main()
