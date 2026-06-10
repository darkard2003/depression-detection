#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

COMBINED_CSV = "datasets/combined_dataset.csv"
DATA_DIR = "data_processed/combined"
OUTPUT_REPORT_PATH = "evaluation_combined_report.md"
SEED = 42

def evaluate_predictions(y_true, y_pred_probs):
    y_pred = (y_pred_probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, f1

def main():
    print("=" * 60)
    # 1. Load Combined Dataset and compute split indices
    if not os.path.exists(COMBINED_CSV):
        print(f"Error: {COMBINED_CSV} not found!")
        sys.exit(1)
        
    df = pd.read_csv(COMBINED_CSV)
    labels = df['label'].astype(int).values
    sources = df['source'].values
    indices = np.arange(len(df))
    
    # Replicate train_test_split to get exact test indices
    _, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=SEED
    )
    
    # 2. Load precomputed features and soft targets
    print("Loading precomputed feature matrices and target arrays...")
    X_lite = np.load(os.path.join(DATA_DIR, "X_lite_combined.npy"))
    X_hybrid = np.load(os.path.join(DATA_DIR, "X_hybrid_combined.npy"))
    y_teacher_orig = np.load(os.path.join(DATA_DIR, "y_teacher_soft_original.npy"))
    y_teacher_ft = np.load(os.path.join(DATA_DIR, "y_teacher_soft_finetuned.npy"))
    
    # Test slices
    X_lite_test = X_lite[test_idx]
    X_hybrid_test = X_hybrid[test_idx]
    y_true_test = labels[test_idx]
    sources_test = sources[test_idx]
    
    y_teacher_orig_test = y_teacher_orig[test_idx]
    y_teacher_ft_test = y_teacher_ft[test_idx]
    
    # 3. Load Keras Student Models
    print("Loading trained student models...")
    models = {}
    
    lite_orig_path = "outputs/distilled_lite/reddit_mlp_distilled_lite_combined_orig/best_model.keras"
    lite_ft_path = "outputs/distilled_lite/reddit_mlp_distilled_lite_combined_ft/best_model.keras"
    hybrid_orig_path = "outputs/distilled_hybrid_gated/reddit_mlp_distilled_hybrid_gated_combined_orig/best_model.keras"
    hybrid_ft_path = "outputs/distilled_hybrid_gated/reddit_mlp_distilled_hybrid_gated_combined_ft/best_model.keras"
    
    if os.path.exists(lite_orig_path):
        models["Distilled Lite (Variant A - Orig Teacher)"] = (tf.keras.models.load_model(lite_orig_path, safe_mode=False), "lite")
    if os.path.exists(lite_ft_path):
        models["Distilled Lite (Variant B - FT Teacher)"] = (tf.keras.models.load_model(lite_ft_path, safe_mode=False), "lite")
    if os.path.exists(hybrid_orig_path):
        models["Gated Hybrid (Variant A - Orig Teacher)"] = (tf.keras.models.load_model(hybrid_orig_path, safe_mode=False), "hybrid")
    if os.path.exists(hybrid_ft_path):
        models["Gated Hybrid (Variant B - FT Teacher)"] = (tf.keras.models.load_model(hybrid_ft_path, safe_mode=False), "hybrid")
        
    print(f"Loaded {len(models)} student models successfully.")
    
    # 4. Perform Evaluations
    eval_sources = ["all", "thepixel42", "shreya", "ourafla"]
    results = {}
    
    # Define teacher model predictions on the test set
    teacher_preds = {
        "Teacher (Original)": y_teacher_orig_test,
        "Teacher (Fine-Tuned)": y_teacher_ft_test
    }
    
    for src in eval_sources:
        results[src] = {}
        
        # Filter test indices based on source
        if src == "all":
            mask = np.ones(len(test_idx), dtype=bool)
        else:
            mask = (sources_test == src)
            
        y_true_src = y_true_test[mask]
        
        # Evaluate teachers
        for t_name, t_probs in teacher_preds.items():
            acc, f1 = evaluate_predictions(y_true_src, t_probs[mask])
            results[src][t_name] = {"accuracy": acc, "f1": f1}
            
        # Evaluate students
        for s_name, (model, m_type) in models.items():
            X_test_src = X_lite_test[mask] if m_type == "lite" else X_hybrid_test[mask]
            
            # Predict
            pred_probs = model.predict(X_test_src, verbose=0).flatten()
            acc, f1 = evaluate_predictions(y_true_src, pred_probs)
            results[src][s_name] = {"accuracy": acc, "f1": f1}
            
    # 5. Build Markdown Report
    print("Compiling results into evaluation report...")
    
    report = "# Evaluation Report: Combined Dataset Student Distillation\n\n"
    report += "This report compares the performance of the distilled student models (Distilled Lite and Gated Hybrid) trained on the combined dataset using targets from either the original teacher or the fine-tuned teacher.\n\n"
    report += f"* **Combined Dataset Size**: {len(df):,} samples\n"
    report += f"* **Test Split Size**: {len(test_idx):,} samples (20% split, stratified, seed 42)\n\n"
    
    for src in eval_sources:
        src_title = src.upper() if src != "all" else "OVERALL COMBINED TEST SPLIT"
        report += f"## {src_title}\n"
        if src != "all":
            subset_size = np.sum(sources_test == src)
            report += f"*Slice size in test split: {subset_size:,} samples*\n\n"
        else:
            report += "*All source datasets merged.*\n\n"
            
        report += "| Model | Accuracy | F1-Score |\n"
        report += "| :--- | :---: | :---: |\n"
        
        # Sort models so teachers are at the top, then hybrid, then lite
        sorted_model_names = sorted(
            results[src].keys(), 
            key=lambda x: (0 if "Teacher" in x else (1 if "Hybrid" in x else 2), x)
        )
        
        for name in sorted_model_names:
            metrics = results[src][name]
            acc_str = f"{metrics['accuracy']*100:.2f}%"
            f1_str = f"{metrics['f1']:.5f}"
            
            # Highlight best student
            if "Teacher" not in name:
                name_formatted = f"**{name}**"
            else:
                name_formatted = name
                
            report += f"| {name_formatted} | {acc_str} | {f1_str} |\n"
            
        report += "\n"
        
    # Write report file
    with open(OUTPUT_REPORT_PATH, "w") as f:
        f.write(report)
        
    print(f"\n✓ Saved evaluation report to '{OUTPUT_REPORT_PATH}'")
    print("=" * 60)

if __name__ == "__main__":
    main()
