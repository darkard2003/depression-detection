#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

# Directories
LOG_DIR = "/Users/dark/.gemini/antigravity-cli/brain/b4ad9358-6b2e-44ba-92a7-f04eed82bbf9/.system_generated/tasks"
REPORT_DIR = "report"
DATA_DIR = "data_processed/combined"
SEED = 42

def parse_log_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    parts = re.split(r'Training (?:Distilled Lite|Gated Hybrid):', content)
    variant_a_data = []
    variant_b_data = []
    
    pattern = re.compile(
        r'loss:\s*([\d\.]+)\s*-\s*mae:\s*([\d\.]+)\s*-\s*val_loss:\s*([\d\.]+)\s*-\s*val_mae:\s*([\d\.]+)'
    )
    
    if len(parts) > 1:
        # Variant A
        matches_a = pattern.findall(parts[1])
        for m in matches_a:
            variant_a_data.append({
                'loss': float(m[0]),
                'mae': float(m[1]),
                'val_loss': float(m[2]),
                'val_mae': float(m[3])
            })
            
    if len(parts) > 2:
        # Variant B
        matches_b = pattern.findall(parts[2])
        for m in matches_b:
            variant_b_data.append({
                'loss': float(m[0]),
                'mae': float(m[1]),
                'val_loss': float(m[2]),
                'val_mae': float(m[3])
            })
            
    return pd.DataFrame(variant_a_data), pd.DataFrame(variant_b_data)

def plot_learning_curves(df_a, df_b, model_name, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    if not df_a.empty:
        axes[0].plot(df_a['loss'], label='Variant A Train Loss', linestyle='--', color='blue', alpha=0.5)
        axes[0].plot(df_a['val_loss'], label='Variant A Val Loss', linestyle='-', color='blue')
    if not df_b.empty:
        axes[0].plot(df_b['loss'], label='Variant B Train Loss', linestyle='--', color='red', alpha=0.5)
        axes[0].plot(df_b['val_loss'], label='Variant B Val Loss', linestyle='-', color='red')
        
    axes[0].set_title(f'{model_name}: Binary Crossentropy Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()
    
    # MAE plot
    if not df_a.empty:
        axes[1].plot(df_a['mae'], label='Variant A Train MAE', linestyle='--', color='blue', alpha=0.5)
        axes[1].plot(df_a['val_mae'], label='Variant A Val MAE', linestyle='-', color='blue')
    if not df_b.empty:
        axes[1].plot(df_b['mae'], label='Variant B Train MAE', linestyle='--', color='red', alpha=0.5)
        axes[1].plot(df_b['val_mae'], label='Variant B Val MAE', linestyle='-', color='red')
        
    axes[1].set_title(f'{model_name}: Mean Absolute Error')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('MAE')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved learning curves to {output_path}")

def plot_confusion_matrix(y_true, y_pred, title, output_path):
    cm = confusion_matrix(y_true, y_pred)
    # Norm to percent
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    labels = ['Non-Depressed (0)', 'Depressed (1)']
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=labels, 
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    
    # Add percentage label in cells
    for i in range(2):
        for j in range(2):
            plt.text(
                j + 0.5, 
                i + 0.7, 
                f"({cm_percent[i, j]:.1f}%)", 
                ha="center", 
                va="center", 
                color="black" if cm[i, j] < (cm.max() / 2) else "white",
                fontsize=11
            )
            
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel('Actual Category', fontsize=12)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved Confusion Matrix to {output_path}")

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # 1. Generate Learning Curves from logs
    print("=" * 60)
    print("Parsing task execution logs for learning curves...")
    print("=" * 60)
    
    lite_log = os.path.join(LOG_DIR, "task-1879.log")
    gated_log = os.path.join(LOG_DIR, "task-1921.log")
    
    if os.path.exists(lite_log):
        df_lite_a, df_lite_b = parse_log_file(lite_log)
        plot_learning_curves(df_lite_a, df_lite_b, "Distilled Lite", os.path.join(REPORT_DIR, "learning_curves_lite.png"))
    else:
        print(f"Warning: {lite_log} not found.")
        
    if os.path.exists(gated_log):
        df_gated_a, df_gated_b = parse_log_file(gated_log)
        plot_learning_curves(df_gated_a, df_gated_b, "Gated Hybrid", os.path.join(REPORT_DIR, "learning_curves_gated.png"))
    else:
        print(f"Warning: {gated_log} not found.")
        
    # 2. Load Combined Dataset and Models for ROC/PR/Confusion Matrix
    print("\n" + "=" * 60)
    print("Loading datasets and running model evaluations...")
    print("=" * 60)
    
    df = pd.read_csv("datasets/combined_dataset.csv")
    labels = df['label'].astype(int).values
    indices = np.arange(len(df))
    
    _, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=SEED
    )
    
    X_lite = np.load(os.path.join(DATA_DIR, "X_lite_combined.npy"))
    X_hybrid = np.load(os.path.join(DATA_DIR, "X_hybrid_combined.npy"))
    y_true_test = labels[test_idx]
    
    y_teacher_orig = np.load(os.path.join(DATA_DIR, "y_teacher_soft_original.npy"))[test_idx]
    y_teacher_ft = np.load(os.path.join(DATA_DIR, "y_teacher_soft_finetuned.npy"))[test_idx]
    
    print("Loading models...")
    lite_model = tf.keras.models.load_model(
        "outputs/distilled_lite/reddit_mlp_distilled_lite_combined_ft/best_model.keras", 
        safe_mode=False
    )
    gated_model = tf.keras.models.load_model(
        "outputs/distilled_hybrid_gated/reddit_mlp_distilled_hybrid_gated_combined_ft/best_model.keras", 
        safe_mode=False
    )
    
    print("Running predictions...")
    lite_preds = lite_model.predict(X_lite[test_idx], verbose=0).flatten()
    gated_preds = gated_model.predict(X_hybrid[test_idx], verbose=0).flatten()
    
    # Save Confusion Matrices
    plot_confusion_matrix(
        y_true_test, 
        (lite_preds >= 0.5).astype(int), 
        "Confusion Matrix: Distilled Lite (Variant B)", 
        os.path.join(REPORT_DIR, "confusion_matrix_lite.png")
    )
    plot_confusion_matrix(
        y_true_test, 
        (gated_preds >= 0.5).astype(int), 
        "Confusion Matrix: Gated Hybrid (Variant B)", 
        os.path.join(REPORT_DIR, "confusion_matrix_gated.png")
    )
    
    # 3. Plot ROC curves
    print("Plotting ROC Curves...")
    plt.figure(figsize=(8, 6))
    
    for name, preds in [
        ("Teacher (Original)", y_teacher_orig),
        ("Teacher (Fine-Tuned)", y_teacher_ft),
        ("Gated Hybrid (Variant B)", gated_preds),
        ("Distilled Lite (Variant B)", lite_preds)
    ]:
        fpr, tpr, _ = roc_curve(y_true_test, preds)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", lw=2)
        
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve comparison', fontsize=14, pad=15)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "roc_curves.png"), dpi=300)
    plt.close()
    print("✓ Saved ROC curves to report/roc_curves.png")
    
    # 4. Plot Precision-Recall curves
    print("Plotting Precision-Recall Curves...")
    plt.figure(figsize=(8, 6))
    
    for name, preds in [
        ("Teacher (Original)", y_teacher_orig),
        ("Teacher (Fine-Tuned)", y_teacher_ft),
        ("Gated Hybrid (Variant B)", gated_preds),
        ("Distilled Lite (Variant B)", lite_preds)
    ]:
        precision, recall, _ = precision_recall_curve(y_true_test, preds)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.4f})", lw=2)
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
    plt.title('Precision-Recall (PR) Curve comparison', fontsize=14, pad=15)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "pr_curves.png"), dpi=300)
    plt.close()
    print("✓ Saved PR curves to report/pr_curves.png")
    
    print("\n✓ All report graph images successfully generated and saved to 'report/' folder.")
    print("=" * 60)

if __name__ == "__main__":
    main()
