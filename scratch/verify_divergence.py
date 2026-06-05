#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd

CSV_FILE = "datasets/bin_reddit1.csv"
SOFT_LABELS_FILE = "processed_chi2/y_teacher_soft.npy"
TRUE_LABELS_FILE = "processed_chi2/y.npy"
OUTPUT_CSV = "processed_chi2/teacher_divergence.csv"

def main():
    print("=" * 60)
    print("Extracting Teacher Model Mistakes & Divergences")
    print("=" * 60)
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
        
    if not os.path.exists(SOFT_LABELS_FILE):
        print(f"Error: {SOFT_LABELS_FILE} not found! Run 'make teacher-labels' first.")
        sys.exit(1)
        
    # Load dataset, true labels, and teacher soft predictions
    df = pd.read_csv(CSV_FILE)
    y_true = np.load(TRUE_LABELS_FILE)
    y_soft = np.load(SOFT_LABELS_FILE)
    
    # Calculate binary prediction
    y_pred = (y_soft >= 0.5).astype(int)
    
    # Filter only mistakes
    errors_mask = y_true != y_pred
    error_indices = np.where(errors_mask)[0]
    
    if len(error_indices) == 0:
        print("Wow! Model got 100% accuracy. No mistakes found.")
        sys.exit(0)
        
    # Build dataframe of errors
    errors_df = pd.DataFrame({
        'original_index': error_indices,
        'text': df.iloc[error_indices]['text'].values,
        'true_label': y_true[error_indices].astype(int),
        'predicted_prob': y_soft[error_indices],
        'error_margin': np.abs(y_true[error_indices] - y_soft[error_indices])
    })
    
    # Sort by error_margin descending (most confident mistakes first)
    errors_df = errors_df.sort_values(by='error_margin', ascending=False)
    
    # Save top 100 to CSV
    errors_df.head(100).to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Top 100 confident errors exported to '{OUTPUT_CSV}'")
    
    # Print top 5 False Positives (True = 0, Pred Prob close to 1)
    fps = errors_df[errors_df['true_label'] == 0].head(5)
    print("\n" + "-"*40)
    print("TOP 5 CONFIDENT FALSE POSITIVES (True: Normal, Predict: Depression)")
    print("-"*40)
    for idx, row in fps.iterrows():
        print(f"\n[Index {row['original_index']}] Prob: {row['predicted_prob']:.4f}")
        print(f"Text snippet: {str(row['text'])[:300]}...")
        
    # Print top 5 False Negatives (True = 1, Pred Prob close to 0)
    fns = errors_df[errors_df['true_label'] == 1].head(5)
    print("\n" + "-"*40)
    print("TOP 5 CONFIDENT FALSE NEGATIVES (True: Depression, Predict: Normal)")
    print("-"*40)
    for idx, row in fns.iterrows():
        print(f"\n[Index {row['original_index']}] Prob: {row['predicted_prob']:.4f}")
        print(f"Text snippet: {str(row['text'])[:300]}...")
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
