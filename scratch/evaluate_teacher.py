#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

CSV_FILE = "datasets/bin_reddit1.csv"
SOFT_LABELS_FILE = "processed_chi2/y_teacher_soft.npy"

def main():
    print("=" * 60)
    print("Evaluating Teacher Model Accuracy on Dataset")
    print("=" * 60)
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
        
    if not os.path.exists(SOFT_LABELS_FILE):
        print(f"Error: {SOFT_LABELS_FILE} not found!")
        print("Please run 'make teacher-labels' first to generate soft targets.")
        sys.exit(1)
        
    # Load true labels
    df = pd.read_csv(CSV_FILE)
    y_true = df['label'].values
    
    # Load teacher predictions
    y_soft = np.load(SOFT_LABELS_FILE)
    
    if len(y_true) != len(y_soft):
        print(f"Error: Row count mismatch! CSV has {len(y_true)} rows, but soft labels file has {len(y_soft)} predictions.")
        sys.exit(1)
        
    # Apply 0.5 threshold
    y_pred = (y_soft >= 0.5).astype(int)
    
    # Calculate metrics
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Total Samples: {total}")
    print(f"Correct Predictions: {correct} / {total} ({accuracy*100:.2f}%)")
    print(f"F1-Score: {f1:.5f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("=" * 60)

if __name__ == "__main__":
    main()
