#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def main():
    y_true = np.load("data_processed/combined/y_combined.npy")
    y_soft_orig = np.load("data_processed/combined/y_teacher_soft_original.npy")

    print("=" * 60)
    print("Evaluating Pre-Tuned (Original) Teacher on Combined Dataset")
    print("=" * 60)
    y_pred_orig = (y_soft_orig >= 0.5).astype(int)
    print(f"Accuracy: {accuracy_score(y_true, y_pred_orig)*100:.2f}%")
    print(f"F1-Score: {f1_score(y_true, y_pred_orig):.5f}")

    try:
        y_soft_ft = np.load("data_processed/combined/y_teacher_soft_finetuned.npy")
        print("\n" + "=" * 60)
        print("Evaluating Fine-Tuned Teacher on Combined Dataset")
        print("=" * 60)
        y_pred_ft = (y_soft_ft >= 0.5).astype(int)
        print(f"Accuracy: {accuracy_score(y_true, y_pred_ft)*100:.2f}%")
        print(f"F1-Score: {f1_score(y_true, y_pred_ft):.5f}")
        
        diff_acc = (accuracy_score(y_true, y_pred_ft) - accuracy_score(y_true, y_pred_orig)) * 100
        diff_f1 = f1_score(y_true, y_pred_ft) - f1_score(y_true, y_pred_orig)
        print("\n" + "=" * 60)
        print(f"Improvement: Accuracy: +{diff_acc:.2f}%, F1-Score: +{diff_f1:.5f}")
        print("=" * 60)
    except FileNotFoundError:
        print("\n[Pending] Fine-tuned soft targets not generated yet.")

if __name__ == "__main__":
    main()
