#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

DATA_DIR = "data_processed/combined"
OUTPUT_DIR = "src/distilled_hybrid_gated"
MODEL_SAVE_DIR = "outputs/distilled_hybrid_gated/reddit_mlp_distilled_hybrid_gated_combined_ft"
SEED = 42

def build_gated_model(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    
    # Slice inputs into SBERT (first 384 dimensions) and TF-IDF (remaining dimensions)
    sbert_in = layers.Lambda(lambda x: x[:, :384])(inputs)
    tfidf_in = layers.Lambda(lambda x: x[:, 384:])(inputs)
    
    # SBERT branch
    sbert_dense = layers.Dense(128, activation='relu')(sbert_in)
    sbert_dense = layers.Dropout(0.2)(sbert_dense)
    
    # TF-IDF branch
    tfidf_dense = layers.Dense(256, activation='relu')(tfidf_in)
    tfidf_dense = layers.Dropout(0.2)(tfidf_dense)
    
    # Gate weight calculation
    gate_in = layers.concatenate([sbert_dense, tfidf_dense])
    gate = layers.Dense(1, activation='sigmoid', name='gating_weight')(gate_in)
    
    # Projection to common dimension
    common_dim = 64
    sbert_proj = layers.Dense(common_dim, activation='relu')(sbert_dense)
    tfidf_proj = layers.Dense(common_dim, activation='relu')(tfidf_dense)
    
    # Weighted combination
    weighted_sbert = layers.Multiply()([sbert_proj, gate])
    one_minus_gate = layers.Lambda(lambda x: 1.0 - x)(gate)
    weighted_tfidf = layers.Multiply()([tfidf_proj, one_minus_gate])
    
    fused = layers.Add()([weighted_sbert, weighted_tfidf])
    fused = layers.Dropout(0.2)(fused)
    
    outputs = layers.Dense(1, activation='sigmoid')(fused)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['mae']
    )
    return model

def main():
    print("=" * 60)
    print("Retraining Gated Hybrid (Variant B) and Generating Report Plots")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    X_path = os.path.join(DATA_DIR, "X_hybrid_combined.npy")
    y_hard_path = os.path.join(DATA_DIR, "y_combined.npy")
    y_soft_path = os.path.join(DATA_DIR, "y_teacher_soft_finetuned.npy")
    
    if not (os.path.exists(X_path) and os.path.exists(y_hard_path) and os.path.exists(y_soft_path)):
        print("Error: Missing required numpy feature/target matrices in data_processed/combined!")
        sys.exit(1)
        
    X = np.load(X_path)
    y_hard = np.load(y_hard_path)
    y_soft = np.load(y_soft_path)
    
    # Train-test split (80/20)
    X_train_val, X_test, y_hard_train_val, y_hard_test, y_soft_train_val, y_soft_test = train_test_split(
        X, y_hard, y_soft, test_size=0.2, stratify=y_hard, random_state=SEED
    )
    
    # Train-val split (80/20 of train_val)
    X_train, X_val, y_hard_train, y_hard_val, y_soft_train, y_soft_val = train_test_split(
        X_train_val, y_hard_train_val, y_soft_train_val, test_size=0.2, stratify=y_hard_train_val, random_state=SEED
    )
    
    model = build_gated_model(X.shape[1])
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    print("Starting training...")
    history = model.fit(
        X_train, 
        y_soft_train, 
        epochs=100, 
        validation_data=(X_val, y_soft_val), 
        batch_size=128, 
        callbacks=[early_stop]
    )
    
    # Save the model
    model_save_path = os.path.join(MODEL_SAVE_DIR, "best_model.keras")
    model.save(model_save_path)
    print(f"✓ Saved retrained model to {model_save_path}")
    
    # 1. Plot Learning Curves
    print("Generating learning curves...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], color='black', linestyle='-', label='Train Loss')
    plt.plot(history.history['val_loss'], color='dimgray', linestyle='--', label='Val Loss')
    plt.title('Gated Hybrid: Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Binary Crossentropy')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], color='black', linestyle='-', label='Train MAE')
    plt.plot(history.history['val_mae'], color='dimgray', linestyle='--', label='Val MAE')
    plt.title('Gated Hybrid: MAE Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    learning_curve_path = os.path.join(OUTPUT_DIR, "learning_curve_gated.png")
    plt.savefig(learning_curve_path, dpi=300)
    plt.close()
    print(f"✓ Saved learning curve to {learning_curve_path}")
    
    # Run test predictions
    y_pred_probs = model.predict(X_test).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    # Print classification report
    print("\nClassification Report (Test Set - Hard Targets):")
    print(classification_report(y_hard_test, y_pred))
    
    # 2. Plot ROC and PR Curves
    print("Generating ROC and Precision-Recall curves...")
    fpr, tpr, _ = roc_curve(y_hard_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_hard_test, y_pred_probs)
    ap = average_precision_score(y_hard_test, y_pred_probs)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='black', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='darkgray', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Gated Hybrid: ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='black', lw=2, label=f'PR Curve (AP = {ap:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Gated Hybrid: Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    roc_pr_path = os.path.join(OUTPUT_DIR, "roc_pr_gated.png")
    plt.savefig(roc_pr_path, dpi=300)
    plt.close()
    print(f"✓ Saved ROC/PR curves to {roc_pr_path}")
    
    # 3. Plot Confusion Matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_hard_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', cbar=False,
                xticklabels=['Non-Depressed', 'Depressed'],
                yticklabels=['Non-Depressed', 'Depressed'],
                linecolor='black', linewidths=0.5)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Gated Hybrid: Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_gated.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"✓ Saved confusion matrix to {cm_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
