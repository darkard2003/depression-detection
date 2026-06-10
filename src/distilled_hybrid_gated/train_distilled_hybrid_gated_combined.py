#!/usr/bin/env python3
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_DIR = "data_processed/combined"
OUTPUT_DIR = "outputs/distilled_hybrid_gated"
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

def train_variant(variant_name, X, y_hard, y_soft, output_subdir):
    print("\n" + "=" * 50)
    print(f"Training Gated Hybrid: {variant_name}")
    print("=" * 50)
    
    # Train-test split
    X_train_val, X_test, y_hard_train_val, y_hard_test, y_soft_train_val, y_soft_test = train_test_split(
        X, y_hard, y_soft, test_size=0.2, stratify=y_hard, random_state=SEED
    )
    
    # Train-val split
    X_train, X_val, y_hard_train, y_hard_val, y_soft_train, y_soft_val = train_test_split(
        X_train_val, y_hard_train_val, y_soft_train_val, test_size=0.2, stratify=y_hard_train_val, random_state=SEED
    )
    
    print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")
    
    model = build_gated_model(X.shape[1])
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, 
        y_soft_train, 
        epochs=100, 
        validation_data=(X_val, y_soft_val), 
        batch_size=128, 
        callbacks=[early_stop]
    )
    
    # Save the model
    os.makedirs(output_subdir, exist_ok=True)
    model_save_path = os.path.join(output_subdir, "best_model.keras")
    model.save(model_save_path)
    print(f"✓ Saved best model to {model_save_path}")
    
    # Evaluate against hard targets on the test set
    y_pred_probs = model.predict(X_test).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    print(f"\nClassification Report (Test Set - Hard Targets) for {variant_name}:")
    print(classification_report(y_hard_test, y_pred))
    
    return model

def main():
    print("Loading features and targets...")
    X_path = os.path.join(DATA_DIR, "X_hybrid_combined.npy")
    y_hard_path = os.path.join(DATA_DIR, "y_combined.npy")
    y_soft_orig_path = os.path.join(DATA_DIR, "y_teacher_soft_original.npy")
    y_soft_ft_path = os.path.join(DATA_DIR, "y_teacher_soft_finetuned.npy")
    
    if not (os.path.exists(X_path) and os.path.exists(y_hard_path)):
        print("Error: Hybrid features not generated yet!")
        sys.exit(1)
        
    X = np.load(X_path)
    y_hard = np.load(y_hard_path)
    
    # Train Variant A (Original Teacher)
    if os.path.exists(y_soft_orig_path):
        y_soft_orig = np.load(y_soft_orig_path)
        train_variant(
            variant_name="Variant A (Original Teacher Targets)",
            X=X,
            y_hard=y_hard,
            y_soft=y_soft_orig,
            output_subdir=os.path.join(OUTPUT_DIR, "reddit_mlp_distilled_hybrid_gated_combined_orig")
        )
    else:
        print(f"Warning: Original teacher soft targets '{y_soft_orig_path}' not found, skipping Variant A.")
        
    # Train Variant B (Fine-Tuned Teacher)
    if os.path.exists(y_soft_ft_path):
        y_soft_ft = np.load(y_soft_ft_path)
        train_variant(
            variant_name="Variant B (Fine-Tuned Teacher Targets)",
            X=X,
            y_hard=y_hard,
            y_soft=y_soft_ft,
            output_subdir=os.path.join(OUTPUT_DIR, "reddit_mlp_distilled_hybrid_gated_combined_ft")
        )
    else:
        print(f"Warning: Fine-tuned teacher soft targets '{y_soft_ft_path}' not found, skipping Variant B.")

if __name__ == "__main__":
    main()
