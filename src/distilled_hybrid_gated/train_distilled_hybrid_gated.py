#!/usr/bin/env python3
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import keras_tuner as kt
from sklearn.metrics import classification_report

DATA_DIR = "data_processed/processed_hybrid"
OUTPUT_DIR = "outputs/distilled_hybrid_gated"
PROJECT_NAME = "reddit_mlp_distilled_hybrid_gated"
ALPHA = 0.1
SEED = 42

def build_model(hp):
    # Dynamic dimension detection (384 SBERT features + 1000 TF-IDF features = 1384 dimensions)
    inputs = layers.Input(shape=(INPUT_DIM,))
    
    # Slice the input tensor into SBERT (first 384 dimensions) and TF-IDF (remaining dimensions)
    sbert_in = layers.Lambda(lambda x: x[:, :384])(inputs)
    tfidf_in = layers.Lambda(lambda x: x[:, 384:])(inputs)
    
    # SBERT Sub-network Branch
    sbert_units = hp.Int('sbert_units', 32, 256, step=32)
    sbert_dense = layers.Dense(sbert_units, activation='relu')(sbert_in)
    sbert_dense = layers.Dropout(hp.Float('sbert_dropout', 0.0, 0.5, step=0.1))(sbert_dense)
    
    # TF-IDF Sub-network Branch
    tfidf_units = hp.Int('tfidf_units', 64, 512, step=64)
    tfidf_dense = layers.Dense(tfidf_units, activation='relu')(tfidf_in)
    tfidf_dense = layers.Dropout(hp.Float('tfidf_dropout', 0.0, 0.5, step=0.1))(tfidf_dense)
    
    # Gating Mechanism: predicts a weight between 0 (trust TF-IDF) and 1 (trust SBERT)
    gate_in = layers.concatenate([sbert_dense, tfidf_dense])
    gate = layers.Dense(1, activation='sigmoid', name='gating_weight')(gate_in)
    
    # Projections to common space for weighted summation
    common_dim = hp.Int('common_dim', 32, 128, step=32)
    sbert_proj = layers.Dense(common_dim, activation='relu')(sbert_dense)
    tfidf_proj = layers.Dense(common_dim, activation='relu')(tfidf_dense)
    
    # Weighted element-wise fusion: gate * sbert_proj + (1 - gate) * tfidf_proj
    weighted_sbert = layers.Multiply()([sbert_proj, gate])
    
    one_minus_gate = layers.Lambda(lambda x: 1.0 - x)(gate)
    weighted_tfidf = layers.Multiply()([tfidf_proj, one_minus_gate])
    
    fused = layers.Add()([weighted_sbert, weighted_tfidf])
    fused = layers.Dropout(hp.Float('fused_dropout', 0.0, 0.5, step=0.1))(fused)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(fused)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['mae']
    )
    return model

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading pre-computed hybrid features and targets...")
    X = np.load(os.path.join(DATA_DIR, "X_hybrid.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    y_teacher = np.load(os.path.join(DATA_DIR, "y_teacher_soft.npy"))
    
    # Blended targets: alpha * y_true + (1 - alpha) * y_teacher
    y_blend = ALPHA * y + (1.0 - ALPHA) * y_teacher

    global INPUT_DIM
    INPUT_DIM = X.shape[1]
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    X_train_val, X_test, y_train_val, y_test, y_blend_train_val, y_blend_test = train_test_split(
        X, y, y_blend, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val, y_blend_train, y_blend_val = train_test_split(
        X_train_val, y_train_val, y_blend_train_val, test_size=0.2, stratify=y_train_val, random_state=SEED
    )

    print("Initializing Hyperband search...")
    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=20,
        factor=3,
        directory='outputs/distilled_hybrid_gated/tuning',
        project_name=PROJECT_NAME,
        overwrite=False
    )

    stop_early = EarlyStopping(monitor='val_loss', patience=5)
    
    print("Starting hyperparameter tuning...")
    tuner.search(
        X_train, 
        y_blend_train, 
        epochs=20, 
        validation_data=(X_val, y_blend_val), 
        callbacks=[stop_early]
    )
    
    best_hps = tuner.get_best_hyperparameters(1)[0]
    print("Best Hyperparameters:")
    for param in best_hps.values:
        print(f"  {param}: {best_hps.get(param)}")
        
    print("Retraining the best model on full train-val dataset...")
    best_model = tuner.hypermodel.build(best_hps)
    fit_early = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    best_model.fit(
        X_train_val, 
        y_blend_train_val, 
        epochs=150, 
        validation_data=(X_val, y_val), 
        callbacks=[fit_early], 
        batch_size=64
    )
    
    print("Evaluating on test split...")
    y_pred_probs = best_model.predict(X_test).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    print("\nClassification Report (Hard Targets):")
    print(classification_report(y_test, y_pred))
    
    # Save best student model to its own folder named after PROJECT_NAME
    project_dir = os.path.join(OUTPUT_DIR, PROJECT_NAME)
    os.makedirs(project_dir, exist_ok=True)
    model_save_path = os.path.join(project_dir, "best_model.keras")
    best_model.save(model_save_path)
    print(f"✓ Saved best model to {model_save_path}")

if __name__ == "__main__":
    main()
