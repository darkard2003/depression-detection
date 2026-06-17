#!/usr/bin/env python3
import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import shap

DATA_DIR = "data_processed/combined"
OUTPUT_DIR = "src/distilled_hybrid_gated"
MODEL_SAVE_DIR = "outputs/distilled_hybrid_gated/reddit_mlp_distilled_hybrid_gated_combined_ft"
SEED = 42

def main():
    print("=" * 60)
    print("Running SHAP Explainability for Gated Hybrid")
    print("=" * 60)
    
    # 1. Load model
    model_path = os.path.join(MODEL_SAVE_DIR, "best_model.keras")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}! Run training first.")
        sys.exit(1)
        
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path, safe_mode=False)
    
    # 2. Load data
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
    
    # 3. Load and map feature names
    print("Loading feature names mapping...")
    try:
        with open("outputs/distilled_hybrid_gated/combined_assets/tfidf_vectorizer.pkl", "rb") as f:
            tfidf_vectorizer = pickle.load(f)
            
        sbert_names = [f"SBERT_dim_{i}" for i in range(384)]
        tfidf_names = list(tfidf_vectorizer.get_feature_names_out())
        hybrid_feature_names = np.array(sbert_names + tfidf_names)
    except Exception as e:
        print(f"Error loading feature names: {e}")
        sys.exit(1)
        
    # 4. Generate SHAP Explainability Plot
    print("Generating SHAP feature importances...")
    try:
        # Sample background and explain data
        np.random.seed(SEED)
        bg_idx = np.random.choice(X_train.shape[0], size=min(100, X_train.shape[0]), replace=False)
        X_background = X_train[bg_idx].astype(np.float32)
        
        explain_idx = np.random.choice(X_test.shape[0], size=min(200, X_test.shape[0]), replace=False)
        X_explain = X_test[explain_idx].astype(np.float32)
        
        explainer = shap.DeepExplainer(model, X_background)
        shap_values = explainer.shap_values(X_explain)
        
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values
        sv = np.squeeze(sv)
        
        # Plot bar chart for feature importances
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, features=X_explain, feature_names=hybrid_feature_names, plot_type='bar', max_display=20, show=False)
        plt.title('Gated Hybrid: SHAP Feature Importance (Top 20)')
        plt.tight_layout()
        shap_path = os.path.join(OUTPUT_DIR, "xai_shap_gated.png")
        plt.savefig(shap_path, dpi=300)
        plt.close()
        print(f"✓ Saved SHAP feature importance plot to {shap_path}")
    except Exception as e:
        print(f"Error generating SHAP plot: {e}")
        import traceback
        traceback.print_exc()
        
    print("=" * 60)

if __name__ == "__main__":
    main()
