#!/usr/bin/env python3
import os
import pickle
import numpy as np
from scipy.sparse import load_npz, save_npz, hstack
from sklearn.feature_selection import SelectKBest, chi2

# Configuration
DATA_DIR = 'data_processed/processed_dirty'
SAVE_DIR = 'data_processed/processed_chi2'
PREPROCESSORS_DIR = 'preprocessors'
K_FEATURES = 1000

def main():
    print("=" * 60)
    print("Starting Chi-Square Feature Selection & Preprocessing")
    print("=" * 60)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Load TF-IDF features, scaled NRC features, and labels
    print(f"Loading features from '{DATA_DIR}'...")
    X_tfidf_path = os.path.join(DATA_DIR, 'X_tfidf.npz')
    X_nrc_path = os.path.join(DATA_DIR, 'X_nrc_features_scaled.npy')
    y_path = os.path.join(DATA_DIR, 'y.npy')
    
    X_tfidf = load_npz(X_tfidf_path)
    X_nrc = np.load(X_nrc_path)
    y = np.load(y_path)
    
    print(f"Loaded TF-IDF shape: {X_tfidf.shape}")
    print(f"Loaded NRC shape: {X_nrc.shape}")
    print(f"Loaded labels shape: {y.shape}")
    
    # 2. Perform Chi-Square selection on TF-IDF features
    print(f"Selecting top {K_FEATURES} TF-IDF features using Chi-Square...")
    selector = SelectKBest(score_func=chi2, k=K_FEATURES)
    X_tfidf_selected = selector.fit_transform(X_tfidf, y)
    
    # Get indices of selected features
    selected_indices = selector.get_support(indices=True)
    indices_path = os.path.join(PREPROCESSORS_DIR, 'selected_tfidf_indices.npy')
    np.save(indices_path, selected_indices)
    print(f"✓ Saved selected TF-IDF indices to '{indices_path}'")
    
    # 3. Load TF-IDF vectorizer vocabulary & map to words
    vectorizer_path = os.path.join(PREPROCESSORS_DIR, 'tfidf_vectorizer.pkl')
    print(f"Loading TF-IDF vectorizer from '{vectorizer_path}'...")
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    tfidf_words = vectorizer.get_feature_names_out()
    selected_words = [tfidf_words[idx] for idx in selected_indices]
    
    # NRC emotion column names (matching order used in fit_preprocessors.py)
    nrc_emotions = ['anticipation', 'joy', 'positive', 'anger', 'fear', 'negative', 'sadness', 'surprise', 'disgust', 'trust']
    
    # Combine feature names
    feature_names = selected_words + nrc_emotions
    
    # Save feature names to text file
    feature_names_path = os.path.join(PREPROCESSORS_DIR, 'feature_names_chi2.txt')
    with open(feature_names_path, 'w', encoding='utf-8') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    print(f"✓ Saved feature name list to '{feature_names_path}'")
    
    # 4. Combine selected TF-IDF features and scaled NRC features
    print("Combining selected TF-IDF and NRC features...")
    # Convert NRC array to sparse matrix to allow hstack
    X_nrc_sparse = X_tfidf_selected.__class__(X_nrc)
    X_combined_sparse = hstack([X_tfidf_selected, X_nrc_sparse]).tocsr()
    
    # 5. Save output files to new processed directory
    X_combined_path = os.path.join(SAVE_DIR, 'X_combined_sparse.npz')
    y_out_path = os.path.join(SAVE_DIR, 'y.npy')
    
    save_npz(X_combined_path, X_combined_sparse)
    np.save(y_out_path, y)
    
    print(f"✓ Saved combined sparse matrix ({X_combined_sparse.shape}) to '{X_combined_path}'")
    print(f"✓ Saved labels copy to '{y_out_path}'")
    print("=" * 60)
    print("🎉 Feature selection and preprocessing finished successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()
