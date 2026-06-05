#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

# Configuration
CSV_FILE = "datasets/thepixel42_depression-detection.csv"
SAVE_DIR = "data_processed/processed_chi2"
PREPROCESSORS_DIR = "outputs/tfidf_mlp"
K_FEATURES = 5000
SEED = 42

from src.utils.text_cleaning import clean_text_for_tfidf

def main():
    print("=" * 60)
    print(f"Preprocessing & Feature Selection for {CSV_FILE}")
    print("=" * 60)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(PREPROCESSORS_DIR, exist_ok=True)
    
    # 1. Load dataset
    print(f"Loading dataset: {CSV_FILE}...")
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
        
    df = pd.read_csv(CSV_FILE)
    labels = df['label'].astype(int).values
    
    # 2. Clean text
    print("Cleaning texts...")
    df['text_clean'] = df['text'].fillna("").apply(clean_text_for_tfidf)
    texts_cleaned = df['text_clean'].tolist()
    
    # 3. Fit TF-IDF Vectorizer
    print("Fitting TF-IDF Vectorizer (5000 max features)...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english',
        token_pattern=r'\b\w+\b',
        strip_accents='unicode',
        lowercase=True,
        sublinear_tf=True
    )
    X_tfidf = tfidf_vectorizer.fit_transform(texts_cleaned)
    print(f"✓ TF-IDF fitted. Shape: {X_tfidf.shape}")
    
    # Save vectorizer
    vectorizer_path = os.path.join(PREPROCESSORS_DIR, "tfidf_vectorizer.pkl")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    print(f"✓ Saved vectorizer to '{vectorizer_path}'")
    
    # 4. Chi-Square feature selection
    print(f"Selecting top {K_FEATURES} features using Chi-Square...")
    selector = SelectKBest(score_func=chi2, k=K_FEATURES)
    X_tfidf_selected = selector.fit_transform(X_tfidf, labels)
    
    # Get and save indices
    selected_indices = selector.get_support(indices=True)
    indices_path = os.path.join(PREPROCESSORS_DIR, 'selected_tfidf_indices.npy')
    np.save(indices_path, selected_indices)
    print(f"✓ Saved selected indices to '{indices_path}'")
    
    # 5. Extract and save feature names
    vocab = tfidf_vectorizer.get_feature_names_out()
    selected_words = [vocab[idx] for idx in selected_indices]
    
    feature_names_path = os.path.join(PREPROCESSORS_DIR, 'feature_names_chi2.txt')
    with open(feature_names_path, 'w', encoding='utf-8') as f:
        for word in selected_words:
            f.write(f"{word}\n")
    print(f"✓ Saved feature name list to '{feature_names_path}'")
    
    # 6. Save final sparse matrix and labels to processed_chi2/
    X_combined_path = os.path.join(SAVE_DIR, 'X_combined_sparse.npz')
    y_out_path = os.path.join(SAVE_DIR, 'y.npy')
    
    save_npz(X_combined_path, X_tfidf_selected)
    np.save(y_out_path, labels)
    
    print(f"✓ Saved combined sparse matrix ({X_tfidf_selected.shape}) to '{X_combined_path}'")
    print(f"✓ Saved labels to '{y_out_path}'")
    print("=" * 60)
    print("🎉 Preprocessing and feature selection finished successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()
