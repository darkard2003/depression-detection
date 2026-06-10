#!/usr/bin/env python3
import os
import re
import sys
import emoji
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import hstack
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from nrclex import NRCLex

# Configuration
CSV_FILE = "datasets/thepixel42_depression-detection.csv"
TEACHER_LABELS_PATH = "data_processed/processed_chi2/y_teacher_soft.npy"
OUTPUT_DATA_DIR = "data_processed/processed_hybrid_gated_ft"
OUTPUT_MODEL_DIR = "outputs/distilled_hybrid_gated_ft/reddit_mlp_distilled_hybrid_gated_ft"
K_FEATURES = 1000

# --- Text Cleaning functions placed directly here for zero dependency ---
def process_hashtags(text):
    if not isinstance(text, str):
        return ""
    hashtags = re.findall(r'#(\w+)', text)
    for hashtag in hashtags:
        processed = hashtag
        processed = processed.replace('_', ' ')
        processed = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', processed)
        processed = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', processed)
        processed = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', processed)
        processed = re.sub(r'(?<=\d)(?=[a-zA-Z])', ' ', processed)
        processed = ' '.join(processed.lower().split())
        text = text.replace(f'#{hashtag}', processed)
    return text

def clean_text_for_tfidf(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = process_hashtags(text)
    return text

def extract_single_nrc(text):
    emotion_object = NRCLex(text)
    return emotion_object.raw_emotion_scores

def main():
    print("=" * 60)
    print("Generating Upgraded Gated FT features from raw dataset")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
        
    if not os.path.exists(TEACHER_LABELS_PATH):
        print(f"Error: Teacher soft labels '{TEACHER_LABELS_PATH}' not found!")
        sys.exit(1)
        
    # Load dataset
    print(f"Loading dataset: {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    raw_texts = df['text'].fillna("").astype(str).tolist()
    labels = df['label'].astype(int).values
    total_samples = len(raw_texts)
    
    # 1. Clean texts
    print("Cleaning texts...")
    cleaned_texts = [clean_text_for_tfidf(t) for t in tqdm(raw_texts, desc="Text Cleaning")]
    
    # 2. Fit TF-IDF Vectorizer (5000 max features)
    print("Fitting TF-IDF Vectorizer...")
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
    X_tfidf = tfidf_vectorizer.fit_transform(cleaned_texts)
    print(f"✓ TF-IDF features shape: {X_tfidf.shape}")
    
    # 3. Parallel NRC Emotion feature extraction
    print("Extracting NRC emotion features in parallel...")
    with Pool(os.cpu_count() or 4) as pool:
        raw_scores = pool.map(extract_single_nrc, cleaned_texts)
        
    emotion_cols = ['anticipation', 'joy', 'positive', 'anger', 'fear', 'negative', 'sadness', 'surprise', 'disgust', 'trust']
    nrc_df = pd.DataFrame(raw_scores).reindex(columns=emotion_cols).fillna(0)
    
    print("Fitting NRC MinMaxScaler...")
    nrc_scaler = MinMaxScaler()
    X_nrc = nrc_scaler.fit_transform(nrc_df)
    print(f"✓ NRC features shape: {X_nrc.shape}")
    
    # 4. Combine TF-IDF and NRC features
    print("Combining TF-IDF and NRC features...")
    # Convert dense NRC array to sparse matrix to match TF-IDF format for hstack
    X_nrc_sparse = X_tfidf.__class__(X_nrc)
    X_combined = hstack([X_tfidf, X_nrc_sparse]).tocsr()
    print(f"✓ Combined raw features shape: {X_combined.shape}")
    
    # 5. Chi-Square feature selection
    print(f"Selecting top {K_FEATURES} features using Chi-Square...")
    selector = SelectKBest(score_func=chi2, k=K_FEATURES)
    X_selected = selector.fit_transform(X_combined, labels).toarray()
    selected_indices = selector.get_support(indices=True)
    print(f"✓ Selected features shape: {X_selected.shape}")
    
    # 6. Save preprocessors
    print("Saving fitted preprocessors...")
    with open(os.path.join(OUTPUT_MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(os.path.join(OUTPUT_MODEL_DIR, "nrc_scaler.pkl"), "wb") as f:
        pickle.dump(nrc_scaler, f)
    np.save(os.path.join(OUTPUT_MODEL_DIR, "selected_chi2_indices.npy"), selected_indices)
    
    # 7. Save output features and target labels
    print("Saving processed features and targets...")
    np.save(os.path.join(OUTPUT_DATA_DIR, "X_tfidf_selected.npy"), X_selected)
    np.save(os.path.join(OUTPUT_DATA_DIR, "y.npy"), labels)
    
    # Copy teacher soft labels
    print("Copying teacher soft labels...")
    y_teacher = np.load(TEACHER_LABELS_PATH)
    np.save(os.path.join(OUTPUT_DATA_DIR, "y_teacher_soft.npy"), y_teacher)
    
    print("=" * 60)
    print("🎉 Upgraded hybrid features generated successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
