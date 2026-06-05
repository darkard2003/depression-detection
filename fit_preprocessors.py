import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import emoji
from nrclex import NRCLex
from multiprocessing import Pool

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
CSV_FILE = "datasets/bin_reddit1.csv"
SAVE_DIR = "preprocessors"
# ==============================================================================

# Preprocessing helpers
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

# Parallel NRC Emotion Feature Extraction Helper
def extract_single_nrc(text):
    emotion_object = NRCLex(text)
    return emotion_object.raw_emotion_scores

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"=========================================================")
    print(f"Starting Preprocessor Fitting")
    print(f"Saving preprocessors directly to: {SAVE_DIR}")
    print(f"=========================================================")

    # Step 1: Load df
    print(f"Loading dataset {CSV_FILE}...")
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
    df = pd.read_csv(CSV_FILE)

    print("Cleaning text...")
    df['text_clean'] = df['text'].fillna("").apply(clean_text_for_tfidf)

    # Fit and Save TF-IDF Vectorizer
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
    tfidf_vectorizer.fit(df['text_clean'])

    vectorizer_path = os.path.join(SAVE_DIR, "tfidf_vectorizer.pkl")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    print(f"✓ TF-IDF Vectorizer saved to {vectorizer_path}")

    # Parallel NRC Emotion Feature Extraction
    print("Extracting NRC emotion features using multiprocessing...")
    texts = df['text_clean'].tolist()
    with Pool(os.cpu_count() or 4) as pool:
        raw_scores = pool.map(extract_single_nrc, texts)

    print("Converting NRC scores to DataFrame...")
    emotion_cols = ['anticipation', 'joy', 'positive', 'anger', 'fear', 'negative', 'sadness', 'surprise', 'disgust', 'trust']
    nrc_df = pd.DataFrame(raw_scores).reindex(columns=emotion_cols).fillna(0)

    print("Fitting MinMaxScaler on NRC features...")
    scaler = MinMaxScaler()
    scaler.fit(nrc_df)

    scaler_path = os.path.join(SAVE_DIR, "nrc_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✓ NRC MinMaxScaler saved to {scaler_path}")
    print("=========================================================")
    print("🎉 Preprocessing pipeline fully fitted and saved!")
    print("=========================================================")

if __name__ == "__main__":
    main()
