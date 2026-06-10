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
from textblob import TextBlob

# Configuration
CSV_FILE = "datasets/combined_dataset.csv"
OUTPUT_DATA_DIR = "data_processed/combined"
OUTPUT_MODEL_DIR = "outputs/distilled_lite/combined_assets"
K_FEATURES = 2000

# Text cleaning
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

# Feature extraction for a single post
def extract_single_text_features(text):
    clean_t = clean_text_for_tfidf(text)
    
    # 1. NRClex
    emotion_object = NRCLex(clean_t)
    nrc_scores = emotion_object.raw_emotion_scores
    
    # 2. TextBlob Sentiment
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # 3. Linguistic Counts
    words = clean_t.split()
    word_count = len(words)
    char_count = len(clean_t)
    avg_word_length = char_count / max(1, word_count)
    
    # Pronouns
    singular_pronouns = {"i", "me", "my", "myself", "mine"}
    plural_pronouns = {"we", "us", "our", "ourselves", "ours"}
    sing_count = sum(1 for w in words if w in singular_pronouns)
    plural_count = sum(1 for w in words if w in plural_pronouns)
    
    # Tenses
    past_verbs = {"was", "were", "did", "had", "went", "felt", "thought", "said", "got", "came"}
    future_markers = {"will", "would", "shall", "going", "tomorrow"}
    past_count = sum(1 for w in words if w in past_verbs)
    future_count = sum(1 for w in words if w in future_markers)
    
    # Negations & Absolute Thinking
    negations = {"no", "not", "never", "none", "neither", "nor", "cant", "cannot", "wont", "dont", "isnt", "arent", "wasnt", "werent"}
    absolutes = {"always", "never", "completely", "totally", "absolutely", "forever", "nothing", "everything"}
    neg_count = sum(1 for w in words if w in negations)
    abs_count = sum(1 for w in words if w in absolutes)
    
    # Crisis Keyword Flag
    crisis_keywords = ["kill myself", "end it all", "want to die", "suicide", "self harm", "overdose", "end my life"]
    crisis_flag = 1.0 if any(cw in clean_t for cw in crisis_keywords) else 0.0
    
    # Punctuation Densities
    exclamation_count = text.count('!') / max(1, word_count)
    question_count = text.count('?') / max(1, word_count)
    
    # Densities
    sing_ratio = sing_count / max(1, word_count)
    plural_ratio = plural_count / max(1, word_count)
    past_ratio = past_count / max(1, word_count)
    future_ratio = future_count / max(1, word_count)
    neg_ratio = neg_count / max(1, word_count)
    abs_ratio = abs_count / max(1, word_count)
    
    # Collect features
    features = {
        'anticipation': nrc_scores.get('anticipation', 0.0),
        'joy': nrc_scores.get('joy', 0.0),
        'positive': nrc_scores.get('positive', 0.0),
        'anger': nrc_scores.get('anger', 0.0),
        'fear': nrc_scores.get('fear', 0.0),
        'negative': nrc_scores.get('negative', 0.0),
        'sadness': nrc_scores.get('sadness', 0.0),
        'surprise': nrc_scores.get('surprise', 0.0),
        'disgust': nrc_scores.get('disgust', 0.0),
        'trust': nrc_scores.get('trust', 0.0),
        'polarity': polarity,
        'subjectivity': subjectivity,
        'word_count': float(word_count),
        'avg_word_length': avg_word_length,
        'sing_ratio': sing_ratio,
        'plural_ratio': plural_ratio,
        'past_ratio': past_ratio,
        'future_ratio': future_ratio,
        'neg_ratio': neg_ratio,
        'abs_ratio': abs_ratio,
        'crisis_flag': crisis_flag,
        'exclamation_ratio': exclamation_count,
        'question_ratio': question_count
    }
    return features, clean_t

def main():
    print("=" * 60)
    print("Generating Combined Lite Model Features From Scratch")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
        
    print(f"Loading dataset: {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    raw_texts = df['text'].fillna("").astype(str).tolist()
    labels = df['label'].astype(int).values
    
    # 1. Parallel Lexical Extraction
    print("Extracting emotions, sentiment, and stylistic features in parallel...")
    with Pool(os.cpu_count() or 4) as pool:
        results = pool.map(extract_single_text_features, raw_texts)
        
    features_list = [r[0] for r in results]
    cleaned_texts = [r[1] for r in results]
    
    extra_features_df = pd.DataFrame(features_list)
    print(f"Extracted features shape: {extra_features_df.shape}")
    
    # 2. Fit TF-IDF Vectorizer
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
    
    # 3. Fit MinMaxScaler
    print("Fitting MinMaxScaler on engineered features...")
    feature_scaler = MinMaxScaler()
    X_extra = feature_scaler.fit_transform(extra_features_df)
    
    # 4. Combine TF-IDF and Extra features
    print("Combining TF-IDF and engineered features...")
    X_extra_sparse = X_tfidf.__class__(X_extra)
    X_combined = hstack([X_tfidf, X_extra_sparse]).tocsr()
    print(f"✓ Combined raw features shape: {X_combined.shape}")
    
    # 5. Chi-Square feature selection
    print(f"Selecting top {K_FEATURES} features using Chi-Square...")
    selector = SelectKBest(score_func=chi2, k=K_FEATURES)
    X_selected = selector.fit_transform(X_combined, labels).toarray()
    selected_indices = selector.get_support(indices=True)
    print(f"✓ Selected features shape: {X_selected.shape}")
    
    # 6. Save preprocessors
    print("Saving preprocessor assets...")
    with open(os.path.join(OUTPUT_MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(os.path.join(OUTPUT_MODEL_DIR, "feature_scaler.pkl"), "wb") as f:
        pickle.dump(feature_scaler, f)
    np.save(os.path.join(OUTPUT_MODEL_DIR, "selected_chi2_indices.npy"), selected_indices)
    
    # 7. Save output features
    np.save(os.path.join(OUTPUT_DATA_DIR, "X_lite_combined.npy"), X_selected)
    np.save(os.path.join(OUTPUT_DATA_DIR, "y_combined.npy"), labels)
    print(f"✓ Saved Lite features to '{os.path.join(OUTPUT_DATA_DIR, 'X_lite_combined.npy')}'")
    print("=" * 60)

if __name__ == "__main__":
    main()
