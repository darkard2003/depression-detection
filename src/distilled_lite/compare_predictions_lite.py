#!/usr/bin/env python3
import os
import re
import sys
import time
import emoji
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nrclex import NRCLex
from textblob import TextBlob

CSV_FILE = "datasets/bin_reddit1.csv"
STUDENT_MODEL_PATH = "outputs/distilled_lite/reddit_mlp_distilled_lite/best_model.keras"
TEACHER_MODEL_NAME = "TRT1000/depression-detection-model"

# Lite Model Preprocessor Paths
VECTORIZER_PATH = "outputs/distilled_lite/reddit_mlp_distilled_lite/tfidf_vectorizer.pkl"
SCALER_PATH = "outputs/distilled_lite/reddit_mlp_distilled_lite/feature_scaler.pkl"
INDICES_PATH = "outputs/distilled_lite/reddit_mlp_distilled_lite/selected_chi2_indices.npy"
BATCH_SIZE = 64

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

# Parallel Helper for NRC & TextBlob & Stylistic Extraction
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
    
    # Densities (normalise by word count)
    sing_ratio = sing_count / max(1, word_count)
    plural_ratio = plural_count / max(1, word_count)
    past_ratio = past_count / max(1, word_count)
    future_ratio = future_count / max(1, word_count)
    neg_ratio = neg_count / max(1, word_count)
    abs_ratio = abs_count / max(1, word_count)
    
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
    print("Evaluating Lite Student Model Fidelity to Teacher")
    print("=" * 60)
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
        
    if not os.path.exists(STUDENT_MODEL_PATH):
        print(f"Error: Student model {STUDENT_MODEL_PATH} not found. Please train first.")
        sys.exit(1)
        
    if not os.path.exists(VECTORIZER_PATH):
        print(f"Error: TF-IDF vectorizer {VECTORIZER_PATH} not found.")
        sys.exit(1)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset: {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    raw_texts = df['text'].fillna("").astype(str).tolist()
    total_samples = len(raw_texts)

    # 1. Generate Teacher predictions
    print(f"\n[1/3] Loading Teacher model: {TEACHER_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL_NAME).to(device)
    teacher_model.eval()
    
    y_teacher_probs = []
    print("Running batch inference for Teacher...")
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="Teacher Inference"):
            batch = raw_texts[i : i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            outputs = teacher_model(**inputs)
            logits = outputs.logits
            if logits.shape[1] == 2:
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            else:
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            y_teacher_probs.extend(probs)
            
    y_teacher_probs = np.array(y_teacher_probs)
    y_teacher_pred = (y_teacher_probs >= 0.5).astype(int)

    # 2. Process features for Student
    print(f"\n[2/3] Processing text features for Lite Student model...")
    start_feat_time = time.time()
    
    with Pool(os.cpu_count() or 4) as pool:
        results = pool.map(extract_single_text_features, raw_texts)
        
    features_list = [r[0] for r in results]
    cleaned_texts = [r[1] for r in results]
    extra_features_df = pd.DataFrame(features_list)
    
    print("Loading vectorizer, scaler, and selected indices...")
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    selected_indices = np.load(INDICES_PATH)
        
    X_tfidf_all = vectorizer.transform(cleaned_texts)
    X_extra = scaler.transform(extra_features_df)
    
    # Combine TF-IDF and Extra features
    X_extra_sparse = X_tfidf_all.__class__(X_extra)
    from scipy.sparse import hstack
    X_combined = hstack([X_tfidf_all, X_extra_sparse]).tocsr()
    
    X_lite = X_combined[:, selected_indices].toarray()
    feat_time = time.time() - start_feat_time
    print(f"✓ Feature processing completed in {feat_time:.2f}s (Average: {feat_time*1000/total_samples:.4f} ms/sample)")

    print(f"Loading Student model: {STUDENT_MODEL_PATH} (safe_mode=True)...")
    student_model = tf.keras.models.load_model(STUDENT_MODEL_PATH)
    
    print("Running Student predictions...")
    start_pred_time = time.time()
    y_student_probs = []
    for i in range(0, total_samples, BATCH_SIZE):
        batch_x = X_lite[i : i + BATCH_SIZE]
        probs = student_model.predict(batch_x, verbose=0).flatten()
        y_student_probs.extend(probs)
        
    y_student_probs = np.array(y_student_probs)
    y_student_pred = (y_student_probs >= 0.5).astype(int)
    pred_time = time.time() - start_pred_time
    print(f"✓ Prediction completed in {pred_time:.2f}s (Average: {pred_time*1000/total_samples:.4f} ms/sample)")

    # 3. Evaluate Fidelity (Student vs Teacher)
    print(f"\n[3/3] Evaluating Student Fidelity to Teacher...")
    accuracy = accuracy_score(y_teacher_pred, y_student_pred)
    f1 = f1_score(y_teacher_pred, y_student_pred)
    
    print("\n" + "=" * 60)
    print("FIDELITY METRICS (Teacher = Ground Truth, Student = Prediction)")
    print("=" * 60)
    print(f"Total Samples: {total_samples}")
    print(f"Fidelity Agreement: {np.sum(y_teacher_pred == y_student_pred)} / {total_samples} ({accuracy*100:.2f}%)")
    print(f"Fidelity F1-Score: {f1:.5f}")
    print("\nClassification Report (Fidelity):")
    print(classification_report(y_teacher_pred, y_student_pred))
    print(f"Total Inference Latency (Extract + Pred): {(feat_time + pred_time)*1000/total_samples:.4f} ms/sample")
    print("=" * 60)

if __name__ == "__main__":
    main()
