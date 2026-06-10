#!/usr/bin/env python3
import os
import re
import sys
import emoji
import pickle
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

# Configuration
CSV_FILE = "datasets/thepixel42_depression-detection.csv"
TEACHER_LABELS_PATH = "data_processed/processed_chi2/y_teacher_soft.npy"
OUTPUT_DATA_DIR = "data_processed/processed_hybrid_augmented"
OUTPUT_MODEL_DIR = "outputs/distilled_hybrid_augmented/reddit_mlp_distilled_hybrid_augmented"
BATCH_SIZE = 256
DEGRADE_PROB = 0.5  # Degrade 50% of training texts

# Rule-based word stemmer/lemmatizer to avoid nltk downloads
def simple_lemmatize(word):
    if len(word) <= 3:
        return word
    # Common verb/noun endings
    if word.endswith("ing"):
        return word[:-3]
    if word.endswith("ed"):
        return word[:-2]
    if word.endswith("es"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    # Irregular forms
    irregulars = {
        "am": "be", "is": "be", "are": "be", "was": "be", "were": "be", "been": "be",
        "has": "have", "had": "have", "does": "do", "did": "do", "done": "do",
        "goes": "go", "went": "go", "gone": "go", "came": "come", "got": "get",
        "feeling": "feel", "felt": "feel", "thinking": "think", "thought": "think"
    }
    return irregulars.get(word, word)

def degrade_text(text, prob=DEGRADE_PROB):
    if not isinstance(text, str) or not text:
        return ""
    if random.random() > prob:
        return text
        
    words = text.split()
    degraded_words = []
    
    # Grammatical stopwords to remove
    stopwords_to_remove = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for", "on", "with", "at", "by", "and", "but", "or", "it", "this", "that"}
    
    for w in words:
        w_clean = re.sub(r'\W+', '', w).lower()
        if w_clean in stopwords_to_remove and random.random() < 0.6:
            continue
        w_degraded = simple_lemmatize(w_clean) if w_clean else w
        degraded_words.append(w_degraded)
        
    return " ".join(degraded_words)

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

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    print("=" * 60)
    print("Generating Augmented Hybrid Features From Scratch")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
        
    if not os.path.exists(TEACHER_LABELS_PATH):
        print(f"Error: Teacher soft labels '{TEACHER_LABELS_PATH}' not found!")
        print("Please run the teacher labels generation first: 'make process-data-teacher-labels'")
        sys.exit(1)
        
    # Load dataset
    print(f"Loading dataset: {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    raw_texts = df['text'].fillna("").astype(str).tolist()
    labels = df['label'].astype(int).values
    total_samples = len(raw_texts)
    
    # 1. Degrade syntax for SBERT training robustness
    print("\n[1/3] Applying simulated grammatical degradation to SBERT training texts...")
    random.seed(42)
    sbert_texts = [degrade_text(t) for t in tqdm(raw_texts, desc="Text Degradation")]
    
    # 2. TF-IDF feature extraction
    print("\n[2/3] Processing TF-IDF features...")
    print("Cleaning texts for TF-IDF...")
    cleaned_texts = [clean_text_for_tfidf(t) for t in tqdm(raw_texts, desc="Text Cleaning")]
    
    print("Fitting TF-IDF Vectorizer (1000 features)...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english',
        token_pattern=r'\b\w+\b',
        strip_accents='unicode',
        lowercase=True,
        sublinear_tf=True
    )
    X_tfidf = tfidf_vectorizer.fit_transform(cleaned_texts).toarray()
    print(f"✓ TF-IDF features shape: {X_tfidf.shape}")
    
    # Save the vectorizer to the hybrid model directory
    vectorizer_path = os.path.join(OUTPUT_MODEL_DIR, "tfidf_vectorizer.pkl")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    print(f"✓ Saved TF-IDF vectorizer to {vectorizer_path}")
    
    # 3. SBERT embeddings extraction (on degraded text)
    print("\n[3/3] Generating SBERT embeddings on degraded text...")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using PyTorch device: {device}")
    
    print("Loading MiniLM model...")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
    model.eval()
    
    embeddings = []
    print("Extracting embeddings in batches...")
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="SBERT Inference"):
            batch = sbert_texts[i : i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            outputs = model(**inputs)
            pooled = mean_pooling(outputs, inputs['attention_mask'])
            embeddings.append(pooled.cpu().numpy())
            
    X_embeddings = np.vstack(embeddings)
    print(f"✓ SBERT embeddings shape: {X_embeddings.shape}")
    
    # 4. Concatenate and save
    print("\nConcatenating hybrid features...")
    X_hybrid = np.hstack([X_embeddings, X_tfidf])
    print(f"✓ Hybrid feature matrix shape: {X_hybrid.shape}")
    
    # Save hybrid features
    np.save(os.path.join(OUTPUT_DATA_DIR, "X_hybrid.npy"), X_hybrid)
    np.save(os.path.join(OUTPUT_DATA_DIR, "y.npy"), labels)
    
    # Load and copy teacher soft labels
    print("Loading and copying teacher soft labels...")
    y_teacher = np.load(TEACHER_LABELS_PATH)
    np.save(os.path.join(OUTPUT_DATA_DIR, "y_teacher_soft.npy"), y_teacher)
        
    print("\n" + "=" * 60)
    print("🎉 Augmented hybrid features generation completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
