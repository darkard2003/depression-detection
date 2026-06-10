#!/usr/bin/env python3
import os
import re
import sys
import emoji
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

# Configuration
CSV_FILE = "datasets/combined_dataset.csv"
OUTPUT_DATA_DIR = "data_processed/combined"
OUTPUT_MODEL_DIR = "outputs/distilled_hybrid_gated/combined_assets"
BATCH_SIZE = 256

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
    print("Generating Combined Hybrid Model Features From Scratch")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
        
    print(f"Loading dataset: {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    raw_texts = df['text'].fillna("").astype(str).tolist()
    total_samples = len(raw_texts)
    
    # 1. SBERT embeddings extraction
    print("\n[1/2] Generating SBERT embeddings from scratch...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    print("Loading MiniLM model...")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
    model.eval()
    
    embeddings = []
    print("Extracting SBERT embeddings in batches...")
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="SBERT Inference"):
            batch = raw_texts[i : i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            outputs = model(**inputs)
            pooled = mean_pooling(outputs, inputs['attention_mask'])
            embeddings.append(pooled.cpu().numpy())
            
    X_embeddings = np.vstack(embeddings)
    print(f"✓ SBERT embeddings shape: {X_embeddings.shape}")
    
    # Save SBERT embeddings separately in case we need them
    np.save(os.path.join(OUTPUT_DATA_DIR, "X_embeddings_combined.npy"), X_embeddings)
    
    # 2. TF-IDF feature extraction
    print("\n[2/2] Processing TF-IDF features...")
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
    
    # Save TF-IDF preprocessor
    with open(os.path.join(OUTPUT_MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
        
    # Combine SBERT and TF-IDF
    X_hybrid = np.hstack([X_embeddings, X_tfidf])
    print(f"✓ Hybrid feature matrix shape: {X_hybrid.shape}")
    
    # Save Hybrid features
    np.save(os.path.join(OUTPUT_DATA_DIR, "X_hybrid_combined.npy"), X_hybrid)
    print(f"✓ Saved Hybrid features to '{os.path.join(OUTPUT_DATA_DIR, 'X_hybrid_combined.npy')}'")
    print("=" * 60)

if __name__ == "__main__":
    main()
