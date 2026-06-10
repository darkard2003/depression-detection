#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

CSV_FILE = "datasets/thepixel42_depression-detection.csv"
OUTPUT_DIR = "data_processed/processed_embeddings"
BATCH_SIZE = 128

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load model
    print("Loading MiniLM model...")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
    model.eval()

    # Load dataset
    print(f"Loading dataset: {CSV_FILE}...")
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
        
    df = pd.read_csv(CSV_FILE)
    texts = df['text'].fillna("").astype(str).tolist()
    
    # Generate embeddings
    embeddings = []
    print("Generating embeddings in batches...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch = texts[i : i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            outputs = model(**inputs)
            pooled = mean_pooling(outputs, inputs['attention_mask'])
            embeddings.append(pooled.cpu().numpy())
            
    X_embeddings = np.vstack(embeddings)
    np.save(os.path.join(OUTPUT_DIR, "X_embeddings.npy"), X_embeddings)
    print(f"✓ Saved embeddings to {OUTPUT_DIR}/X_embeddings.npy. Shape: {X_embeddings.shape}")

if __name__ == "__main__":
    main()
