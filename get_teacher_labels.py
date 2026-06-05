#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configuration
CSV_FILE = "datasets/thepixel42_depression-detection.csv"
OUTPUT_DIR = "data_processed/processed_chi2"
MODEL_NAME = "TRT1000/depression-detection-model" # SOTA binary classifier on HuggingFace
BATCH_SIZE = 64

def main():
    print("=" * 60)
    print("Generating Teacher Soft Labels for Distillation")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    # 2. Load tokenizer and model
    print(f"Loading tokenizer and model: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # 3. Load dataset
    print(f"Loading dataset: {CSV_FILE}...")
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
        
    df = pd.read_csv(CSV_FILE)
    # Ensure cleaned text or raw text is used. The model expects raw text for tokenization.
    texts = df['text'].fillna("").astype(str).tolist()
    total_samples = len(texts)
    
    # 4. Generate soft labels in batches
    soft_labels = []
    print(f"Generating soft labels for {total_samples} samples...")
    
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="Inference"):
            batch_texts = texts[i : i + BATCH_SIZE]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get logits & convert to probabilities (sigmoid for binary classification)
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Check if output is binary classification (2 outputs for softmax, or 1 for sigmoid)
            if logits.shape[1] == 2:
                probs = torch.softmax(logits, dim=1)[:, 1] # Probability of class 1 (depression)
            else:
                probs = torch.sigmoid(logits).squeeze(1)
                
            soft_labels.extend(probs.cpu().numpy())
            
    # 5. Save teacher predictions
    soft_labels = np.array(soft_labels, dtype=np.float32)
    output_path = os.path.join(OUTPUT_DIR, "y_teacher_soft.npy")
    np.save(output_path, soft_labels)
    
    print(f"✓ Saved teacher soft labels to '{output_path}'")
    print("=" * 60)
    print("🎉 Soft labels successfully generated!")
    print("=" * 60)

if __name__ == "__main__":
    main()
