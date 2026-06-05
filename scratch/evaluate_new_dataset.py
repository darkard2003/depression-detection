#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Configuration
CSV_FILE = "datasets/thepixel42_depression-detection.csv"
MODEL_NAME = "TRT1000/depression-detection-model" # SOTA binary classifier on HuggingFace
SAMPLE_SIZE = 5000
BATCH_SIZE = 64
SEED = 42

def main():
    print("=" * 60)
    print(f"Evaluating {MODEL_NAME} on New Dataset {CSV_FILE}")
    print("=" * 60)
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
        
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
        
    # 2. Load dataset and sample
    print(f"Loading and sampling {SAMPLE_SIZE} rows from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    df_sampled = df.sample(n=SAMPLE_SIZE, random_state=SEED)
    
    texts = df_sampled['text'].fillna("").astype(str).tolist()
    y_true = df_sampled['label'].astype(int).values
    
    # 3. Load tokenizer and model
    print(f"Loading tokenizer and model: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # 4. Run inference
    y_pred = []
    print("Running batch inference...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Inference"):
            batch_texts = texts[i : i + BATCH_SIZE]
            
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            
            if logits.shape[1] == 2:
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            else:
                probs = torch.sigmoid(logits).squeeze(1)
                preds = (probs >= 0.5).int().cpu().numpy()
                
            y_pred.extend(preds)
            
    y_pred = np.array(y_pred)
    
    # 5. Calculate metrics
    correct = np.sum(y_true == y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Evaluated Samples: {SAMPLE_SIZE}")
    print(f"Correct Predictions: {correct} / {SAMPLE_SIZE} ({accuracy*100:.2f}%)")
    print(f"F1-Score: {f1:.5f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("=" * 60)

if __name__ == "__main__":
    main()
