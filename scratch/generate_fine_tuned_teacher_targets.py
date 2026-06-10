#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CSV_FILE = "datasets/combined_dataset.csv"
OUTPUT_DIR = "data_processed/combined"
MODEL_PATH = "models/teacher_fine_tuned_combined"
BATCH_SIZE = 256

def main():
    print("=" * 60)
    print("Generating Fine-Tuned Teacher Soft Targets From Scratch")
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
    print(f"Loading tokenizer and model from: {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found! Please run teacher fine-tuning first.")
        sys.exit(1)
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        
        # Optimize with Float16 precision for CUDA and MPS devices
        if device.type in ["cuda", "mps"]:
            print("Converting model to Float16 (Half Precision)...")
            model = model.half()
            
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # 3. Load combined dataset
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
        
    print(f"Loading dataset: {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    texts = df['text'].fillna("").astype(str).tolist()
    total_samples = len(texts)
    
    # 4. Generate soft labels in batches
    soft_labels = []
    print(f"Running fine-tuned teacher inference on {total_samples} samples...")
    
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="Fine-tuned Teacher Inference"):
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
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Convert logits to float32 before computing probabilities to avoid numerical issues
            logits = logits.float()
            
            # Check logits dimensions
            if logits.shape[1] == 2:
                probs = torch.softmax(logits, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(logits).squeeze(1)
                
            soft_labels.extend(probs.cpu().numpy())
            
    # 5. Save teacher predictions
    soft_labels = np.array(soft_labels, dtype=np.float32)
    output_path = os.path.join(OUTPUT_DIR, "y_teacher_soft_finetuned.npy")
    np.save(output_path, soft_labels)
    
    print(f"✓ Saved fine-tuned teacher soft labels to '{output_path}'")
    print(f"Soft labels shape: {soft_labels.shape}")
    print("=" * 60)

if __name__ == "__main__":
    main()
