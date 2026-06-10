#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

CSV_FILE = "datasets/combined_dataset.csv"
MODEL_NAME = "TRT1000/depression-detection-model"
OUTPUT_DIR = "models/teacher_fine_tuned_combined"
SEED = 42
BATCH_SIZE = 64
LR = 2e-5

class RedditDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # handle token_type_ids if present
            extra_args = {}
            if 'token_type_ids' in batch:
                extra_args['token_type_ids'] = batch['token_type_ids'].to(device)
                
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, **extra_args)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    return acc, f1

def main():
    print("=" * 60)
    print(f"Fine-Tuning Teacher Model via Custom PyTorch Loop: {MODEL_NAME}")
    print("=" * 60)

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

    # 2. Load dataset
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)

    df = pd.read_csv(CSV_FILE)
    texts = df['text'].fillna("").astype(str).tolist()
    labels = df['label'].astype(int).tolist()

    # Split into 90% train, 10% validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=SEED
    )
    print(f"Train size: {len(train_labels)}, Val size: {len(val_labels)}")

    # 3. Load Tokenizer & Model
    print(f"Loading tokenizer and model: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Pre-tokenize all texts
    print("Pre-tokenizing train texts...")
    train_encodings = tokenizer(
        train_texts,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    print("Pre-tokenizing val texts...")
    val_encodings = tokenizer(
        val_texts,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # 4. Create DataLoader
    train_dataset = RedditDataset(train_encodings, train_labels)
    val_dataset = RedditDataset(val_encodings, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Define Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    # 6. Training loop
    print("Starting fine-tuning...")
    model.train()
    
    total_steps = len(train_loader)
    running_loss = 0.0
    
    for step, batch in enumerate(tqdm(train_loader, desc="Training")):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # handle token_type_ids if present
        extra_args = {}
        if 'token_type_ids' in batch:
            extra_args['token_type_ids'] = batch['token_type_ids'].to(device)
            
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **extra_args)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Print intermediate loss and evaluate every 1000 steps
        if (step + 1) % 1000 == 0:
            avg_loss = running_loss / 1000
            print(f"\nStep [{step+1}/{total_steps}] - Average Loss: {avg_loss:.4f}")
            running_loss = 0.0
            
            # Evaluate on validation subset to track progress
            val_acc, val_f1 = evaluate(model, val_loader, device)
            print(f"Validation metrics at step {step+1}: Accuracy={val_acc*100:.2f}%, F1={val_f1:.5f}")
            model.train()
            
    # Final evaluation
    print("\nRunning final validation evaluation...")
    val_acc, val_f1 = evaluate(model, val_loader, device)
    print(f"Final Validation Metrics: Accuracy={val_acc*100:.2f}%, F1={val_f1:.5f}")
    
    # 7. Save model and tokenizer
    print(f"Saving fine-tuned teacher model to: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("=" * 60)
    print("🎉 Teacher model fine-tuned successfully via custom loop!")
    print("=" * 60)

if __name__ == "__main__":
    main()
