#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# Configuration
CSV_FILE = "datasets/bin_reddit1.csv"
MODEL_NAME = "mental/mentalbert-base-uncased"
OUTPUT_DIR = "models/mentalbert_teacher"
LOGS_DIR = "logs/mentalbert_teacher"
SEED = 42

# Define compute metrics helper
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    return {"accuracy": acc, "f1": f1}

# Dataset helper class
class RedditDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def main():
    print("=" * 60)
    print(f"Fine-tuning {MODEL_NAME} as Teacher Model")
    print("=" * 60)

    # 1. Load raw data
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)

    df = pd.read_csv(CSV_FILE)
    texts = df['text'].fillna("").astype(str).tolist()
    labels = df['label'].astype(int).tolist()

    # Split to train and validation sets (80/20)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=SEED
    )

    print(f"Dataset loaded. Train size: {len(train_labels)}, Val size: {len(val_labels)}")

    # 2. Load Tokenizer & Model
    print(f"Loading pre-trained {MODEL_NAME} tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # 3. Tokenize data
    print("Tokenizing datasets...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

    train_dataset = RedditDataset(train_encodings, train_labels)
    val_dataset = RedditDataset(val_encodings, val_labels)

    # Determine GPU/MPS/CPU device
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    print(f"Training will run on: {device_name.upper()}")

    # 4. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOGS_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
        report_to="none" # Disable W&B logging
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    # 6. Run Training
    print("Starting fine-tuning...")
    trainer.train()

    # 7. Save the best model
    print(f"Saving best teacher model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("=" * 60)
    print("🎉 MentalBERT Teacher model fine-tuned successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
