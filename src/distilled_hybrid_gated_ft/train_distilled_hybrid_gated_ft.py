#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report

CSV_FILE = "datasets/thepixel42_depression-detection.csv"
DATA_DIR = "data_processed/processed_hybrid_gated_ft"
OUTPUT_DIR = "outputs/distilled_hybrid_gated_ft"
PROJECT_NAME = "reddit_mlp_distilled_hybrid_gated_ft"
ALPHA = 0.1
SEED = 42
BATCH_SIZE = 128
EPOCHS = 3
LR = 2e-5

class GatedHybridModel(nn.Module):
    def __init__(self, transformer_name='sentence-transformers/all-MiniLM-L6-v2', tfidf_dim=1000):
        super().__init__()
        # Load pre-trained PyTorch MiniLM
        self.transformer = AutoModel.from_pretrained(transformer_name)
        
        # Freeze early layers (0-3), leave last 2 trainable (4 and 5)
        for param in self.transformer.parameters():
            param.requires_grad = False
        if hasattr(self.transformer, "encoder") and hasattr(self.transformer.encoder, "layer"):
            for layer in self.transformer.encoder.layer[4:]:
                for param in layer.parameters():
                    param.requires_grad = True
                    
        # SBERT Branch
        self.sbert_dense = nn.Linear(384, 224)
        self.sbert_proj = nn.Linear(224, 128)
        self.sbert_dropout = nn.Dropout(0.2)
        
        # TF-IDF Branch
        self.tfidf_dense = nn.Linear(tfidf_dim, 64)
        self.tfidf_proj = nn.Linear(64, 128)
        self.tfidf_dropout = nn.Dropout(0.2)
        
        # Gating Node
        self.gate_dense = nn.Linear(224 + 64, 1)
        
        # Output layers
        self.out_dense = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout_fused = nn.Dropout(0.2)
        
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask, tfidf_input):
        # 1. MiniLM Forward Pass & Mean Pooling
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = transformer_outputs[0]
        sbert_embeddings = self.mean_pooling(token_embeddings, attention_mask)
        
        # 2. SBERT Branch
        sbert_h = self.relu(self.sbert_dense(sbert_embeddings))
        sbert_h = self.sbert_dropout(sbert_h)
        sbert_proj = self.relu(self.sbert_proj(sbert_h))
        
        # 3. TF-IDF Branch
        tfidf_h = self.relu(self.tfidf_dense(tfidf_input))
        tfidf_h = self.tfidf_dropout(tfidf_h)
        tfidf_proj = self.relu(self.tfidf_proj(tfidf_h))
        
        # 4. Gating weight calculation
        gate_in = torch.cat([sbert_h, tfidf_h], dim=1)
        gate = self.sigmoid(self.gate_dense(gate_in))
        
        # 5. Fusion: gate * sbert + (1 - gate) * tfidf
        weighted_sbert = sbert_proj * gate
        weighted_tfidf = tfidf_proj * (1.0 - gate)
        fused = weighted_sbert + weighted_tfidf
        fused = self.dropout_fused(fused)
        
        # 6. Output Sigmoid prediction
        out = self.sigmoid(self.out_dense(fused))
        return out.squeeze(-1)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for ids, mask, tfidf, target in tqdm(loader, desc="Training Batch"):
        ids = ids.to(device)
        mask = mask.to(device)
        tfidf = tfidf.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(ids, mask, tfidf)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for ids, mask, tfidf, target in loader:
            ids = ids.to(device)
            mask = mask.to(device)
            tfidf = tfidf.to(device)
            
            output = model(ids, mask, tfidf)
            preds = (output >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target.numpy())
    return all_targets, all_preds

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using PyTorch training device: {device}")
    
    # Load targets
    print("Loading targets...")
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    y_teacher = np.load(os.path.join(DATA_DIR, "y_teacher_soft.npy"))
    y_blend = ALPHA * y + (1.0 - ALPHA) * y_teacher
    
    # Load TF-IDF features
    print("Loading TF-IDF features...")
    X_tfidf = np.load(os.path.join(DATA_DIR, "X_tfidf_selected.npy"))
    
    # Load raw text and tokenize
    print(f"Loading raw texts from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    raw_texts = df['text'].fillna("").astype(str).tolist()
    
    print("Tokenizing raw texts for SBERT...")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    tokens = tokenizer(raw_texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    
    # Split datasets
    indices = np.arange(len(y))
    idx_train_val, idx_test = train_test_split(indices, test_size=0.2, stratify=y, random_state=SEED)
    idx_train, idx_val = train_test_split(idx_train_val, test_size=0.2, stratify=y[idx_train_val], random_state=SEED)
    
    # PyTorch Datasets
    X_tfidf_tensor = torch.tensor(X_tfidf, dtype=torch.float32)
    y_blend_tensor = torch.tensor(y_blend, dtype=torch.float32)
    y_hard_tensor = torch.tensor(y, dtype=torch.float32)
    
    train_dataset = TensorDataset(
        input_ids[idx_train], 
        attention_mask[idx_train], 
        X_tfidf_tensor[idx_train], 
        y_blend_tensor[idx_train]
    )
    val_dataset = TensorDataset(
        input_ids[idx_val], 
        attention_mask[idx_val], 
        X_tfidf_tensor[idx_val], 
        y_hard_tensor[idx_val]
    )
    test_dataset = TensorDataset(
        input_ids[idx_test], 
        attention_mask[idx_test], 
        X_tfidf_tensor[idx_test], 
        y_hard_tensor[idx_test]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Build Model
    model = GatedHybridModel(tfidf_dim=X_tfidf.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    
    # Training Loop
    best_val_loss = float('inf')
    best_model_state = None
    
    print("Starting end-to-end model training...")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ids, mask, tfidf, target in val_loader:
                ids = ids.to(device)
                mask = mask.to(device)
                tfidf = tfidf.to(device)
                target = target.to(device)
                
                output = model(ids, mask, tfidf)
                val_loss += criterion(output, target).item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {loss:.5f} | Val Loss: {val_loss:.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            
    # Load best weights
    model.load_state_dict(best_model_state)
    
    # Final Evaluate on test split
    print("Evaluating on test split...")
    targets, preds = evaluate_model(model, test_loader, device)
    
    print("\nClassification Report (Hard Targets):")
    print(classification_report(targets, preds))
    
    # Save best student model weights
    project_dir = os.path.join(OUTPUT_DIR, PROJECT_NAME)
    os.makedirs(project_dir, exist_ok=True)
    model_save_path = os.path.join(project_dir, "best_model.pt")
    
    print(f"Saving model to {model_save_path}...")
    torch.save(best_model_state, model_save_path)
    print(f"✓ Saved best model weights to {model_save_path}")

if __name__ == "__main__":
    main()
