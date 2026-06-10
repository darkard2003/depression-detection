#!/usr/bin/env python3
import os
import re
import sys
import emoji
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel

CSV_FILE = "datasets/bin_reddit1.csv"
STUDENT_MODEL_PATH = "outputs/distilled_hybrid_gated_ft/reddit_mlp_distilled_hybrid_gated_ft/best_model.pt"
TEACHER_MODEL_NAME = "TRT1000/depression-detection-model"

# Upgraded TF-IDF preprocessor paths
VECTORIZER_PATH = "outputs/distilled_hybrid_gated_ft/reddit_mlp_distilled_hybrid_gated_ft/tfidf_vectorizer.pkl"
SCALER_PATH = "outputs/distilled_hybrid_gated_ft/reddit_mlp_distilled_hybrid_gated_ft/nrc_scaler.pkl"
INDICES_PATH = "outputs/distilled_hybrid_gated_ft/reddit_mlp_distilled_hybrid_gated_ft/selected_chi2_indices.npy"
BATCH_SIZE = 64

class GatedHybridModel(nn.Module):
    def __init__(self, transformer_name='sentence-transformers/all-MiniLM-L6-v2', tfidf_dim=1000):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        
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
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = transformer_outputs[0]
        sbert_embeddings = self.mean_pooling(token_embeddings, attention_mask)
        
        sbert_h = self.relu(self.sbert_dense(sbert_embeddings))
        sbert_h = self.sbert_dropout(sbert_h)
        sbert_proj = self.relu(self.sbert_proj(sbert_h))
        
        tfidf_h = self.relu(self.tfidf_dense(tfidf_input))
        tfidf_h = self.tfidf_dropout(tfidf_h)
        tfidf_proj = self.relu(self.tfidf_proj(tfidf_h))
        
        gate_in = torch.cat([sbert_h, tfidf_h], dim=1)
        gate = self.sigmoid(self.gate_dense(gate_in))
        
        weighted_sbert = sbert_proj * gate
        weighted_tfidf = tfidf_proj * (1.0 - gate)
        fused = weighted_sbert + weighted_tfidf
        fused = self.dropout_fused(fused)
        
        out = self.sigmoid(self.out_dense(fused))
        return out.squeeze(-1)

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

# Parallel NRC Emotion Feature Extraction Helper
from nrclex import NRCLex
def extract_single_nrc(text):
    emotion_object = NRCLex(text)
    return emotion_object.raw_emotion_scores

def main():
    print("=" * 60)
    print("Evaluating Gated Fine-Tuned PyTorch Hybrid Student Model")
    print("=" * 60)
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
        
    if not os.path.exists(STUDENT_MODEL_PATH):
        print(f"Error: Student model weights {STUDENT_MODEL_PATH} not found. Please train first.")
        sys.exit(1)
        
    if not os.path.exists(VECTORIZER_PATH):
        print(f"Error: TF-IDF vectorizer {VECTORIZER_PATH} not found.")
        sys.exit(1)

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
    print(f"\n[2/3] Processing text features for Student model...")
    print("Cleaning text...")
    cleaned_texts = [clean_text_for_tfidf(t) for t in tqdm(raw_texts, desc="Text Cleaning")]
    
    print("Loading TF-IDF vectorizer, scaler, and selected indices...")
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    selected_indices = np.load(INDICES_PATH)
        
    print("Extracting TF-IDF features...")
    X_tfidf_all = vectorizer.transform(cleaned_texts)
    
    print("Extracting NRC emotion features in parallel...")
    with Pool(os.cpu_count() or 4) as pool:
        raw_scores = pool.map(extract_single_nrc, cleaned_texts)
        
    emotion_cols = ['anticipation', 'joy', 'positive', 'anger', 'fear', 'negative', 'sadness', 'surprise', 'disgust', 'trust']
    nrc_df = pd.DataFrame(raw_scores).reindex(columns=emotion_cols).fillna(0)
    X_nrc_scaled = scaler.transform(nrc_df)
    
    print("Combining TF-IDF and NRC features...")
    X_nrc_sparse = X_tfidf_all.__class__(X_nrc_scaled)
    from scipy.sparse import hstack
    X_combined = hstack([X_tfidf_all, X_nrc_sparse]).tocsr()
    
    print("Slicing selected Chi-Square features...")
    X_tfidf_selected = X_combined[:, selected_indices].toarray()
    print(f"✓ Final TF-IDF/NRC shape: {X_tfidf_selected.shape}")
    
    print("Tokenizing texts for SBERT...")
    sbert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    tokens = sbert_tokenizer(raw_texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    # Load Student PyTorch model
    print(f"Loading Student model weights from {STUDENT_MODEL_PATH}...")
    student_model = GatedHybridModel(tfidf_dim=X_tfidf_selected.shape[1]).to(device)
    student_model.load_state_dict(torch.load(STUDENT_MODEL_PATH, map_location=device))
    student_model.eval()
    
    print("Running Student predictions...")
    y_student_probs = []
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="Student Inference"):
            batch_ids = input_ids[i : i + BATCH_SIZE].to(device)
            batch_mask = attention_mask[i : i + BATCH_SIZE].to(device)
            batch_tfidf = torch.tensor(X_tfidf_selected[i : i + BATCH_SIZE], dtype=torch.float32).to(device)
            
            probs = student_model(batch_ids, batch_mask, batch_tfidf).cpu().numpy()
            y_student_probs.extend(probs)
        
    y_student_probs = np.array(y_student_probs)
    y_student_pred = (y_student_probs >= 0.5).astype(int)

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
    print("=" * 60)

if __name__ == "__main__":
    main()
