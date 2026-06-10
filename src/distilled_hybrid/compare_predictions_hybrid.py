#!/usr/bin/env python3
import os
import re
import sys
import emoji
import pickle
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel

CSV_FILE = "datasets/bin_reddit1.csv"
STUDENT_MODEL_PATH = "outputs/distilled_hybrid/reddit_mlp_distilled_hybrid/best_model.keras"
TEACHER_MODEL_NAME = "TRT1000/depression-detection-model"
VECTORIZER_PATH = "outputs/distilled_hybrid/reddit_mlp_distilled_hybrid/tfidf_vectorizer.pkl"
BATCH_SIZE = 64

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
    print("Evaluating Hybrid Student Model Fidelity to Teacher")
    print("=" * 60)
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
        
    if not os.path.exists(STUDENT_MODEL_PATH):
        print(f"Error: Student model {STUDENT_MODEL_PATH} not found. Please train it first.")
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
    
    print("Loading TF-IDF vectorizer...")
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
        
    print("Extracting TF-IDF features...")
    X_tfidf = vectorizer.transform(cleaned_texts).toarray()
    
    print(f"Loading MiniLM model for Student embeddings...")
    embed_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    embed_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
    embed_model.eval()

    embeddings = []
    print("Generating Student embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="Student Embedding"):
            batch = raw_texts[i : i + BATCH_SIZE]
            inputs = embed_tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            outputs = embed_model(**inputs)
            pooled = mean_pooling(outputs, inputs['attention_mask'])
            embeddings.append(pooled.cpu().numpy())
    X_embeddings = np.vstack(embeddings)

    print("Concatenating hybrid features...")
    X_hybrid = np.hstack([X_embeddings, X_tfidf])
    print(f"✓ Hybrid feature matrix shape: {X_hybrid.shape}")

    print(f"Loading Student model: {STUDENT_MODEL_PATH}...")
    student_model = tf.keras.models.load_model(STUDENT_MODEL_PATH)
    
    print("Running Student predictions...")
    y_student_probs = []
    for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="Student Inference"):
        batch_x = X_hybrid[i : i + BATCH_SIZE]
        probs = student_model.predict(batch_x, verbose=0).flatten()
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
    
    # 4. Extract Disagreements
    disagreements_mask = y_teacher_pred != y_student_pred
    disagreements_idx = np.where(disagreements_mask)[0]
    
    if len(disagreements_idx) == 0:
        print("Student and Teacher are 100% aligned! No disagreements.")
        sys.exit(0)
        
    dis_df = pd.DataFrame({
        'original_index': disagreements_idx,
        'text': df.iloc[disagreements_idx]['text'].values,
        'teacher_prob': y_teacher_probs[disagreements_idx],
        'student_prob': y_student_probs[disagreements_idx],
        'teacher_pred': y_teacher_pred[disagreements_idx],
        'student_pred': y_student_pred[disagreements_idx]
    })
    
    # Show Top 5 False Positives relative to Teacher (Teacher = 0, Student = 1)
    fps = dis_df[(dis_df['teacher_pred'] == 0) & (dis_df['student_pred'] == 1)].head(5)
    print("\n" + "-"*65)
    print("TOP 5 DISAGREEMENTS: Teacher = Normal (0), Student = Depression (1)")
    print("-"*65)
    for _, row in fps.iterrows():
        print(f"\n[Index {row['original_index']}] Teacher Prob: {row['teacher_prob']:.4f} | Student Prob: {row['student_prob']:.4f}")
        print(f"Text: {str(row['text'])[:300]}...")
        
    # Show Top 5 False Negatives relative to Teacher (Teacher = 1, Student = 0)
    fns = dis_df[(dis_df['teacher_pred'] == 1) & (dis_df['student_pred'] == 0)].head(5)
    print("\n" + "-"*65)
    print("TOP 5 DISAGREEMENTS: Teacher = Depression (1), Student = Normal (0)")
    print("-"*65)
    for _, row in fns.iterrows():
        print(f"\n[Index {row['original_index']}] Teacher Prob: {row['teacher_prob']:.4f} | Student Prob: {row['student_prob']:.4f}")
        print(f"Text: {str(row['text'])[:300]}...")
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
