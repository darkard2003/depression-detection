#!/usr/bin/env python3
import os
import re
import sys
import pickle
import torch
import emoji
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CSV_FILE = "datasets/bin_reddit1.csv"
STUDENT_MODEL_PATH = "models/reddit_mlp_distilled_best.keras"
TEACHER_MODEL_NAME = "TRT1000/depression-detection-model"
VECTORIZER_PATH = "preprocessors/tfidf_vectorizer.pkl"
INDICES_PATH = "preprocessors/selected_tfidf_indices.npy"
BATCH_SIZE = 64

# Cleaning functions
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

def main():
    print("=" * 60)
    print("Evaluating Student Model Fidelity to Teacher on Dataset")
    print("=" * 60)
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
        
    # 1. Device selection for Teacher
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU for Teacher")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS) for Teacher")
    else:
        device = torch.device("cpu")
        print("Using CPU for Teacher")

    # 2. Load dataset
    print(f"Loading dataset {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    raw_texts = df['text'].fillna("").astype(str).tolist()
    total_samples = len(raw_texts)

    # 3. Generate Teacher predictions
    print(f"\n[1/3] Loading Teacher model: {TEACHER_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL_NAME)
    teacher_model.to(device)
    teacher_model.eval()
    
    y_teacher_probs = []
    print("Running batch inference for Teacher...")
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="Teacher Inference"):
            batch_texts = raw_texts[i : i + BATCH_SIZE]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = teacher_model(**inputs)
            logits = outputs.logits
            if logits.shape[1] == 2:
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            else:
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            y_teacher_probs.extend(probs)
            
    y_teacher_probs = np.array(y_teacher_probs)
    y_teacher_pred = (y_teacher_probs >= 0.5).astype(int)

    # 4. Generate Student predictions
    print(f"\n[2/3] Processing text for Student model...")
    df['text_clean'] = df['text'].fillna("").apply(clean_text_for_tfidf)
    
    print("Loading vectorizer and indices...")
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    selected_indices = np.load(INDICES_PATH)
    
    print("Vectorizing text...")
    X_tfidf = vectorizer.transform(df['text_clean'].tolist())
    X_student = X_tfidf[:, selected_indices]
    
    print(f"Loading Student model: {STUDENT_MODEL_PATH}...")
    student_model = tf.keras.models.load_model(STUDENT_MODEL_PATH)
    
    print("Running Student predictions...")
    y_student_probs = []
    # Predict in batches to avoid RAM crashes
    for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="Student Inference"):
        batch_x = X_student[i : i + BATCH_SIZE].toarray()
        probs = student_model.predict(batch_x, verbose=0).flatten()
        y_student_probs.extend(probs)
        
    y_student_probs = np.array(y_student_probs)
    y_student_pred = (y_student_probs >= 0.5).astype(int)

    # 5. Evaluate Fidelity (Student vs Teacher)
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
    
    # 6. Extract Disagreements
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
