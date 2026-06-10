#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, f1_score

CSV_FILE = "datasets/bin_reddit1.csv"
STUDENT_MODEL_PATH = "outputs/distilled_embeddings/reddit_mlp_distilled_embeddings/best_model.keras"
TEACHER_MODEL_NAME = "TRT1000/depression-detection-model"
BATCH_SIZE = 64

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    print("=" * 60)
    print("Evaluating Embedding Student Model Fidelity to Teacher")
    print("=" * 60)
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        sys.exit(1)
        
    if not os.path.exists(STUDENT_MODEL_PATH):
        print(f"Error: Student model {STUDENT_MODEL_PATH} not found. Please train it first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset: {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    texts = df['text'].fillna("").astype(str).tolist()
    total_samples = len(texts)

    # 1. Generate Teacher predictions
    print(f"\n[1/3] Loading Teacher model: {TEACHER_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL_NAME).to(device)
    teacher_model.eval()
    
    y_teacher_probs = []
    print("Running batch inference for Teacher...")
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="Teacher Inference"):
            batch = texts[i : i + BATCH_SIZE]
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

    # 2. Student embeddings & inference
    print(f"\n[2/3] Loading MiniLM model for Student embeddings...")
    embed_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    embed_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
    embed_model.eval()

    embeddings = []
    print("Generating Student embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="Student Embedding"):
            batch = texts[i : i + BATCH_SIZE]
            inputs = embed_tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            outputs = embed_model(**inputs)
            pooled = mean_pooling(outputs, inputs['attention_mask'])
            embeddings.append(pooled.cpu().numpy())
    X_embeddings = np.vstack(embeddings)

    print(f"Loading Student model: {STUDENT_MODEL_PATH}...")
    student_model = tf.keras.models.load_model(STUDENT_MODEL_PATH)
    
    print("Running Student predictions...")
    y_student_probs = student_model.predict(X_embeddings, batch_size=128).flatten()
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
