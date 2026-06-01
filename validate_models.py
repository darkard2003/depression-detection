import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import emoji
from nrclex import NRCLex

# ==============================================================================
# GLOBAL CONFIGURATIONS - EDIT TO CHOOSE MODEL & PATH DIRECTLY
# ==============================================================================
MODEL_PATH = "api/models/reddit_mlp_hyperband_v3.keras"  # Direct path to Keras model (.keras)
MODEL_TYPE = "tfidf"                                      # Type of model: "tfidf" or "bert"
PREPROCESSORS_DIR = "api/preprocessors"                   # Folder containing tfidf_vectorizer.pkl and nrc_scaler.pkl
# ==============================================================================

# Preprocessing helpers
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

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = process_hashtags(text)
    return text

def get_nrc_features(text, scaler):
    emotion_object = NRCLex(text)
    raw_scores = emotion_object.raw_emotion_scores
    emotion_cols = ['anticipation', 'joy', 'positive', 'anger', 'fear', 'negative', 'sadness', 'surprise', 'disgust', 'trust']
    df = pd.DataFrame([raw_scores]).reindex(columns=emotion_cols).fillna(0)
    return scaler.transform(df)

def main():
    model_type = MODEL_TYPE.lower().strip()
    if model_type not in ["tfidf", "bert"]:
        print("Error: MODEL_TYPE must be 'tfidf' or 'bert'!")
        return

    print("=========================================================")
    print("Initializing Direct Model Validation Script...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model type: {model_type.upper()}")
    print(f"Preprocessors folder: {PREPROCESSORS_DIR}")
    print("=========================================================")

    # 1. Check if Model Path exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at: {MODEL_PATH}")
        return

    # 2. Load Common NRCLex MinMaxScaler
    nrc_scaler_path = os.path.join(PREPROCESSORS_DIR, "nrc_scaler.pkl")
    if not os.path.exists(nrc_scaler_path):
        print(f"Error: Fitted NRCLex scaler not found in {PREPROCESSORS_DIR} directory!")
        return
    with open(nrc_scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("✓ Loaded NRCLex MinMaxScaler.")

    model = None
    threshold = 0.5
    vectorizer = None
    sbert = None

    # 3. Setup model-specific assets
    if model_type == "tfidf":
        # Load TF-IDF Vectorizer
        tfidf_vectorizer_path = os.path.join(PREPROCESSORS_DIR, "tfidf_vectorizer.pkl")
        if not os.path.exists(tfidf_vectorizer_path):
            print(f"Error: Fitted TF-IDF Vectorizer not found in {PREPROCESSORS_DIR} directory!")
            return
        with open(tfidf_vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        print("✓ Loaded TF-IDF Vectorizer.")
    else:  # bert
        # Load SentenceTransformer
        print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        print("✓ Loaded SentenceTransformer.")

    # 4. Load Keras Model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✓ Keras Model loaded successfully.")

    # Try loading metadata to resolve optimal decision threshold
    meta_path = MODEL_PATH.replace(".keras", "_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                threshold = json.load(f).get("decision_threshold", 0.5)
            print(f"✓ Loaded Decision Threshold from metadata: {threshold:.3f}")
        except Exception:
            print("✓ Metadata load failed. Decision Threshold defaulted to 0.500")
    else:
        print("✓ Decision Threshold defaulted to 0.500")

    # 5. Unforeseen Test Cases
    test_cases = [
        "I just feel so tired all the time. I do not want to get out of bed. Nothing brings me joy anymore and I feel completely empty inside.",
        "Had a wonderful day at the park today! The weather was perfect and the dogs loved running around. Feeling very grateful.",
        "I cannot keep doing this. I feel like an absolute burden to my family and friends. It would be better if I just was not here.",
        "Studying for my exams next week. Feeling a bit stressed, but I think I am prepared. Just going to keep working hard.",
        "It is hard to describe the feeling. It is like being underwater while everyone else is breathing normally. I just want to escape this pain."
    ]

    print("\n=========================================================")
    print(f"Running Validation on Unforeseen Texts (Model: {model_type.upper()})")
    print("=========================================================\n")

    results = []
    for idx, raw_text in enumerate(test_cases):
        cleaned_text = clean_text(raw_text)
        nrc_feats = get_nrc_features(cleaned_text, scaler)

        if model_type == "tfidf":
            tfidf_feats = vectorizer.transform([cleaned_text]).toarray()
            combined_features = np.hstack([tfidf_feats, nrc_feats])
        else:  # bert
            bert_embeddings = sbert.encode([cleaned_text])
            combined_features = np.hstack([bert_embeddings, nrc_feats])

        # Run prediction
        prob = float(model.predict(combined_features, verbose=0)[0][0])
        pred = 1 if prob >= threshold else 0
        label = "Depression" if pred == 1 else "Control"

        results.append({
            "id": idx + 1,
            "text": raw_text[:60] + "...",
            "prob": prob,
            "label": label
        })

    # Display results as a beautiful table
    print(f"{'ID':<3} | {'Unforeseen Text snippet':<63} | {'Probability':<12} | {'Prediction':<10}")
    print("-" * 98)
    for r in results:
        print(f"{r['id']:<3} | {r['text']:<63} | {r['prob']:<12.4f} | {r['label']:<10}")

    print("\n=========================================================")
    print(f"🎉 {model_type.upper()} Validation successfully complete!")
    print("=========================================================")

if __name__ == "__main__":
    main()
