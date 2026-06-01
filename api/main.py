import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import emoji
from nrclex import NRCLex

# Disable GPU dynamic growth warnings or preallocation issues
gpu_devices = tf.config.list_physical_devices('GPU')
if len(gpu_devices) > 0:
    for gpu in gpu_devices:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

app = FastAPI(
    title="Depression Detection NLP API",
    description="API for detecting depression level from social media posts using a TF-IDF Keras MLP model.",
    version="1.0.0"
)

# Global variables for models and preprocessors
models = {}
preprocessors = {}
thresholds = {}

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
    # Build dataframe with single row
    df = pd.DataFrame([raw_scores]).reindex(columns=emotion_cols).fillna(0)
    
    # Scale features
    scaled_features = scaler.transform(df)
    return scaled_features

@app.on_event("startup")
def startup_event():
    print("=========================================================")
    print("Loading Preprocessors and Model...")
    print("=========================================================")
    
    # Get current directory path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Preprocessors from api/preprocessors/
    preprocessor_dir = os.path.join(base_dir, "preprocessors")
    tfidf_vectorizer_path = os.path.join(preprocessor_dir, "tfidf_vectorizer.pkl")
    nrc_scaler_path = os.path.join(preprocessor_dir, "nrc_scaler.pkl")
    
    if not os.path.exists(tfidf_vectorizer_path) or not os.path.exists(nrc_scaler_path):
        raise RuntimeError("Fitted preprocessors not found in api/preprocessors/ directory!")
        
    with open(tfidf_vectorizer_path, "rb") as f:
        preprocessors["tfidf_vectorizer"] = pickle.load(f)
        
    with open(nrc_scaler_path, "rb") as f:
        preprocessors["nrc_scaler"] = pickle.load(f)
        
    print("✓ TF-IDF Vectorizer and NRCLex Scaler loaded successfully.")
    
    # 2. Load TF-IDF Model from api/models/
    model_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(model_dir, "reddit_mlp_hyperband_v3.keras")
    meta_path = os.path.join(model_dir, "reddit_mlp_hyperband_v3_metadata.json")
    
    if not os.path.exists(model_path):
        raise RuntimeError(f"Keras model not found at: {model_path}")
        
    models["tfidf"] = tf.keras.models.load_model(model_path)
    print("✓ TF-IDF Keras Model loaded successfully.")
        
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            thresholds["tfidf"] = meta.get("decision_threshold", 0.5)
            print(f"✓ Decision Threshold: {thresholds['tfidf']:.3f}")
    else:
        thresholds["tfidf"] = 0.5
        print("✓ Decision Threshold defaulted to 0.500")
        
    print("=========================================================")
    print("🎉 Startup complete! Single-model API is ready.")
    print("=========================================================")

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    depression_probability: float
    depression_prediction: int
    label: str
    threshold_used: float

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Step 1: Clean raw text
        cleaned_text = clean_text(request.text)
        
        # Step 2: Extract NRCLex scaled features
        nrc_features = get_nrc_features(cleaned_text, preprocessors["nrc_scaler"])
        
        # Step 3: Extract TF-IDF features
        vectorizer = preprocessors["tfidf_vectorizer"]
        tfidf_features = vectorizer.transform([cleaned_text]).toarray()
        
        # Combine: TF-IDF (5000) + NRCLex (10) = 5010 features
        combined_features = np.hstack([tfidf_features, nrc_features])
        
        # Step 4: Run Keras model prediction
        prob = float(models["tfidf"].predict(combined_features, verbose=0)[0][0])
        threshold = thresholds["tfidf"]
        pred = 1 if prob >= threshold else 0
        
        return PredictionResponse(
            text=request.text,
            depression_probability=prob,
            depression_prediction=pred,
            label="Depression" if pred == 1 else "Control",
            threshold_used=threshold
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": "tfidf (reddit_mlp_hyperband_v3.keras)",
        "preprocessors_loaded": list(preprocessors.keys())
    }
