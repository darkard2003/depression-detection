import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import shap

# Lite model
X = np.load("data_processed/combined/X_lite_combined.npy")
y_hard = np.load("data_processed/combined/y_combined.npy")
y_soft = np.load("data_processed/combined/y_teacher_soft_finetuned.npy")
model = keras.models.load_model("outputs/distilled_lite/reddit_mlp_distilled_lite_combined_ft/best_model.keras")

X_train_val, X_test, y_hard_train_val, y_hard_test, y_soft_train_val, y_soft_test = train_test_split(
    X, y_hard, y_soft, test_size=0.2, stratify=y_hard, random_state=42
)
X_train, X_val, y_hard_train, y_hard_val, y_soft_train, y_soft_val = train_test_split(
    X_train_val, y_hard_train_val, y_soft_train_val, test_size=0.2, stratify=y_hard_train_val, random_state=42
)

with open("outputs/distilled_lite/combined_assets/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)
selected_indices = np.load("outputs/distilled_lite/combined_assets/selected_chi2_indices.npy")

tfidf_names = list(tfidf_vectorizer.get_feature_names_out())
extra_names = [
    'anticipation', 'joy', 'positive', 'anger', 'fear', 'negative', 'sadness', 'surprise', 'disgust', 'trust',
    'polarity', 'subjectivity', 'word_count', 'avg_word_length', 'sing_ratio', 'plural_ratio', 'past_ratio',
    'future_ratio', 'neg_ratio', 'abs_ratio', 'crisis_flag', 'exclamation_ratio', 'question_ratio'
]
all_feature_names = np.array(tfidf_names + extra_names)
selected_feature_names = all_feature_names[selected_indices]

np.random.seed(42)
bg_idx = np.random.choice(X_train.shape[0], size=100, replace=False)
X_background = X_train[bg_idx].astype(np.float32)

explain_idx = np.random.choice(X_test.shape[0], size=200, replace=False)
X_explain = X_test[explain_idx].astype(np.float32)

explainer = shap.DeepExplainer(model, X_background)
shap_values = explainer.shap_values(X_explain)
if isinstance(shap_values, list):
    sv = shap_values[0]
else:
    sv = shap_values
sv = np.squeeze(sv)
mean_abs_shap = np.mean(np.abs(sv), axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1][:15]
print("Distilled Lite Top Features:")
for idx in top_idx:
    print(f"  {selected_feature_names[idx]}: {mean_abs_shap[idx]:.5f}")

# Gated model
X_g = np.load("data_processed/combined/X_hybrid_combined.npy")
model_g = keras.models.load_model("outputs/distilled_hybrid_gated/reddit_mlp_distilled_hybrid_gated_combined_ft/best_model.keras", safe_mode=False)
X_train_val_g, X_test_g, y_hard_train_val_g, y_hard_test_g, y_soft_train_val_g, y_soft_test_g = train_test_split(
    X_g, y_hard, y_soft, test_size=0.2, stratify=y_hard, random_state=42
)
X_train_g, X_val_g, y_hard_train_g, y_hard_val_g, y_soft_train_g, y_soft_val_g = train_test_split(
    X_train_val_g, y_hard_train_val_g, y_soft_train_val_g, test_size=0.2, stratify=y_hard_train_val_g, random_state=42
)

with open("outputs/distilled_hybrid_gated/combined_assets/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer_g = pickle.load(f)
sbert_names = [f"SBERT_dim_{i}" for i in range(384)]
tfidf_names_g = list(tfidf_vectorizer_g.get_feature_names_out())
hybrid_feature_names = np.array(sbert_names + tfidf_names_g)

bg_idx_g = np.random.choice(X_train_g.shape[0], size=100, replace=False)
X_background_g = X_train_g[bg_idx_g].astype(np.float32)

explain_idx_g = np.random.choice(X_test_g.shape[0], size=200, replace=False)
X_explain_g = X_test_g[explain_idx_g].astype(np.float32)

explainer_g = shap.DeepExplainer(model_g, X_background_g)
shap_values_g = explainer_g.shap_values(X_explain_g)
if isinstance(shap_values_g, list):
    sv_g = shap_values_g[0]
else:
    sv_g = shap_values_g
sv_g = np.squeeze(sv_g)
mean_abs_shap_g = np.mean(np.abs(sv_g), axis=0)
top_idx_g = np.argsort(mean_abs_shap_g)[::-1][:15]
print("\nGated Hybrid Top Features:")
for idx in top_idx_g:
    print(f"  {hybrid_feature_names[idx]}: {mean_abs_shap_g[idx]:.5f}")
