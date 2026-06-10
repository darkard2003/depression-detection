# Distillation Plan: Hybrid (Context + Keyword) Student Model

This document outlines the architecture, pipeline stages, and execution plan to create a distilled student model utilizing a hybrid feature set of **Sentence Transformer embeddings** and **TF-IDF features**.

---

## 1. Architectural Concept

The hybrid student model is designed to solve the limitations of both individual feature sets:
* **Sentence Transformers (MiniLM):** Capture deep semantics, synonyms, and negations, but are highly sensitive to ungrammatical or pre-lemmatized input.
* **TF-IDF:** Immune to grammar/syntax degradation, but cannot resolve word order, negations, or context.

By concatenating both features, the student MLP learns a robust decision boundary:

```
[Raw Input Text]
       │
       ├───────────────────────────────┐
       ▼                               ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│  Sentence Transformer       │ │  TF-IDF Vectorizer          │
│  (384 dense features)       │ │  (1000 sparse features)     │
└─────────────────────────────┘ └─────────────────────────────┘
       │                               │
       └──────────────┬────────────────┘
                      ▼
             [Concatenated Vector] (1,384 dimensions)
                      │
                      ▼
             ┌─────────────────┐
             │   Student MLP   │ (Sigmoid Output)
             └─────────────────┘
```

### Key Models
1. **Teacher:** `TRT1000/depression-detection-model` (DistilBERT base, 110M parameters).
2. **Student Text Encoder:** `sentence-transformers/all-MiniLM-L6-v2` (MiniLM, 384 dimensions).
3. **Student Keyword Extractor:** Fitted TF-IDF Vectorizer (1,000 features).
4. **Student Classifier:** Keras MLP designed for 1,384 inputs.

---

## 2. Pipeline Execution Steps

### Phase 1: Feature Concatenation
We will reuse the pre-computed sentence embeddings and TF-IDF matrices to create the hybrid feature space.

* **Script:** `src/distilled_hybrid/generate_hybrid_features.py`
* **Task:**
  1. Load `data_processed/processed_embeddings/X_embeddings.npy` (384 dimensions).
  2. Load `data_processed/processed_chi2/X_combined_sparse.npz` (1,000 dimensions) and convert it to a dense array.
  3. Concatenate the features: `X_hybrid = np.hstack([X_embeddings, X_tfidf_dense])`.
  4. Save as `data_processed/processed_hybrid/X_hybrid.npy` (shape `[140000, 1384]`).
  5. Copy the corresponding targets `y.npy` and `y_teacher_soft.npy`.

### Phase 2: Hyperparameter Tuning & Training
We will tune the MLP architecture specifically to handle the larger, combined input size (1,384 dimensions).

* **Script:** `src/distilled_hybrid/train_distilled_hybrid.py`
* **Task:**
  1. Load `X_hybrid.npy`, `y.npy`, and `y_teacher_soft.npy`.
  2. Perform a Hyperband search over hidden layers, dropout rates, and learning rate.
  3. Train on blended soft targets.
  4. Save the trained model to `outputs/distilled_hybrid/reddit_mlp_distilled_hybrid_best.keras`.

### Phase 3: Fidelity Evaluation
We will measure the student model's agreement against the teacher on `bin_reddit1.csv` using the hybrid inputs.

* **Script:** `src/distilled_hybrid/compare_predictions_hybrid.py`
* **Task:**
  1. Load `datasets/bin_reddit1.csv`.
  2. Run teacher predictions.
  3. Extract both SBERT embeddings and TF-IDF features from raw text, merge them, and run predictions.
  4. Print agreement percentage (Fidelity Accuracy), F1-score, and top 5 disagreements.

---

## 3. Makefile Targets to Add

```makefile
process-data-hybrid:
	uv run src/distilled_hybrid/generate_hybrid_features.py

train-distilled-hybrid:
	uv run src/distilled_hybrid/train_distilled_hybrid.py

compare-models-hybrid:
	uv run src/distilled_hybrid/compare_predictions_hybrid.py
```
