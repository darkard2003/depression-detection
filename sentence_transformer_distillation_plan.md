# Sentence Transformer Distillation Plan

This document outlines the architecture, pipeline stages, and execution plan to transition the distilled student model from a sparse TF-IDF representation to a dense **Sentence Transformer + MLP** architecture. This transition aims to improve the student model's fidelity to the teacher from ~78% to >90% by preserving semantic context and negations.

---

## 1. Architecture Overview

Rather than training the student MLP on 5,000 sparse TF-IDF word counts, we will use a lightweight pre-trained **Sentence Transformer** as a feature extractor.

```
[Raw Input Text] 
       │
       ▼
┌──────────────────────────────┐
│  Sentence Transformer        │ (all-MiniLM-L6-v2)
│  (Frozen Feature Extractor)  │ (384-dimensional dense vector output)
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  Student MLP Classifier      │ (Tuned via Keras Tuner)
│  (Dense Input: 384)          │ (Sigmoid probability output)
└──────────────────────────────┘
```

### Key Models
1. **Teacher:** `TRT1000/depression-detection-model` (DistilBERT base, 110M parameters, ~440MB).
2. **Student Feature Extractor:** `sentence-transformers/all-MiniLM-L6-v2` (MiniLM base, 22M parameters, ~80MB).
3. **Student Classifier:** Keras MLP (~1MB).

---

## 2. Pipeline Execution Steps

### Phase 1: Feature Extraction
We will pre-compute and cache the sentence embeddings for the training dataset (`thepixel42_depression-detection.csv`) to speed up model training and hyperparameter search.

* **Script:** `generate_embeddings.py`
* **Task:**
  1. Load the dataset.
  2. Batch-encode the clean texts using `all-MiniLM-L6-v2`.
  3. Save the resulting embeddings array as `processed_embeddings/X_embeddings.npy`.
  4. Save the corresponding labels and teacher soft targets (`y.npy` and `y_teacher_soft.npy`).

### Phase 2: Hyperparameter Tuning & Training
Using the pre-computed embeddings, we will run a Hyperband search to optimize the architecture of the student classifier.

* **Script:** `train_distilled_embeddings.py`
* **Task:**
  1. Load `X_embeddings.npy`, `y.npy`, and `y_teacher_soft.npy`.
  2. Split into Train/Val/Test (stratified).
  3. Run `keras-tuner` to optimize the hidden layers, units, and dropout rates of the MLP.
  4. Train on the blended distillation target:
     $$y_{\text{distill}} = \alpha \cdot y_{\text{true}} + (1 - \alpha) \cdot y_{\text{teacher}}$$
  5. Save the best model to `models/reddit_mlp_distilled_embeddings_best.keras`.

### Phase 3: Fidelity Evaluation
We will measure how closely the new embedding-based student matches the teacher on the evaluation dataset (`bin_reddit1.csv`).

* **Script:** `compare_predictions_embeddings.py`
* **Task:**
  1. Load the evaluation dataset.
  2. Generate teacher predictions.
  3. Generate student embeddings and predict using the new student classifier.
  4. Print agreement percentage (Fidelity Accuracy), F1-score, and extract top 5 disagreements.

---

## 3. Production Optimization (ONNX Quantization)

To keep the inference footprint as light as possible, we can compile the Sentence Transformer and the student MLP into a single **ONNX graph** and apply **INT8 quantization**:

* **Size Reduction:** Shrinks the Sentence Transformer from ~80MB to **~20MB**.
* **Speedup:** ONNX Runtime executes the Transformer layers on CPU significantly faster than PyTorch.
* **Simplification:** Allows deploying a single `.onnx` model file in production with zero PyTorch/TensorFlow dependencies.
