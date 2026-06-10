# Gated Fine-Tuned (FT) Hybrid Model

This directory contains the PyTorch implementation of the **Gated Fine-Tuned (FT) Hybrid** model. This is an end-to-end gating model that unfreezes and fine-tunes the transformer layers of SBERT during feature distillation.

---

## 1. Core Paradigm & SBERT Fine-Tuning
While other hybrid models use static, frozen sentence embeddings, this model fine-tunes the transformer representation:
* **Base Encoder:** `sentence-transformers/all-MiniLM-L6-v2`.
* **Fine-Tuning:** Unfreezes and updates the weights of the **last 2 transformer layers** of MiniLM end-to-end during training.
* **Objective:** Adapt the sentence encoder's representations to handle lemmatized, grammatically degraded social media posts (e.g. `bin_reddit1.csv`).

---

## 2. Gated Fusion Architecture (PyTorch)
Implemented in PyTorch, the network processes both sparse TF-IDF and deep semantic features:
* **Gated Fusion Layer:** Dynamically calculates a weighting scalar $g \in (0, 1)$ based on SBERT and TF-IDF outputs, performing the fusion:
  $$\text{fused} = g \cdot \text{sbert\_proj} + (1 - g) \cdot \text{tfidf\_proj}$$
* **Optimizer:** AdamW (learning rate optimized for both MLP head and transformer layers).
* **Framework:** PyTorch running on Apple Silicon GPU (MPS) for accelerated training.

---

## 3. Scripts
* [generate_hybrid_features.py](file:///Users/dark/code/project/depression/src/distilled_hybrid_gated_ft/generate_hybrid_features.py): Extracts SBERT embeddings and TF-IDF features.
* [train_distilled_hybrid_gated_ft.py](file:///Users/dark/code/project/depression/src/distilled_hybrid_gated_ft/train_distilled_hybrid_gated_ft.py): Runs end-to-end PyTorch training loops (unfreezing MiniLM layers) on GPU.
* [compare_predictions_hybrid_gated_ft.py](file:///Users/dark/code/project/depression/src/distilled_hybrid_gated_ft/compare_predictions_hybrid_gated_ft.py): Computes fidelity agreement and latency.

---

## 4. Performance & The Overfitting Trade-Off
* **Fidelity Agreement (with Teacher):** **80.62%** (F1: **0.63792**) on degraded [bin_reddit1.csv](file:///Users/dark/code/project/depression/datasets/bin_reddit1.csv). This was our **highest baseline fidelity** on degraded text.
* **The Overfitting Issue (Generalization Drop):**
  * **Ourafla Dataset Accuracy:** **82.97%** (Lower than the frozen SBERT Gated Hybrid's **88.34%**).
  * **Shreya Dataset Accuracy:** **69.16%** (Severe drop compared to the frozen Gated Hybrid's **85.27%**).
* **Key Finding:** Fine-tuning the final layers of SBERT on grammatically degraded training targets caused it to overfit to domain-specific spelling and grammatical syntax, severely breaking its generalization on clean, grammatically natural general-domain texts (like the Shreya dataset).
