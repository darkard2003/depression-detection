# Concatenated Hybrid Student Model

This directory contains the **Concatenated Hybrid Student** model. This student combines dense semantic embeddings and sparse lexical keywords via simple horizontal concatenation.

---

## 1. Feature Representation
The input is a horizontally concatenated vector of **1,384 dimensions**:
1. **Dense SBERT Embeddings (384 dimensions):** Semantic representations from `all-MiniLM-L6-v2`.
2. **Sparse TF-IDF Keywords (1,000 dimensions):** Term-frequency representations.
* **Script:** [generate_hybrid_features.py](file:///Users/dark/code/project/depression/src/distilled_hybrid/generate_hybrid_features.py) handles tokenization, SBERT encoding, and sparse feature concatenation.

---

## 2. Model Architecture
Unlike the gating fusion architecture, this model merges features directly:
* **MLP Structure:** Takes the 1,384 concatenated vector directly as input into a single feedforward pipeline (1-2 layers, optimized using Keras Tuner).
* **Loss:** Trained using Binary Crossentropy against soft teacher targets.
* **Script:** [train_distilled_hybrid.py](file:///Users/dark/code/project/depression/src/distilled_hybrid/train_distilled_hybrid.py) fits the classifier.

---

## 3. Performance Results
* **Fidelity Agreement (with Teacher):** **76.86%** (F1: **0.62196**) on degraded [bin_reddit1.csv](file:///Users/dark/code/project/depression/datasets/bin_reddit1.csv).
* **Generalization Accuracy (Ground Truth):**
  * **Ourafla Dataset:** **86.43%**
  * **Shreya Dataset:** **82.55%**
* **Inference Latency:** **~1.50 - 2.50 ms/sample** (including MiniLM text encoding).
