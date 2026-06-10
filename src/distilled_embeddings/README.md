# Distilled SBERT Embeddings Model

This directory contains the **Distilled SBERT Embeddings** student model. This model attempts to replicate the teacher's classification boundary using dense, contextually aware sentence embeddings instead of lexical keywords.

---

## 1. Feature Representation
* **Feature Type:** 384-dimensional dense sentence embeddings.
* **Extraction Model:** Pre-trained frozen `sentence-transformers/all-MiniLM-L6-v2`.
* **Script:** [generate_embeddings.py](file:///Users/dark/code/project/depression/src/distilled_embeddings/generate_embeddings.py) extracts the static sentence representations from text.

---

## 2. Model Architecture
Built in TensorFlow/Keras, optimized via Hyperband tuning:
* **MLP Structure:** 1-2 dense layers mapping the 384-dimensional SBERT input through hidden layers to a single classification node.
* **Optimizer:** Adam (tuned learning rate).
* **Loss:** Binary Crossentropy against soft targets.

---

## 3. Training & Evaluation
* **Target:** Trained on teacher soft targets to mimic its prediction confidence.
* **Script:** [train_distilled_embeddings.py](file:///Users/dark/code/project/depression/src/distilled_embeddings/train_distilled_embeddings.py) fits the student MLP.
* **Fidelity agreement check:** Evaluated using [compare_predictions_embeddings.py](file:///Users/dark/code/project/depression/src/distilled_embeddings/compare_predictions_embeddings.py).

---

## 4. Performance Results
* **Fidelity Agreement (with Teacher):** **77.81%** (F1: **0.61193**) on degraded [bin_reddit1.csv](file:///Users/dark/code/project/depression/datasets/bin_reddit1.csv).
* **Generalization Accuracy (Ground Truth):**
  * **Ourafla Dataset:** **86.11%**
  * **Shreya Dataset:** **80.29%**
* **Inference Latency:** **~1.20 ms/sample** (includes SBERT encoding time).
