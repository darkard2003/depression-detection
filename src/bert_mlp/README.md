# BERT MLP Classifier Model

This directory contains the Keras **BERT MLP Classifier** model. It uses dense context representation to evaluate standard transformer classification baselines.

---

## 1. Feature Representation
* **Feature Type:** 768-dimensional dense BERT sentence embeddings.
* **Extraction Model:** Extracted using pretrained representations of the teacher transformer (e.g. `TRT1000/depression-detection-model` or similar BERT-family models).
* **Script:** [train_bert.py](file:///Users/dark/code/project/depression/src/bert_mlp/train_bert.py) orchestrates BERT feature extraction and subsequent training.

---

## 2. Model Architecture
Built in TensorFlow/Keras:
* **MLP Structure:** Dense layers mapping the 768-dimensional inputs through standard hidden layers to a single classification node.
* **Optimizer:** Adam (tuned using Hyperband tuning)
* **Loss:** Binary Crossentropy

---

## 3. Scripts
* [train.py](file:///Users/dark/code/project/depression/src/bert_mlp/train.py): Hyperband search and training using precomputed SBERT/BERT representations.
* [train_bert.py](file:///Users/dark/code/project/depression/src/bert_mlp/train_bert.py): End-to-end training including SBERT inference mapping.
* [validate_models.py](file:///Users/dark/code/project/depression/src/bert_mlp/validate_models.py): Performs evaluation of saved BERT MLP configurations.
