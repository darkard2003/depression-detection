# TF-IDF MLP Baseline Model

This directory contains the standard, non-distilled **TF-IDF MLP Baseline** model. It serves as a traditional lexical baseline trained directly on hard ground-truth labels.

---

## 1. Feature Extraction & Preprocessing
* **Feature Type:** Sparse lexical term-frequency inverse-document-frequency (TF-IDF).
* **Dimensionality:** Selected top **5,000 features** via Chi-Square (`chi2`) selection from a raw vocabulary of unigrams and bigrams.
* **Preprocessors Fit:**
  * `tfidf_vectorizer.pkl` (TF-IDF Vectorizer fit on training corpus text)
  * `feature_scaler.pkl` (MinMaxScaler to scale inputs to $[0, 1]$ range)
  * `selected_chi2_indices.npy` (Selected feature indices)
* **Scripts:**
  * [fit_preprocessors.py](file:///Users/dark/code/project/depression/src/tfidf_mlp/fit_preprocessors.py): Fits the vectorizers and scalers.
  * [process_data_chi2.py](file:///Users/dark/code/project/depression/src/tfidf_mlp/process_data_chi2.py): Runs feature selection.
  * [process_new_data.py](file:///Users/dark/code/project/depression/src/tfidf_mlp/process_new_data.py): Helper script to preprocess unseen data.

---

## 2. Model Architecture
Built in Keras/TensorFlow, optimized via Hyperband tuning:
* **Shallow MLP Structure:** Typically utilizes 1 to 2 Dense layers with ReLU activations and Dropout layers to reduce overfitting.
* **Optimizer:** Adam (Dynamic tuned learning rate)
* **Loss:** Binary Crossentropy

---

## 3. Performance Results
* **Fidelity Agreement (with Teacher):** **69.01%** (F1: **0.54020**) on degraded [bin_reddit1.csv](file:///Users/dark/code/project/depression/datasets/bin_reddit1.csv).
* **Generalization Accuracy (Ground Truth):**
  * **Ourafla Dataset:** **84.89%**
  * **Shreya Dataset:** **82.96%**
* **Inference Latency:** **~0.20 ms/sample** on CPU (highly efficient due to sparse matrix operations and shallow structure).
