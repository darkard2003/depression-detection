# Session Summary: Depression Detection Distillation Pipeline

This document summarizes the pipeline targets, model architectures, and results achieved during this session to build high-fidelity distilled student models.

---

## 1. Key Findings & Architecture Shifts

### Feature Mismatch on Degraded Text
* **The Problem:** Sentence Transformers (e.g., SBERT) are trained on natural grammar. Pre-lemmatized, grammatically broken text (such as in `bin_reddit1.csv`) pushes SBERT embeddings out-of-distribution, capping baseline SBERT student model fidelity at **77.81%**.
* **The Solution:** We designed a **Gated Fusion Architecture** that combines dense SBERT context and sparse TF-IDF keywords, achieving our highest fidelity on degraded text.

### Gated Fine-Tuning Performance & Overfitting
* **Model:** Gated Hybrid Student (1,384 dimensions) unfreezing the last 2 transformer layers of MiniLM.
* **Result:** Achieved SOTA **80.62% fidelity agreement** on the degraded dataset. However, fine-tuning SBERT end-to-end on target degraded text caused it to overfit, dropping its accuracy to **69.16%** on clean grammatical natural language text (Shreya dataset).

### Generalization Robustness of Gated Hybrid and Distilled Lite
* **Models:** Gated Hybrid (frozen SBERT) and Distilled Lite (lexical features only).
* **Result:** These models achieved the highest generalization robustness, actually **outperforming the Teacher model** against ground-truth labels on the Ourafla datasets (e.g., Lite model achieved **88.36%** accuracy vs. Teacher's **85.07%**). 
* **Key Insight:** Relying on frozen pretrained embeddings and domain-invariant lexical features (polarity, emotions, pronoun densities) makes models resilient to domain shifts.

---

## 2. Pipeline Scripts Created

### A. Gated Fine-Tuned Hybrid (PyTorch)
* **[generate_hybrid_features.py](file:///Users/dark/code/project/depression/src/distilled_hybrid_gated_ft/generate_hybrid_features.py)**: Cleans text, fits TF-IDF, extracts parallelized NRClex scales, combines them, and runs Chi-Square selection to extract the top 1,000 features.
* **[train_distilled_hybrid_gated_ft.py](file:///Users/dark/code/project/depression/src/distilled_hybrid_gated_ft/train_distilled_hybrid_gated_ft.py)**: Unfreezes the last 2 layers of MiniLM in a PyTorch Gated Fusion network.
* **[compare_predictions_hybrid_gated_ft.py](file:///Users/dark/code/project/depression/src/distilled_hybrid_gated_ft/compare_predictions_hybrid_gated_ft.py)**: Evaluates the fine-tuned model's fidelity.

### B. Distilled Lite (TensorFlow/Keras)
* **[generate_lite_features.py](file:///Users/dark/code/project/depression/src/distilled_lite/generate_lite_features.py)**: Extracts TF-IDF, NRC, TextBlob sentiment, and stylistic markers, and selects the top 2,000 features via Chi-Square.
* **[train_distilled_lite.py](file:///Users/dark/code/project/depression/src/distilled_lite/train_distilled_lite.py)**: Fits a shallow Keras MLP optimized for resource-constrained deployment.
* **[compare_predictions_lite.py](file:///Users/dark/code/project/depression/src/distilled_lite/compare_predictions_lite.py)**: Evaluates lite student fidelity and measures CPU inference latency.

### C. Generalization & Cross-Dataset Evaluation
* **[evaluate_on_extra_datasets.py](file:///Users/dark/code/project/depression/scratch/evaluate_on_extra_datasets.py)**: An optimized multi-dataset validation script evaluating all 6 distilled students and the teacher model across three distinct subsets of the Shreya and Ourafla datasets.

---

## 3. Makefile Targets

A unified **[Makefile](file:///Users/dark/code/project/depression/Makefile)** was maintained to run all targets:

| Target | Command | Description |
| :--- | :--- | :--- |
| `process-data-hybrid-gated-ft` | `uv run src/distilled_hybrid_gated_ft/generate_hybrid_features.py` | Generate Chi-Square selected upgraded TF-IDF/NRC features. |
| `train-distilled-hybrid-gated-ft` | `uv run src/distilled_hybrid_gated_ft/train_distilled_hybrid_gated_ft.py` | Train end-to-end PyTorch gated model on M4 GPU. |
| `compare-models-hybrid-gated-ft` | `uv run src/distilled_hybrid_gated_ft/compare_predictions_hybrid_gated_ft.py` | Evaluate gated model fidelity on evaluation dataset. |
| `process-data-lite` | `uv run src/distilled_lite/generate_lite_features.py` | Extract TF-IDF + NRC + Sentiment + Stylistic markers. |
| `train-distilled-lite` | `uv run src/distilled_lite/train_distilled_lite.py` | Train lightweight Keras MLP model. |
| `compare-models-lite` | `uv run src/distilled_lite/compare_predictions_lite.py` | Evaluate lite model fidelity and latency. |
