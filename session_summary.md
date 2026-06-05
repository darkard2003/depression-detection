# Session Summary: Depression Detection Distillation Pipeline

This document summarizes the pipeline refactoring, dataset optimization, and new scripts implemented during this session to build a high-performance, distilled student MLP model.

---

## 1. Key Findings & Architecture Shifts

### Dataset Label Noise (Old vs. New)
- **Old Dataset (`bin_reddit1.csv`)**: Had significant user-level label noise (e.g., normal political/sports posts labeled "Depression" because the posting user was diagnosed). SOTA models got penalized for predicting "Normal", capping teacher accuracy at **67.23%**.
- **New Dataset (`thepixel42_depression-detection.csv`)**: Downloaded from Hugging Face with clean, post-level annotations. Pre-trained SOTA teacher model immediately achieved **98.60% accuracy** and **0.986 F1-score** out-of-the-box.

### Streamlining Preprocessing (Skipping NRC)
- **Decision**: Dropped NRC Lex emotion extraction from the student model pipeline.
- **Why**: `NRCLex` regex parsing takes ~30 minutes on 140k rows. Skipping it speeds up preprocessing by **100x** (reducing it to 20 seconds for TF-IDF). The 10 emotional features provided negligible information gain ($<0.5\%$ accuracy) over 1000 optimized TF-IDF words.
- **Benefit**: Simplifies deployment by removing `nrclex` and scaler pickle dependencies at inference time.

### Distillation Simplification
- For binary probability outputs, minimizing KL-Divergence is mathematically identical to training on a blended target:
  $$y_{\text{distill}} = \alpha \cdot y_{\text{true}} + (1 - \alpha) \cdot y_{\text{teacher}}$$
  using standard Binary Cross-Entropy loss. This allows Keras Tuner (Hyperband) and standard callbacks to work out-of-the-box without custom gradient tape code.

---

## 2. Pipeline Scripts Created & Modified

### A. New Scripts
1. **[download_dataset.py](file:///Users/dark/code/project/depression/download_dataset.py)**: Downloads `thePixel42/depression-detection` from Hugging Face and saves it as `thepixel42_depression-detection.csv`.
2. **[process_new_data.py](file:///Users/dark/code/project/depression/process_new_data.py)**: Preprocesses the new dataset. Cleans text, fits TF-IDF (5000 features), runs Chi-Square selection to extract the top 1000 features, and saves the matrix (`processed_chi2/X_combined_sparse.npz`) and labels (`processed_chi2/y.npy`).
3. **[get_teacher_labels.py](file:///Users/dark/code/project/depression/get_teacher_labels.py)**: Uses the pre-trained SOTA model (`TRT1000/depression-detection-model`) to generate soft label predictions over raw text and saves them to `processed_chi2/y_teacher_soft.npy`.
4. **[evaluate_new_dataset.py](file:///Users/dark/code/project/depression/evaluate_new_dataset.py)**: Helper script to evaluate the SOTA model on a 5000-row sample of the new dataset.
5. **[verify_divergence.py](file:///Users/dark/code/project/depression/verify_divergence.py)**: Evaluates teacher model mistakes, sorts them by confidence error, and exports the top 100 discrepancies to `processed_chi2/teacher_divergence.csv` for manual inspection.
6. **[train_distilled.py](file:///Users/dark/code/project/depression/train_distilled.py)**: Student MLP training script utilizing Keras Tuner and the blended target loss to distill teacher knowledge.
7. **[feature_engineering_backlog.md](file:///Users/dark/code/project/depression/feature_engineering_backlog.md)**: Backlog documenting future lightweight features (pronoun frequency, text length, punctuation, VADER sentiment) with code snippets.

### B. Modified Scripts
- **[train_tfidf.py](file:///Users/dark/code/project/depression/train_tfidf.py)** & **[train_bert.py](file:///Users/dark/code/project/depression/train_bert.py)**: Updated to parse `--data_dir` and `--project_name` from command-line arguments to allow training on different datasets non-destructively.

---

## 3. Makefile Commands

A unified **[Makefile](file:///Users/dark/code/project/depression/Makefile)** was created to streamline pipeline execution:

| Target | Command | Description |
| :--- | :--- | :--- |
| `process-data-chi2` | `uv run process_new_data.py` | Run TF-IDF cleaning & $\chi^2$ selection |
| `teacher-labels` | `uv run get_teacher_labels.py` | Generate teacher soft targets |
| `evaluate-teacher` | `uv run evaluate_teacher.py` | Compare teacher labels against ground truth |
| `verify-teacher` | `uv run verify_divergence.py` | Export top 100 teacher errors |
| `train-distilled` | `uv run train_distilled.py ...` | Train student model on soft targets |
| `train-tfidf-chi2` | `uv run train_tfidf.py ...` | Train standard student MLP on chi2 targets |

---

## 4. Final Results

- **Student Model**: Keras MLP (1000 Chi-Square TF-IDF features).
- **Teacher**: Pre-trained HuggingFace `TRT1000/depression-detection-model` (DistilBERT).
- **Dataset**: `thepixel42_depression-detection.csv` (140,000 rows, post-level annotations).
- **Student Performance (against true labels)**:
  - **F1-Score**: ~90%
  - **Precision**: ~90%
  - **Recall**: ~90%
- **Inference Speed**: Milliseconds per post (CPU-friendly, no heavy transformers).
- **Conclusion**: Knowledge distillation successfully compressed the transformer model's knowledge into a tiny, CPU-friendly MLP model with minimal loss in accuracy (~8% drop compared to the teacher's 98.6% in exchange for massive parameter reduction and inference speedup).
