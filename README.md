# Depression detection using NLP

In this project, we experiment the possibility of detecting depression in social media with simple NLP structure.

## Project Structure

The project has been organized into a model-centric layout:

* **`datasets/`**: Directory containing raw, read-only CSV datasets.
  * `bin_reddit1.csv` - The original Reddit dataset (evaluation set).
  * `thepixel42_depression-detection.csv` - The new, cleaned training dataset.
* **`data_processed/`**: Directory for cached intermediate feature matrices.
  * `processed_chi2/` - Pre-processed sparse TF-IDF feature matrices and labels.
  * `processed_bert/` - Cached BERT embeddings and labels.
* **`notebooks/`**: Directory housing Jupyter Notebooks for exploratory data analysis, experimental models, and feature prototyping.
* **`src/`**: Source code of the core pipelines:
  * `src/utils/` - Shared helper utilities (e.g., `text_cleaning.py`).
  * `src/tfidf_mlp/` - Preprocessing, feature selection, and training for the base TF-IDF MLP model.
  * `src/distilled_tfidf/` - Teacher label generation, student distillation training, and model comparison.
  * `src/bert_mlp/` - Scripts for BERT/Transformer-based models.
* **`outputs/`**: Model-specific run artifacts (trained models, matching preprocessors, logs, and evaluation reports).
  * `outputs/tfidf_mlp/` - Saved TF-IDF vectorizers, selected feature index files, and base models.
  * `outputs/distilled_tfidf/` - Saved distilled student models and comparison results.
* **`scratch/`**: Temporary helper scripts for testing, ad-hoc evaluations, and validation.
* **`Makefile`**: Commands to easily execute pipeline steps (e.g., `make process-data-chi2`, `make train-distilled`, `make compare-models`).
