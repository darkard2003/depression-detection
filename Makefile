.PHONY: process-data-tfidf process-data-teacher-labels process-data-embeddings process-data-hybrid process-data-hybrid-augmented process-data-hybrid-gated-ft process-data-lite train-base-tfidf-chi2 train-base-tfidf-dirty train-base-bert-nrclex train-base-teacher-bert train-distilled-tfidf train-distilled-embeddings train-distilled-hybrid train-distilled-hybrid-augmented train-distilled-hybrid-gated train-distilled-hybrid-gated-ft train-distilled-lite evaluate-teacher-old evaluate-teacher-new verify-teacher-divergence compare-models-tfidf compare-models-embeddings compare-models-hybrid compare-models-hybrid-augmented compare-models-hybrid-gated compare-models-hybrid-gated-ft compare-models-lite

# --- Data Processing / Prep ---
process-data-tfidf:
	uv run src/tfidf_mlp/process_new_data.py

process-data-teacher-labels:
	uv run src/distilled_tfidf/get_teacher_labels.py

process-data-embeddings:
	uv run src/distilled_embeddings/generate_embeddings.py

# --- Base Model Training ---
train-base-tfidf-chi2:
	uv run src/tfidf_mlp/train_tfidf.py --data_dir data_processed/processed_chi2 --project_name reddit_mlp_hyperband_chi2

train-base-tfidf-dirty:
	uv run src/tfidf_mlp/train_tfidf.py --data_dir data_processed/processed_dirty --project_name reddit_mlp_hyperband_v3

train-base-bert-nrclex:
	uv run src/bert_mlp/train_bert.py --data_dir data_processed/processed_bert --project_name reddit_mlp_bert_nrclex_v5_smote

train-base-teacher-bert:
	uv run scratch/train_teacher_bert.py

# --- Distilled Student Training ---
train-distilled-tfidf:
	uv run src/distilled_tfidf/train_distilled.py --data_dir data_processed/processed_chi2 --project_name reddit_mlp_distilled

train-distilled-embeddings:
	uv run src/distilled_embeddings/train_distilled_embeddings.py

# --- Evaluation / Validation ---
evaluate-teacher-old:
	uv run scratch/evaluate_teacher.py

evaluate-teacher-new:
	uv run scratch/evaluate_new_dataset.py

verify-teacher-divergence:
	uv run scratch/verify_divergence.py

compare-models-tfidf:
	uv run src/distilled_tfidf/compare_predictions.py

compare-models-embeddings:
	uv run src/distilled_embeddings/compare_predictions_embeddings.py

process-data-hybrid:
	uv run src/distilled_hybrid/generate_hybrid_features.py

train-distilled-hybrid:
	uv run src/distilled_hybrid/train_distilled_hybrid.py

compare-models-hybrid:
	uv run src/distilled_hybrid/compare_predictions_hybrid.py

# --- Augmented Hybrid Project ---
process-data-hybrid-augmented:
	uv run src/distilled_hybrid_augmented/generate_hybrid_features.py

train-distilled-hybrid-augmented:
	uv run src/distilled_hybrid_augmented/train_distilled_hybrid_augmented.py

compare-models-hybrid-augmented:
	uv run src/distilled_hybrid_augmented/compare_predictions_hybrid_augmented.py

# --- Gated Hybrid Project ---
train-distilled-hybrid-gated:
	uv run src/distilled_hybrid_gated/train_distilled_hybrid_gated.py

compare-models-hybrid-gated:
	uv run src/distilled_hybrid_gated/compare_predictions_hybrid_gated.py

# --- Gated Fine-Tuned (FT) Hybrid Project ---
process-data-hybrid-gated-ft:
	uv run src/distilled_hybrid_gated_ft/generate_hybrid_features.py

train-distilled-hybrid-gated-ft:
	uv run src/distilled_hybrid_gated_ft/train_distilled_hybrid_gated_ft.py

compare-models-hybrid-gated-ft:
	uv run src/distilled_hybrid_gated_ft/compare_predictions_hybrid_gated_ft.py

# --- Lite Distilled Project ---
process-data-lite:
	uv run src/distilled_lite/generate_lite_features.py

train-distilled-lite:
	uv run src/distilled_lite/train_distilled_lite.py

compare-models-lite:
	uv run src/distilled_lite/compare_predictions_lite.py
