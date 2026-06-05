.PHONY: process-data-chi2 train-tfidf-chi2 train-tfidf-dirty train-bert train-teacher-bert teacher-labels evaluate-teacher evaluate-new-teacher verify-teacher train-distilled compare-models

process-data-chi2:
	uv run src/tfidf_mlp/process_new_data.py

train-tfidf-chi2:
	uv run src/tfidf_mlp/train_tfidf.py --data_dir data_processed/processed_chi2 --project_name reddit_mlp_hyperband_chi2

train-tfidf-dirty:
	uv run src/tfidf_mlp/train_tfidf.py --data_dir data_processed/processed_dirty --project_name reddit_mlp_hyperband_v3

train-bert:
	uv run src/bert_mlp/train_bert.py --data_dir data_processed/processed_bert --project_name reddit_mlp_bert_nrclex_v5_smote

train-teacher-bert:
	uv run scratch/train_teacher_bert.py

teacher-labels:
	uv run src/distilled_tfidf/get_teacher_labels.py

evaluate-teacher:
	uv run scratch/evaluate_teacher.py

evaluate-new-teacher:
	uv run scratch/evaluate_new_dataset.py

verify-teacher:
	uv run scratch/verify_divergence.py

train-distilled:
	uv run src/distilled_tfidf/train_distilled.py --data_dir data_processed/processed_chi2 --project_name reddit_mlp_distilled

compare-models:
	uv run src/distilled_tfidf/compare_predictions.py




