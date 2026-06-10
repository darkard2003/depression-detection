# Augmented Hybrid Student Model

This directory contains the **Augmented Hybrid Student** model. This variant applies synthetic data augmentation and oversampling techniques (such as SMOTE/RandomOverSampler) to combat class imbalances in the SBERT embedding space during distillation.

---

## 1. Feature Representation & Augmentation
* **Input Features:** 1,384 dimensions (384 SBERT + 1,000 TF-IDF concatenated).
* **Augmentation Method:** Generates synthetic samples in the hybrid representation space to balance non-depressed and depressed classes. 
* **Script:** [generate_hybrid_features.py](file:///Users/dark/code/project/depression/src/distilled_hybrid_augmented/generate_hybrid_features.py) handles embedding extraction and prepares augmented datasets.

---

## 2. Model Architecture
Built in TensorFlow/Keras:
* **MLP Structure:** Dense hidden layers trained on the oversampled balanced dataset.
* **Loss:** Binary Crossentropy against soft teacher targets.
* **Script:** [train_distilled_hybrid_augmented.py](file:///Users/dark/code/project/depression/src/distilled_hybrid_augmented/train_distilled_hybrid_augmented.py) fits the MLP.

---

## 3. Performance Results
* **Fidelity Agreement (with Teacher):** **71.37%** (F1: **0.59239**) on degraded [bin_reddit1.csv](file:///Users/dark/code/project/depression/datasets/bin_reddit1.csv).
* **Fidelity agreement check script:** [compare_predictions_hybrid_augmented.py](file:///Users/dark/code/project/depression/src/distilled_hybrid_augmented/compare_predictions_hybrid_augmented.py)
* **Key Finding:** While oversampling in embedding spaces can balance training class ratios, it can introduce synthetic noise that distorts the soft targets of the teacher, resulting in a drop in fidelity (down to **71.37%** from the non-augmented hybrid's **76.86%**).
