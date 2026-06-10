# Evaluation Report: Combined Dataset Student Distillation

This report compares the performance of the distilled student models (Distilled Lite and Gated Hybrid) trained on the combined dataset using targets from either the original teacher or the fine-tuned teacher.

* **Combined Dataset Size**: 191,840 samples
* **Test Split Size**: 38,368 samples (20% split, stratified, seed 42)

## OVERALL COMBINED TEST SPLIT
*All source datasets merged.*

| Model | Accuracy | F1-Score |
| :--- | :---: | :---: |
| Distilled Lite (Variant A - Orig Teacher) | 91.05% | 0.91192 |
| Distilled Lite (Variant B - FT Teacher) | 91.69% | 0.91948 |
| Gated Hybrid (Variant A - Orig Teacher) | 91.78% | 0.92240 |
| Gated Hybrid (Variant B - FT Teacher) | 92.63% | 0.93026 |
| Teacher (Fine-Tuned) | 97.68% | 0.97758 |
| Teacher (Original) | 95.58% | 0.95722 |

## THEPIXEL42
*Slice size in test split: 28,055 samples*

| Model | Accuracy | F1-Score |
| :--- | :---: | :---: |
| Distilled Lite (Variant A - Orig Teacher) | 91.84% | 0.91788 |
| Distilled Lite (Variant B - FT Teacher) | 91.97% | 0.92050 |
| Gated Hybrid (Variant A - Orig Teacher) | 92.63% | 0.92827 |
| Gated Hybrid (Variant B - FT Teacher) | 92.64% | 0.92867 |
| Teacher (Fine-Tuned) | 98.66% | 0.98668 |
| Teacher (Original) | 98.69% | 0.98697 |

## SHREYA
*Slice size in test split: 1,538 samples*

| Model | Accuracy | F1-Score |
| :--- | :---: | :---: |
| Distilled Lite (Variant A - Orig Teacher) | 92.39% | 0.91981 |
| Distilled Lite (Variant B - FT Teacher) | 93.56% | 0.93279 |
| Gated Hybrid (Variant A - Orig Teacher) | 92.59% | 0.92470 |
| Gated Hybrid (Variant B - FT Teacher) | 93.76% | 0.93617 |
| Teacher (Fine-Tuned) | 98.24% | 0.98215 |
| Teacher (Original) | 98.44% | 0.98404 |

## OURAFLA
*Slice size in test split: 8,775 samples*

| Model | Accuracy | F1-Score |
| :--- | :---: | :---: |
| Distilled Lite (Variant A - Orig Teacher) | 88.27% | 0.89356 |
| Distilled Lite (Variant B - FT Teacher) | 90.46% | 0.91451 |
| Gated Hybrid (Variant A - Orig Teacher) | 88.95% | 0.90555 |
| Gated Hybrid (Variant B - FT Teacher) | 92.39% | 0.93393 |
| Teacher (Fine-Tuned) | 94.44% | 0.95117 |
| Teacher (Original) | 85.12% | 0.86935 |

