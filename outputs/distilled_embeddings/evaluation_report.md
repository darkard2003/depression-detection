# Distillation Evaluation Report: TF-IDF MLP vs. Sentence Transformer MLP

This report compares the performance, student-teacher fidelity, and structural trade-offs of the two distilled student models trained during this session.

---

## 1. Comparative Metrics

Both student models were distilled from the pre-trained SOTA teacher model (`TRT1000/depression-detection-model`, a 110M parameter DistilBERT variant) on the 140,000-sample training dataset (`thepixel42_depression-detection.csv`).

| Metric / Attribute | TF-IDF Distilled Student | SBERT Distilled Student |
| :--- | :--- | :--- |
| **Input Feature Type** | 5,000 Sparse TF-IDF n-grams | 384 Dense Contextual Vectors |
| **Feature Extractor** | `scikit-learn` Vectorizer | `sentence-transformers/all-MiniLM-L6-v2` |
| **Classifier Model Size** | ~31.8 MB (Keras model) | **~1.1 MB** (Keras model) |
| **Feature Cache Size** | ~63 MB (`.npz` sparse matrix) | ~215 MB (`.npy` dense matrix) |
| **Test F1-Score (Hard Targets)**| 91.44% | **93.00%** |
| **Fidelity Agreement (Teacher)** | **78.87%** (78,550 / 99,590) | 77.55% (77,230 / 99,590) |
| **Fidelity F1-Score (Teacher)** | **0.61495** | 0.60273 |
| **Inference Footprint** | **~2 MB** (MLP + Vectorizer) | ~81 MB (MLP + ONNX MiniLM Model) |
| **Dependencies** | `numpy`, `scikit-learn` (Minimal) | `onnxruntime` / `torch` (Heavy) |

---

## 2. Key Finding: The Lemmatization / Grammar Mismatch

An interesting paradox occurred during the evaluation on the target dataset (`bin_reddit1.csv`):
> **Why did the Sentence Transformer student achieve higher test accuracy (93% vs 91.4%) on the training set, but lower fidelity agreement (77.55% vs 78.87%) to the teacher on the evaluation dataset?**

### The Root Cause: Out-of-Distribution broken syntax
The evaluation dataset (`bin_reddit1.csv`) is composed of pre-processed, lemmatized, and grammatically broken texts:
* *Example (Index 49):* `"absolutely no energy depression not eat stress but annoy..."`
* *Example (Index 32):* `"absolutely fine universe stay bed sit sofa sit..."`

#### Contextual Embeddings (SBERT) Sensitivity
Sentence Transformers (`all-MiniLM-L6-v2`) are trained on natural English sentences. They rely on sequence order, attention masks, syntax, and grammatical structures to generate high-quality embeddings. 
When fed broken, lemmatized text:
* The resulting dense vectors fall **out-of-distribution**.
* Synonym and context mappings fail, causing the student MLP to predict `Normal` for texts containing highly predictive words (like `"depression"` and `"no energy"` in lemmatized format).

#### TF-IDF Robustness
TF-IDF is a bag-of-words model. It checks for the frequency of individual tokens (e.g., the presence of `"depression"`, `"energy"`, `"stress"`) completely ignoring grammar, casing, and word order. 
Because it does not rely on syntax, it is highly robust to lemmatized and ungrammatical text.

---

## 3. Deployment Recommendations

### Choose the TF-IDF MLP Student if:
1. **Broken/Lemmatized Input:** Your target runtime text has already gone through aggressive pipeline steps like lemmatization, stemming, or stop-word removal.
2. **Minimal Footprint:** You need to deploy in a resource-constrained environment (e.g., edge devices, browser-side JS, or serverless cold starts) where PyTorch/ONNX runtime overhead is unacceptable.

### Choose the Sentence Transformer MLP Student if:
1. **Raw Natural Text:** Your inputs are raw, un-lemmatized social media posts (which match how the SBERT encoder was pre-trained).
2. **Context and Sarcasm:** You need the student model to handle negations (e.g., `"not depressed"`) and context-dependent phrases (e.g., distinguishing `"suicide bomber"` from `"suicide prevention"`), which bag-of-words models cannot resolve.
