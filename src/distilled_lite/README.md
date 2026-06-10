# Distilled Lite Student Model

The **Distilled Lite** model is an extremely lightweight, pure Multilayer Perceptron (MLP) classification network designed for resource-constrained, high-throughput production environments. 

It completely avoids heavy transformer architectures (like BERT/SBERT) at inference time. Instead, it extracts lexical, sentiment, and stylistic markers using dictionaries, regular expressions, and basic string processors.

---

## 1. Feature Engineering System
The model takes a sparse vector of **2,000 features** selected using Chi-Square (`chi2`) selection from a pool of **5,022 raw features**:

1. **TF-IDF Vocabulary (5,000 features):** Character unigrams and bigrams capturing lexical patterns.
2. **Emotions (10 features):** Continuous scores from NRCLex (anger, anticipation, disgust, fear, joy, sadness, surprise, trust, positive, negative).
3. **Sentiment (2 features):** TextBlob sentiment polarity and subjectivity.
4. **Style & Stylistics (10 features):**
   * **Self-Focus:** First-person singular pronoun density (e.g. `"i"`, `"me"`, `"my"`).
   * **Social Connection:** First-person plural pronoun density (e.g. `"we"`, `"us"`, `"our"`).
   * **Negation Density:** Frequency of negations (e.g. `"not"`, `"never"`, `"cant"`).
   * **Absolute Thinking:** Cognitive distortions using absolutes (e.g. `"always"`, `"never"`, `"completely"`).
   * **Temporal Focus:** Past tense verbs (rumination) vs. future tense verbs (projection).
   * **Punctuation Density:** Ratios of exclamation (`!`) and question (`?`) marks.
   * **Verbosity:** Total word count and average word length.
   * **Crisis Flag:** Binary indicator for explicit risk keywords (e.g., `"kill myself"`, `"suicide"`).

---

## 2. Model Architecture
Built using TensorFlow/Keras, the architecture differs by dataset scale:

* **Single Dataset Baseline:** Keras Tuner Hyperband optimized. Typically yields a shallow MLP (1-2 layers, e.g., `16 -> Dropout(0.0) -> Sigmoid`) with Adam optimizer.
* **Combined Dataset:** Rebuilt as a fixed capacity 2-layer MLP:
  * `Input(2000)` $\rightarrow$ `Dense(128, ReLU)` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Dense(64, ReLU)` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Dense(1, Sigmoid)`
  * **Optimizer:** Adam (Learning Rate: `1e-3`)
  * **Loss:** Binary Crossentropy

---

## 3. Dataset Configurations & Training Methods
* **Baseline Pipeline:** Trained on [bin_reddit1.csv](file:///Users/dark/code/project/depression/datasets/bin_reddit1.csv) using **Blended Targets** ($\alpha = 0.1$):
  $$y_{blend} = 0.1 \cdot y_{true} + 0.9 \cdot y_{teacher}$$
  *Features extracted via:* [generate_lite_features.py](file:///Users/dark/code/project/depression/src/distilled_lite/generate_lite_features.py)  
  *Training script:* [train_distilled_lite.py](file:///Users/dark/code/project/depression/src/distilled_lite/train_distilled_lite.py)
* **Combined Pipeline:** Trained on [combined_dataset.csv](file:///Users/dark/code/project/depression/datasets/combined_dataset.csv) (191,840 samples merging thepixel42, Shreya, and Ourafla). Trained directly on teacher probabilities:
  * **Variant A:** Trained on soft targets from the original teacher (`TRT1000/depression-detection-model`).
  * **Variant B:** Trained on soft targets from the domain fine-tuned teacher.
  *Features extracted via:* [generate_lite_features_combined.py](file:///Users/dark/code/project/depression/src/distilled_lite/generate_lite_features_combined.py)  
  *Training script:* [train_distilled_lite_combined.py](file:///Users/dark/code/project/depression/src/distilled_lite/train_distilled_lite_combined.py)

---

## 4. Evaluation & Results

### Baseline (on degraded bin_reddit1.csv)
* **Teacher Fidelity Agreement:** **78.72%** (F1: **0.63141**)
* **Inference Latency:** **0.36 ms/sample** total (0.10 ms feature extraction + 0.26 ms prediction) on CPU. This is **50x+ faster** than SBERT-based inference.

### Combined Dataset (on test split stratified by source)
* **Overall Test Split:**
  * **Variant A (Original Teacher Targets):** **91.05%** Accuracy (F1: **0.91192**)
  * **Variant B (Fine-Tuned Teacher Targets):** **91.69%** Accuracy (F1: **0.91948**)
* **THEPIXEL42 Slice:**
  * **Variant A:** **91.84%** Accuracy | **Variant B:** **91.97%** Accuracy
* **SHREYA Slice:**
  * **Variant A:** **92.39%** Accuracy | **Variant B:** **93.56%** Accuracy
* **OURAFLA Slice:**
  * **Variant A:** **88.27%** Accuracy | **Variant B:** **90.46%** Accuracy
