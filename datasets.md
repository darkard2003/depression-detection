# Depression & Mental Health NLP Datasets Registry

This registry lists high-quality post-level annotated datasets for depression and mental health classification. It includes links, descriptions, and download instructions.

---

## 1. thePixel42/depression-detection (Hugging Face)
* **Type**: Open-Access (No token required)
* **Annotation**: Post-level binary classification (0: Normal, 1: Depression). Cleaned.
* **Link**: [thePixel42/depression-detection](https://huggingface.co/datasets/thePixel42/depression-detection)
* **Download Python Code**:
  ```python
  from datasets import load_dataset
  dataset = load_dataset("thePixel42/depression-detection")
  df = dataset['train'].to_pandas()
  df.to_csv("datasets/thepixel42_depression-detection.csv", index=False)
  ```

---

## 2. ourafla/Mental-Health_Text-Classification_Dataset (Hugging Face)
* **Type**: Open-Access (No token required)
* **Annotation**: Post-level multi-class classification (labels: Depression, Anxiety, Bipolar, ADHD, PTSD, etc.).
* **Link**: [ourafla/Mental-Health_Text-Classification_Dataset](https://huggingface.co/datasets/ourafla/Mental-Health_Text-Classification_Dataset)
* **Download Python Code**:
  ```python
  from datasets import load_dataset
  dataset = load_dataset("ourafla/Mental-Health_Text-Classification_Dataset")
  df = dataset['train'].to_pandas()
  df.to_csv("datasets/ourafla_mental_health.csv", index=False)
  ```

---

## 3. nikhileswarcomati/suicide-depression-detection (Kaggle)
* **Type**: Open-Access (Requires Kaggle API Credentials)
* **Annotation**: Post-level binary classification (Suicide/Depression vs. Control). Sourced from r/SuicideWatch and r/depression.
* **Link**: [Suicide and Depression Detection Dataset](https://www.kaggle.com/datasets/nikhileswarcomati/suicide-depression-detection)
* **Download Kaggle CLI Command**:
  ```bash
  # Step 1: Install kaggle cli
  uv pip install kaggle
  # Step 2: Download dataset (requires kaggle.json in ~/.kaggle/)
  kaggle datasets download -d nikhileswarcomati/suicide-depression-detection -p datasets/ --unzip
  ```

---

## 4. ReDSM5 (Hugging Face - Gated)
* **Type**: Gated (Hugging Face account and manual request approval required)
* **Annotation**: Sentence-by-sentence annotation by **licensed psychologists** mapping text to DSM-5 depressive symptoms. Gold standard clinical quality.
* **Link**: [irlab-udc/redsm5](https://huggingface.co/datasets/irlab-udc/redsm5)
* **Download Python Code** (Run after obtaining access and setting `HF_TOKEN` environment variable):
  ```python
  import os
  from datasets import load_dataset
  
  hf_token = os.environ.get("HF_TOKEN") # Or paste your token string directly
  dataset = load_dataset("irlab-udc/redsm5", token=hf_token)
  df = dataset['train'].to_pandas()
  df.to_csv("datasets/redsm5_clinical.csv", index=False)
  ```

---

## 5. SWMH (Hugging Face - Gated)
* **Type**: Gated (Requires request approval)
* **Annotation**: 54,000+ Reddit posts labeled for mental health conditions and suicide severity levels.
* **Link**: [AIMH/SWMH](https://huggingface.co/datasets/AIMH/SWMH)
* **Download Python Code** (Run after obtaining access and setting `HF_TOKEN` environment variable):
  ```python
  import os
  from datasets import load_dataset
  
  hf_token = os.environ.get("HF_TOKEN")
  dataset = load_dataset("AIMH/SWMH", token=hf_token)
  df = dataset['train'].to_pandas()
  df.to_csv("datasets/swmh_reddit.csv", index=False)
  ```
