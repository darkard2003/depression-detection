# Feature Engineering Backlog for Depression Detection NLP

This backlog contains advanced, lightweight features to explore for improving the depression detection student MLP model. These features are computationally fast and grounded in psychiatric and linguistic research.

---

## 1. First-Person Pronoun Frequency (Linguistic Self-Focus)

### Theory
Psycholinguistic studies show that individuals with depression display heightened self-attention. This manifests as a significantly higher frequency of first-person singular pronouns (*I, me, my, myself, mine*) and a lower frequency of first-person plural pronouns (*we, us, our*).

### Code Snippet
```python
import re

def extract_pronoun_features(text):
    if not isinstance(text, str):
        return 0.0, 0.0
        
    words = text.lower().split()
    word_count = len(words)
    if word_count == 0:
        return 0.0, 0.0
        
    # First-person singular patterns
    fps_patterns = r'\b(i|me|my|myself|mine)\b'
    fps_count = len(re.findall(fps_patterns, text.lower()))
    
    # First-person plural patterns
    fpp_patterns = r'\b(we|us|our|ours|ourselves)\b'
    fpp_count = len(re.findall(fpp_patterns, text.lower()))
    
    return fps_count / word_count, fpp_count / word_count
```

---

## 2. Text Length and Density Metrics

### Theory
Depressed users online tend to write longer, more descriptive, narrative-like posts (e.g., sharing personal stories) compared to control group users.

### Code Snippet
```python
def extract_length_features(text):
    if not isinstance(text, str):
        return 0, 0, 0.0
        
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    
    avg_word_length = char_count / word_count if word_count > 0 else 0.0
    
    return char_count, word_count, avg_word_length
```

---

## 3. Punctuation and Capitalization Ratios

### Theory
Capitalization and punctuation distributions are proxies for emotional intensity. Frequent use of exclamation marks or capital letters (shouting) can indicate anxiety or manic states, while flat punctuation can indicate emotional flattening.

### Code Snippet
```python
def extract_punctuation_features(text):
    if not isinstance(text, str):
        return 0.0, 0.0, 0.0
        
    char_count = len(text)
    if char_count == 0:
        return 0.0, 0.0, 0.0
        
    excl_count = text.count('!') / char_count
    quest_count = text.count('?') / char_count
    
    # Uppercase ratio (excluding spaces)
    uppercase_chars = sum(1 for c in text if c.isupper())
    upper_ratio = uppercase_chars / char_count
    
    return excl_count, quest_count, upper_ratio
```

---

## 4. VADER Rule-Based Sentiment Analysis

### Theory
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based sentiment tool specifically attuned to social media style text (slang, emojis, capitalization). It outputs positive, negative, neutral, and compound scores and is significantly faster than `NRCLex`.

### Code Snippet
```python
# Requires: pip install nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Ensure vader_lexicon is downloaded
nltk.download('vader_lexicon', quiet=True)
analyzer = SentimentIntensityAnalyzer()

def extract_vader_features(text):
    if not isinstance(text, str):
        return 0.0, 0.0, 0.0, 0.0
        
    scores = analyzer.polarity_scores(text)
    return (
        scores['pos'],
        scores['neg'],
        scores['neu'],
        scores['compound']  # Normalized, weighted composite score (-1 to 1)
    )
```

---

## Integration Plan (Future Exploration)
To integrate these features into the pipeline:
1. Append these extracted features as additional dense columns alongside the TF-IDF sparse matrix before training.
2. Fit a `MinMaxScaler` on these metadata columns to ensure they are on the same scale ($[0, 1]$) as the TF-IDF values.
