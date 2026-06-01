# Depression Detection NLP Project

In this workspace, we experiment with detecting depression in social media posts (e.g., Reddit) using Natural Language Processing (NLP) models.

### Key Components & Pipeline
- **Dataset:** Reddit social media posts (`bin_reddit1.csv`).
- **Feature Extraction (`train.py`):**
  - **TF-IDF (`tfidf`):** Statistical token/ngram features.
  - **BERT (`bert`):** Transformer-based word/sentence embeddings.
- **Model Training:** Keras Multilayer Perceptron (MLP) classification.
- **Hyperparameter Optimization:** Keras Tuner (Hyperband search) to find optimal network architecture and hyperparameters.
- **Imbalance Handling:** SMOTE and RandomOverSampler to mitigate class imbalance.

---

# Agents and Skills in this Workspace

This workspace features a set of specialized, context-saving "caveman" tools and subagent workflows designed to maximize token efficiency, prolong context limits, and keep interactions terse and highly technical.


---

## Subagent Delegation Architecture

When working on large-scale tasks in this workspace, the agent follows the **Locate → Fix → Verify** pattern to conserve context window capacity:

1. **Locate:** Use `cavecrew-investigator` or `research` subagents to find relevant code without loading excess files into the main context.
2. **Fix:** Delegate precise 1-2 file modifications to `cavecrew-builder`.
3. **Verify:** Use `cavecrew-reviewer` to audit diffs and catch errors before finalization.

---

## User Instructions & Guidelines

* **Implementation Flow:** When code modifications or tasks are requested, the agent must:
  1. First outline the exact implementation details/plan.
  2. Ask for user confirmation before performing any file edits or command executions.

