#!/usr/bin/env python3
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from os import path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner import Hyperband
import keras_tuner as kt
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

# ==============================================================================
# GLOBAL CONFIGURATION - EDIT MANUALLY
# ==============================================================================
# Paths and Filenames
DATA_DIR = 'processed_bert'
X_FILE = 'X_combined.npy'
Y_FILE = 'y.npy'

# Tuner and Hyperband settings
TUNER_EPOCHS = 20             # Trial search epochs
MAX_EPOCHS = 30               # Max Hyperband epochs
OVERWRITE_TUNER = False       # False to resume crashed runs, True to start fresh

# Model Training settings
FIT_EPOCHS = 200              # Max epochs for training the best model
BATCH_SIZE = 512              # Mini-batch size
SEED = 42                     # Random state seed

# Project/Model naming to match combined_tensor_bert_v5.ipynb
PROJECT_NAME = "reddit_mlp_bert_nrclex_v5_smote"
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. SETUP LOGGING AND DIRECTORIES
# ------------------------------------------------------------------------------
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

log_filename = "logs/train_bert.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("bert_trainer")

logger.info("=" * 60)
logger.info(f"Starting BERT MLP training pipeline: {PROJECT_NAME}")
logger.info("=" * 60)

# Check GPU availability and enable dynamic memory growth to keep RAM low
gpu_devices = tf.config.list_physical_devices('GPU')
if len(gpu_devices) > 0:
    logger.info(f"✅ GPU is detected: {gpu_devices}")
    try:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("✅ Enabled dynamic GPU memory growth to prevent pre-allocating all RAM.")
    except Exception as e:
        logger.warning(f"⚠️ Could not set dynamic GPU memory growth: {e}")
else:
    logger.info("❌ GPU NOT detected. TensorFlow will run on CPU.")



# ------------------------------------------------------------------------------
# 2. DATASET INPUT DIMENSION
# ------------------------------------------------------------------------------
# Store the input dimension statically
INPUT_DIM = 394


# ------------------------------------------------------------------------------
# 3. DATA LOADING AND PREPROCESSING
# ------------------------------------------------------------------------------
logger.info("Loading BERT embeddings and labels...")
x_path = path.join(DATA_DIR, X_FILE)
y_path = path.join(DATA_DIR, Y_FILE)

X_combined_dataset = np.load(x_path)
y_resampled = np.load(y_path)
y_resampled = pd.Series(y_resampled)

logger.info(f"Dataset loaded. Features shape: {X_combined_dataset.shape}, Labels shape: {y_resampled.shape}")

# Split Train_Val / Test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_combined_dataset, y_resampled, test_size=0.2, stratify=y_resampled, random_state=SEED
)
# Free massive initial dataset immediately since it's never used again
del X_combined_dataset
del y_resampled
import gc
gc.collect()

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=SEED
)

# Apply SMOTE to balance training splits
logger.info("Applying SMOTE oversampling...")
smote = SMOTE(random_state=SEED)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
X_train_val_resampled, y_train_val_resampled = smote.fit_resample(X_train_val, y_train_val)

# Free intermediate raw train splits immediately
del X_train
del y_train
gc.collect()

logger.info(f"Resampled Train split shape: {X_train_resampled.shape}")

# Convert y Series to standard numpy arrays to prevent label indexing mismatches
y_train_resampled_arr = y_train_resampled.to_numpy() if hasattr(y_train_resampled, "to_numpy") else np.asarray(y_train_resampled)
y_val_arr = y_val.to_numpy() if hasattr(y_val, "to_numpy") else np.asarray(y_val)
y_test_arr = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)
y_train_val_resampled_arr = y_train_val_resampled.to_numpy() if hasattr(y_train_val_resampled, "to_numpy") else np.asarray(y_train_val_resampled)

# Set up Native TF Datasets to prevent memory leak inside the GPU loop
logger.info("Setting up native tf.data.Dataset pipelines to prevent memory leaks...")
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_resampled, y_train_resampled_arr))
train_dataset = train_dataset.shuffle(buffer_size=2048, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_arr))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test_arr))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_val_dataset = tf.data.Dataset.from_tensor_slices((X_train_val_resampled, y_train_val_resampled_arr))
train_val_dataset = train_val_dataset.shuffle(buffer_size=2048, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



# ------------------------------------------------------------------------------
# 4. MODEL BUILDING FUNCTION
# ------------------------------------------------------------------------------
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(INPUT_DIM,)))

    # Input batch normalization
    model.add(layers.BatchNormalization())
    
    # Tunable deep dense layers
    num_layers = hp.Int('num_layers', min_value=2, max_value=5)
    for i in range(num_layers):
        model.add(
            layers.Dense(
                units=hp.Int(f'units_hidden_{i}', min_value=64, max_value=768, step=64),
                activation=hp.Choice(f'activation_hidden_{i}', values=['relu', 'tanh', 'elu']),
                kernel_regularizer=regularizers.l2(hp.Float(f'l2_{i}', min_value=1e-5, max_value=1e-2, sampling='log'))
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(layers.Dense(1, activation='sigmoid'))

    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(curve='PR', name='pr_auc')
        ]
    )
    return model


# ------------------------------------------------------------------------------
# 5. HYPERPARAMETER SEARCH
# ------------------------------------------------------------------------------
class MemoryEfficientHyperband(Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        result = super().run_trial(trial, *args, **kwargs)
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        return result

logger.info("Initializing Keras Tuner Hyperband Search...")
tuner = MemoryEfficientHyperband(
    build_model,
    objective=kt.Objective('val_pr_auc', direction='max'),
    max_epochs=MAX_EPOCHS,
    factor=3,
    directory='keras_mlp',
    project_name=PROJECT_NAME,
    overwrite=OVERWRITE_TUNER
)


search_early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

logger.info("Starting hyperparameter search...")
tuner.search(
    train_dataset,
    epochs=TUNER_EPOCHS,
    validation_data=val_dataset,
    callbacks=[search_early_stopping]
)
logger.info("Hyperparameter search finished.")

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
logger.info("Best Hyperparameters found:")
for hp_name, hp_val in best_hps.values.items():
    logger.info(f"  - {hp_name}: {hp_val}")


# ------------------------------------------------------------------------------
# 6. MODEL TRAINING & VAL-LOSS MINIMIZATION
# ------------------------------------------------------------------------------
logger.info("Building model with best hyperparameters...")
model = build_model(best_hps)

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

# Compute class weights for combined SMOTE + weight balancing
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_resampled),
    y=y_train_resampled
)
class_weight_dict = dict(enumerate(weights))

logger.info(f"Fitting validation-minimizing model for max {FIT_EPOCHS} epochs...")
history = model.fit(
    train_dataset,
    epochs=FIT_EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict
)

# Plot training & validation accuracy and loss values
os.makedirs(path.join('plots', PROJECT_NAME), exist_ok=True)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()
learning_curves_path = path.join('plots', PROJECT_NAME, 'learning_curves.png')
plt.savefig(learning_curves_path)
plt.close()
logger.info(f"Learning curves saved to {learning_curves_path}")


# ------------------------------------------------------------------------------
# 7. DECISION THRESHOLD TUNING (F1 optimization)
# ------------------------------------------------------------------------------
logger.info("Optimizing decision threshold on the validation set for F1-score...")
val_probs = model.predict(val_dataset).ravel()

threshold_grid = np.linspace(0.1, 0.9, 81)
f1_scores = [f1_score(y_val, (val_probs >= t).astype(int)) for t in threshold_grid]
best_threshold = float(threshold_grid[int(np.argmax(f1_scores))])
max_f1 = max(f1_scores)

logger.info(f"🎯 Best validation threshold (F1): {best_threshold:.3f}")
logger.info(f"🎯 Best validation F1-score: {max_f1:.4f}")


# ------------------------------------------------------------------------------
# 8. RETRAIN FINAL MODEL ON FULL DATASET
# ------------------------------------------------------------------------------
best_epoch_num = int(np.argmin(history.history['val_loss']) + 1)
logger.info(f"Re-training final model on FULL dataset (Train + Val) for {best_epoch_num} epochs...")

final_model = build_model(best_hps)

weights_full = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_val_resampled),
    y=y_train_val_resampled
)
class_weight_dict_full = dict(enumerate(weights_full))

final_model.fit(
    train_val_dataset,
    epochs=best_epoch_num,
    verbose=True,
    class_weight=class_weight_dict_full
)


# ------------------------------------------------------------------------------
# 9. EVALUATION ON TEST SET & GRAPH GENERATION
# ------------------------------------------------------------------------------
logger.info("Evaluating final model on test set...")
test_eval = final_model.evaluate(test_dataset)
test_loss, test_accuracy = test_eval[0], test_eval[1]
test_auc = test_eval[2] if len(test_eval) > 2 else 0.0
test_pr_auc = test_eval[3] if len(test_eval) > 3 else 0.0

logger.info(f"Test Loss: {test_loss:.4f}")
logger.info(f"Test Accuracy: {test_accuracy:.4f}")
logger.info(f"Test ROC-AUC: {test_auc:.4f}")
logger.info(f"Test PR-AUC: {test_pr_auc:.4f}")

# Predict on test set
y_pred_probs = final_model.predict(test_dataset).ravel()
y_pred = (y_pred_probs >= best_threshold).astype("int32")

# Classification Report
report_str = classification_report(y_test_arr, y_pred)
logger.info("\nClassification Report:\n" + report_str)

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test_arr, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Threshold: {best_threshold:.3f})')
cm_path = path.join('plots', PROJECT_NAME, 'confusion_matrix.png')
plt.savefig(cm_path)
plt.close()
logger.info(f"Confusion matrix saved to {cm_path}")

# ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test_arr, y_pred_probs)
roc_auc_val = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
roc_path = path.join('plots', PROJECT_NAME, 'roc_curve.png')
plt.savefig(roc_path)
plt.close()
logger.info(f"ROC curve saved to {roc_path}")

# PR Curve
precision, recall, pr_thresholds = precision_recall_curve(y_test_arr, y_pred_probs)
average_precision = average_precision_score(y_test_arr, y_pred_probs)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {average_precision:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
pr_path = path.join('plots', PROJECT_NAME, 'pr_curve.png')
plt.savefig(pr_path)
plt.close()
logger.info(f"PR curve saved to {pr_path}")


# ------------------------------------------------------------------------------
# 10. SAVE MODEL & METADATA EXPORT
# ------------------------------------------------------------------------------
logger.info("Saving trained final model...")
os.makedirs(path.join('models', PROJECT_NAME), exist_ok=True)
model_path = path.join('models', PROJECT_NAME, f'{PROJECT_NAME}.keras')
final_model.save(model_path)
logger.info(f"Final model successfully saved to: {model_path}")

# Write tuning/training metadata to JSON
metadata = {
    "mode": "bert",
    "project_name": PROJECT_NAME,
    "best_hyperparameters": best_hps.values,
    "decision_threshold": best_threshold,
    "max_val_f1": float(max_f1),
    "best_epoch_num": best_epoch_num,
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "test_auc": float(test_auc),
    "test_pr_auc": float(test_pr_auc),
    "history": {k: [float(x) for x in v] for k, v in history.history.items()}
}

metadata_path = path.join('models', PROJECT_NAME, f'{PROJECT_NAME}_metadata.json')
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=4)
logger.info(f"Training metadata successfully exported to: {metadata_path}")
logger.info("=" * 60)
logger.info("🎉 Training & tuning pipeline complete. Ready for notebook evaluation!")
logger.info("=" * 60)
