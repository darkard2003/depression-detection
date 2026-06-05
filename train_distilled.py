#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from os import path
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ==============================================================================
# GLOBAL CONFIGURATION & ARGUMENT PARSING
# ==============================================================================
parser = argparse.ArgumentParser(description="Train distilled student MLP model.")
parser.add_argument("--data_dir", type=str, default="data_processed/processed_chi2", help="Data directory containing features and labels.")
parser.add_argument("--project_name", type=str, default="reddit_mlp_distilled", help="Project and model name.")
parser.add_argument("--alpha", type=float, default=0.1, help="Weight for hard labels (0 to 1).")
args = parser.parse_args()

DATA_DIR = args.data_dir
PROJECT_NAME = args.project_name
ALPHA = args.alpha

X_FILE = 'X_combined_sparse.npz'
Y_FILE = 'y.npy'
Y_TEACHER_FILE = 'y_teacher_soft.npy'

# Tuner and Hyperband settings
TUNER_EPOCHS = 20             # Trial search epochs
MAX_EPOCHS = 20               # Max Hyperband epochs
OVERWRITE_TUNER = False

# Model Training settings
FIT_EPOCHS = 200
BATCH_SIZE = 32
SEED = 42
# ==============================================================================

# 1. SETUP LOGGING
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

log_filename = f"logs/train_tfidf_{PROJECT_NAME}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("distilled_trainer")

logger.info("=" * 60)
logger.info(f"Starting Distilled Student Training Pipeline: {PROJECT_NAME}")
logger.info(f"Distillation Weight (Alpha): {ALPHA}")
logger.info("=" * 60)

# 2. SPARSE DATA GENERATOR
class SparseDistillGenerator(tf.keras.utils.Sequence):
    """Generates dense batches incrementally from Scipy sparse matrices.
    Returns blended soft labels: alpha * y_true + (1-alpha) * y_teacher.
    """
    def __init__(self, x, y_true, y_teacher, batch_size=32, shuffle=True, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.x = x
        self.shuffle = shuffle
        self.alpha = alpha
        
        self.y_true = np.asarray(y_true, dtype=np.float32)
        self.y_teacher = np.asarray(y_teacher, dtype=np.float32)
        
        # Blended Target: alpha * y_true + (1 - alpha) * y_teacher
        self.y_blend = self.alpha * self.y_true + (1.0 - self.alpha) * self.y_teacher

        self.indexes = np.arange(self.x.shape[0])
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.x.shape[0])
        indexes = self.indexes[start_idx:end_idx]

        x_batch_dense = self.x[indexes].toarray()
        y_batch = self.y_blend[indexes]
        return x_batch_dense, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# 4. MODEL BUILDER
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(INPUT_DIM,)))

    num_layers = hp.Int('num_layers', min_value=1, max_value=3)
    for i in range(num_layers):
        model.add(
            layers.Dense(
                units=hp.Int(f'units_hidden_{i}', min_value=32, max_value=512, step=32),
                activation=hp.Choice(f'activation_hidden_{i}', values=['relu', 'tanh'])
            )
        )
        model.add(layers.Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)))

    model.add(layers.Dense(1, activation='sigmoid'))

    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    # Use BinaryCrossentropy on blended soft target
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['mae'])
    return model

# 3. MAIN TRAINING FUNCTION
def main():
    logger.info("Loading sparse features, true labels, and teacher soft targets...")
    X_combined = load_npz(path.join(DATA_DIR, X_FILE))
    y_true = np.load(path.join(DATA_DIR, Y_FILE))
    y_teacher = np.load(path.join(DATA_DIR, Y_TEACHER_FILE))

    logger.info(f"Loaded Features: {X_combined.shape}")
    logger.info(f"True Labels Distribution: {np.bincount(y_true.astype(int))}")

    # Train/Val/Test Split (stratified by true label index)
    X_train_val, X_test, y_train_val, y_test, y_teach_train_val, y_teach_test = train_test_split(
        X_combined, y_true, y_teacher, test_size=0.2, stratify=y_true, random_state=SEED
    )
    global INPUT_DIM
    INPUT_DIM = X_train_val.shape[1]

    # Split train/val
    X_train, X_val, y_train, y_val, y_teach_train, y_teach_val = train_test_split(
        X_train_val, y_train_val, y_teach_train_val, test_size=0.2, stratify=y_train_val, random_state=SEED
    )

    # Generators
    train_generator = SparseDistillGenerator(X_train, y_train, y_teach_train, batch_size=BATCH_SIZE, alpha=ALPHA)
    val_generator = SparseDistillGenerator(X_val, y_val, y_teach_val, batch_size=BATCH_SIZE, shuffle=False, alpha=ALPHA)
    train_val_generator = SparseDistillGenerator(X_train_val, y_train_val, y_teach_train_val, batch_size=BATCH_SIZE, alpha=ALPHA)

    # 4. TUNING
    logger.info("Initializing Hyperband Tuner...")
    tuner = kt.Hyperband(
        build_model,
        objective=kt.Objective("val_loss", direction="min"),
        max_epochs=MAX_EPOCHS,
        factor=3,
        directory='models',
        project_name=PROJECT_NAME,
        overwrite=OVERWRITE_TUNER
    )

    stop_early = EarlyStopping(monitor='val_loss', patience=5)
    logger.info("Starting hyperparameter search...")
    tuner.search(
        train_generator,
        epochs=TUNER_EPOCHS,
        validation_data=val_generator,
        callbacks=[stop_early]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logger.info("Best Hyperparameters:")
    for param in best_hps.values:
        logger.info(f"  {param}: {best_hps.get(param)}")

    # 5. RETRAIN ON FULL TRAIN-VAL
    logger.info("Retraining best architecture on combined Train-Val split...")
    best_model = tuner.hypermodel.build(best_hps)
    fit_early = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # Validation generator uses pure true targets for final validation metrics checks
    test_generator_true = SparseDistillGenerator(X_test, y_test, y_test, batch_size=BATCH_SIZE, shuffle=False, alpha=1.0)
    val_generator_true = SparseDistillGenerator(X_val, y_val, y_val, batch_size=BATCH_SIZE, shuffle=False, alpha=1.0)

    history = best_model.fit(
        train_val_generator,
        epochs=FIT_EPOCHS,
        validation_data=val_generator_true,
        callbacks=[fit_early]
    )

    # 6. EVALUATE
    logger.info("Running evaluation on Test Split (against hard targets)...")
    y_pred_probs = []
    for i in range(len(test_generator_true)):
        x_b, _ = test_generator_true[i]
        y_pred_probs.extend(best_model.predict(x_b, verbose=0).flatten())

    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs >= 0.5).astype(int)

    f1 = f1_score(y_test, y_pred)
    logger.info(f"🎯 Test F1-Score: {f1:.5f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save best student model
    model_save_path = f"models/{PROJECT_NAME}_best.keras"
    best_model.save(model_save_path)
    logger.info(f"✓ Best student model saved to '{model_save_path}'")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()
