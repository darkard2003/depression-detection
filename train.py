#!/usr/bin/env python3
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from os import path
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner import Hyperband
import keras_tuner as kt
from sklearn.utils import class_weight
from sklearn.metrics import f1_score


# ==============================================================================
# GLOBAL CONFIGURATION - EDIT MANUALLY (NO AUTO-RESOLUTION MAGIC)
# ==============================================================================
MODE = 'bert'  # Options: 'bert' or 'tfidf'

# Paths and Filenames
DATA_DIR = 'processed_bert'  # 'processed_bert' or 'processed_dirty'
X_FILE = 'X_combined.npy'    # 'X_combined.npy' or 'X_combined_sparse.npz'
Y_FILE = 'y.npy'             # 'y.npy'

# Tuner and Hyperband settings
TUNER_EPOCHS = 20                                 # Trial search epochs
MAX_EPOCHS = 30                                   # Max Hyperband epochs (30 for bert, 20 for tfidf)
OVERWRITE_TUNER = False                           # False to resume crashed runs, True to start fresh

# Model Training settings
FIT_EPOCHS = 200        # Max epochs for training the best model
BATCH_SIZE = 512         # Mini-batch size
SEED = 42               # Random state seed

# Dynamic Project/Model naming based on manual parameters above
PROJECT_NAME = f"reddit_mlp_{MODE}_bs_{BATCH_SIZE}_seed_{SEED}"
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. SETUP LOGGING AND DIRECTORIES
# ------------------------------------------------------------------------------
# Create directories if they don't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Configure logging to print to both stdout and a file
log_filename = f"logs/train_{MODE}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("trainer")

logger.info("=" * 60)
logger.info(f"Starting depression classifier training script in MODE: {MODE}")
logger.info(f"Config: DATA_DIR={DATA_DIR}, X_FILE={X_FILE}, Y_FILE={Y_FILE}")
logger.info(f"Tuner Config: PROJECT_NAME={PROJECT_NAME}, TUNER_EPOCHS={TUNER_EPOCHS}, MAX_EPOCHS={MAX_EPOCHS}")
logger.info(f"Fit Config: FIT_EPOCHS={FIT_EPOCHS}, BATCH_SIZE={BATCH_SIZE}, SEED={SEED}")
logger.info("=" * 60)


# Check GPU availability
gpu_devices = tf.config.list_physical_devices('GPU')
if len(gpu_devices) > 0:
    logger.info(f"✅ GPU is detected and will be used by TensorFlow: {gpu_devices}")
else:
    logger.info("❌ GPU NOT detected. TensorFlow will run on CPU.")


# ------------------------------------------------------------------------------
# 2. UNIFIED DATA GENERATOR (To conserve RAM in both modes)
# ------------------------------------------------------------------------------
class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras dynamically from dense arrays or sparse matrices to conserve RAM."""
    def __init__(self, x, y, batch_size=32, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.x = x
        
        # Convert y to a flat numpy array immediately to prevent Pandas Series label index mismatches
        if hasattr(y, "to_numpy"):
            self.y = y.to_numpy()
        else:
            self.y = np.asarray(y)

        self.shuffle = shuffle
        self.indexes = np.arange(self.x.shape[0])
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, self.x.shape[0])
        indexes = self.indexes[start_index:end_index]

        x_batch = self.x[indexes]
        y_batch = self.y[indexes]

        # Convert sparse subset to dense array just for this batch
        if hasattr(x_batch, "toarray"):
            x_batch = x_batch.toarray()

        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)



# ------------------------------------------------------------------------------
# 3. DATA LOADING AND PREPROCESSING
# ------------------------------------------------------------------------------
logger.info("Loading features and labels...")
x_path = path.join(DATA_DIR, X_FILE)
y_path = path.join(DATA_DIR, Y_FILE)

# Load labels
y_resampled = np.load(y_path)
y_resampled = pd.Series(y_resampled)

# Load features dynamically based on dense (.npy) or sparse (.npz) format
if X_FILE.endswith('.npz'):
    logger.info("Loading sparse feature matrix (.npz)...")
    X_combined_dataset = load_npz(x_path)
else:
    logger.info("Loading dense feature matrix (.npy)...")
    X_combined_dataset = np.load(x_path)

logger.info(f"Dataset loaded. Features shape: {X_combined_dataset.shape}, Labels shape: {y_resampled.shape}")


# Split train_val/test first (20% test size)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_combined_dataset, y_resampled, test_size=0.2, stratify=y_resampled, random_state=SEED
)
logger.info(f"Train/Val split shape: {X_train_val.shape}, Test split shape: {X_test.shape}")
logger.info(f"Test split class counts:\n{y_test.value_counts()}")

# Split train/val (20% validation size)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=SEED
)
logger.info(f"Train split shape: {X_train.shape}, Val split shape: {X_val.shape}")

# Resampling to balance training set
logger.info(f"Applying oversampling/resampling for MODE: {MODE}...")
if MODE == 'bert':
    # BERT mode uses SMOTE for synthetic sample creation
    smote = SMOTE(random_state=SEED)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    X_train_val_resampled, y_train_val_resampled = smote.fit_resample(X_train_val, y_train_val)
else:
    # TF-IDF mode uses RandomOverSampler to keep it sparse-compatible
    ros = RandomOverSampler(random_state=SEED)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    X_train_val_resampled, y_train_val_resampled = ros.fit_resample(X_train_val, y_train_val)

logger.info(f"Resampled Train split shape: {X_train_resampled.shape}")
logger.info(f"Resampled Train class counts:\n{y_train_resampled.value_counts()}")

# Format Data for Keras (Generators)
logger.info("Setting up unified PyDataset generators for training flow to minimize RAM footprint...")
train_generator = DataGenerator(X_train_resampled, y_train_resampled, batch_size=BATCH_SIZE)
val_generator = DataGenerator(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
test_generator = DataGenerator(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
train_val_generator = DataGenerator(X_train_val_resampled, y_train_val_resampled, batch_size=BATCH_SIZE)





# ------------------------------------------------------------------------------
# 4. MODEL BUILDING FUNCTION
# ------------------------------------------------------------------------------
def build_bert_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))

    # BERT-based deep model with batch normalization and input-normalization
    model.add(layers.BatchNormalization())
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


def build_tfidf_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))

    # TF-IDF-based compact model optimized for high-dimensional sparse inputs
    l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-3, sampling='log')
    num_layers = hp.Int('num_layers', min_value=1, max_value=3)
    for i in range(num_layers):
        model.add(
            layers.Dense(
                units=hp.Int(f'units_hidden_{i}', min_value=32, max_value=512, step=32),
                activation=hp.Choice(f'activation_hidden_{i}', values=['relu', 'tanh']),
                kernel_regularizer=keras.regularizers.l2(l2_reg)
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

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(curve='PR', name='pr_auc')
        ]
    )
    return model


# Dynamically bind build_model function based on MODE
build_model = build_bert_model if MODE == 'bert' else build_tfidf_model



# ------------------------------------------------------------------------------
# 5. HYPERPARAMETER SEARCH
# ------------------------------------------------------------------------------
logger.info("Initializing Keras Tuner Hyperband Search...")
tuner = Hyperband(
    build_model,
    objective=kt.Objective('val_pr_auc', direction='max'),
    max_epochs=MAX_EPOCHS,
    factor=3,
    directory='keras_mlp',
    project_name=PROJECT_NAME,
    overwrite=OVERWRITE_TUNER
)

# Early stopping callback for search phase
search_early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

logger.info("Starting hyperparameter search...")
tuner.search(
    train_generator,
    epochs=TUNER_EPOCHS,
    validation_data=val_generator,
    callbacks=[search_early_stopping]
)

logger.info("Hyperparameter search finished.")


# Retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
logger.info("Best Hyperparameters found:")
for hp_name, hp_val in best_hps.values.items():
    logger.info(f"  - {hp_name}: {hp_val}")


# ------------------------------------------------------------------------------
# 6. MODEL TRAINING & VAL-LOSS MINIMIZATION
# ------------------------------------------------------------------------------
logger.info("Building the primary model with the best hyperparameters...")
model = build_model(best_hps)
logger.info(f"Model Summary:\n")
model.summary(print_fn=lambda x: logger.info(x))

# Training Callbacks
early_stop_patience = 15 if MODE == 'bert' else 10
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=early_stop_patience,
    restore_best_weights=True
)

callbacks_list = [early_stop]
class_weight_dict = None

if MODE == 'bert':
    # Add learning rate scheduler for BERT
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    callbacks_list.append(reduce_lr)

    # Compute class weights for combined SMOTE + weight balancing
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_resampled),
        y=y_train_resampled
    )
    class_weight_dict = dict(enumerate(weights))
    logger.info(f"Computed BERT class weights: {class_weight_dict}")


logger.info(f"Fitting validation-minimizing model for max {FIT_EPOCHS} epochs...")
history = model.fit(
    train_generator,
    epochs=FIT_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_list,
    class_weight=class_weight_dict
)


# ------------------------------------------------------------------------------
# 7. DECISION THRESHOLD TUNING (F1 optimization)
# ------------------------------------------------------------------------------
logger.info("Optimizing decision threshold on the validation set for F1-score...")
val_probs = model.predict(val_generator).ravel()

threshold_grid = np.linspace(0.1, 0.9, 81)
f1_scores = [f1_score(y_val, (val_probs >= t).astype(int)) for t in threshold_grid]
best_threshold = float(threshold_grid[int(np.argmax(f1_scores))])
max_f1 = max(f1_scores)

logger.info(f"🎯 Best validation threshold (F1): {best_threshold:.3f}")
logger.info(f"🎯 Best validation F1-score: {max_f1:.4f}")


# ------------------------------------------------------------------------------
# 8. RETRAIN FINAL MODEL ON FULL DATASET
# ------------------------------------------------------------------------------
# Find best training epoch index
best_epoch_num = int(np.argmin(history.history['val_loss']) + 1)
logger.info(f"Re-training final model on FULL dataset (Train + Val) for {best_epoch_num} epochs...")

final_model = build_model(best_hps)

class_weight_dict_full = None
if MODE == 'bert':
    # Class weights for final full train
    weights_full = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_val_resampled),
        y=y_train_val_resampled
    )
    class_weight_dict_full = dict(enumerate(weights_full))

final_model.fit(
    train_val_generator,
    epochs=best_epoch_num,
    verbose=True,
    class_weight=class_weight_dict_full
)


# ------------------------------------------------------------------------------
# 9. SAVE MODEL & METADATA EXPORT
# ------------------------------------------------------------------------------
logger.info("Saving trained final model...")
model_path = path.join('models', f'{PROJECT_NAME}.keras')
final_model.save(model_path)
logger.info(f"Final model successfully saved to: {model_path}")

# Write tuning/training metadata to JSON (so you can load them in a notebook to plot/evaluate)
metadata = {
    "mode": MODE,
    "project_name": PROJECT_NAME,
    "best_hyperparameters": best_hps.values,
    "decision_threshold": best_threshold,
    "max_val_f1": float(max_f1),
    "best_epoch_num": best_epoch_num,
    "history": {k: [float(x) for x in v] for k, v in history.history.items()}
}

metadata_path = path.join('models', f'{PROJECT_NAME}_metadata.json')
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=4)
logger.info(f"Training metadata successfully exported to: {metadata_path}")
logger.info("=" * 60)
logger.info("🎉 Training & tuning pipeline complete. Ready for notebook evaluation!")
logger.info("=" * 60)

