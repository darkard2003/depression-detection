"""Data generators for handling sparse matrix data efficiently with Keras/TensorFlow."""

import numpy as np
import tensorflow as tf


class SparseDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data for Keras from sparse matrices.
    
    This generator efficiently handles sparse data by only converting
    individual batches to dense format, saving memory.
    """
    
    def __init__(self, x, y, batch_size=32, shuffle=True):
        """Initialize the data generator.
        
        Args:
            x: Sparse matrix (scipy.sparse) of features
            y: Numpy array of labels
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data at the end of each epoch
        """
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.shuffle = shuffle
        self.indexes = np.arange(self.x.shape[0])
        self.on_epoch_end()  # Perform initial shuffle if enabled

    def __len__(self):
        """Return the number of batches per epoch."""
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data.
        
        Args:
            index: Batch index
            
        Returns:
            Tuple of (x_batch, y_batch) as numpy arrays
        """
        # Calculate batch boundaries
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, self.x.shape[0])
        
        # Get shuffled indexes for this batch
        batch_indexes = self.indexes[start_index:end_index]
        
        # Extract batch data
        x_batch_sparse = self.x[batch_indexes]
        y_batch = self.y[batch_indexes]
        
        # Convert sparse batch to dense (this is memory-efficient)
        x_batch_dense = x_batch_sparse.toarray()
        
        return x_batch_dense, y_batch

    def on_epoch_end(self):
        """Update indexes after each epoch (shuffle if enabled)."""
        if self.shuffle:
            np.random.shuffle(self.indexes)
