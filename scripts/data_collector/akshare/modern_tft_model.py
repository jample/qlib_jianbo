#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modern TFT Model Implementation

A TensorFlow 2.x compatible implementation for time series forecasting
that works with the enhanced Alpha158 features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from loguru import logger

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    logger.error("TensorFlow not available for modern TFT model")


if TENSORFLOW_AVAILABLE:
    class ModernTFTModel:
        """
        Modern TensorFlow 2.x compatible TFT-inspired model

        This is a simplified implementation that captures the key ideas of TFT
        but uses modern TensorFlow 2.x APIs and works with Python 3.12.
        """
    
        def __init__(self,
                     feature_dim: int = 75,
                     hidden_dim: int = 128,
                     num_heads: int = 8,
                     num_layers: int = 4,
                     dropout_rate: float = 0.1,
                     learning_rate: float = 0.001,
                     sequence_length: int = 30,
                     prediction_horizon: int = 1):
            """
            Initialize Modern TFT Model

            Args:
                feature_dim: Number of input features
                hidden_dim: Hidden dimension size
                num_heads: Number of attention heads
                num_layers: Number of transformer layers
                dropout_rate: Dropout rate
                learning_rate: Learning rate
                sequence_length: Input sequence length
                prediction_horizon: Prediction horizon
            """
            self.feature_dim = feature_dim
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.dropout_rate = dropout_rate
            self.learning_rate = learning_rate
            self.sequence_length = sequence_length
            self.prediction_horizon = prediction_horizon

            self.model = None
            self.history = None

            logger.info(f"Initialized Modern TFT Model with {feature_dim} features")
    
        def _build_model(self, num_targets: int = 2):
            """Build the TFT-inspired model architecture"""

            # Input layer
            inputs = keras.Input(shape=(self.sequence_length, self.feature_dim), name='features')

            # Feature embedding and normalization
            x = layers.LayerNormalization()(inputs)
            x = layers.Dense(self.hidden_dim, activation='relu')(x)
            x = layers.Dropout(self.dropout_rate)(x)

            # Multi-head attention layers (TFT-inspired)
            for i in range(self.num_layers):
                # Self-attention
                attention_output = layers.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.hidden_dim // self.num_heads,
                    dropout=self.dropout_rate,
                    name=f'attention_{i}'
                )(x, x)

                # Add & Norm
                x = layers.Add()([x, attention_output])
                x = layers.LayerNormalization()(x)

                # Feed forward
                ff_output = layers.Dense(self.hidden_dim * 2, activation='relu')(x)
                ff_output = layers.Dropout(self.dropout_rate)(ff_output)
                ff_output = layers.Dense(self.hidden_dim)(ff_output)

                # Add & Norm
                x = layers.Add()([x, ff_output])
                x = layers.LayerNormalization()(x)

            # Global average pooling to get sequence representation
            x = layers.GlobalAveragePooling1D()(x)

            # Final prediction layers
            x = layers.Dense(self.hidden_dim, activation='relu')(x)
            x = layers.Dropout(self.dropout_rate)(x)

            # Multi-target outputs
            outputs = []
            output_names = ['roi_ratio', 'boll_vol_ratio']

            for i in range(num_targets):
                output = layers.Dense(1, name=output_names[i] if i < len(output_names) else f'target_{i}')(x)
                outputs.append(output)

            model = keras.Model(inputs=inputs, outputs=outputs, name='ModernTFT')

            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )

            return model
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 1) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X_train: Training features [samples, sequence_length, features]
            y_train: Training targets [samples, num_targets]
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        logger.info("Training Modern TFT Model...")
        
        # Determine number of targets
        num_targets = y_train.shape[1] if len(y_train.shape) > 1 else 1
        
        # Build model
        self.model = self._build_model(num_targets)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            if num_targets > 1:
                validation_data = (X_val, [y_val[:, i:i+1] for i in range(num_targets)])
            else:
                validation_data = (X_val, y_val)
        
        # Prepare training targets
        if num_targets > 1:
            y_train_list = [y_train[:, i:i+1] for i in range(num_targets)]
        else:
            y_train_list = y_train
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train_list,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        logger.info("✅ Modern TFT Model training completed")
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            X: Input features [samples, sequence_length, features]
            
        Returns:
            Predictions [samples, num_targets]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        predictions = self.model.predict(X)
        
        # Handle single vs multi-target outputs
        if isinstance(predictions, list):
            return np.concatenate(predictions, axis=1)
        else:
            return predictions
    
    def save(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            logger.warning("Model not built yet")
            return
        
        self.model.summary()


# Dataset setting for compatibility
DATASET_SETTING = {}


def process_qlib_data(dataset, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process qlib dataset for modern TFT model
    
    Args:
        dataset: Qlib dataset
        sequence_length: Sequence length for time series
        
    Returns:
        X_train, y_train, X_val, y_val
    """
    logger.info("Processing qlib data for Modern TFT...")
    
    # Get data from qlib dataset
    train_data = dataset.prepare("train")
    val_data = dataset.prepare("valid")
    
    # Extract features and labels
    train_features = train_data[0].values
    train_labels = train_data[1].values
    
    val_features = val_data[0].values
    val_labels = val_data[1].values
    
    # Create sequences
    def create_sequences(features, labels, seq_len):
        X, y = [], []
        for i in range(seq_len, len(features)):
            X.append(features[i-seq_len:i])
            y.append(labels[i])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences(train_features, train_labels, sequence_length)
    X_val, y_val = create_sequences(val_features, val_labels, sequence_length)
    
    logger.info(f"Created sequences: Train {X_train.shape}, Val {X_val.shape}")
    
    return X_train, y_train, X_val, y_val

else:
    # Fallback class when TensorFlow is not available
    class ModernTFTModel:
        """Fallback ModernTFTModel when TensorFlow is not available"""

        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow is required for ModernTFTModel. Install with: pip install tensorflow")

        def fit(self, *args, **kwargs):
            raise ImportError("TensorFlow is required for ModernTFTModel")

        def predict(self, *args, **kwargs):
            raise ImportError("TensorFlow is required for ModernTFTModel")

        def save(self, *args, **kwargs):
            raise ImportError("TensorFlow is required for ModernTFTModel")

        def load(self, *args, **kwargs):
            raise ImportError("TensorFlow is required for ModernTFTModel")

    def process_qlib_data(*args, **kwargs):
        raise ImportError("TensorFlow is required for data processing")
