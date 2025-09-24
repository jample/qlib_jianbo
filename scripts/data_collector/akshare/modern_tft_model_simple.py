#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modern TFT Model Implementation - Qlib Compatible

A TensorFlow 2.x compatible implementation that follows qlib's ModelFT interface
and works with the enhanced Alpha158 features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from loguru import logger

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow available for Modern TFT")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available for Modern TFT")

# Dataset configuration compatible with original TFT - available regardless of TensorFlow
DATASET_SETTING = {
    "Alpha158": {
        "feature_col": [
            "RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10",
            "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5",
            "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"
        ],
        "label_col": "LABEL0",
    },
    "Shanghai_Alpha158": {
        "feature_col": [
            # Enhanced Alpha158 features including ROI and BOLL
            "RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10",
            "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5",
            "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW",
            # ROI features
            "ROI_1D", "ROI_VOL_1D", "ROI_5D", "ROI_VOL_5D", "ROI_RATIO_1D_5D",
            "ROI_CUM_1D", "ROI_CUM_5D", "ROI_SHARPE_1D", "ROI_SHARPE_5D",
            # BOLL features
            "BOLL_VOL_1D", "BOLL_MOMENTUM_1D", "BOLL_VOL_5D", "BOLL_MOMENTUM_5D",
            "BOLL_TREND_5D", "BOLL_VOL_20D", "BOLL_MOMENTUM_20D", "BOLL_TREND_20D",
            "BOLL_VOL_RATIO_1D_5D", "BB_UPPER", "BB_LOWER", "BB_WIDTH", "BB_POSITION"
        ],
        "label_col": "LABEL0",
    }
}

# Qlib imports
try:
    from qlib.model.base import ModelFT
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    logger.warning("Qlib not available - using fallback base class")

    # Fallback base class when qlib not available
    class ModelFT:
        def fit(self, dataset, **kwargs):
            raise NotImplementedError
        def predict(self, dataset):
            raise NotImplementedError


def get_shifted_label(data_df, shifts=5, col_shift="LABEL0"):
    """Get shifted labels for prediction targets"""
    return data_df[[col_shift]].groupby("instrument", group_keys=False).apply(lambda df: df.shift(shifts))


def fill_test_na(test_df):
    """Fill NaN values in test data"""
    test_df_res = test_df.copy()
    feature_cols = ~test_df_res.columns.str.contains("label", case=False)
    test_feature_fna = (
        test_df_res.loc[:, feature_cols].groupby("datetime", group_keys=False).apply(lambda df: df.fillna(df.mean()))
    )
    test_df_res.loc[:, feature_cols] = test_feature_fna
    return test_df_res


def transform_df(df, col_name="LABEL0"):
    """Transform qlib dataset format"""
    df_res = df["feature"]
    df_res[col_name] = df["label"]
    return df_res


class ModernTFTModel(ModelFT):
    """
    Modern TensorFlow 2.x compatible TFT-inspired model

    This is a qlib-compatible implementation that captures the key ideas of TFT
    but uses modern TensorFlow 2.x APIs and works with Python 3.12.
    Inherits from qlib's ModelFT for proper integration.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Modern TFT Model with qlib-compatible interface

        Args:
            **kwargs: Model parameters including DATASET, label_shift, etc.
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for ModernTFTModel. Install with: pip install tensorflow")

        # Default parameters compatible with original TFT
        self.params = {
            "DATASET": "Shanghai_Alpha158",
            "label_shift": 5,
            "feature_dim": 75,
            "hidden_dim": 128,
            "num_heads": 8,
            "num_layers": 4,
            "dropout_rate": 0.1,
            "learning_rate": 0.001,
            "sequence_length": 30,
            "prediction_horizon": 1
        }
        self.params.update(kwargs)

        # Model components
        self.model = None
        self.history = None
        self.data_formatter = None
        self.model_folder = None
        self.gpu_id = 0
        self.label_shift = self.params["label_shift"]
        self.expt_name = self.params["DATASET"]
        self.label_col = DATASET_SETTING[self.expt_name]["label_col"]

        logger.info(f"Initialized Modern TFT Model for {self.expt_name} with {self.params['feature_dim']} features")

    def _prepare_data(self, dataset: DatasetH):
        """Prepare qlib dataset for training - compatible with original TFT interface"""
        if not QLIB_AVAILABLE:
            raise ImportError("Qlib is required for dataset preparation")

        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        return transform_df(df_train), transform_df(df_valid)

    def _build_model(self, num_targets: int = 2):
        """Build the TFT-inspired model architecture"""

        # Get parameters
        feature_dim = self.params['feature_dim']
        hidden_dim = self.params['hidden_dim']
        num_heads = self.params['num_heads']
        num_layers = self.params['num_layers']
        dropout_rate = self.params['dropout_rate']
        learning_rate = self.params['learning_rate']
        sequence_length = self.params['sequence_length']

        # Input layer
        inputs = keras.Input(shape=(sequence_length, feature_dim), name='features')

        # Feature embedding and normalization
        x = layers.LayerNormalization()(inputs)
        x = layers.Dense(hidden_dim, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Multi-head attention layers (TFT-inspired)
        for i in range(num_layers):
            # Self-attention
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=hidden_dim // num_heads,
                dropout=dropout_rate,
                name=f'attention_{i}'
            )(x, x)

            # Add & Norm
            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization()(x)

            # Feed forward
            ff_output = layers.Dense(hidden_dim * 2, activation='relu')(x)
            ff_output = layers.Dropout(dropout_rate)(ff_output)
            ff_output = layers.Dense(hidden_dim)(ff_output)

            # Add & Norm
            x = layers.Add()([x, ff_output])
            x = layers.LayerNormalization()(x)

        # Global average pooling to get sequence representation
        x = layers.GlobalAveragePooling1D()(x)

        # Final prediction layers
        x = layers.Dense(hidden_dim, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Multi-target outputs
        outputs = []
        output_names = ['roi_ratio', 'boll_vol_ratio']
        
        for i in range(num_targets):
            output = layers.Dense(1, name=output_names[i] if i < len(output_names) else f'target_{i}')(x)
            outputs.append(output)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='ModernTFT')
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def fit(self, dataset: DatasetH, MODEL_FOLDER="qlib_tft_model", USE_GPU_ID=0, **kwargs):
        """Train the model using qlib dataset - compatible with original TFT interface"""
        logger.info("Training Modern TFT Model...")

        DATASET = self.params["DATASET"]
        LABEL_SHIFT = self.params["label_shift"]
        LABEL_COL = DATASET_SETTING[DATASET]["label_col"]

        # Store training parameters
        self.model_folder = MODEL_FOLDER
        self.gpu_id = USE_GPU_ID

        # Prepare data using qlib interface
        dtrain, dvalid = self._prepare_data(dataset)
        dtrain.loc[:, LABEL_COL] = get_shifted_label(dtrain, shifts=LABEL_SHIFT, col_shift=LABEL_COL)
        dvalid.loc[:, LABEL_COL] = get_shifted_label(dvalid, shifts=LABEL_SHIFT, col_shift=LABEL_COL)

        # Process data for modern TFT
        train = process_qlib_data(dtrain, DATASET, fillna=True).dropna()
        valid = process_qlib_data(dvalid, DATASET, fillna=True).dropna()

        # Convert to sequences for time series modeling
        X_train, y_train, X_val, y_val = process_qlib_data_for_modern_tft(
            train, valid,
            sequence_length=self.params['sequence_length'],
            feature_cols=DATASET_SETTING[DATASET]["feature_col"],
            label_col=LABEL_COL
        )

        # Determine number of targets (ROI + BOLL = 2 targets)
        num_targets = 2 if "Shanghai" in DATASET else 1

        # Build model
        self.model = self._build_model(num_targets)

        # Prepare training targets for multi-target
        if num_targets > 1:
            # For Shanghai dataset, create ROI and BOLL targets
            y_train_list = [y_train[:, 0:1], y_train[:, 0:1]]  # Duplicate for now, can be enhanced
            y_val_list = [y_val[:, 0:1], y_val[:, 0:1]]
            validation_data = (X_val, y_val_list)
        else:
            y_train_list = y_train
            validation_data = (X_val, y_val)

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # Train model
        self.history = self.model.fit(
            X_train, y_train_list,
            validation_data=validation_data,
            epochs=kwargs.get('epochs', 100),
            batch_size=kwargs.get('batch_size', 32),
            callbacks=callbacks,
            verbose=kwargs.get('verbose', 1)
        )

        logger.info("✅ Modern TFT Model training completed")
        return self.history.history

    def predict(self, dataset):
        """Generate predictions using qlib dataset - compatible with original TFT interface"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Prepare test data
        d_test = dataset.prepare("test", col_set=["feature", "label"])
        d_test = transform_df(d_test)
        d_test.loc[:, self.label_col] = get_shifted_label(d_test, shifts=self.label_shift, col_shift=self.label_col)
        test = process_qlib_data(d_test, self.expt_name, fillna=True).dropna()

        # Convert to sequences
        X_test, y_test = process_qlib_data_for_prediction(
            test,
            sequence_length=self.params['sequence_length'],
            feature_cols=DATASET_SETTING[self.expt_name]["feature_col"],
            label_col=self.label_col
        )

        # Generate predictions
        predictions = self.model.predict(X_test)

        # Handle multi-target outputs
        if isinstance(predictions, list):
            # Take average of multiple targets for compatibility
            predictions = np.mean(predictions, axis=0)

        # Format predictions to match original TFT output format
        predictions_series = format_predictions_for_qlib(predictions, test.index[-len(predictions):])

        logger.info(f"Generated {len(predictions_series)} predictions")
        return predictions_series
    
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


# Dataset setting already defined above - no need to redefine


def process_qlib_data(df, dataset, fillna=False):
    """Process qlib data for TFT - compatible with original interface"""
    # Several features selected manually
    feature_col = DATASET_SETTING[dataset]["feature_col"]
    label_col = [DATASET_SETTING[dataset]["label_col"]]
    temp_df = df.loc[:, feature_col + label_col]
    if fillna:
        temp_df = fill_test_na(temp_df)
    temp_df = temp_df.swaplevel()
    temp_df = temp_df.sort_index()
    temp_df = temp_df.reset_index(level=0)
    dates = pd.to_datetime(temp_df.index)
    temp_df["date"] = dates
    temp_df["day_of_week"] = dates.dayofweek
    temp_df["month"] = dates.month
    temp_df["year"] = dates.year
    temp_df["const"] = 1.0
    return temp_df


def process_qlib_data_for_modern_tft(train_df, valid_df, sequence_length=30, feature_cols=None, label_col="LABEL0"):
    """Process qlib data specifically for modern TFT model"""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for data processing")

    logger.info("Processing qlib data for Modern TFT...")

    # Extract features and labels
    if feature_cols is None:
        feature_cols = [col for col in train_df.columns if col not in [label_col, 'date', 'day_of_week', 'month', 'year', 'const', 'instrument']]

    train_features = train_df[feature_cols].values
    train_labels = train_df[label_col].values

    valid_features = valid_df[feature_cols].values
    valid_labels = valid_df[label_col].values

    # Create sequences
    def create_sequences(features, labels, seq_len):
        X, y = [], []
        for i in range(seq_len, len(features)):
            X.append(features[i-seq_len:i])
            y.append(labels[i])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_features, train_labels, sequence_length)
    X_val, y_val = create_sequences(valid_features, valid_labels, sequence_length)

    logger.info(f"Created sequences: Train {X_train.shape}, Val {X_val.shape}")

    return X_train, y_train, X_val, y_val


def process_qlib_data_for_prediction(test_df, sequence_length=30, feature_cols=None, label_col="LABEL0"):
    """Process qlib data for prediction"""
    if feature_cols is None:
        feature_cols = [col for col in test_df.columns if col not in [label_col, 'date', 'day_of_week', 'month', 'year', 'const', 'instrument']]

    test_features = test_df[feature_cols].values
    test_labels = test_df[label_col].values

    # Create sequences
    X_test, y_test = [], []
    for i in range(sequence_length, len(test_features)):
        X_test.append(test_features[i-sequence_length:i])
        y_test.append(test_labels[i])

    return np.array(X_test), np.array(y_test)


def format_predictions_for_qlib(predictions, index):
    """Format predictions to match qlib/original TFT output format"""
    # Convert to pandas Series with proper index
    if len(predictions.shape) > 1:
        predictions = predictions.flatten()

    # Create a series with the last N indices from the test data
    pred_series = pd.Series(predictions, index=index)
    return pred_series
