#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training Data Preparation Module

This module combines all components to prepare clean, normalized, feature-rich
datasets ready for machine learning model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TrainingDataPreparator:
    """Prepare training-ready datasets from processed stock data"""
    
    def __init__(self, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 min_history_days: int = 60,
                 prediction_horizon: int = 1):
        """
        Initialize training data preparator
        
        Args:
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            min_history_days: Minimum history required for each sample
            prediction_horizon: Days ahead to predict (for labels)
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_history_days = min_history_days
        self.prediction_horizon = prediction_horizon
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        logger.info(f"Initialized training data preparator with ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    def create_labels(self, df: pd.DataFrame, label_type: str = 'return') -> pd.DataFrame:
        """
        Create prediction labels
        
        Args:
            df: DataFrame with stock data
            label_type: Type of label ('return', 'direction', 'volatility')
            
        Returns:
            DataFrame with labels added
        """
        try:
            logger.info(f"Creating {label_type} labels with horizon={self.prediction_horizon}")
            
            df_with_labels = df.copy()
            
            if 'symbol' in df.columns:
                # Create labels by symbol group
                for symbol in df['symbol'].unique():
                    mask = df_with_labels['symbol'] == symbol
                    symbol_data = df_with_labels.loc[mask].copy()
                    
                    if label_type == 'return':
                        # Future return
                        symbol_data['label'] = symbol_data['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)

                    elif label_type == 'direction':
                        # Future price direction (1 for up, 0 for down)
                        future_return = symbol_data['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
                        symbol_data['label'] = (future_return > 0).astype(int)

                    elif label_type == 'volatility':
                        # Future volatility (rolling std of returns)
                        returns = symbol_data['close'].pct_change()
                        symbol_data['label'] = returns.rolling(self.prediction_horizon).std().shift(-self.prediction_horizon)

                    # Update the main dataframe
                    df_with_labels.loc[mask, symbol_data.columns] = symbol_data
            else:
                # Single symbol
                if label_type == 'return':
                    df_with_labels['label'] = df_with_labels['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
                
                elif label_type == 'direction':
                    future_return = df_with_labels['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
                    df_with_labels['label'] = (future_return > 0).astype(int)
                
                elif label_type == 'volatility':
                    returns = df_with_labels['close'].pct_change()
                    df_with_labels['label'] = returns.rolling(self.prediction_horizon).std().shift(-self.prediction_horizon)
            
            # Remove rows without labels (at the end of each symbol's data)
            if 'label' in df_with_labels.columns:
                initial_count = len(df_with_labels)
                df_with_labels = df_with_labels.dropna(subset=['label'])
                final_count = len(df_with_labels)
                logger.info(f"Removed {initial_count - final_count} rows without labels")
            else:
                logger.error("Label column not found in DataFrame")
                raise ValueError("Label column not created successfully")

            logger.info(f"Created labels for {len(df_with_labels)} samples")

            return df_with_labels
            
        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            raise
    
    def create_time_based_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-based train/validation/test splits
        
        Args:
            df: DataFrame with date column
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        try:
            logger.info("Creating time-based data splits...")
            
            # Sort by date
            df_sorted = df.sort_values('date').reset_index(drop=True)
            
            # Calculate split points
            total_samples = len(df_sorted)
            train_end = int(total_samples * self.train_ratio)
            val_end = int(total_samples * (self.train_ratio + self.val_ratio))
            
            # Create splits
            train_df = df_sorted.iloc[:train_end].copy()
            val_df = df_sorted.iloc[train_end:val_end].copy()
            test_df = df_sorted.iloc[val_end:].copy()
            
            logger.info(f"Data splits created: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
            logger.info(f"Date ranges - Train: {train_df['date'].min()} to {train_df['date'].max()}")
            logger.info(f"Date ranges - Val: {val_df['date'].min()} to {val_df['date'].max()}")
            logger.info(f"Date ranges - Test: {test_df['date'].min()} to {test_df['date'].max()}")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"Error creating time-based splits: {e}")
            raise
    
    def prepare_feature_matrix(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix and labels for training
        
        Args:
            df: DataFrame with features and labels
            feature_columns: Columns to use as features (None for auto-detection)
            
        Returns:
            Tuple of (features array, labels array, feature names)
        """
        try:
            logger.info("Preparing feature matrix...")
            
            if feature_columns is None:
                # Auto-detect feature columns (exclude basic columns and labels)
                exclude_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'label']
                feature_columns = [col for col in df.columns if col not in exclude_cols]
            
            # Extract features and labels
            X = df[feature_columns].values
            y = df['label'].values
            
            # Handle any remaining missing values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.info(f"Feature matrix prepared: {X.shape[0]} samples, {X.shape[1]} features")
            
            return X, y, feature_columns
            
        except Exception as e:
            logger.error(f"Error preparing feature matrix: {e}")
            raise
    
    def create_sequence_data(self, df: pd.DataFrame, sequence_length: int = 20, feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create sequence data for time series models (LSTM, etc.)
        
        Args:
            df: DataFrame with features and labels
            sequence_length: Length of input sequences
            feature_columns: Columns to use as features
            
        Returns:
            Tuple of (sequence features, labels, feature names)
        """
        try:
            logger.info(f"Creating sequence data with length={sequence_length}")
            
            if feature_columns is None:
                exclude_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'label']
                feature_columns = [col for col in df.columns if col not in exclude_cols]
            
            sequences = []
            labels = []
            
            if 'symbol' in df.columns:
                # Create sequences by symbol
                for symbol in df['symbol'].unique():
                    symbol_data = df[df['symbol'] == symbol].sort_values('date')
                    
                    if len(symbol_data) < sequence_length + 1:
                        continue
                    
                    # Extract features and labels for this symbol
                    symbol_features = symbol_data[feature_columns].values
                    symbol_labels = symbol_data['label'].values
                    
                    # Create sequences
                    for i in range(len(symbol_data) - sequence_length):
                        sequences.append(symbol_features[i:i+sequence_length])
                        labels.append(symbol_labels[i+sequence_length])
            else:
                # Single symbol
                df_sorted = df.sort_values('date')
                features = df_sorted[feature_columns].values
                labels_array = df_sorted['label'].values
                
                for i in range(len(df_sorted) - sequence_length):
                    sequences.append(features[i:i+sequence_length])
                    labels.append(labels_array[i+sequence_length])
            
            X_seq = np.array(sequences)
            y_seq = np.array(labels)
            
            # Handle missing values
            X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
            y_seq = np.nan_to_num(y_seq, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.info(f"Sequence data created: {X_seq.shape[0]} sequences, {X_seq.shape[1]} timesteps, {X_seq.shape[2]} features")
            
            return X_seq, y_seq, feature_columns
            
        except Exception as e:
            logger.error(f"Error creating sequence data: {e}")
            raise
    
    def save_training_data(self, 
                          train_data: Tuple[np.ndarray, np.ndarray], 
                          val_data: Tuple[np.ndarray, np.ndarray], 
                          test_data: Tuple[np.ndarray, np.ndarray],
                          feature_names: List[str],
                          output_dir: str) -> Dict[str, str]:
        """
        Save training data to files
        
        Args:
            train_data: Training features and labels
            val_data: Validation features and labels
            test_data: Test features and labels
            feature_names: List of feature names
            output_dir: Directory to save files
            
        Returns:
            Dictionary with file paths
        """
        try:
            logger.info("Saving training data...")
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save arrays
            file_paths = {}
            
            # Training data
            np.save(output_path / 'X_train.npy', train_data[0])
            np.save(output_path / 'y_train.npy', train_data[1])
            file_paths['X_train'] = str(output_path / 'X_train.npy')
            file_paths['y_train'] = str(output_path / 'y_train.npy')
            
            # Validation data
            np.save(output_path / 'X_val.npy', val_data[0])
            np.save(output_path / 'y_val.npy', val_data[1])
            file_paths['X_val'] = str(output_path / 'X_val.npy')
            file_paths['y_val'] = str(output_path / 'y_val.npy')
            
            # Test data
            np.save(output_path / 'X_test.npy', test_data[0])
            np.save(output_path / 'y_test.npy', test_data[1])
            file_paths['X_test'] = str(output_path / 'X_test.npy')
            file_paths['y_test'] = str(output_path / 'y_test.npy')
            
            # Save feature names
            with open(output_path / 'feature_names.txt', 'w') as f:
                for name in feature_names:
                    f.write(f"{name}\n")
            file_paths['feature_names'] = str(output_path / 'feature_names.txt')
            
            # Save metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'train_samples': train_data[0].shape[0],
                'val_samples': val_data[0].shape[0],
                'test_samples': test_data[0].shape[0],
                'num_features': len(feature_names),
                'data_shape': train_data[0].shape,
                'config': {
                    'train_ratio': self.train_ratio,
                    'val_ratio': self.val_ratio,
                    'test_ratio': self.test_ratio,
                    'prediction_horizon': self.prediction_horizon
                }
            }
            
            import json
            with open(output_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            file_paths['metadata'] = str(output_path / 'metadata.json')
            
            logger.info(f"Training data saved to {output_path}")
            logger.info(f"Files created: {list(file_paths.keys())}")
            
            return file_paths
            
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
            raise

    def prepare_complete_training_dataset(self,
                                        df: pd.DataFrame,
                                        label_type: str = 'return',
                                        sequence_length: Optional[int] = None,
                                        feature_columns: Optional[List[str]] = None,
                                        output_dir: Optional[str] = None) -> Dict:
        """
        Complete training dataset preparation pipeline

        Args:
            df: Processed DataFrame with features
            label_type: Type of labels to create
            sequence_length: Length for sequence data (None for regular tabular data)
            feature_columns: Columns to use as features
            output_dir: Directory to save data (None to skip saving)

        Returns:
            Dictionary with prepared datasets and metadata
        """
        try:
            logger.info("Starting complete training dataset preparation...")

            # Step 1: Create labels
            df_with_labels = self.create_labels(df, label_type)

            # Step 2: Create time-based splits
            train_df, val_df, test_df = self.create_time_based_splits(df_with_labels)

            # Step 3: Prepare feature matrices
            if sequence_length is not None:
                # Create sequence data
                X_train, y_train, feature_names = self.create_sequence_data(train_df, sequence_length, feature_columns)
                X_val, y_val, _ = self.create_sequence_data(val_df, sequence_length, feature_columns)
                X_test, y_test, _ = self.create_sequence_data(test_df, sequence_length, feature_columns)
                data_type = 'sequence'
            else:
                # Create regular tabular data
                X_train, y_train, feature_names = self.prepare_feature_matrix(train_df, feature_columns)
                X_val, y_val, _ = self.prepare_feature_matrix(val_df, feature_columns)
                X_test, y_test, _ = self.prepare_feature_matrix(test_df, feature_columns)
                data_type = 'tabular'

            # Prepare result dictionary
            result = {
                'train_data': (X_train, y_train),
                'val_data': (X_val, y_val),
                'test_data': (X_test, y_test),
                'feature_names': feature_names,
                'metadata': {
                    'data_type': data_type,
                    'label_type': label_type,
                    'sequence_length': sequence_length,
                    'train_shape': X_train.shape,
                    'val_shape': X_val.shape,
                    'test_shape': X_test.shape,
                    'num_features': len(feature_names),
                    'date_ranges': {
                        'train': (train_df['date'].min(), train_df['date'].max()),
                        'val': (val_df['date'].min(), val_df['date'].max()),
                        'test': (test_df['date'].min(), test_df['date'].max())
                    }
                }
            }

            # Step 4: Save data if output directory provided
            if output_dir is not None:
                file_paths = self.save_training_data(
                    (X_train, y_train),
                    (X_val, y_val),
                    (X_test, y_test),
                    feature_names,
                    output_dir
                )
                result['file_paths'] = file_paths

            logger.info("Training dataset preparation completed successfully")
            logger.info(f"Dataset summary: {result['metadata']}")

            return result

        except Exception as e:
            logger.error(f"Error in complete training dataset preparation: {e}")
            raise

    def get_data_summary(self, result: Dict) -> Dict:
        """Get summary of prepared training data"""
        try:
            X_train, y_train = result['train_data']
            X_val, y_val = result['val_data']
            X_test, y_test = result['test_data']

            summary = {
                'dataset_info': {
                    'total_samples': X_train.shape[0] + X_val.shape[0] + X_test.shape[0],
                    'train_samples': X_train.shape[0],
                    'val_samples': X_val.shape[0],
                    'test_samples': X_test.shape[0],
                    'num_features': len(result['feature_names']),
                    'data_type': result['metadata']['data_type']
                },
                'label_statistics': {
                    'train_mean': float(np.mean(y_train)),
                    'train_std': float(np.std(y_train)),
                    'val_mean': float(np.mean(y_val)),
                    'val_std': float(np.std(y_val)),
                    'test_mean': float(np.mean(y_test)),
                    'test_std': float(np.std(y_test))
                },
                'feature_statistics': {
                    'train_feature_mean': float(np.mean(X_train)),
                    'train_feature_std': float(np.std(X_train)),
                    'feature_names_sample': result['feature_names'][:10]  # First 10 features
                }
            }

            logger.info(f"Data summary: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {}
