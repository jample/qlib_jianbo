#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Normalization and Cleaning Pipeline

This module implements robust data cleaning, outlier detection, normalization
(z-score, rank-based), and feature scaling for training preparation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
import warnings
warnings.filterwarnings('ignore')


class DataNormalizer:
    """Normalize and clean stock data for training"""
    
    def __init__(self, 
                 outlier_method: str = 'iqr',
                 outlier_threshold: float = 3.0,
                 normalization_method: str = 'zscore',
                 fill_method: str = 'forward'):
        """
        Initialize data normalizer
        
        Args:
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'quantile')
            outlier_threshold: Threshold for outlier detection
            normalization_method: Normalization method ('zscore', 'robust', 'minmax', 'quantile', 'rank')
            fill_method: Method for filling missing values ('forward', 'backward', 'mean', 'median')
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.normalization_method = normalization_method
        self.fill_method = fill_method
        
        # Store scalers for inverse transformation
        self.scalers = {}
        self.feature_stats = {}
        
        logger.info(f"Initialized data normalizer with method={normalization_method}, outlier_method={outlier_method}")
    
    def detect_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Detect outliers in the data
        
        Args:
            df: DataFrame to check for outliers
            columns: Columns to check (None for all numeric columns)
            
        Returns:
            DataFrame with outlier flags
        """
        try:
            logger.info(f"Detecting outliers using {self.outlier_method} method...")
            
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
                # Exclude date and symbol columns
                columns = [col for col in columns if col not in ['date', 'symbol']]
            
            outlier_flags = pd.DataFrame(index=df.index)
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                
                if self.outlier_method == 'iqr':
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.outlier_threshold * IQR
                    upper_bound = Q3 + self.outlier_threshold * IQR
                    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                
                elif self.outlier_method == 'zscore':
                    z_scores = np.abs((df[col] - series.mean()) / series.std())
                    outliers = z_scores > self.outlier_threshold
                
                elif self.outlier_method == 'quantile':
                    lower_bound = series.quantile(self.outlier_threshold / 100)
                    upper_bound = series.quantile(1 - self.outlier_threshold / 100)
                    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                
                else:
                    raise ValueError(f"Unknown outlier method: {self.outlier_method}")
                
                outlier_flags[f'{col}_outlier'] = outliers
            
            total_outliers = outlier_flags.sum().sum()
            logger.info(f"Detected {total_outliers} outliers across {len(columns)} columns")
            
            return outlier_flags
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            raise
    
    def handle_outliers(self, df: pd.DataFrame, outlier_flags: pd.DataFrame, method: str = 'clip') -> pd.DataFrame:
        """
        Handle outliers in the data
        
        Args:
            df: Original DataFrame
            outlier_flags: DataFrame with outlier flags
            method: Method to handle outliers ('clip', 'remove', 'winsorize')
            
        Returns:
            DataFrame with outliers handled
        """
        try:
            logger.info(f"Handling outliers using {method} method...")
            
            df_clean = df.copy()
            
            for col in outlier_flags.columns:
                if not col.endswith('_outlier'):
                    continue
                
                feature_col = col.replace('_outlier', '')
                if feature_col not in df.columns:
                    continue
                
                outlier_mask = outlier_flags[col]
                
                if method == 'clip':
                    # Clip to 1st and 99th percentiles
                    lower_bound = df[feature_col].quantile(0.01)
                    upper_bound = df[feature_col].quantile(0.99)
                    df_clean.loc[outlier_mask, feature_col] = np.clip(
                        df_clean.loc[outlier_mask, feature_col], 
                        lower_bound, 
                        upper_bound
                    )
                
                elif method == 'winsorize':
                    # Winsorize to 5th and 95th percentiles
                    lower_bound = df[feature_col].quantile(0.05)
                    upper_bound = df[feature_col].quantile(0.95)
                    df_clean.loc[outlier_mask & (df_clean[feature_col] < lower_bound), feature_col] = lower_bound
                    df_clean.loc[outlier_mask & (df_clean[feature_col] > upper_bound), feature_col] = upper_bound
                
                elif method == 'remove':
                    # Mark for removal (will be handled later)
                    df_clean.loc[outlier_mask, feature_col] = np.nan
            
            if method == 'remove':
                # Remove rows with too many missing values
                missing_threshold = 0.5  # Remove rows with >50% missing features
                df_clean = df_clean.dropna(thresh=int(len(df_clean.columns) * missing_threshold))
            
            logger.info(f"Outlier handling completed. Remaining records: {len(df_clean)}")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error handling outliers: {e}")
            raise
    
    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in the data
        
        Args:
            df: DataFrame with missing values
            
        Returns:
            DataFrame with missing values filled
        """
        try:
            logger.info(f"Filling missing values using {self.fill_method} method...")
            
            df_filled = df.copy()
            
            # Get numeric columns (exclude date and symbol)
            numeric_cols = df_filled.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ['date', 'symbol']]
            
            if 'symbol' in df.columns:
                # Fill by symbol group
                for symbol in df['symbol'].unique():
                    mask = df_filled['symbol'] == symbol
                    symbol_data = df_filled.loc[mask, numeric_cols]
                    
                    if self.fill_method == 'forward':
                        symbol_data = symbol_data.fillna(method='ffill').fillna(method='bfill')
                    elif self.fill_method == 'backward':
                        symbol_data = symbol_data.fillna(method='bfill').fillna(method='ffill')
                    elif self.fill_method == 'mean':
                        symbol_data = symbol_data.fillna(symbol_data.mean())
                    elif self.fill_method == 'median':
                        symbol_data = symbol_data.fillna(symbol_data.median())
                    
                    df_filled.loc[mask, numeric_cols] = symbol_data
            else:
                # Fill entire DataFrame
                if self.fill_method == 'forward':
                    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(method='ffill').fillna(method='bfill')
                elif self.fill_method == 'backward':
                    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(method='bfill').fillna(method='ffill')
                elif self.fill_method == 'mean':
                    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())
                elif self.fill_method == 'median':
                    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].median())
            
            # Final fill with 0 for any remaining missing values
            df_filled[numeric_cols] = df_filled[numeric_cols].fillna(0)
            
            remaining_missing = df_filled[numeric_cols].isnull().sum().sum()
            logger.info(f"Missing value filling completed. Remaining missing values: {remaining_missing}")
            
            return df_filled
            
        except Exception as e:
            logger.error(f"Error filling missing values: {e}")
            raise
    
    def normalize_features(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalize features using the specified method
        
        Args:
            df: DataFrame to normalize
            feature_columns: Columns to normalize (None for all numeric columns)
            
        Returns:
            Normalized DataFrame
        """
        try:
            logger.info(f"Normalizing features using {self.normalization_method} method...")
            
            if feature_columns is None:
                feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                # Exclude basic price/volume columns and date/symbol
                exclude_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']
                feature_columns = [col for col in feature_columns if col not in exclude_cols]
            
            df_normalized = df.copy()
            
            if self.normalization_method == 'zscore':
                scaler = StandardScaler()
            elif self.normalization_method == 'robust':
                scaler = RobustScaler()
            elif self.normalization_method == 'minmax':
                scaler = MinMaxScaler()
            elif self.normalization_method == 'quantile':
                scaler = QuantileTransformer(output_distribution='normal', random_state=42)
            elif self.normalization_method == 'rank':
                # Rank-based normalization (cross-sectional)
                if 'symbol' in df.columns and 'date' in df.columns:
                    for col in feature_columns:
                        df_normalized[col] = df.groupby('date')[col].transform(
                            lambda x: x.rank(pct=True) - 0.5  # Center around 0
                        )
                else:
                    for col in feature_columns:
                        df_normalized[col] = df[col].rank(pct=True) - 0.5
                
                logger.info(f"Applied rank normalization to {len(feature_columns)} features")
                return df_normalized
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")
            
            # Apply sklearn scaler
            if 'symbol' in df.columns:
                # Normalize by symbol group to preserve time series properties
                for symbol in df['symbol'].unique():
                    mask = df_normalized['symbol'] == symbol
                    symbol_data = df_normalized.loc[mask, feature_columns]
                    
                    if len(symbol_data) > 1:  # Need at least 2 points for normalization
                        normalized_data = scaler.fit_transform(symbol_data)
                        df_normalized.loc[mask, feature_columns] = normalized_data
                        
                        # Store scaler for this symbol
                        self.scalers[symbol] = scaler
            else:
                # Normalize entire dataset
                normalized_data = scaler.fit_transform(df_normalized[feature_columns])
                df_normalized[feature_columns] = normalized_data
                self.scalers['global'] = scaler
            
            # Store feature statistics
            self.feature_stats = {
                'columns': feature_columns,
                'method': self.normalization_method,
                'stats': {
                    col: {
                        'mean': float(df_normalized[col].mean()),
                        'std': float(df_normalized[col].std()),
                        'min': float(df_normalized[col].min()),
                        'max': float(df_normalized[col].max())
                    } for col in feature_columns
                }
            }
            
            logger.info(f"Feature normalization completed for {len(feature_columns)} features")
            
            return df_normalized
            
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            raise

    def cross_sectional_normalize(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply cross-sectional normalization (normalize across stocks at each time point)

        Args:
            df: DataFrame with multi-stock data
            feature_columns: Columns to normalize

        Returns:
            Cross-sectionally normalized DataFrame
        """
        try:
            if 'symbol' not in df.columns or 'date' not in df.columns:
                logger.warning("Cross-sectional normalization requires 'symbol' and 'date' columns")
                return df

            logger.info("Applying cross-sectional normalization...")

            if feature_columns is None:
                feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                exclude_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']
                feature_columns = [col for col in feature_columns if col not in exclude_cols]

            df_cs_norm = df.copy()

            for col in feature_columns:
                if col in df.columns:
                    # Z-score normalization across stocks at each date
                    df_cs_norm[f'{col}_cs_zscore'] = df.groupby('date')[col].transform(
                        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                    )

                    # Rank normalization across stocks at each date
                    df_cs_norm[f'{col}_cs_rank'] = df.groupby('date')[col].transform(
                        lambda x: x.rank(pct=True) - 0.5
                    )

            logger.info(f"Cross-sectional normalization completed for {len(feature_columns)} features")

            return df_cs_norm

        except Exception as e:
            logger.error(f"Error in cross-sectional normalization: {e}")
            raise

    def clean_and_normalize_pipeline(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete data cleaning and normalization pipeline

        Args:
            df: Raw DataFrame with features
            feature_columns: Columns to process (None for auto-detection)

        Returns:
            Tuple of (cleaned and normalized DataFrame, validation results)
        """
        try:
            logger.info("Starting complete data cleaning and normalization pipeline...")

            # Step 1: Detect outliers
            outlier_flags = self.detect_outliers(df, feature_columns)

            # Step 2: Handle outliers
            df_clean = self.handle_outliers(df, outlier_flags, method='clip')

            # Step 3: Fill missing values
            df_filled = self.fill_missing_values(df_clean)

            # Step 4: Normalize features
            df_normalized = self.normalize_features(df_filled, feature_columns)

            # Step 5: Apply cross-sectional normalization if applicable
            if 'symbol' in df.columns and len(df['symbol'].unique()) > 1:
                df_normalized = self.cross_sectional_normalize(df_normalized, feature_columns)

            # Step 6: Validate results
            validation_results = self.validate_normalized_data(df_normalized)

            logger.info("Data cleaning and normalization pipeline completed successfully")

            return df_normalized, validation_results

        except Exception as e:
            logger.error(f"Error in cleaning and normalization pipeline: {e}")
            raise

    def validate_normalized_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate the normalized data quality

        Args:
            df: Normalized DataFrame

        Returns:
            Validation results dictionary
        """
        try:
            logger.info("Validating normalized data...")

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ['date', 'symbol']]

            validation_results = {
                'total_records': len(df),
                'total_features': len(numeric_cols),
                'missing_values': df[numeric_cols].isnull().sum().sum(),
                'infinite_values': np.isinf(df[numeric_cols]).sum().sum(),
                'feature_stats': {},
                'data_quality_score': 0.0
            }

            # Calculate feature statistics
            for col in numeric_cols[:20]:  # First 20 features for summary
                if col in df.columns:
                    validation_results['feature_stats'][col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'skewness': float(df[col].skew()),
                        'kurtosis': float(df[col].kurtosis())
                    }

            # Calculate data quality score
            missing_ratio = validation_results['missing_values'] / (len(df) * len(numeric_cols))
            infinite_ratio = validation_results['infinite_values'] / (len(df) * len(numeric_cols))
            quality_score = max(0, 1 - missing_ratio - infinite_ratio)
            validation_results['data_quality_score'] = quality_score

            logger.info(f"Data validation completed. Quality score: {quality_score:.3f}")

            return validation_results

        except Exception as e:
            logger.error(f"Error validating normalized data: {e}")
            return {}

    def save_normalization_params(self, filepath: str) -> None:
        """Save normalization parameters for later use"""
        try:
            import pickle

            params = {
                'scalers': self.scalers,
                'feature_stats': self.feature_stats,
                'config': {
                    'outlier_method': self.outlier_method,
                    'outlier_threshold': self.outlier_threshold,
                    'normalization_method': self.normalization_method,
                    'fill_method': self.fill_method
                }
            }

            with open(filepath, 'wb') as f:
                pickle.dump(params, f)

            logger.info(f"Normalization parameters saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving normalization parameters: {e}")
            raise

    def load_normalization_params(self, filepath: str) -> None:
        """Load normalization parameters"""
        try:
            import pickle

            with open(filepath, 'rb') as f:
                params = pickle.load(f)

            self.scalers = params['scalers']
            self.feature_stats = params['feature_stats']

            # Update config
            config = params['config']
            self.outlier_method = config['outlier_method']
            self.outlier_threshold = config['outlier_threshold']
            self.normalization_method = config['normalization_method']
            self.fill_method = config['fill_method']

            logger.info(f"Normalization parameters loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading normalization parameters: {e}")
            raise
