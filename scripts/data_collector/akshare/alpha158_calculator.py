#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alpha 158 Feature Calculator

This module implements the Alpha 158 feature set for stock market analysis,
including technical indicators, rolling statistics, and cross-sectional features.
Based on the qlib Alpha158 implementation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class Alpha158Calculator:
    """Calculate Alpha 158 features for stock data"""
    
    def __init__(self):
        """Initialize Alpha 158 calculator"""
        self.feature_names = []
        logger.info("Initialized Alpha 158 feature calculator")
    
    def _rolling_rank(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling rank"""
        return series.rolling(window=window, min_periods=1).rank(pct=True)
    
    def _ts_rank(self, series: pd.Series, window: int) -> pd.Series:
        """Time series rank (percentile rank over rolling window)"""
        return series.rolling(window=window, min_periods=1).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / len(x) if len(x) > 0 else np.nan
        )
    
    def _correlation(self, x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """Rolling correlation"""
        return x.rolling(window=window, min_periods=window//2).corr(y)
    
    def _covariance(self, x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """Rolling covariance"""
        return x.rolling(window=window, min_periods=window//2).cov(y)
    
    def _stddev(self, series: pd.Series, window: int) -> pd.Series:
        """Rolling standard deviation"""
        return series.rolling(window=window, min_periods=1).std()
    
    def _zscore(self, series: pd.Series, window: int) -> pd.Series:
        """Rolling z-score"""
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        return (series - rolling_mean) / rolling_std.replace(0, np.nan)
    
    def _return_features(self, df: pd.DataFrame, windows: List[int] = [1, 2, 3, 4, 5, 10, 20, 30, 60]) -> pd.DataFrame:
        """Calculate return-based features"""

        for window in windows:
            # Simple returns
            df[f'ROC{window}'] = df['close'].pct_change(window)
            
            # Log returns
            df[f'LOG_ROC{window}'] = np.log(df['close'] / df['close'].shift(window))
            
            # Maximum return in window
            df[f'MAX_ROC{window}'] = df['close'].rolling(window).apply(
                lambda x: (x.max() / x.iloc[0] - 1) if len(x) > 0 and x.iloc[0] != 0 else np.nan
            )
            
            # Minimum return in window
            df[f'MIN_ROC{window}'] = df['close'].rolling(window).apply(
                lambda x: (x.min() / x.iloc[0] - 1) if len(x) > 0 and x.iloc[0] != 0 else np.nan
            )
        
        return df
    
    def _moving_average_features(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20, 30, 60]) -> pd.DataFrame:
        """Calculate moving average features"""

        for window in windows:
            # Simple moving average
            df[f'MA{window}'] = df['close'].rolling(window).mean()
            
            # Exponential moving average
            df[f'EMA{window}'] = df['close'].ewm(span=window).mean()
            
            # Volume weighted moving average
            if 'volume' in df.columns:
                df[f'VWMA{window}'] = (df['close'] * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
            
            # Price relative to moving average
            df[f'MA_RATIO{window}'] = df['close'] / df[f'MA{window}'] - 1
            
            # Moving average slope
            df[f'MA_SLOPE{window}'] = (df[f'MA{window}'] - df[f'MA{window}'].shift(1)) / df[f'MA{window}'].shift(1)
        
        return df
    
    def _volatility_features(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20, 30, 60]) -> pd.DataFrame:
        """Calculate volatility features"""

        for window in windows:
            # Standard deviation of returns
            df[f'STD{window}'] = df['return'].rolling(window).std()
            
            # Standard deviation of prices
            df[f'PRICE_STD{window}'] = df['close'].rolling(window).std()
            
            # Average True Range (ATR)
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift(1))
            low_close = abs(df['low'] - df['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'ATR{window}'] = true_range.rolling(window).mean()
            
            # Realized volatility
            df[f'RVOL{window}'] = np.sqrt(df['return'].rolling(window).var() * 252)
        
        return df
    
    def _momentum_features(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20, 30, 60]) -> pd.DataFrame:
        """Calculate momentum features"""

        for window in windows:
            # RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = (-delta).where(delta < 0, 0).rolling(window).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f'RSI{window}'] = 100 - (100 / (1 + rs))
            
            # Time series rank
            df[f'RANK{window}'] = self._ts_rank(df['close'], window)
            
            # Quantile
            df[f'QTLU{window}'] = df['close'].rolling(window).quantile(0.8)
            df[f'QTLD{window}'] = df['close'].rolling(window).quantile(0.2)
            
            # Price position in range
            df[f'POSITION{window}'] = (df['close'] - df['close'].rolling(window).min()) / (
                df['close'].rolling(window).max() - df['close'].rolling(window).min()
            )
        
        return df
    
    def _volume_features(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20, 30, 60]) -> pd.DataFrame:
        """Calculate volume features"""
        if 'volume' not in df.columns:
            logger.warning("Volume column not found, skipping volume features")
            return df
        

        for window in windows:
            # Volume moving average
            df[f'VOL_MA{window}'] = df['volume'].rolling(window).mean()
            
            # Volume ratio
            df[f'VOL_RATIO{window}'] = df['volume'] / df[f'VOL_MA{window}']
            
            # Volume standard deviation
            df[f'VOL_STD{window}'] = df['volume'].rolling(window).std()
            
            # On Balance Volume
            obv = (df['volume'] * np.sign(df['close'].diff())).cumsum()
            df[f'OBV{window}'] = obv.rolling(window).mean()
            
            # Volume Price Trend
            vpt = (df['volume'] * df['close'].pct_change()).cumsum()
            df[f'VPT{window}'] = vpt.rolling(window).mean()
        
        return df
    
    def _correlation_features(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20, 30, 60]) -> pd.DataFrame:
        """Calculate correlation features"""

        for window in windows:
            # Price-Volume correlation
            if 'volume' in df.columns:
                df[f'CORR{window}'] = self._correlation(df['close'], df['volume'], window)
            
            # High-Low correlation
            df[f'CORD{window}'] = self._correlation(df['high'], df['low'], window)
            
            # Return autocorrelation
            df[f'AUTOCORR{window}'] = self._correlation(df['return'], df['return'].shift(1), window)
        
        return df
    
    def _technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional technical indicators"""

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_SIGNAL'] = df['MACD'].ewm(span=9).mean()
        df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']
        
        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        bb_ma = df['close'].rolling(bb_window).mean()
        bb_std_val = df['close'].rolling(bb_window).std()
        df['BB_UPPER'] = bb_ma + (bb_std_val * bb_std)
        df['BB_LOWER'] = bb_ma - (bb_std_val * bb_std)
        df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / bb_ma
        df['BB_POSITION'] = (df['close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])
        
        # Stochastic Oscillator
        stoch_window = 14
        lowest_low = df['low'].rolling(stoch_window).min()
        highest_high = df['high'].rolling(stoch_window).max()
        df['STOCH_K'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()
        
        # Williams %R
        df['WILLIAMS_R'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        
        # Commodity Channel Index (CCI)
        cci_window = 20
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(cci_window).mean()
        mad = typical_price.rolling(cci_window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
        
        return df
    
    def _cross_sectional_features(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Calculate cross-sectional features (requires multiple symbols)"""

        if 'symbol' not in df.columns:
            logger.warning("Symbol column not found, skipping cross-sectional features")
            return df
        
        for window in windows:
            # Cross-sectional rank
            df[f'CS_RANK{window}'] = df.groupby('date')['close'].transform(
                lambda x: x.rank(pct=True)
            )
            
            # Cross-sectional z-score
            df[f'CS_ZSCORE{window}'] = df.groupby('date')['close'].transform(
                lambda x: (x - x.mean()) / x.std()
            )
            
            # Industry relative (simplified - using all stocks as industry)
            df[f'IND_REL{window}'] = df.groupby('date')['return'].transform(
                lambda x: x - x.mean()
            )
        
        return df

    def calculate_alpha158_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all Alpha 158 features

        Args:
            df: DataFrame with stock data (must have OHLCV data)

        Returns:
            DataFrame with Alpha 158 features added
        """
        try:
            logger.info("Starting Alpha 158 feature calculation...")

            # Validate required columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Ensure data is sorted properly
            if 'symbol' in df.columns:
                df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
            else:
                df = df.sort_values('date').reset_index(drop=True)

            # Calculate basic return if not present
            if 'return' not in df.columns:
                if 'symbol' in df.columns:
                    df['return'] = df.groupby('symbol')['close'].pct_change()
                else:
                    df['return'] = df['close'].pct_change()

            # Process by symbol if multiple symbols present
            if 'symbol' in df.columns:
                logger.info(f"Processing {df['symbol'].nunique()} symbols...")

                # Process each symbol separately for time series features
                symbol_dfs = []
                for symbol in df['symbol'].unique():
                    symbol_df = df[df['symbol'] == symbol].copy()
                    symbol_df = self._calculate_symbol_features(symbol_df)
                    symbol_dfs.append(symbol_df)

                # Combine all symbols
                df = pd.concat(symbol_dfs, ignore_index=True)

                # Calculate cross-sectional features
                df = self._cross_sectional_features(df)
            else:
                # Single symbol processing
                df = self._calculate_symbol_features(df)

            # Get list of feature columns (exclude original columns)
            original_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']
            self.feature_names = [col for col in df.columns if col not in original_cols]

            logger.info(f"Alpha 158 calculation completed. Generated {len(self.feature_names)} features")

            return df

        except Exception as e:
            logger.error(f"Error calculating Alpha 158 features: {e}")
            raise

    def _calculate_symbol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for a single symbol"""
        # Calculate all feature groups
        df = self._return_features(df)
        df = self._moving_average_features(df)
        df = self._volatility_features(df)
        df = self._momentum_features(df)
        df = self._volume_features(df)
        df = self._correlation_features(df)
        df = self._technical_indicators(df)

        return df

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get features grouped by importance/type"""
        groups = {
            'price_features': [f for f in self.feature_names if any(x in f for x in ['MA', 'EMA', 'ROC', 'RATIO'])],
            'volatility_features': [f for f in self.feature_names if any(x in f for x in ['STD', 'ATR', 'RVOL'])],
            'momentum_features': [f for f in self.feature_names if any(x in f for x in ['RSI', 'RANK', 'POSITION'])],
            'volume_features': [f for f in self.feature_names if any(x in f for x in ['VOL', 'OBV', 'VPT'])],
            'correlation_features': [f for f in self.feature_names if any(x in f for x in ['CORR', 'CORD'])],
            'technical_features': [f for f in self.feature_names if any(x in f for x in ['MACD', 'BB', 'STOCH', 'CCI'])],
            'cross_sectional_features': [f for f in self.feature_names if any(x in f for x in ['CS_', 'IND_'])]
        }

        return groups

    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of calculated features"""
        try:
            feature_cols = [col for col in df.columns if col in self.feature_names]

            summary = {
                'total_features': len(feature_cols),
                'feature_groups': self.get_feature_importance_groups(),
                'missing_values': df[feature_cols].isnull().sum().to_dict(),
                'feature_stats': {
                    col: {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max())
                    } for col in feature_cols[:10]  # First 10 features as example
                }
            }

            logger.info(f"Feature summary: {len(feature_cols)} features calculated")

            return summary

        except Exception as e:
            logger.error(f"Error getting feature summary: {e}")
            return {}
