#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Initializer for Stock Market Analysis

This module prepares stock data structure, handles missing values, and sets up
proper indexing for time series analysis and feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataInitializer:
    """Initialize and prepare stock data for analysis"""
    
    def __init__(self, min_trading_days: int = 100, min_price: float = 0.01):
        """
        Initialize data initializer
        
        Args:
            min_trading_days: Minimum number of trading days required for a stock
            min_price: Minimum price threshold for valid data
        """
        self.min_trading_days = min_trading_days
        self.min_price = min_price
        
        logger.info(f"Initialized data initializer with min_trading_days={min_trading_days}, min_price={min_price}")
    
    def create_complete_date_range(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """
        Create complete date range for all trading days
        
        Args:
            df: DataFrame with date column
            
        Returns:
            Complete date range index
        """
        try:
            # Get min and max dates
            start_date = df['date'].min()
            end_date = df['date'].max()
            
            # Create business day range (excludes weekends)
            date_range = pd.bdate_range(start=start_date, end=end_date, freq='B')
            
            # Filter to only include dates that actually exist in the data
            # (to handle holidays and other non-trading days)
            actual_dates = set(df['date'].dt.date)
            trading_dates = [d for d in date_range if d.date() in actual_dates]
            
            logger.info(f"Created complete date range: {len(trading_dates)} trading days")
            logger.info(f"Date range: {trading_dates[0].date()} to {trading_dates[-1].date()}")
            
            return pd.DatetimeIndex(trading_dates)
            
        except Exception as e:
            logger.error(f"Error creating date range: {e}")
            raise
    
    def filter_valid_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter stocks that meet minimum requirements
        
        Args:
            df: Raw stock data DataFrame
            
        Returns:
            Filtered DataFrame with valid stocks only
        """
        try:
            logger.info("Filtering valid stocks...")
            original_symbols = df['symbol'].nunique()
            
            # Group by symbol and calculate statistics
            stock_stats = df.groupby('symbol').agg({
                'date': 'count',  # Number of trading days
                'close': ['mean', 'min', 'max'],
                'volume': 'mean'
            }).round(4)
            
            # Flatten column names
            stock_stats.columns = ['trading_days', 'avg_close', 'min_close', 'max_close', 'avg_volume']
            
            # Apply filters
            valid_stocks = stock_stats[
                (stock_stats['trading_days'] >= self.min_trading_days) &
                (stock_stats['min_close'] >= self.min_price) &
                (stock_stats['avg_volume'] > 0)
            ].index.tolist()
            
            # Filter original DataFrame
            filtered_df = df[df['symbol'].isin(valid_stocks)].copy()
            
            removed_symbols = original_symbols - len(valid_stocks)
            logger.info(f"Filtered stocks: kept {len(valid_stocks)}, removed {removed_symbols}")
            logger.info(f"Remaining data: {len(filtered_df)} records")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering valid stocks: {e}")
            raise
    
    def create_panel_data_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create panel data structure with complete date index for all stocks
        
        Args:
            df: Filtered stock data
            
        Returns:
            Panel DataFrame with complete date structure
        """
        try:
            logger.info("Creating panel data structure...")
            
            # Get complete date range and symbols
            complete_dates = self.create_complete_date_range(df)
            symbols = sorted(df['symbol'].unique())
            
            logger.info(f"Panel structure: {len(symbols)} symbols × {len(complete_dates)} dates")
            
            # Create MultiIndex for complete panel
            panel_index = pd.MultiIndex.from_product(
                [symbols, complete_dates],
                names=['symbol', 'date']
            )
            
            # Create empty DataFrame with complete structure
            panel_df = pd.DataFrame(index=panel_index)
            
            # Set date as regular column for easier merging
            df_for_merge = df.set_index(['symbol', 'date'])
            
            # Merge with original data
            panel_df = panel_df.join(df_for_merge, how='left')
            
            # Reset index to make symbol and date regular columns
            panel_df = panel_df.reset_index()
            
            logger.info(f"Created panel data: {len(panel_df)} total records")
            logger.info(f"Missing data ratio: {panel_df.isnull().sum().sum() / (len(panel_df) * len(panel_df.columns)):.2%}")
            
            return panel_df
            
        except Exception as e:
            logger.error(f"Error creating panel data structure: {e}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the panel data
        
        Args:
            df: Panel DataFrame with missing values
            
        Returns:
            DataFrame with missing values handled
        """
        try:
            logger.info("Handling missing values...")
            
            df = df.copy()
            
            # Sort by symbol and date for proper forward/backward fill
            df = df.sort_values(['symbol', 'date'])
            
            # Handle missing values by symbol group
            price_cols = ['open', 'high', 'low', 'close']
            volume_cols = ['volume', 'amount']
            other_cols = ['amplitude', 'change_percent', 'change_amount', 'turnover_rate']
            
            for symbol in df['symbol'].unique():
                mask = df['symbol'] == symbol
                symbol_data = df.loc[mask].copy()
                
                # Forward fill prices (use last known price)
                for col in price_cols:
                    if col in symbol_data.columns:
                        symbol_data[col] = symbol_data[col].fillna(method='ffill')
                        # If still missing at the beginning, backward fill
                        symbol_data[col] = symbol_data[col].fillna(method='bfill')
                
                # For volume, use 0 for missing values (no trading)
                for col in volume_cols:
                    if col in symbol_data.columns:
                        symbol_data[col] = symbol_data[col].fillna(0)
                
                # For other indicators, forward fill then set remaining to 0
                for col in other_cols:
                    if col in symbol_data.columns:
                        symbol_data[col] = symbol_data[col].fillna(method='ffill').fillna(0)
                
                # Update the main DataFrame
                df.loc[mask] = symbol_data
            
            # Final check: remove any rows that still have missing critical data
            critical_cols = ['open', 'high', 'low', 'close']
            before_drop = len(df)
            df = df.dropna(subset=critical_cols)
            after_drop = len(df)
            
            if before_drop > after_drop:
                logger.warning(f"Dropped {before_drop - after_drop} rows with missing critical data")
            
            logger.info("Missing value handling completed")
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise
    
    def calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic features needed for analysis
        
        Args:
            df: Panel DataFrame
            
        Returns:
            DataFrame with basic features added
        """
        try:
            logger.info("Calculating basic features...")
            
            df = df.copy()
            df = df.sort_values(['symbol', 'date'])
            
            # Calculate features by symbol group
            for symbol in df['symbol'].unique():
                mask = df['symbol'] == symbol
                symbol_data = df.loc[mask].copy()
                
                # Calculate returns
                symbol_data['return'] = symbol_data['close'].pct_change()
                symbol_data['log_return'] = np.log(symbol_data['close'] / symbol_data['close'].shift(1))
                
                # Calculate VWAP (Volume Weighted Average Price)
                if 'amount' in symbol_data.columns and symbol_data['amount'].sum() > 0:
                    symbol_data['vwap'] = symbol_data['amount'] / symbol_data['volume'].replace(0, np.nan)
                else:
                    symbol_data['vwap'] = (symbol_data['high'] + symbol_data['low'] + symbol_data['close']) / 3
                
                # Calculate price ranges
                symbol_data['price_range'] = symbol_data['high'] - symbol_data['low']
                symbol_data['price_range_pct'] = symbol_data['price_range'] / symbol_data['close']
                
                # Calculate gap (difference between open and previous close)
                symbol_data['gap'] = symbol_data['open'] / symbol_data['close'].shift(1) - 1
                
                # Update the main DataFrame
                df.loc[mask] = symbol_data
            
            # Fill any remaining NaN values in new features
            new_features = ['return', 'log_return', 'vwap', 'price_range', 'price_range_pct', 'gap']
            for col in new_features:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            
            logger.info(f"Added basic features: {new_features}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating basic features: {e}")
            raise
    
    def validate_data_integrity(self, df: pd.DataFrame) -> Dict:
        """
        Validate data integrity and return summary
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Validation summary dictionary
        """
        try:
            logger.info("Validating data integrity...")
            
            validation_results = {
                'total_records': len(df),
                'unique_symbols': df['symbol'].nunique(),
                'date_range': {
                    'start': df['date'].min(),
                    'end': df['date'].max(),
                    'trading_days': df['date'].nunique()
                },
                'missing_data': df.isnull().sum().to_dict(),
                'data_quality': {
                    'negative_prices': (df[['open', 'high', 'low', 'close']] <= 0).sum().sum(),
                    'invalid_price_relationships': 0,
                    'extreme_returns': (abs(df['return']) > 0.5).sum() if 'return' in df.columns else 0
                }
            }
            
            # Check price relationships
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                invalid_high = (df['high'] < df[['open', 'low', 'close']].max(axis=1)).sum()
                invalid_low = (df['low'] > df[['open', 'high', 'close']].min(axis=1)).sum()
                validation_results['data_quality']['invalid_price_relationships'] = invalid_high + invalid_low
            
            # Log validation results
            logger.info(f"Data validation completed: {validation_results}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating data integrity: {e}")
            raise
    
    def initialize_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete data initialization process
        
        Args:
            df: Raw stock data DataFrame
            
        Returns:
            Tuple of (initialized DataFrame, validation summary)
        """
        try:
            logger.info("Starting complete data initialization...")
            
            # Step 1: Filter valid stocks
            filtered_df = self.filter_valid_stocks(df)
            
            # Step 2: Create panel data structure
            panel_df = self.create_panel_data_structure(filtered_df)
            
            # Step 3: Handle missing values
            clean_df = self.handle_missing_values(panel_df)
            
            # Step 4: Calculate basic features
            feature_df = self.calculate_basic_features(clean_df)
            
            # Step 5: Validate data integrity
            validation_summary = self.validate_data_integrity(feature_df)
            
            logger.info("Data initialization completed successfully")
            
            return feature_df, validation_summary
            
        except Exception as e:
            logger.error(f"Error in data initialization: {e}")
            raise
