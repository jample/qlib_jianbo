#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Binary Data Converter for Qlib

This module converts pandas DataFrame stock data to qlib-compatible binary format
for efficient storage and fast access during training and inference.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from loguru import logger
import pickle
import struct
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class BinaryDataConverter:
    """Convert stock data to qlib binary format"""
    
    def __init__(self, output_dir: str = "scripts/data_collector/akshare/qlib_data"):
        """
        Initialize binary data converter
        
        Args:
            output_dir: Directory to save binary data files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for qlib structure
        self.features_dir = self.output_dir / "features"
        self.calendars_dir = self.output_dir / "calendars"
        self.instruments_dir = self.output_dir / "instruments"
        
        for dir_path in [self.features_dir, self.calendars_dir, self.instruments_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized binary converter with output dir: {self.output_dir}")
    
    def create_calendar(self, df: pd.DataFrame) -> List[str]:
        """
        Create trading calendar from the data
        
        Args:
            df: DataFrame with date column
            
        Returns:
            List of trading dates in string format
        """
        try:
            # Get unique trading dates and sort them
            trading_dates = sorted(df['date'].dt.strftime('%Y-%m-%d').unique())
            
            # Save calendar to file
            calendar_file = self.calendars_dir / "trading_calendar.txt"
            with open(calendar_file, 'w') as f:
                for date in trading_dates:
                    f.write(f"{date}\n")
            
            logger.info(f"Created trading calendar with {len(trading_dates)} dates")
            logger.info(f"Date range: {trading_dates[0]} to {trading_dates[-1]}")
            
            return trading_dates
            
        except Exception as e:
            logger.error(f"Error creating calendar: {e}")
            raise
    
    def create_instruments_list(self, df: pd.DataFrame) -> List[str]:
        """
        Create instruments list from the data
        
        Args:
            df: DataFrame with symbol column
            
        Returns:
            List of stock symbols
        """
        try:
            # Get unique symbols and sort them
            symbols = sorted(df['symbol'].unique())
            
            # Save instruments to file
            instruments_file = self.instruments_dir / "all.txt"
            with open(instruments_file, 'w') as f:
                for symbol in symbols:
                    f.write(f"{symbol}\n")
            
            logger.info(f"Created instruments list with {len(symbols)} symbols")
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error creating instruments list: {e}")
            raise
    
    def convert_to_binary_format(self, df: pd.DataFrame, field_name: str) -> None:
        """
        Convert a specific field to binary format
        
        Args:
            df: DataFrame with stock data
            field_name: Name of the field to convert (e.g., 'close', 'volume')
        """
        try:
            if field_name not in df.columns:
                logger.warning(f"Field {field_name} not found in data")
                return
            
            # Create field directory
            field_dir = self.features_dir / field_name
            field_dir.mkdir(parents=True, exist_ok=True)
            
            # Get unique symbols and dates
            symbols = sorted(df['symbol'].unique())
            dates = sorted(df['date'].unique())
            
            logger.info(f"Converting {field_name} for {len(symbols)} symbols and {len(dates)} dates")
            
            # Create date index mapping
            date_to_index = {date: idx for idx, date in enumerate(dates)}
            
            # Process each symbol
            for symbol in symbols:
                symbol_data = df[df['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_values('date')
                
                # Create array for this symbol (filled with NaN initially)
                symbol_array = np.full(len(dates), np.nan, dtype=np.float32)
                
                # Fill in the actual data
                for _, row in symbol_data.iterrows():
                    date_idx = date_to_index[row['date']]
                    symbol_array[date_idx] = row[field_name]
                
                # Save binary file for this symbol
                bin_file = field_dir / f"{symbol}.bin"
                with open(bin_file, 'wb') as f:
                    # Write header: number of records
                    f.write(struct.pack('I', len(symbol_array)))
                    # Write data
                    symbol_array.tofile(f)
            
            logger.info(f"Successfully converted {field_name} to binary format")
            
        except Exception as e:
            logger.error(f"Error converting {field_name} to binary: {e}")
            raise
    
    def convert_all_fields(self, df: pd.DataFrame) -> None:
        """
        Convert all numeric fields to binary format
        
        Args:
            df: DataFrame with stock data
        """
        try:
            # Define fields to convert
            fields_to_convert = ['open', 'high', 'low', 'close', 'volume', 'amount']
            
            # Add optional fields if they exist
            optional_fields = ['amplitude', 'change_percent', 'change_amount', 'turnover_rate']
            for field in optional_fields:
                if field in df.columns and not df[field].isna().all():
                    fields_to_convert.append(field)
            
            logger.info(f"Converting fields: {fields_to_convert}")
            
            # Convert each field
            for field in fields_to_convert:
                self.convert_to_binary_format(df, field)
            
            logger.info("All fields converted to binary format")
            
        except Exception as e:
            logger.error(f"Error converting all fields: {e}")
            raise
    
    def create_metadata(self, df: pd.DataFrame, trading_dates: List[str], symbols: List[str]) -> None:
        """
        Create metadata files for the binary data
        
        Args:
            df: Original DataFrame
            trading_dates: List of trading dates
            symbols: List of symbols
        """
        try:
            # Create metadata dictionary
            metadata = {
                'created_at': datetime.now().isoformat(),
                'total_records': len(df),
                'unique_symbols': len(symbols),
                'trading_days': len(trading_dates),
                'date_range': {
                    'start': trading_dates[0],
                    'end': trading_dates[-1]
                },
                'symbols': symbols,
                'fields': [col for col in df.columns if col not in ['symbol', 'date']],
                'data_stats': {
                    'avg_close': float(df['close'].mean()),
                    'min_close': float(df['close'].min()),
                    'max_close': float(df['close'].max()),
                    'avg_volume': float(df['volume'].mean()),
                    'total_volume': float(df['volume'].sum())
                }
            }
            
            # Save metadata
            metadata_file = self.output_dir / "metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Also save as JSON for human readability
            import json
            json_file = self.output_dir / "metadata.json"
            with open(json_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info("Created metadata files")
            
        except Exception as e:
            logger.error(f"Error creating metadata: {e}")
            raise
    
    def convert_dataframe_to_qlib_format(self, df: pd.DataFrame) -> str:
        """
        Convert entire DataFrame to qlib binary format
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            Path to the created qlib data directory
        """
        try:
            logger.info("Starting conversion to qlib binary format")
            
            # Validate input data
            required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Ensure date is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Create calendar and instruments
            trading_dates = self.create_calendar(df)
            symbols = self.create_instruments_list(df)
            
            # Convert all fields to binary
            self.convert_all_fields(df)
            
            # Create metadata
            self.create_metadata(df, trading_dates, symbols)
            
            logger.info(f"Successfully converted data to qlib format at: {self.output_dir}")
            return str(self.output_dir)
            
        except Exception as e:
            logger.error(f"Error converting to qlib format: {e}")
            raise
    
    def verify_binary_data(self, symbol: str, field: str) -> bool:
        """
        Verify that binary data was created correctly
        
        Args:
            symbol: Stock symbol to verify
            field: Field name to verify
            
        Returns:
            True if verification passes
        """
        try:
            bin_file = self.features_dir / field / f"{symbol}.bin"
            if not bin_file.exists():
                logger.error(f"Binary file not found: {bin_file}")
                return False
            
            # Read and verify the binary file
            with open(bin_file, 'rb') as f:
                # Read header
                num_records = struct.unpack('I', f.read(4))[0]
                # Read data
                data = np.frombuffer(f.read(), dtype=np.float32)
                
                if len(data) != num_records:
                    logger.error(f"Data length mismatch for {symbol}/{field}")
                    return False
            
            logger.info(f"Verification passed for {symbol}/{field}: {num_records} records")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying binary data: {e}")
            return False
    
    def get_conversion_summary(self) -> Dict:
        """Get summary of the conversion process"""
        try:
            summary = {
                'output_directory': str(self.output_dir),
                'created_files': {
                    'features': len(list(self.features_dir.rglob('*.bin'))),
                    'calendar': len(list(self.calendars_dir.glob('*.txt'))),
                    'instruments': len(list(self.instruments_dir.glob('*.txt'))),
                    'metadata': len(list(self.output_dir.glob('metadata.*')))
                },
                'total_size_mb': sum(f.stat().st_size for f in self.output_dir.rglob('*') if f.is_file()) / (1024 * 1024)
            }
            
            logger.info(f"Conversion summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error getting conversion summary: {e}")
            return {}
