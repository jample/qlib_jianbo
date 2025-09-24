#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DuckDB Data Extractor for Shanghai Stock Data

This module extracts Shanghai stock data from DuckDB and converts it to 
pandas DataFrame format with proper data types and validation for qlib processing.
"""

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict, Tuple, Union
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class DuckDBDataExtractor:
    """Extract and process stock/fund data from DuckDB with configurable data scope"""

    def __init__(self,
                 db_path: str = "/root/autodl-tmp/code/duckdb/shanghai_stock_data.duckdb",
                 data_type: str = "stock",
                 exchange_filter: Optional[str] = "shanghai",
                 interval: str = "1d"):
        """
        Initialize the DuckDB data extractor

        Args:
            db_path: Path to the DuckDB database file
            data_type: Type of data to extract ("stock" or "fund")
            exchange_filter: Exchange filter ("shanghai", "shenzhen", "all", None)
            interval: Data interval ("1d", "1w", "1m") - currently supports 1d
        """
        self.db_path = Path(db_path)
        self.data_type = data_type.lower()
        self.exchange_filter = exchange_filter.lower() if exchange_filter else None
        self.interval = interval.lower()

        # Validate parameters
        if self.data_type not in ["stock", "fund"]:
            raise ValueError(f"data_type must be 'stock' or 'fund', got: {data_type}")

        if self.exchange_filter and self.exchange_filter not in ["shanghai", "shenzhen", "all"]:
            raise ValueError(f"exchange_filter must be 'shanghai', 'shenzhen', 'all', or None, got: {exchange_filter}")

        if self.interval not in ["1d", "1w", "1m"]:
            logger.warning(f"interval '{interval}' not fully supported yet, using '1d'")
            self.interval = "1d"

        if not self.db_path.exists():
            raise FileNotFoundError(f"DuckDB file not found: {self.db_path}")

        # Set table names based on data type
        self.main_table = f"{self.data_type}_data"
        self.metadata_table = f"{self.data_type}_update_metadata"
        self.symbols_table = f"{self._get_exchange_prefix()}{self.data_type}s" if self.exchange_filter else f"{self.data_type}s"

        logger.info(f"Initialized DuckDB extractor: data_type={self.data_type}, exchange={self.exchange_filter}, interval={self.interval}")
        logger.info(f"Using tables: {self.main_table}, {self.metadata_table}, {self.symbols_table}")

    def _get_exchange_prefix(self) -> str:
        """Get exchange prefix for table names"""
        if self.exchange_filter == "shanghai":
            return "shanghai_"
        elif self.exchange_filter == "shenzhen":
            return "shenzhen_"
        else:
            return ""
    
    def get_database_info(self) -> Dict:
        """Get basic information about the database"""
        try:
            conn = duckdb.connect(str(self.db_path))

            # Get table info
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]

            # Check if main table exists
            if self.main_table not in table_names:
                logger.warning(f"Main table '{self.main_table}' not found in database. Available tables: {table_names}")
                conn.close()
                return {
                    'database_path': str(self.db_path),
                    'data_type': self.data_type,
                    'exchange_filter': self.exchange_filter,
                    'tables': table_names,
                    'main_table': self.main_table,
                    'table_exists': False,
                    'error': f"Table '{self.main_table}' not found"
                }

            # Get main data statistics
            stats = conn.execute(f"""
                SELECT
                    COUNT(DISTINCT symbol) as total_symbols,
                    COUNT(*) as total_records,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    COUNT(DISTINCT date) as trading_days
                FROM {self.main_table}
            """).fetchone()

            # Get sample of available symbols
            symbols = conn.execute(f"""
                SELECT DISTINCT symbol
                FROM {self.main_table}
                ORDER BY symbol
                LIMIT 20
            """).fetchall()

            conn.close()

            info = {
                'database_path': str(self.db_path),
                'data_type': self.data_type,
                'exchange_filter': self.exchange_filter,
                'interval': self.interval,
                'tables': table_names,
                'main_table': self.main_table,
                'table_exists': True,
                'total_symbols': stats[0],
                'total_records': stats[1],
                'earliest_date': stats[2],
                'latest_date': stats[3],
                'trading_days': stats[4],
                'sample_symbols': [s[0] for s in symbols]
            }

            logger.info(f"Database info: {info}")
            return info

        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            raise
    
    def get_available_symbols(self, prefix_filter: Optional[List[str]] = None) -> List[str]:
        """
        Get list of available stock symbols
        
        Args:
            prefix_filter: List of prefixes to filter symbols (e.g., ['600', '601'])
            
        Returns:
            List of stock symbols
        """
        try:
            conn = duckdb.connect(str(self.db_path))
            
            # Apply exchange filter based on configuration
            exchange_conditions = []
            if self.exchange_filter == "shanghai":
                # Shanghai stock codes: 600xxx, 601xxx, 603xxx, 605xxx, 688xxx, 689xxx
                exchange_conditions = ["symbol LIKE '600%'", "symbol LIKE '601%'", "symbol LIKE '603%'",
                                     "symbol LIKE '605%'", "symbol LIKE '688%'", "symbol LIKE '689%'"]
            elif self.exchange_filter == "shenzhen":
                # Shenzhen stock codes: 000xxx, 001xxx, 002xxx, 003xxx, 300xxx
                exchange_conditions = ["symbol LIKE '000%'", "symbol LIKE '001%'", "symbol LIKE '002%'",
                                     "symbol LIKE '003%'", "symbol LIKE '300%'"]

            # Apply additional prefix filter if provided
            if prefix_filter:
                prefix_conditions = [f"symbol LIKE '{prefix}%'" for prefix in prefix_filter]
                if exchange_conditions:
                    # Combine exchange and prefix filters (intersection)
                    combined_conditions = []
                    for prefix in prefix_filter:
                        for exchange_pattern in ['600', '601', '603', '605', '688', '689', '000', '001', '002', '003', '300']:
                            if prefix.startswith(exchange_pattern[:len(prefix)]) or exchange_pattern.startswith(prefix):
                                combined_conditions.append(f"symbol LIKE '{prefix}%'")
                                break
                    conditions = combined_conditions if combined_conditions else prefix_conditions
                else:
                    conditions = prefix_conditions
            else:
                conditions = exchange_conditions

            # Build query
            if conditions:
                where_clause = " OR ".join(conditions)
                query = f"""
                    SELECT DISTINCT symbol
                    FROM {self.main_table}
                    WHERE {where_clause}
                    ORDER BY symbol
                """
            else:
                query = f"""
                    SELECT DISTINCT symbol
                    FROM {self.main_table}
                    ORDER BY symbol
                """
            
            symbols = conn.execute(query).fetchall()
            conn.close()
            
            symbol_list = [s[0] for s in symbols]

            filter_desc = []
            if self.exchange_filter:
                filter_desc.append(f"exchange={self.exchange_filter}")
            if prefix_filter:
                filter_desc.append(f"prefix={prefix_filter}")

            filter_str = f" with filters: {', '.join(filter_desc)}" if filter_desc else ""
            logger.info(f"Found {len(symbol_list)} {self.data_type} symbols{filter_str}")

            return symbol_list
            
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            raise
    
    def extract_stock_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract data from DuckDB

        Args:
            symbols: List of symbols to extract (None for all)
            start_date: Start date for data extraction
            end_date: End date for data extraction
            limit: Maximum number of records to return

        Returns:
            DataFrame with data
        """
        try:
            conn = duckdb.connect(str(self.db_path))
            
            # Build WHERE conditions
            conditions = []
            params = []
            
            if symbols:
                # Check if symbols are prefixes (like "600", "601") or exact symbols
                if any(len(s) <= 3 for s in symbols):
                    # Handle as prefixes
                    prefix_conditions = []
                    for symbol in symbols:
                        prefix_conditions.append("sd.symbol LIKE ?")
                        params.append(f"{symbol}%")
                    conditions.append(f"({' OR '.join(prefix_conditions)})")
                else:
                    # Handle as exact symbols
                    symbol_placeholders = ','.join(['?' for _ in symbols])
                    conditions.append(f"sd.symbol IN ({symbol_placeholders})")
                    params.extend(symbols)
            
            if start_date:
                conditions.append("sd.date >= ?")
                params.append(str(start_date))

            if end_date:
                conditions.append("sd.date <= ?")
                params.append(str(end_date))
            
            # Note: WHERE clause is now handled in the main query with JOIN
            
            limit_clause = ""
            if limit:
                limit_clause = f"LIMIT {limit}"
            
            # Execute query with JOIN to filter only active stocks
            query = f"""
                SELECT
                    sd.symbol,
                    sd.date,
                    sd.open,
                    sd.high,
                    sd.low,
                    sd.close,
                    sd.volume,
                    sd.amount,
                    sd.amplitude,
                    sd.change_percent,
                    sd.change_amount,
                    sd.turnover_rate
                FROM {self.main_table} sd
                INNER JOIN stock_update_metadata sm ON sd.symbol = sm.symbol
                WHERE sm.status = 'active'
                {' AND ' + ' AND '.join(conditions) if conditions else ''}
                ORDER BY sd.symbol, sd.date
                {limit_clause}
            """

            logger.info(f"Executing {self.data_type} data query with {len(params)} parameters")
            logger.info(f"Query filters: active stocks only, date range: {start_date} to {end_date}")
            df = conn.execute(query, params).df()
            conn.close()

            logger.info(f"Extracted {len(df)} records for {df['symbol'].nunique()} symbols (active stocks only)")

            # Validate time series continuity
            if not df.empty:
                self._validate_time_series_continuity(df)

            return df
            
        except Exception as e:
            logger.error(f"Error extracting stock data: {e}")
            raise

    def _validate_time_series_continuity(self, df: pd.DataFrame) -> None:
        """Validate time series data continuity and report any issues"""
        try:
            # Check date range and continuity for each symbol
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].sort_values('date')

                if len(symbol_data) < 2:
                    logger.warning(f"Symbol {symbol}: Only {len(symbol_data)} data points")
                    continue

                # Check for date gaps (more than 7 days between consecutive records)
                dates = pd.to_datetime(symbol_data['date'])
                date_diffs = dates.diff().dt.days
                large_gaps = date_diffs[date_diffs > 7]

                if not large_gaps.empty:
                    logger.warning(f"Symbol {symbol}: {len(large_gaps)} date gaps > 7 days detected")

                # Check for missing OHLCV data
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    null_count = symbol_data[col].isnull().sum()
                    if null_count > 0:
                        logger.warning(f"Symbol {symbol}: {null_count} null values in {col}")

                logger.debug(f"Symbol {symbol}: {len(symbol_data)} records from {symbol_data['date'].min()} to {symbol_data['date'].max()}")

        except Exception as e:
            logger.warning(f"Time series validation failed: {e}")
    
    def validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the extracted data
        
        Args:
            df: Raw DataFrame from DuckDB
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data validation and cleaning")
        original_len = len(df)
        
        # Convert data types
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 
                       'amplitude', 'change_percent', 'change_amount', 'turnover_rate']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing critical data
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df.dropna(subset=critical_cols)
        
        # Remove invalid price data
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df = df[df[col] > 0]  # Remove zero or negative prices
        
        # Validate price relationships
        df = df[df['high'] >= df['low']]  # High should be >= Low
        df = df[df['high'] >= df['open']]  # High should be >= Open  
        df = df[df['high'] >= df['close']]  # High should be >= Close
        df = df[df['low'] <= df['open']]  # Low should be <= Open
        df = df[df['low'] <= df['close']]  # Low should be <= Close
        
        # Remove extreme outliers (prices that changed more than 50% in a day)
        df = df[df['amplitude'].fillna(0) <= 50]
        
        # Sort by symbol and date
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        cleaned_len = len(df)
        removed = original_len - cleaned_len
        
        logger.info(f"Data cleaning completed: {removed} records removed ({removed/original_len*100:.2f}%)")
        logger.info(f"Final dataset: {cleaned_len} records for {df['symbol'].nunique()} symbols")
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics of the data"""
        summary = {
            'total_records': len(df),
            'unique_symbols': df['symbol'].nunique(),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max(),
                'days': df['date'].nunique()
            },
            'price_stats': {
                'avg_close': df['close'].mean(),
                'min_close': df['close'].min(),
                'max_close': df['close'].max()
            },
            'volume_stats': {
                'avg_volume': df['volume'].mean(),
                'min_volume': df['volume'].min(),
                'max_volume': df['volume'].max()
            },
            'missing_data': df.isnull().sum().to_dict()
        }
        
        logger.info(f"Data summary: {summary}")
        return summary
