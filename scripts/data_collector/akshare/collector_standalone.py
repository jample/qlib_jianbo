# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
AkShare Data Collector following Fund Collector Pattern

This collector follows the same design pattern as the fund collector but works
independently without complex qlib dependencies that cause import issues.
"""

import abc
import sys
import time
from abc import ABC
from pathlib import Path
from typing import Union
from os import PathLike
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime, timedelta

import fire
import pandas as pd
from loguru import logger
from dateutil.tz import tzlocal
from tqdm import tqdm
import duckdb

try:
    import akshare as ak
except ImportError:
    logger.error("akshare is not installed. Please install it using: pip install akshare")
    sys.exit(1)

# Define region constant locally
REGION_CN = "CN"
CUR_DIR = Path(__file__).resolve().parent


class AkShareCollector(ABC):
    """Base AkShare collector following fund collector pattern"""
    
    DEFAULT_START_DATETIME_1D = pd.Timestamp("2000-01-01")
    DEFAULT_START_DATETIME_1MIN = pd.Timestamp(datetime.now() - pd.Timedelta(days=5 * 6 - 1)).date()
    DEFAULT_END_DATETIME_1D = pd.Timestamp(datetime.now() + pd.Timedelta(days=1)).date()
    DEFAULT_END_DATETIME_1MIN = DEFAULT_END_DATETIME_1D

    INTERVAL_1min = "1min"
    INTERVAL_1d = "1d"
    
    def __init__(
        self,
        save_dir: PathLike = None,
        delay: float = 12,
        interval: str = "1d",
        start: str = None,
        end: str = None,
        check_data_length: int = None,
        limit_nums: int = None,
        adjust: str = "qfq",
        data_type: str = "stock",
        shanghai_only: bool = False,
        **kwargs
    ):
        """
        Parameters
        ----------
        save_dir: str
            akshare save dir
        max_workers: int
            workers, forced to 1 for rate limiting
        max_collector_count: int
            forced to 1 for rate limiting
        delay: float
            time.sleep(delay), forced to 12 seconds for rate limiting
        interval: str
            freq, value from [1min, 1d], default 1d
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: int
            check data length, default None
        limit_nums: int
            using for debug, by default None
        adjust: str
            adjustment type, qfq/hfq/empty, default qfq
        data_type: str
            data type to collect, value from ["fund", "stock"], default "fund"
        shanghai_only: bool
            if True, collect only Shanghai Stock Exchange stocks (all 6xx codes: 600xxx, 601xxx, 603xxx, 605xxx, 688xxx, etc.), default False
        """
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Force rate limiting settings
        self.delay = max(12, delay)  # Minimum 12 seconds delay
        self.max_workers = 1  # Single thread only
        self.max_collector_count = 1  # Single collector only
        self.interval = interval
        self.check_data_length = max(int(check_data_length) if check_data_length is not None else 0, 0)
        self.adjust_type = adjust
        self.data_type = data_type.lower()
        self.shanghai_only = shanghai_only

        if self.data_type not in ["fund", "stock"]:
            raise ValueError(f"data_type must be 'fund' or 'stock', got '{data_type}'")

        logger.info(f"AkShare collector initialized for {self.data_type} data")
        logger.info(f"Rate limiting: {self.delay} seconds delay between requests")
        logger.info(f"Data will be saved to: {self.save_dir}")
        
        if self.shanghai_only and self.data_type == "stock":
            logger.info("Shanghai Stock Exchange filter enabled")
        
        # Normalize datetime
        self.start_datetime = self.normalize_start_datetime(start)
        self.end_datetime = self.normalize_end_datetime(end)
        
        # Initialize datetime
        self.init_datetime()
        
        # Get instrument list
        self.instrument_list = sorted(set(self.get_instrument_list()))
        
        # Show total count
        logger.info(f"Total stocks to download: {len(self.instrument_list)}")
        
        if limit_nums is not None:
            try:
                self.instrument_list = self.instrument_list[:int(limit_nums)]
                logger.info(f"Limited to first {len(self.instrument_list)} stocks for testing")
            except Exception as e:
                logger.warning(f"Cannot use limit_nums={limit_nums}, the parameter will be ignored")

    def normalize_start_datetime(self, start_datetime: Union[str, pd.Timestamp] = None):
        return (
            pd.Timestamp(str(start_datetime))
            if start_datetime
            else getattr(self, f"DEFAULT_START_DATETIME_{self.interval.upper()}")
        )

    def normalize_end_datetime(self, end_datetime: Union[str, pd.Timestamp] = None):
        return (
            pd.Timestamp(str(end_datetime))
            if end_datetime
            else getattr(self, f"DEFAULT_END_DATETIME_{self.interval.upper()}")
        )

    def init_datetime(self):
        if self.interval == self.INTERVAL_1min:
            self.start_datetime = max(self.start_datetime, self.DEFAULT_START_DATETIME_1MIN)
        elif self.interval == self.INTERVAL_1d:
            pass
        else:
            raise ValueError(f"interval error: {self.interval}")

        self.start_datetime = self.convert_datetime(self.start_datetime, self._timezone)
        self.end_datetime = self.convert_datetime(self.end_datetime, self._timezone)

    @staticmethod
    def convert_datetime(dt: Union[pd.Timestamp, datetime.date, str], timezone):
        try:
            dt = pd.Timestamp(dt, tz=timezone).timestamp()
            dt = pd.Timestamp(dt, tz=tzlocal(), unit="s")
        except ValueError:
            pass
        return dt

    @property
    @abc.abstractmethod
    def _timezone(self):
        raise NotImplementedError("rewrite get_timezone")

    @abc.abstractmethod
    def get_instrument_list(self):
        raise NotImplementedError("rewrite get_instrument_list")

    @staticmethod
    def get_data_from_remote(symbol, interval, start_datetime, end_datetime, adjust="qfq", data_type="fund"):
        error_msg = f"{symbol}-{interval}-{start_datetime}-{end_datetime}-{data_type}"
        
        try:
            if data_type == "fund":
                # Get fund data
                df = ak.fund_open_fund_info_em(symbol=symbol, indicator="累计净值走势")

                if df.empty:
                    return pd.DataFrame()

                # Normalize fund data columns (adjust based on actual API response)
                fund_column_mapping = {
                    '净值日期': 'date',
                    '单位净值': 'close',
                    '累计净值': 'cumulative_nav',
                    '日增长率': 'change'
                }

                # Check what columns are actually available
                available_mapping = {k: v for k, v in fund_column_mapping.items() if k in df.columns}

                # If cumulative_nav is available but not close, use it as close
                if '累计净值' in df.columns and '单位净值' not in df.columns:
                    available_mapping['累计净值'] = 'close'

                df = df.rename(columns=available_mapping)

                # Ensure date column is datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])

                    # Convert start and end datetime to proper format for comparison
                    try:
                        start_date = pd.to_datetime(start_datetime.date() if hasattr(start_datetime, 'date') else start_datetime)
                        end_date = pd.to_datetime(end_datetime.date() if hasattr(end_datetime, 'date') else end_datetime)

                        # Filter by date range
                        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                    except Exception as e:
                        logger.warning(f"Date filtering error: {e}, skipping date filter")

                # Add required columns for consistency
                if 'close' in df.columns:
                    df['open'] = df['close']  # For funds, use close as open
                    df['high'] = df['close']  # For funds, use close as high/low
                    df['low'] = df['close']
                    df['volume'] = 0  # Funds don't have volume
                    df['money'] = 0   # Funds don't have money

                # Select standard columns
                standard_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'money']
                available_cols = [col for col in standard_cols if col in df.columns]

                # Add change and cumulative_nav if available
                if 'change' in df.columns:
                    available_cols.append('change')
                if 'cumulative_nav' in df.columns:
                    available_cols.append('cumulative_nav')

                df = df[available_cols].copy()

            elif data_type == "stock":
                # Get stock data
                start_date = start_datetime.strftime('%Y%m%d')
                end_date = end_datetime.strftime('%Y%m%d')

                if interval == "1d":
                    # Get daily data
                    df = ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust
                    )
                elif interval == "1min":
                    # Get 1-minute data (recent data only)
                    df = ak.stock_zh_a_hist_min_em(
                        symbol=symbol,
                        period="1",
                        adjust=adjust
                    )
                    # Filter by date range
                    if not df.empty:
                        df['日期'] = pd.to_datetime(df['时间'])
                        df = df[(df['日期'] >= start_datetime) & (df['日期'] <= end_datetime)]
                else:
                    raise ValueError(f"Unsupported interval: {interval}")

                if df.empty:
                    return pd.DataFrame()

                # Normalize stock data column names to English
                stock_column_mapping = {
                    '日期': 'date',
                    '时间': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'money',
                    '振幅': 'amplitude',
                    '涨跌幅': 'change',
                    '涨跌额': 'change_amount',
                    '换手率': 'turnover'
                }

                df = df.rename(columns=stock_column_mapping)

                # Ensure date column is datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])

                # Select standard columns for qlib compatibility
                standard_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'money']
                available_cols = [col for col in standard_cols if col in df.columns]

                # Add change column if available
                if 'change' in df.columns:
                    available_cols.append('change')

                df = df[available_cols].copy()
            else:
                raise ValueError(f"Unsupported data_type: {data_type}")

            if isinstance(df, pd.DataFrame):
                return df.reset_index(drop=True)

        except Exception as e:
            logger.warning(f"{error_msg}:{e}")

        return pd.DataFrame()

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        def _get_simple(start_, end_):
            self.sleep()
            return self.get_data_from_remote(
                symbol,
                interval=interval,
                start_datetime=start_,
                end_datetime=end_,
                adjust=self.adjust_type,
                data_type=self.data_type
            )

        if interval == self.INTERVAL_1d:
            _result = _get_simple(start_datetime, end_datetime)
        else:
            raise ValueError(f"cannot support {interval}")
        return _result

    def sleep(self):
        """Sleep for delay time with rate limiting info"""
        if self.delay > 0:
            logger.info(f"Rate limiting: Waiting {self.delay} seconds before next request...")
            time.sleep(self.delay)

    def save_instrument(self, symbol: str, df: pd.DataFrame):
        """Save data for a single instrument"""
        if df.empty:
            logger.warning(f"No data to save for {symbol}")
            return
            
        # Convert to qlib format filename  
        filename = f"{symbol}.csv"
        filepath = self.save_dir / filename
        
        # Save with proper encoding
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"Saved {len(df)} records to {filename}")

    def collector_data(self, incremental=False):
        """Main data collection method with DuckDB storage and incremental updates"""
        logger.info("=" * 80)
        logger.info("SHANGHAI STOCK DATA COLLECTION STARTED")
        logger.info("=" * 80)
        logger.info(f"Total stocks to download: {len(self.instrument_list)}")
        logger.info(f"Data storage path: {self.save_dir}")
        logger.info(f"Date range: {self.start_datetime.date()} to {self.end_datetime.date()}")
        logger.info(f"Rate limiting: 1 request per {self.delay} seconds")
        logger.info(f"Incremental mode: {'Enabled' if incremental else 'Disabled'}")
        
        # Initialize DuckDB for stock data storage
        stock_data_conn = self._init_stock_data_db()
        self._create_stock_data_tables(stock_data_conn)
        
        logger.info("=" * 80)
        
        if not self.instrument_list:
            logger.error("No symbols to collect")
            stock_data_conn.close()
            return
        
        # Calculate estimated time (less accurate for incremental mode)
        if incremental:
            logger.info("Estimated time: Variable (depends on existing data)")
        else:
            total_time_minutes = (len(self.instrument_list) * self.delay) / 60
            logger.info(f"Estimated completion time: {total_time_minutes:.1f} minutes ({total_time_minutes/60:.1f} hours)")
        
        # Ask for confirmation
        try:
            response = input(f"\nProceed with downloading {len(self.instrument_list)} stocks? [y/N]: ").strip().lower()
            if response not in ['y', 'yes']:
                logger.info("Download cancelled by user")
                stock_data_conn.close()
                return
        except KeyboardInterrupt:
            logger.info("\nDownload cancelled by user")
            stock_data_conn.close()
            return
        
        logger.info("\nStarting data collection...")
        
        # Collect data for each symbol with progress bar
        success_count = 0
        failed_stocks = []
        skipped_na_count = 0
        incremental_stats = {"full": 0, "incremental": 0, "up_to_date": 0, "gap_fill": 0}
        
        # Use tqdm for progress bar, but fall back if not available
        try:
            from tqdm import tqdm
            progress_bar = tqdm(self.instrument_list, desc="Downloading", unit="stock")
        except ImportError:
            logger.warning("tqdm not available, using simple progress counter")
            progress_bar = self.instrument_list
        
        for i, symbol in enumerate(progress_bar):
            try:
                # Update progress description if using tqdm
                if hasattr(progress_bar, 'set_description'):
                    progress_bar.set_description(f"Downloading {symbol}")
                
                if incremental:
                    # Calculate incremental date range
                    actual_start, actual_end, update_type = self._calculate_incremental_date_range(
                        stock_data_conn, symbol, self.start_datetime, self.end_datetime)
                    
                    incremental_stats[update_type] += 1
                    
                    if update_type == "up_to_date":
                        logger.info(f"⚡ {symbol} ({i+1}/{len(self.instrument_list)}) - Already up to date")
                        success_count += 1
                        continue
                    
                    logger.info(f"📊 {symbol} ({i+1}/{len(self.instrument_list)}) - {update_type.upper()}: {actual_start} to {actual_end}")
                    
                    # Create temporary datetime objects for the actual range
                    temp_start = pd.Timestamp(actual_start)
                    temp_end = pd.Timestamp(actual_end)
                    df, result_status = self._get_data_with_retry(symbol, self.interval, temp_start, temp_end)
                else:
                    # Full range download
                    logger.info(f"📈 Processing {symbol} ({i+1}/{len(self.instrument_list)}) - Progress: {(i+1)/len(self.instrument_list)*100:.1f}%")
                    df, result_status = self._get_data_with_retry(symbol, self.interval, self.start_datetime, self.end_datetime)
                    incremental_stats["full"] += 1
                
                # Handle different result statuses
                if result_status == "skipped_na":
                    skipped_na_count += 1
                    continue
                elif result_status == "success" and not df.empty:
                    # Save to CSV (backward compatibility)
                    self.save_instrument(symbol, df)
                    
                    # Save to DuckDB
                    stock_name = self._get_stock_name(symbol)
                    records_stored = self._store_stock_data_to_duckdb(stock_data_conn, symbol, df, stock_name)
                    
                    success_count += 1
                    update_info = f" ({update_type})" if incremental else ""
                    logger.info(f"✅ Successfully downloaded {symbol}: {len(df)} records{update_info}")
                elif result_status == "failed_max_retries":
                    failed_stocks.append(f"{symbol} (Failed after 3 attempts)")
                else:
                    # Other failure cases - use simple diagnosis
                    failure_reason = self._get_simple_failure_reason(symbol)
                    logger.warning(f"❌ No data retrieved for {symbol} - {failure_reason}")
                    failed_stocks.append(f"{symbol} ({failure_reason})")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                failed_stocks.append(f"{symbol} (API Error: {str(e)[:50]})")
                continue
        
        # Close progress bar if using tqdm
        if hasattr(progress_bar, 'close'):
            progress_bar.close()
        
        # Close DuckDB connection
        stock_data_conn.close()
        
        # Final summary
        logger.info("=" * 80)
        logger.info("DATA COLLECTION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Successfully processed: {success_count}/{len(self.instrument_list)} stocks")
        logger.info(f"Skipped (NA status): {skipped_na_count} stocks")
        logger.info(f"Failed: {len(failed_stocks)} stocks")
        logger.info(f"Success rate: {success_count/len(self.instrument_list)*100:.1f}%")
        logger.info(f"Data saved to CSV: {self.save_dir}")
        logger.info(f"Data stored in DuckDB: {self.save_dir / 'shanghai_stock_data.duckdb'}")
        
        if incremental:
            logger.info("Incremental update summary:")
            for update_type, count in incremental_stats.items():
                if count > 0:
                    logger.info(f"  {update_type.replace('_', ' ').title()}: {count} stocks")
        
        if failed_stocks:
            logger.warning(f"Failed stocks ({len(failed_stocks)}): {failed_stocks[:10]}{'...' if len(failed_stocks) > 10 else ''}")
            
            # Save failed stocks list
            failed_file = self.save_dir / "failed_stocks.txt"
            with open(failed_file, 'w', encoding='utf-8') as f:
                for stock in failed_stocks:
                    f.write(f"{stock}\n")
            logger.info(f"Failed stocks list saved to: {failed_file}")
        
        logger.info("=" * 80)


class AkShareCollectorCN(AkShareCollector):
    def get_instrument_list(self):
        if self.data_type == "fund":
            return self._get_fund_list()
        elif self.data_type == "stock":
            if self.shanghai_only:
                return self._get_active_shanghai_stocks()  # Use active stocks only
            else:
                return self._get_stock_list()
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

    def _get_fund_list(self):
        logger.info("get cn fund symbols......")
        try:
            # Get fund list
            fund_list = ak.fund_name_em()
            symbols = fund_list['基金代码'].tolist()
            logger.info(f"get {len(symbols)} fund symbols.")
            return symbols
        except Exception as e:
            logger.error(f"Error getting fund list: {e}")
            # Return some default fund codes if API fails
            default_funds = ["015198", "110022", "161725", "000001", "519066", "110011",
                           "000300", "519674", "000905", "110020", "161017", "000831"]
            logger.info(f"Using default fund list: {len(default_funds)} funds")
            return default_funds

    def _get_stock_list(self):
        logger.info("get cn stock symbols......")
        try:
            # Get A-share stock list
            stock_list = ak.stock_info_a_code_name()
            symbols = stock_list['code'].tolist()
            logger.info(f"get {len(symbols)} stock symbols.")
            return symbols
        except Exception as e:
            logger.error(f"Error getting stock list: {e}")
            return []
    
    def _init_stock_data_db(self):
        """Initialize DuckDB database for stock data storage"""
        db_path = self.save_dir / "shanghai_stock_data.duckdb"
        return duckdb.connect(str(db_path))
    
    def _create_stock_data_tables(self, conn):
        """Create tables for stock data if they don't exist"""
        # Create main stock data table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                open DECIMAL(10,2),
                high DECIMAL(10,2),
                low DECIMAL(10,2),
                close DECIMAL(10,2),
                volume BIGINT,
                amount DECIMAL(18,2),
                amplitude DECIMAL(8,4),
                change_percent DECIMAL(8,4),
                change_amount DECIMAL(10,2),
                turnover_rate DECIMAL(8,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """)
        
        # Create metadata table to track update status
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_update_metadata (
                symbol VARCHAR PRIMARY KEY,
                name VARCHAR,
                first_date DATE,
                last_date DATE,
                total_records INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR DEFAULT 'active'
            )
        """)
        
        # Create indexes for better query performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_data_date ON stock_data(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_data_symbol_date ON stock_data(symbol, date)")
    
    def _get_stock_last_update_date(self, conn, symbol):
        """Get the last update date for a specific stock"""
        try:
            # First try to get from metadata table
            result = conn.execute("""
                SELECT last_date FROM stock_update_metadata 
                WHERE symbol = ?
            """, [symbol]).fetchone()
            
            if result and result[0]:
                metadata_date = pd.to_datetime(result[0]).date()
            else:
                metadata_date = None
            
            # Also check the actual stock_data table for the real latest date
            actual_result = conn.execute("""
                SELECT MAX(date) FROM stock_data WHERE symbol = ?
            """, [symbol]).fetchone()
            
            if actual_result and actual_result[0]:
                actual_date = pd.to_datetime(actual_result[0]).date()
            else:
                actual_date = None
            
            # Return the later of the two dates (prioritize actual data)
            if actual_date and metadata_date:
                return max(actual_date, metadata_date)
            elif actual_date:
                return actual_date
            elif metadata_date:
                return metadata_date
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Error getting last update date for {symbol}: {e}")
            return None
    
    def _update_stock_metadata(self, conn, symbol, name, first_date, last_date, record_count):
        """Update or insert stock metadata"""
        conn.execute("""
            INSERT OR REPLACE INTO stock_update_metadata 
            (symbol, name, first_date, last_date, total_records, last_updated, status)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 'active')
        """, [symbol, name, first_date, last_date, record_count])
    
    def _store_stock_data_to_duckdb(self, conn, symbol, df, stock_name=""):
        """Store stock data to DuckDB"""
        if df.empty:
            return 0
            
        try:
            # Prepare data for insertion
            df_clean = df.copy()
            df_clean['symbol'] = symbol
            df_clean['created_at'] = pd.Timestamp.now()
            df_clean['updated_at'] = pd.Timestamp.now()
            
            # Rename columns to match database schema
            column_mapping = {
                '日期': 'date', 'date': 'date',
                '开盘': 'open', 'open': 'open', 
                '最高': 'high', 'high': 'high',
                '最低': 'low', 'low': 'low',
                '收盘': 'close', 'close': 'close',
                '成交量': 'volume', 'volume': 'volume',
                '成交额': 'amount', 'amount': 'amount',
                '振幅': 'amplitude', 'amplitude': 'amplitude', 
                '涨跌幅': 'change_percent', 'change_percent': 'change_percent',
                '涨跌额': 'change_amount', 'change_amount': 'change_amount',
                '换手率': 'turnover_rate', 'turnover_rate': 'turnover_rate'
            }
            
            # Apply column mapping
            for old_col, new_col in column_mapping.items():
                if old_col in df_clean.columns:
                    df_clean = df_clean.rename(columns={old_col: new_col})
            
            # Ensure required columns exist
            required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df_clean.columns:
                    if col == 'volume':
                        df_clean[col] = 0
                    elif col in ['open', 'high', 'low', 'close']:
                        df_clean[col] = None
            
            # Convert date column
            if 'date' in df_clean.columns:
                df_clean['date'] = pd.to_datetime(df_clean['date']).dt.date
            
            # Select only columns that exist in the table
            table_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 
                           'volume', 'amount', 'amplitude', 'change_percent', 
                           'change_amount', 'turnover_rate', 'created_at', 'updated_at']
            
            available_columns = [col for col in table_columns if col in df_clean.columns]
            df_final = df_clean[available_columns]
            
            # Insert data using DuckDB's efficient bulk insert
            # Use ON CONFLICT to handle duplicates (update existing records)
            conn.execute("BEGIN TRANSACTION")
            
            try:
                # Delete existing records for this symbol and date range
                if not df_final.empty:
                    date_list = df_final['date'].tolist()
                    placeholders = ','.join(['?' for _ in date_list])
                    conn.execute(f"""
                        DELETE FROM stock_data 
                        WHERE symbol = ? AND date IN ({placeholders})
                    """, [symbol] + date_list)
                
                # Insert new records
                conn.executemany(f"""
                    INSERT INTO stock_data ({','.join(available_columns)})
                    VALUES ({','.join(['?' for _ in available_columns])})
                """, df_final.values.tolist())
                
                # Update metadata
                first_date = df_final['date'].min()
                last_date = df_final['date'].max()
                record_count = len(df_final)
                
                self._update_stock_metadata(conn, symbol, stock_name, 
                                          first_date, last_date, record_count)
                
                conn.execute("COMMIT")
                logger.info(f"Stored {record_count} records for {symbol} in DuckDB")
                return record_count
                
            except Exception as e:
                conn.execute("ROLLBACK")
                logger.error(f"Error storing data for {symbol}: {e}")
                return 0
                
        except Exception as e:
            logger.error(f"Error preparing data for {symbol}: {e}")
            return 0
    
    def _calculate_incremental_date_range(self, conn, symbol, requested_start, requested_end):
        """Calculate the actual date range needed for incremental update"""
        last_update_date = self._get_stock_last_update_date(conn, symbol)
        
        if last_update_date is None:
            # No existing data, download full range
            logger.debug(f"{symbol}: No existing data, downloading full range {requested_start.date()} to {requested_end.date()}")
            return requested_start.date(), requested_end.date(), "full"
        
        # Convert to date objects for comparison
        req_start_date = requested_start.date()
        req_end_date = requested_end.date()
        
        if last_update_date >= req_end_date:
            # Data is already up to date
            logger.debug(f"{symbol}: Data up to date (last: {last_update_date}, requested end: {req_end_date})")
            return None, None, "up_to_date"
        
        if last_update_date < req_start_date:
            # Gap in data, download full requested range
            logger.debug(f"{symbol}: Gap in data (last: {last_update_date}, requested start: {req_start_date})")
            return req_start_date, req_end_date, "gap_fill"
        
        # Normal incremental update: from day after last update to requested end
        incremental_start = last_update_date + timedelta(days=1)
        logger.debug(f"{symbol}: Incremental update from {incremental_start} to {req_end_date}")
        return incremental_start, req_end_date, "incremental"
    
    def _init_stock_symbols_cache(self):
        """Initialize DuckDB cache for stock symbols"""
        db_path = self.save_dir / "stock_symbols_cache.duckdb"
        return duckdb.connect(str(db_path))
    
    def _is_cache_valid(self, conn, table_name="shanghai_stocks", max_age_days=7):
        """Check if cached stock symbols are still valid (not older than max_age_days)"""
        try:
            result = conn.execute(f"""
                SELECT COUNT(*) as count, MAX(cached_date) as last_update 
                FROM {table_name}
            """).fetchone()
            
            if result[0] == 0:  # No data
                return False
                
            last_update = pd.to_datetime(result[1])
            age_days = (pd.Timestamp.now() - last_update).days
            
            logger.info(f"Cache last updated: {last_update.strftime('%Y-%m-%d')}, age: {age_days} days")
            return age_days <= max_age_days
            
        except Exception:
            return False
    
    def _filter_active_stocks(self, symbols_df):
        """Filter out obviously delisted/suspended stocks from the symbols dataframe based on market data only"""
        logger.info("Filtering stocks based on market status indicators...")
        
        original_count = len(symbols_df)
        active_stocks = []
        filtered_count = 0
        
        try:
            # Check each stock's basic information from the market data (no downloads)
            for idx, row in symbols_df.iterrows():
                symbol = row['代码'] if '代码' in row else row.get('symbol', row.iloc[0])
                name = row['名称'] if '名称' in row else row.get('name', row.iloc[1] if len(row) > 1 else 'Unknown')
                latest_price = row.get('最新价', None)
                
                # Basic filtering based on available market data only
                is_active = True
                reason = ""
                
                # Check if stock code follows valid patterns
                if not str(symbol).isdigit() or len(str(symbol)) != 6:
                    is_active = False
                    reason = "Invalid stock code format"
                
                # Check name for obvious delisting indicators
                elif isinstance(name, str):
                    if any(indicator in name for indicator in ['退市', '终止', '摘牌']):
                        is_active = False
                        reason = f"Delisted indicator in name ({name})"
                
                # Check if price data suggests the stock is trading (but don't make API calls)
                elif pd.notna(latest_price) and latest_price == 0:
                    is_active = False
                    reason = "Zero price in market data"
                
                if is_active:
                    active_stocks.append(symbol)
                else:
                    filtered_count += 1
                    if filtered_count <= 5:  # Log first 5 filtered stocks
                        logger.info(f"  Filtered {symbol} ({name}): {reason}")
                    elif filtered_count == 6:
                        logger.info("  ... (additional filtered stocks not shown)")
        
        except Exception as e:
            logger.warning(f"Error in basic stock filtering: {e}, using all stocks")
            return symbols_df['代码'].astype(str).tolist() if '代码' in symbols_df.columns else symbols_df.iloc[:, 0].astype(str).tolist()
        
        logger.info(f"Basic filtering completed: {original_count} -> {len(active_stocks)} (filtered {filtered_count})")
        return active_stocks
    
    def _cache_shanghai_stocks(self, conn):
        """Fetch and cache Shanghai stock symbols in DuckDB with basic filtering (no downloads)"""
        logger.info("Fetching Shanghai stock symbols from AkShare market data...")
        try:
            # Get market snapshot (this is a single API call for all stocks)
            stock_list = ak.stock_zh_a_spot_em()
            # Filter Shanghai stocks: All codes starting with 6
            shanghai_stocks = stock_list[
                stock_list['代码'].astype(str).str.startswith('6')
            ].copy()
            
            logger.info(f"Found {len(shanghai_stocks)} Shanghai stocks from market data")
            
            # Apply basic filtering (no additional API calls)
            active_symbols = self._filter_active_stocks(shanghai_stocks)
            
            # Create clean dataframe with only basically active stocks
            active_shanghai_stocks = shanghai_stocks[
                shanghai_stocks['代码'].isin(active_symbols)
            ].copy()
            
            # Add metadata
            active_shanghai_stocks['cached_date'] = pd.Timestamp.now()
            active_shanghai_stocks['exchange'] = 'Shanghai'
            active_shanghai_stocks['status'] = 'active'  # All start as active, will be updated during download
            
            # Rename columns for consistency
            active_shanghai_stocks = active_shanghai_stocks.rename(columns={
                '代码': 'symbol',
                '名称': 'name'
            })
            
            # Select only needed columns
            columns_to_keep = ['symbol', 'name', 'cached_date', 'exchange', 'status']
            available_columns = [col for col in columns_to_keep if col in active_shanghai_stocks.columns]
            if 'name' not in available_columns:
                active_shanghai_stocks['name'] = active_shanghai_stocks['symbol']  # fallback
                available_columns.append('name')
            
            # Add default values for new columns (no validation, just defaults)
            active_shanghai_stocks['retry_count'] = 0
            active_shanghai_stocks['last_attempt_date'] = None
            active_shanghai_stocks['failure_reason'] = None
            available_columns.extend(['retry_count', 'last_attempt_date', 'failure_reason'])
            
            shanghai_stocks_clean = active_shanghai_stocks[available_columns]
            
            # Create table and insert data
            conn.execute("DROP TABLE IF EXISTS shanghai_stocks")
            conn.execute("""
                CREATE TABLE shanghai_stocks (
                    symbol VARCHAR PRIMARY KEY,
                    name VARCHAR,
                    cached_date TIMESTAMP,
                    exchange VARCHAR,
                    status VARCHAR DEFAULT 'active',
                    retry_count INTEGER DEFAULT 0,
                    last_attempt_date TIMESTAMP,
                    failure_reason VARCHAR
                )
            """)
            
            # Insert data
            conn.executemany(
                "INSERT INTO shanghai_stocks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                shanghai_stocks_clean.values.tolist()
            )
            
            symbols = shanghai_stocks_clean['symbol'].tolist()
            logger.info(f"Cached {len(symbols)} Shanghai stock symbols to DuckDB (all marked as active initially)")
            
            # Log breakdown by prefix
            prefixes = ['600', '601', '603', '605', '688', '689']
            for prefix in prefixes:
                count = sum(1 for s in symbols if s.startswith(prefix))
                if count > 0:
                    logger.info(f"  {prefix}xxx stocks: {count}")
                    
            return symbols
            
        except Exception as e:
            logger.error(f"Error caching Shanghai stock list: {e}")
            return []
    
    def _get_cached_shanghai_stocks(self, conn):
        """Get Shanghai stock symbols from DuckDB cache"""
        try:
            result = conn.execute("""
                SELECT symbol, name, cached_date 
                FROM shanghai_stocks 
                ORDER BY symbol
            """).fetchall()
            
            symbols = [row[0] for row in result]
            cache_date = result[0][2] if result else None
            
            logger.info(f"Retrieved {len(symbols)} Shanghai stocks from cache (updated: {cache_date})")
            logger.info(f"Sample cached stocks: {symbols[:10]}")
            
            # Log breakdown by prefix
            prefixes = ['600', '601', '603', '605', '688', '689']
            for prefix in prefixes:
                count = sum(1 for s in symbols if s.startswith(prefix))
                if count > 0:
                    logger.info(f"  {prefix}xxx stocks: {count}")
                    
            return symbols
            
        except Exception as e:
            logger.error(f"Error retrieving cached Shanghai stocks: {e}")
            return []
    
    def _get_shanghai_stock_list(self):
        """Get Shanghai Stock Exchange stocks from DuckDB cache or fetch fresh if needed"""
        logger.info("Getting Shanghai stock symbols from cache...")
        
        try:
            # Initialize cache connection
            conn = self._init_stock_symbols_cache()
            
            # Check if we have valid cached data
            try:
                conn.execute("SELECT 1 FROM shanghai_stocks LIMIT 1")
                cache_exists = True
            except:
                cache_exists = False
                logger.info("No existing cache found, will fetch fresh data")
            
            if cache_exists and self._is_cache_valid(conn):
                # Use cached data
                symbols = self._get_cached_shanghai_stocks(conn)
                if symbols:  # If cache retrieval successful
                    conn.close()
                    return symbols
            
            # Cache is invalid or empty, fetch fresh data
            logger.info("Cache invalid or empty, fetching fresh data from AkShare...")
            symbols = self._cache_shanghai_stocks(conn)
            conn.close()
            return symbols
            
        except Exception as e:
            logger.error(f"Error in Shanghai stock list management: {e}")
            # Fallback to direct AkShare query
            logger.info("Falling back to direct AkShare query...")
            return self._get_shanghai_stock_list_direct()

    def _get_active_shanghai_stocks(self):
        """Get Shanghai stocks excluding those with NA status"""
        try:
            conn = self._init_stock_symbols_cache()
            
            # Get stocks that are not marked as NA
            result = conn.execute("""
                SELECT symbol, name, status, retry_count 
                FROM shanghai_stocks 
                WHERE status != 'NA' OR status IS NULL
                ORDER BY symbol
            """).fetchall()
            
            symbols = [row[0] for row in result]
            na_count = conn.execute("""
                SELECT COUNT(*) FROM shanghai_stocks WHERE status = 'NA'
            """).fetchone()[0]
            
            conn.close()
            
            logger.info(f"Active stocks (excluding NA): {len(symbols)}")
            logger.info(f"Stocks marked as NA: {na_count}")
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error getting active stocks: {e}")
            # Fallback to get all stocks
            return self._get_shanghai_stock_list()
    
    def _get_shanghai_stock_list_direct(self):
        """Direct AkShare query fallback method"""
        try:
            stock_list = ak.stock_zh_a_spot_em()
            shanghai_stocks = stock_list[
                stock_list['代码'].astype(str).str.startswith('6')
            ]
            symbols = shanghai_stocks['代码'].astype(str).tolist()
            logger.info(f"Direct query returned {len(symbols)} Shanghai stock symbols")
            return symbols
        except Exception as e:
            logger.error(f"Direct AkShare query also failed: {e}")
            return []

    def _get_stock_status(self, symbol):
        """Get stock status from cache"""
        try:
            conn = self._init_stock_symbols_cache()
            result = conn.execute("""
                SELECT status, retry_count, failure_reason 
                FROM shanghai_stocks WHERE symbol = ?
            """, [symbol]).fetchone()
            conn.close()
            
            if result:
                return {
                    'status': result[0] or 'active',
                    'retry_count': result[1] or 0,
                    'failure_reason': result[2] or None
                }
            return {'status': 'active', 'retry_count': 0, 'failure_reason': None}
        except Exception:
            return {'status': 'active', 'retry_count': 0, 'failure_reason': None}

    def _update_stock_status(self, symbol, status, retry_count=None, failure_reason=None):
        """Update stock status in cache"""
        try:
            conn = self._init_stock_symbols_cache()
            
            if retry_count is not None:
                conn.execute("""
                    UPDATE shanghai_stocks 
                    SET status = ?, retry_count = ?, last_attempt_date = CURRENT_TIMESTAMP, failure_reason = ?
                    WHERE symbol = ?
                """, [status, retry_count, failure_reason, symbol])
            else:
                conn.execute("""
                    UPDATE shanghai_stocks 
                    SET status = ?, last_attempt_date = CURRENT_TIMESTAMP, failure_reason = ?
                    WHERE symbol = ?
                """, [status, failure_reason, symbol])
            
            conn.close()
            logger.info(f"Updated status for {symbol}: {status} (retry_count: {retry_count}, reason: {failure_reason})")
        except Exception as e:
            logger.error(f"Error updating stock status for {symbol}: {e}")

    def _get_stock_name(self, symbol):
        """Get stock name from cache or return symbol if not found"""
        try:
            conn = self._init_stock_symbols_cache()
            result = conn.execute("""
                SELECT name FROM shanghai_stocks WHERE symbol = ?
            """, [symbol]).fetchone()
            conn.close()
            
            if result:
                return result[0]
            return symbol
        except Exception:
            return symbol

    def _get_data_with_retry(self, symbol, interval, start_datetime, end_datetime, max_retries=3):
        """Get data with retry logic"""
        stock_status = self._get_stock_status(symbol)
        
        # Skip if stock is already marked as NA
        if stock_status['status'] == 'NA':
            logger.info(f"⏭️ Skipping {symbol} - already marked as NA ({stock_status['failure_reason']})")
            return pd.DataFrame(), "skipped_na"
        
        retry_count = stock_status['retry_count']
        
        for attempt in range(max_retries):
            current_attempt = retry_count + attempt + 1
            
            try:
                logger.info(f"🔄 Attempt {current_attempt}/{max_retries} for {symbol}")
                
                # Sleep before request (except first attempt if retry_count is 0)
                if current_attempt > 1:
                    self.sleep()
                else:
                    self.sleep()
                
                df = self.get_data_from_remote(
                    symbol,
                    interval=interval,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    adjust=self.adjust_type,
                    data_type=self.data_type
                )
                
                if not df.empty:
                    # Success - reset status to active if it was previously failed
                    if stock_status['status'] != 'active':
                        self._update_stock_status(symbol, 'active', 0, None)
                    return df, "success"
                else:
                    # Empty data returned
                    logger.warning(f"❌ Attempt {current_attempt}: No data returned for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Attempt {current_attempt}: Error for {symbol}: {str(e)[:100]}")
                
        # All attempts failed
        total_retry_count = retry_count + max_retries
        failure_reason = f"Failed after {total_retry_count} attempts"
        
        # Mark as NA after max retries
        self._update_stock_status(symbol, 'NA', total_retry_count, failure_reason)
        logger.error(f"🚫 {symbol} marked as NA after {total_retry_count} failed attempts")
        
        return pd.DataFrame(), "failed_max_retries"

    def _get_simple_failure_reason(self, symbol):
        """Get a simple failure reason without making API calls"""
        # Check if it's a valid Shanghai stock code
        if not str(symbol).startswith('6') or len(str(symbol)) != 6:
            return "Invalid Shanghai stock code"
        
        # Check if it's in known problematic ranges
        if str(symbol).startswith('689'):  # STAR Market
            return "STAR Market stock - may have limited data availability"
        
        return "No data available for requested date range"

    def normalize_symbol(self, symbol):
        return symbol

    def _diagnose_stock_failure(self, symbol):
        """Diagnose why a stock failed to retrieve data"""
        try:
            # Check if stock exists in current listings
            stock_list = ak.stock_zh_a_spot_em()
            stock_info = stock_list[stock_list['代码'] == symbol]
            
            if stock_info.empty:
                return "Stock not found in current listings"
            
            # Check if price is NaN (often indicates delisted/suspended)
            latest_price = stock_info.iloc[0]['最新价']
            if pd.isna(latest_price) or latest_price == 0:
                return "Possibly delisted/suspended (no current price)"
            
            # Check trading status indicators
            stock_name = stock_info.iloc[0]['名称']
            if 'ST' in stock_name or '退' in stock_name:
                return "Special treatment or delisting stock"
            
            return "No data for requested date range"
            
        except Exception:
            return "Unable to diagnose (API issue)"

    @property
    def _timezone(self):
        return "Asia/Shanghai"


class AkShareCollectorCN1d(AkShareCollectorCN):
    pass


class Run:
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=4, interval="1d", region=REGION_CN):
        """
        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved
        normalize_dir: str
            Directory for normalize data
        max_workers: int
            Concurrent number, default is 4
        interval: str
            freq, value from [1min, 1d], default 1d
        region: str
            region, value from ["CN"], default "CN"
        """
        if source_dir is None:
            source_dir = CUR_DIR / "source"
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.source_dir.mkdir(parents=True, exist_ok=True)
        
        if normalize_dir is None:
            normalize_dir = CUR_DIR / "normalize"
        self.normalize_dir = Path(normalize_dir).expanduser().resolve()
        self.normalize_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_workers = max_workers
        self.interval = interval
        self.region = region

    def get_collector_class(self):
        """Get collector class based on region and interval"""
        class_name = f"AkShareCollector{self.region.upper()}{self.interval}"
        return globals().get(class_name)

    def download_data(
        self,
        max_collector_count=1,  # Force single collector
        delay=12,  # Force 12 seconds delay
        start=None,
        end=None,
        check_data_length: int = None,
        limit_nums=None,
        adjust="qfq",
        data_type="fund",  # Default to fund data
        shanghai_only=False,  # New parameter for Shanghai stocks only
        force=False,  # New parameter to force full download from beginning
    ):
        """download data from Internet with strict rate limiting

        Parameters
        ----------
        max_collector_count: int
            forced to 1 for rate limiting
        delay: float
            time.sleep(delay), forced minimum 12 seconds for rate limiting
        start: str
            start datetime, default "2000-01-01"
        end: str
            end datetime, default current date + 1 day
        check_data_length: int
            check data length, default None
        limit_nums: int
            using for debug, by default None
        adjust: str
            adjustment type, qfq/hfq/empty, default qfq
        data_type: str
            data type to collect, value from ["fund", "stock"], default "fund"
        shanghai_only: bool
            if True, collect only Shanghai Stock Exchange stocks, default False
        force: bool
            if True, force full download from beginning (ignore existing data), default False

        Examples
        ---------
            # Resume interrupted download (RECOMMENDED - will continue from last position)
            $ python collector_standalone.py download_data --start 2022-01-01 --end 2025-01-01 --data_type stock --shanghai_only True --delay 12
            
            # Force complete redownload from beginning (ignores existing data)
            $ python collector_standalone.py download_data --start 2022-01-01 --end 2025-01-01 --data_type stock --shanghai_only True --delay 12 --force True

            # test with few stocks first
            $ python collector_standalone.py download_data --start 2022-02-02 --end 2025-01-01 --limit_nums 5 --data_type stock --shanghai_only True --delay 12
        """
        # Enforce rate limiting
        delay = max(12, delay)  # Minimum 12 seconds
        max_collector_count = 1  # Single thread only
        
        collector_class = self.get_collector_class()
        if collector_class is None:
            logger.error(f"Collector class not found for {self.region.upper()}{self.interval}")
            return

        logger.info("=" * 80)
        logger.info("AKSHARE DATA COLLECTOR - RATE LIMITED VERSION")
        logger.info("=" * 80)
        logger.info("Settings:")
        logger.info(f"  - Single thread download only")
        logger.info(f"  - Minimum {delay} seconds between requests")
        logger.info(f"  - Data type: {data_type}")
        logger.info(f"  - Shanghai only: {shanghai_only}")
        logger.info(f"  - Date range: {start} to {end}")
        if limit_nums:
            logger.info(f"  - Limited to {limit_nums} stocks (testing mode)")
        logger.info(f"  - Force mode: {'Enabled (full redownload)' if force else 'Disabled (resume from existing data)'}")
        logger.info("=" * 80)

        # Check if database exists and has data (for resume capability)
        db_path = self.source_dir / "shanghai_stock_data.duckdb"
        has_existing_data = db_path.exists()
        
        if not force and has_existing_data and data_type == "stock" and shanghai_only:
            try:
                # Quick check to see if we have existing data
                import duckdb
                conn = duckdb.connect(str(db_path))
                result = conn.execute("SELECT COUNT(*) FROM stock_data").fetchone()
                existing_records = result[0] if result else 0
                conn.close()
                
                if existing_records > 0:
                    logger.info(f"📊 Found existing database with {existing_records:,} records")
                    logger.info("🔄 Resuming from existing data (incremental mode)")
                    logger.info("💡 Use --force True to ignore existing data and restart from beginning")
                    logger.info("=" * 80)
                    
                    # Use incremental mode instead of full download
                    collector = collector_class(
                        self.source_dir,
                        max_workers=1,  # Force single worker
                        max_collector_count=max_collector_count,
                        delay=delay,
                        start=start,
                        end=end,
                        interval=self.interval,
                        check_data_length=check_data_length,
                        limit_nums=limit_nums,
                        adjust=adjust,
                        data_type=data_type,
                        shanghai_only=shanghai_only,
                    )
                    
                    collector.collector_data(incremental=True)
                    return
            except Exception as e:
                logger.warning(f"Could not check existing data: {e}, proceeding with full download")
        
        if force:
            logger.info("🚨 FORCE MODE: Ignoring existing data, starting fresh download")
            logger.info("=" * 80)

        collector = collector_class(
            self.source_dir,
            max_workers=1,  # Force single worker
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            interval=self.interval,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            adjust=adjust,
            data_type=data_type,
            shanghai_only=shanghai_only,
        )
        
        collector.collector_data(incremental=False)

    def update_data(
        self,
        max_collector_count=1,  # Force single collector
        delay=12,  # Force 12 seconds delay
        start=None,
        end=None,
        check_data_length: int = None,
        limit_nums=None,
        adjust="qfq",
        data_type="stock",  # Default to stock data for updates
        shanghai_only=True,  # Default to Shanghai stocks for updates
    ):
        """Incremental update of stock data in DuckDB

        This method checks the last update date for each stock and only downloads
        data from the last update date to the specified end date.

        Parameters
        ----------
        max_collector_count: int
            forced to 1 for rate limiting
        delay: float
            time.sleep(delay), forced minimum 12 seconds for rate limiting
        start: str
            start datetime, default "2022-01-01"
        end: str
            end datetime, default current date + 1 day
        check_data_length: int
            check data length, default None
        limit_nums: int
            using for debug, by default None
        adjust: str
            adjustment type, qfq/hfq/empty, default qfq
        data_type: str
            data type to collect, value from ["fund", "stock"], default "stock"
        shanghai_only: bool
            if True, collect only Shanghai Stock Exchange stocks, default True

        Examples
        ---------
            # Incremental update to current date
            $ python collector_standalone.py update_data --end $(date +%Y-%m-%d)
            
            # Update specific date range
            $ python collector_standalone.py update_data --start 2024-01-01 --end 2024-12-31
        """
        # Enforce rate limiting
        delay = max(12, delay)  # Minimum 12 seconds
        max_collector_count = 1  # Single thread only
        
        collector_class = self.get_collector_class()
        if collector_class is None:
            logger.error(f"Collector class not found for {self.region.upper()}{self.interval}")
            return

        # Set default dates for incremental updates
        if start is None:
            start = "2022-01-01"  # Conservative start date
        if end is None:
            end = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        logger.info("=" * 80)
        logger.info("INCREMENTAL DATA UPDATE")
        logger.info("=" * 80)
        logger.info("Settings:")
        logger.info("  - Single thread download only")
        logger.info(f"  - Minimum {delay} seconds between requests")
        logger.info(f"  - Data type: {data_type}")
        logger.info(f"  - Shanghai only: {shanghai_only}")
        logger.info(f"  - Date range: {start} to {end}")
        if limit_nums:
            logger.info(f"  - Limited to {limit_nums} stocks (testing mode)")
        logger.info("  - Incremental update: ENABLED")
        logger.info("=" * 80)

        collector = collector_class(
            self.source_dir,
            max_workers=1,  # Force single worker
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            interval=self.interval,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            adjust=adjust,
            data_type=data_type,
            shanghai_only=shanghai_only,
        )
        
        collector.collector_data(incremental=True)

    def query_data(self, source_dir=None, symbol=None, start_date=None, end_date=None, limit=100):
        """Query stock data from DuckDB database
        
        Parameters
        ----------
        source_dir: str, optional
            source directory containing the DuckDB database
        symbol: str, optional
            specific stock symbol to query (e.g., '600000')
        start_date: str, optional
            start date for query (YYYY-MM-DD format)
        end_date: str, optional 
            end date for query (YYYY-MM-DD format)
        limit: int, optional
            maximum number of records to return, default 100
            
        Examples
        --------
            # Query all data (limited to 100 records)
            $ python collector_standalone.py query_data
            
            # Query specific stock
            $ python collector_standalone.py query_data --symbol 600000
            
            # Query date range
            $ python collector_standalone.py query_data --start_date 2024-01-01 --end_date 2024-01-31
            
            # Query specific stock and date range
            $ python collector_standalone.py query_data --symbol 600000 --start_date 2024-01-01 --end_date 2024-01-31 --limit 50
        """
        if source_dir is None:
            source_dir = CUR_DIR / "source"
            
        db_path = source_dir / "shanghai_stock_data.duckdb"
        
        if not db_path.exists():
            logger.error(f"DuckDB database not found at: {db_path}")
            logger.info("Please run download_data or update_data first to create the database.")
            return
            
        logger.info("=" * 80)
        logger.info("QUERYING STOCK DATA FROM DUCKDB")
        logger.info("=" * 80)
        
        try:
            conn = duckdb.connect(str(db_path))
            
            # Build query
            where_conditions = []
            params = []
            
            if symbol:
                where_conditions.append("symbol = ?")
                params.append(symbol)
                
            if start_date:
                where_conditions.append("date >= ?")
                params.append(start_date)
                
            if end_date:
                where_conditions.append("date <= ?") 
                params.append(end_date)
                
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            query = f"""
                SELECT symbol, date, open, high, low, close, volume, change_percent
                FROM stock_data 
                {where_clause}
                ORDER BY symbol, date DESC
                LIMIT {limit}
            """
            
            logger.info(f"Query: {query}")
            logger.info(f"Parameters: {params}")
            logger.info("-" * 80)
            
            result = conn.execute(query, params).fetchall()
            
            if result:
                # Print header
                print(f"{'Symbol':<8} {'Date':<12} {'Open':<8} {'High':<8} {'Low':<8} {'Close':<8} {'Volume':<12} {'Change%':<8}")
                print("-" * 80)
                
                # Print data
                for row in result:
                    symbol, date, open_price, high, low, close, volume, change = row
                    volume_str = f"{volume:,}" if volume else "N/A"
                    change_str = f"{change:.2f}%" if change else "N/A"
                    print(f"{symbol:<8} {date:<12} {open_price:<8.2f} {high:<8.2f} {low:<8.2f} {close:<8.2f} {volume_str:<12} {change_str:<8}")
                
                logger.info(f"\nFound {len(result)} records")
            else:
                logger.info("No data found matching the criteria")
                
            # Show database statistics
            stats_result = conn.execute("""
                SELECT 
                    COUNT(DISTINCT symbol) as total_stocks,
                    COUNT(*) as total_records,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date
                FROM stock_data
            """).fetchone()
            
            if stats_result:
                logger.info(f"\nDatabase Statistics:")
                logger.info(f"  Total stocks: {stats_result[0]:,}")
                logger.info(f"  Total records: {stats_result[1]:,}")
                logger.info(f"  Date range: {stats_result[2]} to {stats_result[3]}")
            
            conn.close()
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error querying database: {e}")

    def refresh_cache(self, source_dir=None):
        """Manually refresh the Shanghai stock symbols cache"""
        if source_dir is None:
            source_dir = CUR_DIR / "source"
        
        logger.info("=" * 80)
        logger.info("SHANGHAI STOCK SYMBOLS CACHE REFRESH")
        logger.info("=" * 80)
        
        # Create temporary collector to access cache methods
        collector_class = AkShareCollectorCN1d
        temp_collector = collector_class(
            save_dir=source_dir,
            data_type="stock",
            shanghai_only=True
        )
        
        try:
            # Force cache refresh
            conn = temp_collector._init_stock_symbols_cache()
            symbols = temp_collector._cache_shanghai_stocks(conn)
            conn.close()
            
            logger.info(f"Successfully refreshed cache with {len(symbols)} symbols")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Cache refresh failed: {e}")
    
    def clean_cache(self, source_dir=None):
        """Clean cache by removing delisted/suspended stocks from existing cache"""
        if source_dir is None:
            source_dir = CUR_DIR / "source"
            
        logger.info("=" * 80)
        logger.info("SHANGHAI STOCK SYMBOLS CACHE CLEANING")
        logger.info("=" * 80)
        
        # Create temporary collector to access cache methods
        collector_class = AkShareCollectorCN1d
        temp_collector = collector_class(
            save_dir=source_dir,
            data_type="stock",
            shanghai_only=True
        )
        
        try:
            conn = temp_collector._init_stock_symbols_cache()
            
            # Check if cache exists
            try:
                result = conn.execute("SELECT COUNT(*) FROM shanghai_stocks").fetchone()
                original_count = result[0]
                logger.info(f"Original cache contains {original_count} stocks")
                
                if original_count == 0:
                    logger.info("Cache is empty, nothing to clean")
                    conn.close()
                    return
                
            except Exception:
                logger.info("No existing cache found")
                conn.close()
                return
            
            # Get fresh stock list to determine active stocks
            logger.info("Fetching current market data to identify active stocks...")
            stock_list = ak.stock_zh_a_spot_em()
            shanghai_stocks = stock_list[
                stock_list['代码'].astype(str).str.startswith('6')
            ].copy()
            
            # Filter active stocks
            active_symbols = temp_collector._filter_active_stocks(shanghai_stocks)
            
            # Remove inactive stocks from cache
            placeholders = ','.join(['?' for _ in active_symbols])
            conn.execute(f"""
                DELETE FROM shanghai_stocks 
                WHERE symbol NOT IN ({placeholders})
            """, active_symbols)
            
            # Update status for remaining stocks
            conn.execute("""
                ALTER TABLE shanghai_stocks 
                ADD COLUMN IF NOT EXISTS status VARCHAR DEFAULT 'active'
            """)
            
            conn.execute("UPDATE shanghai_stocks SET status = 'active'")
            
            # Get final count
            result = conn.execute("SELECT COUNT(*) FROM shanghai_stocks").fetchone()
            final_count = result[0]
            
            cleaned_count = original_count - final_count
            logger.info(f"Cache cleaned: {original_count} -> {final_count} stocks")
            logger.info(f"Removed {cleaned_count} inactive stocks")
            
            # Show breakdown
            prefixes = ['600', '601', '603', '605', '688', '689']
            for prefix in prefixes:
                count = conn.execute(f"""
                    SELECT COUNT(*) FROM shanghai_stocks 
                    WHERE symbol LIKE '{prefix}%'
                """).fetchone()[0]
                if count > 0:
                    logger.info(f"  {prefix}xxx stocks: {count}")
            
            conn.close()
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")

    def show_cache_info(self, source_dir=None):
        """Show information about the current cache"""
        if source_dir is None:
            source_dir = CUR_DIR / "source"
            
        logger.info("=" * 80)
        logger.info("SHANGHAI STOCK SYMBOLS CACHE INFO")
        logger.info("=" * 80)
        
        # Create temporary collector to access cache methods
        collector_class = AkShareCollectorCN1d
        temp_collector = collector_class(
            save_dir=source_dir,
            data_type="stock",
            shanghai_only=True
        )
        
        try:
            conn = temp_collector._init_stock_symbols_cache()
            
            # Check if cache exists
            try:
                result = conn.execute("""
                    SELECT 
                        COUNT(*) as total_symbols,
                        MIN(cached_date) as first_cached,
                        MAX(cached_date) as last_updated
                    FROM shanghai_stocks
                """).fetchone()
                
                logger.info(f"Total cached symbols: {result[0]}")
                logger.info(f"First cached: {result[1]}")
                logger.info(f"Last updated: {result[2]}")
                
                # Show status breakdown
                status_result = conn.execute("""
                    SELECT status, COUNT(*) as count 
                    FROM shanghai_stocks 
                    GROUP BY status 
                    ORDER BY count DESC
                """).fetchall()
                
                if status_result:
                    logger.info("Status breakdown:")
                    for status, count in status_result:
                        status_name = status or "active"
                        logger.info(f"  {status_name}: {count}")
                
                # Show breakdown by prefix
                prefixes = ['600', '601', '603', '605', '688', '689']
                for prefix in prefixes:
                    count = conn.execute(f"""
                        SELECT COUNT(*) FROM shanghai_stocks 
                        WHERE symbol LIKE '{prefix}%'
                    """).fetchone()[0]
                    if count > 0:
                        logger.info(f"  {prefix}xxx stocks: {count}")
                
                # Show recent failures
                failure_result = conn.execute("""
                    SELECT symbol, failure_reason, retry_count, last_attempt_date 
                    FROM shanghai_stocks 
                    WHERE status = 'NA' 
                    ORDER BY last_attempt_date DESC 
                    LIMIT 10
                """).fetchall()
                
                if failure_result:
                    logger.info("Recent failures (top 10):")
                    for symbol, reason, retry_count, last_attempt in failure_result:
                        logger.info(f"  {symbol}: {reason} (retries: {retry_count}, last: {last_attempt})")
                        
            except Exception:
                logger.info("No cache found")
                
            conn.close()
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error checking cache: {e}")

    def reset_failed_stocks(self, source_dir=None, symbols=None):
        """Reset failed stocks status back to active for retry
        
        Parameters
        ----------
        source_dir: str, optional
            source directory containing the cache database
        symbols: str or list, optional
            specific symbols to reset (comma-separated string or list), if None resets all NA status stocks
        """
        if source_dir is None:
            source_dir = CUR_DIR / "source"
            
        logger.info("=" * 80)
        logger.info("RESET FAILED STOCKS STATUS")
        logger.info("=" * 80)
        
        # Create temporary collector to access cache methods
        collector_class = AkShareCollectorCN1d
        temp_collector = collector_class(
            save_dir=source_dir,
            data_type="stock",
            shanghai_only=True
        )
        
        try:
            conn = temp_collector._init_stock_symbols_cache()
            
            if symbols:
                # Handle both string and list input
                if isinstance(symbols, str):
                    symbols_list = [s.strip() for s in symbols.split(',')]
                else:
                    symbols_list = symbols
                
                # Reset specific symbols
                placeholders = ','.join(['?' for _ in symbols_list])
                conn.execute(f"""
                    UPDATE shanghai_stocks 
                    SET status = 'active', retry_count = 0, failure_reason = NULL, last_attempt_date = NULL
                    WHERE symbol IN ({placeholders})
                """, symbols_list)
                logger.info(f"Reset status for {len(symbols_list)} specific stocks: {symbols_list}")
            else:
                # Reset all NA status stocks
                result = conn.execute("""
                    SELECT COUNT(*) FROM shanghai_stocks WHERE status = 'NA'
                """).fetchone()
                na_count = result[0]
                
                conn.execute("""
                    UPDATE shanghai_stocks 
                    SET status = 'active', retry_count = 0, failure_reason = NULL, last_attempt_date = NULL
                    WHERE status = 'NA'
                """)
                logger.info(f"Reset status for {na_count} stocks from NA to active")
            
            conn.close()
            logger.info("Reset completed successfully")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error resetting failed stocks: {e}")


if __name__ == "__main__":
    fire.Fire(Run)
