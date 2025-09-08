# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
AkShare Data Collector following Fund Collector Pattern

This collector follows the same design pattern as the fund collector but works
independently without complex qlib dependencies that cause import issues.
"""

import abc
import sys
import datetime
import time
from abc import ABC
from pathlib import Path
from typing import List, Optional, Union

import fire
import pandas as pd
from loguru import logger
from dateutil.tz import tzlocal

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
    DEFAULT_START_DATETIME_1MIN = pd.Timestamp(datetime.datetime.now() - pd.Timedelta(days=5 * 6 - 1)).date()
    DEFAULT_END_DATETIME_1D = pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1)).date()
    DEFAULT_END_DATETIME_1MIN = DEFAULT_END_DATETIME_1D

    INTERVAL_1min = "1min"
    INTERVAL_1d = "1d"
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
        adjust="qfq",
        data_type="fund",  # Default to fund data
    ):
        """
        Parameters
        ----------
        save_dir: str
            akshare save dir
        max_workers: int
            workers, default 4
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0
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
        """
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.delay = delay
        self.max_workers = max_workers
        self.max_collector_count = max_collector_count
        self.interval = interval
        self.check_data_length = max(int(check_data_length) if check_data_length is not None else 0, 0)
        self.adjust_type = adjust
        self.data_type = data_type.lower()

        if self.data_type not in ["fund", "stock"]:
            raise ValueError(f"data_type must be 'fund' or 'stock', got '{data_type}'")

        logger.info(f"AkShare collector initialized for {self.data_type} data")
        
        # Normalize datetime
        self.start_datetime = self.normalize_start_datetime(start)
        self.end_datetime = self.normalize_end_datetime(end)
        
        # Initialize datetime
        self.init_datetime()
        
        # Get instrument list
        self.instrument_list = sorted(set(self.get_instrument_list()))
        
        if limit_nums is not None:
            try:
                self.instrument_list = self.instrument_list[:int(limit_nums)]
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
        """Sleep for delay time"""
        if self.delay > 0:
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

    def collector_data(self):
        """Main data collection method"""
        logger.info("Starting data collection...")
        
        if not self.instrument_list:
            logger.error("No symbols to collect")
            return
        
        logger.info(f"Collecting data for {len(self.instrument_list)} symbols")
        
        # Collect data for each symbol
        success_count = 0
        for i, symbol in enumerate(self.instrument_list):
            try:
                logger.info(f"Processing {symbol} ({i+1}/{len(self.instrument_list)})")
                
                # Get data
                df = self.get_data(symbol, self.interval, self.start_datetime, self.end_datetime)
                
                if not df.empty:
                    # Save data
                    self.save_instrument(symbol, df)
                    success_count += 1
                else:
                    logger.warning(f"No data retrieved for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        logger.info(f"Data collection completed. Successfully processed {success_count}/{len(self.instrument_list)} symbols")


class AkShareCollectorCN(AkShareCollector):
    def get_instrument_list(self):
        if self.data_type == "fund":
            return self._get_fund_list()
        elif self.data_type == "stock":
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

    def normalize_symbol(self, symbol):
        return symbol

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
        max_collector_count=2,
        delay=0,
        start=None,
        end=None,
        check_data_length: int = None,
        limit_nums=None,
        adjust="qfq",
        data_type="fund",  # Default to fund data
    ):
        """download data from Internet

        Parameters
        ----------
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0
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

        Examples
        ---------
            # get fund data (default)
            $ python collector_standalone.py download_data --start 2025-01-01 --end 2025-06-01 --limit_nums 10 --delay 1 --data_type fund

            # get stock data
            $ python collector_standalone.py download_data --start 2025-01-01 --end 2025-06-01 --limit_nums 10 --delay 1 --data_type stock
        """
        collector_class = self.get_collector_class()
        if collector_class is None:
            logger.error(f"Collector class not found for {self.region.upper()}{self.interval}")
            return

        collector = collector_class(
            self.source_dir,
            max_workers=self.max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            interval=self.interval,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            adjust=adjust,
            data_type=data_type,
        )
        
        collector.collector_data()


if __name__ == "__main__":
    fire.Fire(Run)
