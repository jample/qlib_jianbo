# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import sys
import datetime
from abc import ABC
from pathlib import Path

import fire
import pandas as pd
from loguru import logger
from dateutil.tz import tzlocal

try:
    import akshare as ak
except ImportError:
    logger.error("akshare is not installed. Please install it using: pip install akshare")
    sys.exit(1)

# Define region constant locally to avoid qlib import issues
REGION_CN = "CN"

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))
from data_collector.base import BaseCollector, BaseNormalize, BaseRun
from data_collector.utils import get_calendar_list


class AkShareCollector(BaseCollector):
    def __init__(
        self,
        save_dir: [str, Path],
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
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        limit_nums: int
            using for debug, by default None
        adjust: str
            adjustment type, qfq/hfq/empty, default qfq
        data_type: str
            data type to collect, value from ["fund", "stock"], default "fund"
        """
        super(AkShareCollector, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )

        self.adjust_type = adjust
        self.data_type = data_type.lower()
        if self.data_type not in ["fund", "stock"]:
            raise ValueError(f"data_type must be 'fund' or 'stock', got '{data_type}'")

        logger.info(f"AkShare collector initialized for {self.data_type} data")
        self.init_datetime()

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
    def convert_datetime(dt: [pd.Timestamp, datetime.date, str], timezone):
        try:
            dt = pd.Timestamp(dt, tz=timezone).timestamp()
            dt = pd.Timestamp(dt, tz=tzlocal(), unit="s")
        except ValueError as e:
            pass
        return dt

    @property
    @abc.abstractmethod
    def _timezone(self):
        raise NotImplementedError("rewrite get_timezone")

    @staticmethod
    def get_data_from_remote(symbol, interval, start_datetime, end_datetime, adjust="qfq", data_type="fund"):
        error_msg = f"{symbol}-{interval}-{start_datetime}-{end_datetime}-{data_type}"

        try:
            if data_type == "fund":
                return AkShareCollector._get_fund_data(symbol, start_datetime, end_datetime)
            elif data_type == "stock":
                return AkShareCollector._get_stock_data(symbol, interval, start_datetime, end_datetime, adjust)
            else:
                raise ValueError(f"Unsupported data_type: {data_type}")

        except Exception as e:
            logger.warning(f"{error_msg}:{e}")
            return pd.DataFrame()

    @staticmethod
    def _get_fund_data(symbol, start_datetime, end_datetime):
        """Get fund data from AkShare"""
        try:
            # Get fund net value history
            df = ak.fund_open_fund_info_em(symbol=symbol, indicator="累计净值走势")

            if df.empty:
                logger.warning(f"No fund data retrieved for {symbol}")
                return pd.DataFrame()

            # Normalize fund data columns
            fund_column_mapping = {
                '净值日期': 'date',
                '单位净值': 'close',
                '累计净值': 'cumulative_nav',
                '日增长率': 'change'
            }

            df = df.rename(columns=fund_column_mapping)

            # Ensure date column is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

                # Filter by date range
                df = df[(df['date'] >= start_datetime) & (df['date'] <= end_datetime)]

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

            return df.reset_index(drop=True)

        except Exception as e:
            logger.warning(f"Error getting fund data for {symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    def _get_stock_data(symbol, interval, start_datetime, end_datetime, adjust="qfq"):
        """Get stock data from AkShare"""
        try:
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
                logger.warning(f"No stock data retrieved for {symbol}")
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

            return df.reset_index(drop=True)

        except Exception as e:
            logger.warning(f"Error getting stock data for {symbol}: {e}")
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

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize Chinese column names to English"""
        column_mapping = {
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
        
        df = df.rename(columns=column_mapping)
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Select standard columns for qlib compatibility
        standard_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'money']
        available_cols = [col for col in standard_cols if col in df.columns]
        
        # Add change column if available
        if 'change' in df.columns:
            available_cols.append('change')
            

class AkShareCollectorCN(AkShareCollector, ABC):
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


class AkShareNormalize(BaseNormalize):
    DAILY_FORMAT = "%Y-%m-%d"

    @staticmethod
    def normalize_akshare(
        df: pd.DataFrame,
        calendar_list: list = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
    ):
        if df.empty:
            return df
        df = df.copy()
        df.set_index(date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="first")]
        if calendar_list is not None:
            df = df.reindex(
                pd.DataFrame(index=calendar_list)
                .loc[
                    pd.Timestamp(df.index.min()).date() : pd.Timestamp(df.index.max()).date()
                    + pd.Timedelta(hours=23, minutes=59)
                ]
                .index
            )
        df.sort_index(inplace=True)

        df.index.names = [date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize
        df = self.normalize_akshare(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        return df


class AkShareNormalize1d(AkShareNormalize):
    pass


class AkShareNormalizeCN:
    def _get_calendar_list(self):
        return get_calendar_list("ALL")


class AkShareNormalizeCN1d(AkShareNormalizeCN, AkShareNormalize1d):
    pass


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=4, interval="1d", region=REGION_CN):
        """
        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
        normalize_dir: str
            Directory for normalize data, default "Path(__file__).parent/normalize"
        max_workers: int
            Concurrent number, default is 4
        interval: str
            freq, value from [1min, 1d], default 1d
        region: str
            region, value from ["CN"], default "CN"
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"AkShareCollector{self.region.upper()}{self.interval}"

    @property
    def normalize_class_name(self):
        return f"AkShareNormalize{self.region.upper()}{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

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
        interval: str
            freq, value from [1min, 1d], default 1d
        start: str
            start datetime, default "2000-01-01"
        end: str
            end datetime, default ``pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))``
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        limit_nums: int
            using for debug, by default None
        adjust: str
            adjustment type, qfq/hfq/empty, default qfq
        data_type: str
            data type to collect, value from ["fund", "stock"], default "fund"

        Examples
        ---------
            # get fund data (default)
            $ python collector.py download_data --start 2020-11-01 --end 2020-11-10 --delay 0.1 --data_type fund

            # get stock data
            $ python collector.py download_data --start 2020-11-01 --end 2020-11-10 --delay 0.1 --data_type stock
        """
        # Get the collector class
        collector_class = getattr(self._cur_module, self.collector_class_name)

        collector_class(
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
        ).collector_data()

    def normalize_data(self, date_field_name: str = "date", symbol_field_name: str = "symbol"):
        """normalize data

        Parameters
        ----------
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol

        Examples
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/akshare_data/source/cn_data --normalize_dir ~/.qlib/akshare_data/source/cn_1d_nor --region CN --interval 1d --date_field_name date
        """
        super(Run, self).normalize_data(date_field_name, symbol_field_name)


if __name__ == "__main__":
    fire.Fire(Run)
