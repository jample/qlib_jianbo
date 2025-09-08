#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AkShare Data Updater for 2025

This script provides comprehensive data collection and updating capabilities using AkShare.
It includes automated data collection, validation, and integration with various data sources.

Features:
- Stock data collection and updates
- Index data monitoring
- Fund data tracking
- Economic indicators
- Real-time data feeds
- Data validation and quality checks
- Automated scheduling capabilities

Usage:
    python data_updater.py --mode daily --symbols 000001,000002,600000
    python data_updater.py --mode realtime --update-interval 60
    python data_updater.py --mode backfill --start-date 20240101 --end-date 20241231
"""

import argparse
import json
import logging
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import akshare as ak
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('akshare_updater.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AkShareDataUpdater:
    """Comprehensive AkShare data updater for 2025"""
    
    def __init__(self, data_dir: str = "./akshare_data", config_file: str = None):
        """
        Initialize the data updater
        
        Parameters
        ----------
        data_dir : str
            Directory to store collected data
        config_file : str, optional
            Configuration file path
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "stocks").mkdir(exist_ok=True)
        (self.data_dir / "indices").mkdir(exist_ok=True)
        (self.data_dir / "funds").mkdir(exist_ok=True)
        (self.data_dir / "economic").mkdir(exist_ok=True)
        (self.data_dir / "realtime").mkdir(exist_ok=True)
        
        self.config = self._load_config(config_file)
        
        logger.info(f"AkShare Data Updater initialized")
        logger.info(f"AkShare version: {ak.__version__}")
        logger.info(f"Data directory: {self.data_dir}")
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "default_symbols": ["000001", "000002", "600000", "600036", "000858"],
            "indices": {
                "CSI300": "sh000300",
                "CSI500": "sh000905", 
                "SSE50": "sh000016",
                "ChiNext": "sz399006"
            },
            "funds": ["015198", "110022", "161725"],
            "delay": 1.0,
            "max_workers": 3,
            "adjust_type": "qfq",
            "data_validation": True
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Error loading config file: {e}, using defaults")
        
        return default_config
    
    def get_stock_list(self, limit: Optional[int] = None) -> List[str]:
        """Get list of A-share stocks"""
        try:
            stock_list = ak.stock_info_a_code_name()
            symbols = stock_list['code'].tolist()
            
            if limit:
                symbols = symbols[:limit]
                
            logger.info(f"Retrieved {len(symbols)} stock symbols")
            return symbols
        except Exception as e:
            logger.error(f"Error getting stock list: {e}")
            return self.config["default_symbols"]
    
    def collect_stock_data(self, symbols: List[str], start_date: str, end_date: str, 
                          interval: str = "daily") -> Dict[str, pd.DataFrame]:
        """
        Collect stock data for multiple symbols
        
        Parameters
        ----------
        symbols : List[str]
            List of stock symbols
        start_date : str
            Start date in YYYYMMDD format
        end_date : str
            End date in YYYYMMDD format
        interval : str
            Data interval ('daily' or '1min')
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of symbol -> DataFrame
        """
        stock_data = {}
        
        def collect_single_stock(symbol):
            try:
                time.sleep(self.config["delay"])  # Rate limiting
                
                if interval == "daily":
                    data = ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust=self.config["adjust_type"]
                    )
                elif interval == "1min":
                    data = ak.stock_zh_a_hist_min_em(
                        symbol=symbol,
                        period="1",
                        adjust=self.config["adjust_type"]
                    )
                else:
                    raise ValueError(f"Unsupported interval: {interval}")
                
                if not data.empty:
                    data['symbol'] = symbol
                    logger.info(f"Collected {len(data)} records for {symbol}")
                    return symbol, data
                else:
                    logger.warning(f"No data for {symbol}")
                    return symbol, None
                    
            except Exception as e:
                logger.error(f"Error collecting {symbol}: {e}")
                return symbol, None
        
        # Use ThreadPoolExecutor for concurrent collection
        with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            future_to_symbol = {executor.submit(collect_single_stock, symbol): symbol 
                              for symbol in symbols}
            
            for future in as_completed(future_to_symbol):
                symbol, data = future.result()
                if data is not None:
                    stock_data[symbol] = data
        
        logger.info(f"Successfully collected data for {len(stock_data)} symbols")
        return stock_data
    
    def collect_index_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Collect major index data"""
        index_data = {}
        
        for name, code in self.config["indices"].items():
            try:
                time.sleep(self.config["delay"])
                data = ak.index_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not data.empty:
                    data['index_name'] = name
                    data['index_code'] = code
                    index_data[name] = data
                    logger.info(f"Collected {len(data)} records for {name}")
                else:
                    logger.warning(f"No data for {name}")
                    
            except Exception as e:
                logger.error(f"Error collecting {name}: {e}")
        
        return index_data
    
    def collect_fund_data(self, fund_codes: List[str] = None) -> Dict[str, Dict]:
        """Collect fund data including net values and holdings"""
        if fund_codes is None:
            fund_codes = self.config["funds"]
        
        fund_data = {}
        
        for fund_code in fund_codes:
            try:
                time.sleep(self.config["delay"])
                
                # Get fund net value history
                net_value = ak.fund_open_fund_info_em(
                    fund=fund_code, 
                    indicator="累计净值走势"
                )
                
                # Get fund holdings for current year
                current_year = str(datetime.now().year)
                holdings = ak.fund_portfolio_hold_em(
                    symbol=fund_code, 
                    date=current_year
                )
                
                fund_data[fund_code] = {
                    'net_value': net_value,
                    'holdings': holdings
                }
                
                logger.info(f"Collected fund data for {fund_code}")
                
            except Exception as e:
                logger.error(f"Error collecting fund {fund_code}: {e}")
        
        return fund_data
    
    def collect_economic_indicators(self) -> Dict[str, pd.DataFrame]:
        """Collect economic indicators"""
        economic_data = {}
        
        indicators = {
            'gdp': lambda: ak.macro_china_gdp(),
            'cpi': lambda: ak.macro_china_cpi(),
            'money_supply': lambda: ak.macro_china_money_supply(),
            'lpr': lambda: ak.macro_china_lpr()
        }
        
        for name, func in indicators.items():
            try:
                time.sleep(self.config["delay"])
                data = func()
                if not data.empty:
                    economic_data[name] = data
                    logger.info(f"Collected {name} data: {len(data)} records")
            except Exception as e:
                logger.error(f"Error collecting {name}: {e}")
        
        return economic_data
    
    def get_realtime_data(self) -> pd.DataFrame:
        """Get real-time market data"""
        try:
            realtime_data = ak.stock_zh_a_spot_em()
            logger.info(f"Retrieved real-time data for {len(realtime_data)} stocks")
            return realtime_data
        except Exception as e:
            logger.error(f"Error getting real-time data: {e}")
            return pd.DataFrame()
    
    def validate_data(self, data: pd.DataFrame, data_type: str = "stock") -> bool:
        """Validate data quality"""
        if not self.config["data_validation"]:
            return True
        
        if data.empty:
            logger.warning(f"Empty {data_type} data")
            return False
        
        # Check for required columns based on data type
        if data_type == "stock":
            required_cols = ['日期', '开盘', '收盘', '最高', '最低', '成交量']
        elif data_type == "index":
            required_cols = ['日期', '开盘', '收盘', '最高', '最低']
        else:
            return True  # Skip validation for other types
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing columns in {data_type} data: {missing_cols}")
            return False
        
        # Check for data anomalies
        numeric_cols = ['开盘', '收盘', '最高', '最低', '成交量']
        for col in numeric_cols:
            if col in data.columns:
                if (data[col] <= 0).any():
                    logger.warning(f"Found non-positive values in {col}")
                    return False
        
        logger.debug(f"{data_type} data validation passed")
        return True
    
    def save_data(self, data_dict: Dict, data_type: str, date_suffix: str = None):
        """Save collected data to files"""
        if date_suffix is None:
            date_suffix = datetime.now().strftime("%Y%m%d")
        
        save_dir = self.data_dir / data_type
        
        for key, data in data_dict.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                filename = f"{key}_{date_suffix}.csv"
                filepath = save_dir / filename
                
                if self.validate_data(data, data_type):
                    data.to_csv(filepath, index=False, encoding='utf-8-sig')
                    logger.info(f"Saved {filename}")
                else:
                    logger.warning(f"Data validation failed for {key}, not saved")
    
    def daily_update(self, symbols: List[str] = None, days_back: int = 5):
        """Perform daily data update"""
        logger.info("Starting daily update...")
        
        if symbols is None:
            symbols = self.config["default_symbols"]
        
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
        
        # Collect stock data
        stock_data = self.collect_stock_data(symbols, start_date, end_date)
        self.save_data(stock_data, "stocks")
        
        # Collect index data
        index_data = self.collect_index_data(start_date, end_date)
        self.save_data(index_data, "indices")
        
        # Collect fund data
        fund_data = self.collect_fund_data()
        for fund_code, fund_info in fund_data.items():
            for data_type, data in fund_info.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    filename = f"{fund_code}_{data_type}_{datetime.now().strftime('%Y%m%d')}.csv"
                    filepath = self.data_dir / "funds" / filename
                    data.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        logger.info("Daily update completed")
    
    def backfill_data(self, symbols: List[str], start_date: str, end_date: str):
        """Backfill historical data"""
        logger.info(f"Starting backfill from {start_date} to {end_date}")
        
        # Split date range into chunks to avoid overwhelming the API
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        chunk_size = timedelta(days=90)  # 3-month chunks
        current_start = start_dt
        
        while current_start < end_dt:
            current_end = min(current_start + chunk_size, end_dt)
            
            chunk_start = current_start.strftime('%Y%m%d')
            chunk_end = current_end.strftime('%Y%m%d')
            
            logger.info(f"Processing chunk: {chunk_start} to {chunk_end}")
            
            stock_data = self.collect_stock_data(symbols, chunk_start, chunk_end)
            self.save_data(stock_data, "stocks", f"backfill_{chunk_start}_{chunk_end}")
            
            current_start = current_end + timedelta(days=1)
        
        logger.info("Backfill completed")


def main():
    parser = argparse.ArgumentParser(description="AkShare Data Updater for 2025")
    parser.add_argument("--mode", choices=["daily", "backfill", "realtime"], 
                       default="daily", help="Update mode")
    parser.add_argument("--symbols", type=str, help="Comma-separated stock symbols")
    parser.add_argument("--start-date", type=str, help="Start date (YYYYMMDD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYYMMDD)")
    parser.add_argument("--data-dir", type=str, default="./akshare_data", 
                       help="Data directory")
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--limit", type=int, help="Limit number of symbols")
    
    args = parser.parse_args()
    
    # Initialize updater
    updater = AkShareDataUpdater(data_dir=args.data_dir, config_file=args.config)
    
    # Parse symbols
    if args.symbols:
        symbols = args.symbols.split(',')
    else:
        symbols = updater.get_stock_list(limit=args.limit)
    
    # Execute based on mode
    if args.mode == "daily":
        updater.daily_update(symbols)
    elif args.mode == "backfill":
        if not args.start_date or not args.end_date:
            logger.error("Backfill mode requires --start-date and --end-date")
            return
        updater.backfill_data(symbols, args.start_date, args.end_date)
    elif args.mode == "realtime":
        logger.info("Real-time mode - collecting current market data")
        realtime_data = updater.get_realtime_data()
        if not realtime_data.empty:
            filepath = updater.data_dir / "realtime" / f"realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            realtime_data.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info(f"Saved real-time data: {filepath}")


if __name__ == "__main__":
    main()
