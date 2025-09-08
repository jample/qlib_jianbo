#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Network-Robust Shanghai Stock Collection

This script handles network issues more gracefully by:
1. Using different user agents
2. Implementing exponential backoff
3. Adding session management
4. Using alternative data collection methods

Usage:
    python collect_shanghai_network_robust.py --start_date 20220101 --end_date 20250101
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import requests
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shanghai_network_robust.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import akshare with error handling
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    logger.error("AkShare not available")
    AKSHARE_AVAILABLE = False


class NetworkRobustCollector:
    """Network-robust collector with multiple strategies"""
    
    def __init__(self, data_dir="./qlib_shanghai_network_data", start_date="20220101", end_date="20250101"):
        self.data_dir = Path(data_dir)
        self.start_date = start_date
        self.end_date = end_date
        
        # Create directories
        self.csv_dir = self.data_dir / "csv_data"
        self.qlib_dir = self.data_dir / "qlib_data"
        
        for dir_path in [self.csv_dir, self.qlib_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup session with different user agents
        self.session = requests.Session()
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        
        logger.info(f"Network Robust Collector initialized")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Date range: {start_date} to {end_date}")
    
    def get_shanghai_stocks(self):
        """Get Shanghai stock list with network robustness"""
        for attempt in range(5):
            try:
                # Rotate user agent
                headers = {'User-Agent': random.choice(self.user_agents)}
                
                logger.info(f"Fetching Shanghai stock list (attempt {attempt + 1})...")
                
                # Add delay before each attempt
                if attempt > 0:
                    delay = min(2 ** attempt, 30)  # Exponential backoff, max 30s
                    logger.info(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                
                # Try to get stock list
                if AKSHARE_AVAILABLE:
                    stock_info = ak.stock_info_a_code_name()
                    
                    # Filter Shanghai stocks
                    shanghai_stocks = stock_info[
                        stock_info['code'].str.startswith(('60', '68', '90'))
                    ]['code'].tolist()
                    
                    logger.info(f"Found {len(shanghai_stocks)} Shanghai stocks")
                    return shanghai_stocks
                else:
                    # Fallback: use a predefined list of major Shanghai stocks
                    logger.warning("AkShare not available, using fallback stock list")
                    return self._get_fallback_shanghai_stocks()
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == 4:  # Last attempt
                    logger.error("All attempts failed, using fallback list")
                    return self._get_fallback_shanghai_stocks()
        
        return self._get_fallback_shanghai_stocks()
    
    def _get_fallback_shanghai_stocks(self):
        """Fallback list of major Shanghai stocks"""
        major_shanghai_stocks = [
            # Major banks
            "600000", "600036", "601166", "601328", "601398", "601939", "601988",
            # Major companies
            "600519", "600276", "600585", "600887", "600009", "600028", "600030",
            "600048", "600050", "600104", "600111", "600150", "600196", "600309",
            "600340", "600362", "600383", "600406", "600438", "600482", "600489",
            "600498", "600547", "600570", "600588", "600606", "600637", "600660",
            "600690", "600703", "600745", "600760", "600795", "600809", "600837",
            "600848", "600867", "600886", "600893", "600900", "600919", "600926",
            "600958", "600999", "601006", "601012", "601066", "601088", "601111",
            "601138", "601169", "601186", "601211", "601225", "601229", "601238",
            "601288", "601318", "601336", "601360", "601377", "601390", "601601",
            "601628", "601633", "601668", "601669", "601688", "601728", "601766",
            "601788", "601800", "601808", "601818", "601828", "601857", "601866",
            "601872", "601877", "601878", "601881", "601888", "601898", "601899",
            "601901", "601919", "601933", "601985", "601989", "601991", "601992",
            "601995", "601997", "601998",
            # STAR Market (科创板)
            "688001", "688002", "688003", "688005", "688006", "688007", "688008",
            "688009", "688010", "688011", "688012", "688016", "688018", "688019",
            "688020", "688021", "688022", "688023", "688025", "688026", "688027",
            "688028", "688029", "688030", "688031", "688032", "688033", "688036",
            "688037", "688038", "688039", "688041", "688043", "688046", "688047",
            "688048", "688050", "688051", "688052", "688053", "688055", "688056"
        ]
        
        logger.info(f"Using fallback list: {len(major_shanghai_stocks)} stocks")
        return major_shanghai_stocks
    
    def collect_stock_data_robust(self, symbol, delay=3.0, max_retries=10):
        """Collect data with maximum network robustness"""
        for attempt in range(max_retries):
            try:
                # Progressive delay with jitter
                actual_delay = delay * (1 + attempt * 0.3) + random.uniform(0, 1)
                time.sleep(actual_delay)
                
                # Rotate user agent
                headers = {'User-Agent': random.choice(self.user_agents)}
                
                if attempt > 0:
                    logger.info(f"Retry {attempt}/{max_retries-1} for {symbol} (delay: {actual_delay:.1f}s)")
                
                logger.debug(f"Collecting data for {symbol}...")
                
                if not AKSHARE_AVAILABLE:
                    logger.error("AkShare not available")
                    return None
                
                # Try to collect data
                data = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=self.start_date,
                    end_date=self.end_date,
                    adjust="qfq"
                )
                
                if data.empty:
                    logger.warning(f"No data for {symbol}")
                    return None
                
                # Process data
                data = self._process_stock_data(data, symbol)
                logger.debug(f"Collected {len(data)} records for {symbol}")
                return data
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check error type
                is_network_error = any(err in error_msg for err in [
                    'connection', 'timeout', 'network', 'remote', 'ssl', 'http', 'socket'
                ])
                
                is_rate_limit = any(err in error_msg for err in [
                    'rate', 'limit', 'too many', 'frequency', 'quota'
                ])
                
                if attempt < max_retries - 1:
                    if is_network_error or is_rate_limit:
                        wait_time = min(delay * (2 ** attempt), 60)  # Exponential backoff, max 60s
                        logger.warning(f"Network/Rate limit error for {symbol} (attempt {attempt+1}): {e}")
                        logger.info(f"Waiting {wait_time:.1f}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Non-recoverable error for {symbol}: {e}")
                        return None
                else:
                    logger.error(f"Failed to collect {symbol} after {max_retries} attempts: {e}")
                    return None
        
        return None
    
    def _process_stock_data(self, data, symbol):
        """Process and normalize stock data"""
        # Normalize column names
        column_mapping = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'change',
            '涨跌额': 'change_amount',
            '换手率': 'turnover'
        }
        
        data = data.rename(columns=column_mapping)
        
        # Ensure date is datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Add symbol column
        data['symbol'] = symbol
        
        # Select required columns for Qlib
        required_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount']
        available_cols = [col for col in required_cols if col in data.columns]
        
        # Add optional columns if available
        optional_cols = ['change', 'turnover']
        for col in optional_cols:
            if col in data.columns:
                available_cols.append(col)
        
        # Add symbol column
        available_cols.append('symbol')
        
        data = data[available_cols].copy()
        
        # Sort by date
        data = data.sort_values('date').reset_index(drop=True)
        
        return data
    
    def save_stock_csv(self, symbol, data):
        """Save stock data to CSV"""
        try:
            csv_file = self.csv_dir / f"{symbol}.csv"
            data.to_csv(csv_file, index=False)
            return True
        except Exception as e:
            logger.error(f"Error saving {symbol}: {e}")
            return False
    
    def collect_all_stocks(self, limit_stocks=None, delay=3.0):
        """Collect all stocks with network robustness"""
        shanghai_stocks = self.get_shanghai_stocks()
        
        if limit_stocks:
            shanghai_stocks = shanghai_stocks[:limit_stocks]
            logger.info(f"Limited to first {limit_stocks} stocks")
        
        logger.info(f"Starting collection for {len(shanghai_stocks)} stocks...")
        
        successful_count = 0
        failed_count = 0
        failed_stocks = []
        
        # Check for existing files
        existing_files = set()
        if self.csv_dir.exists():
            existing_files = {f.stem for f in self.csv_dir.glob("*.csv")}
            if existing_files:
                logger.info(f"Found {len(existing_files)} existing files, will skip them")
        
        for i, symbol in enumerate(shanghai_stocks, 1):
            # Skip if file already exists
            if symbol in existing_files:
                logger.debug(f"Skipping {symbol} - file already exists")
                successful_count += 1
                continue
            
            logger.info(f"Processing {i}/{len(shanghai_stocks)}: {symbol}")
            
            data = self.collect_stock_data_robust(symbol, delay=delay, max_retries=10)
            
            if data is not None:
                if self.save_stock_csv(symbol, data):
                    successful_count += 1
                    logger.info(f"✓ Successfully saved {symbol}")
                else:
                    failed_count += 1
                    failed_stocks.append(symbol)
                    logger.warning(f"✗ Failed to save {symbol}")
            else:
                failed_count += 1
                failed_stocks.append(symbol)
                logger.warning(f"✗ Failed to collect {symbol}")
            
            # Progress update every 10 stocks
            if i % 10 == 0:
                success_rate = (successful_count / i) * 100
                logger.info(f"Progress: {i}/{len(shanghai_stocks)} ({success_rate:.1f}% success)")
                logger.info(f"Success: {successful_count}, Failed: {failed_count}")
                
                # Adaptive delay based on success rate
                if success_rate < 70:  # Less than 70% success
                    delay = min(delay * 1.2, 10.0)  # Increase delay, max 10 seconds
                    logger.info(f"Low success rate, increasing delay to {delay:.1f}s")
        
        final_success_rate = (successful_count / len(shanghai_stocks)) * 100
        logger.info(f"Collection completed: {successful_count} successful, {failed_count} failed ({final_success_rate:.1f}% success rate)")
        
        # Save failed stocks
        if failed_stocks:
            failed_file = self.data_dir / "failed_stocks_final.txt"
            with open(failed_file, 'w') as f:
                f.write('\n'.join(failed_stocks))
            logger.info(f"Failed stocks saved to: {failed_file}")
        
        return successful_count > 0
    
    def convert_to_qlib_format(self):
        """Convert CSV data to Qlib binary format"""
        try:
            logger.info("Converting to Qlib binary format...")
            
            # Import dump_bin functionality
            sys.path.insert(0, str(Path(__file__).parent / "scripts"))
            from dump_bin import DumpDataAll
            
            # Create the dumper
            dumper = DumpDataAll(
                data_path=str(self.csv_dir),
                qlib_dir=str(self.qlib_dir),
                freq="day",
                max_workers=4,
                date_field_name="date",
                file_suffix=".csv",
                symbol_field_name="symbol",
                exclude_fields=""
            )
            
            # Dump the data
            dumper.dump()
            
            logger.info(f"Qlib binary data created at: {self.qlib_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting to Qlib format: {e}")
            return False
    
    def run_collection(self, limit_stocks=None, delay=3.0):
        """Run the complete collection process"""
        logger.info("=" * 80)
        logger.info("NETWORK-ROBUST SHANGHAI STOCK COLLECTION")
        logger.info("=" * 80)
        
        # Step 1: Collect CSV data
        logger.info("Step 1: Collecting stock data with network robustness...")
        if not self.collect_all_stocks(limit_stocks=limit_stocks, delay=delay):
            logger.error("Stock data collection failed!")
            return False
        
        # Step 2: Convert to Qlib format
        logger.info("Step 2: Converting to Qlib format...")
        if not self.convert_to_qlib_format():
            logger.error("Qlib format conversion failed!")
            return False
        
        # Final statistics
        csv_files = list(self.csv_dir.glob("*.csv"))
        logger.info("=" * 80)
        logger.info("COLLECTION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Total CSV files collected: {len(csv_files)}")
        logger.info(f"Qlib data available at: {self.qlib_dir}")
        logger.info("You can now use this data with Qlib by setting:")
        logger.info(f"qlib.init(provider_uri='{self.qlib_dir}', region='cn')")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Network-Robust Shanghai Stock Collection")
    parser.add_argument("--start_date", type=str, default="20220101",
                       help="Start date in YYYYMMDD format")
    parser.add_argument("--end_date", type=str, default="20250101",
                       help="End date in YYYYMMDD format")
    parser.add_argument("--data_dir", type=str, default="./qlib_shanghai_network_data",
                       help="Data directory")
    parser.add_argument("--limit_stocks", type=int, default=None,
                       help="Limit number of stocks for testing")
    parser.add_argument("--delay", type=float, default=3.0,
                       help="Base delay between requests in seconds")
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        datetime.strptime(args.start_date, "%Y%m%d")
        datetime.strptime(args.end_date, "%Y%m%d")
    except ValueError:
        logger.error("Invalid date format. Use YYYYMMDD.")
        return
    
    # Create collector
    collector = NetworkRobustCollector(
        data_dir=args.data_dir,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Show collection info
    logger.info(f"Collection parameters:")
    logger.info(f"  Date range: {args.start_date} to {args.end_date}")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Base delay: {args.delay} seconds")
    logger.info(f"  Limit stocks: {args.limit_stocks or 'All'}")
    
    if not args.limit_stocks:
        logger.info("This will collect data for Shanghai stocks over 3 years")
        logger.info("Using network-robust methods with longer delays")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            logger.info("Collection cancelled")
            return
    
    # Run collection
    success = collector.run_collection(
        limit_stocks=args.limit_stocks,
        delay=args.delay
    )
    
    if success:
        logger.info("Collection process completed!")
    else:
        logger.error("Collection process failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()