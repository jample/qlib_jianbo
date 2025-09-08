#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direct Shanghai Stock Collection using AkShare and Qlib

This script directly collects Shanghai stock data and converts it to Qlib format.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time

# Add qlib paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

try:
    import akshare as ak
    import qlib
    from qlib.data.data import Cal
    from qlib.utils import get_or_create_path
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running in the qlibbase environment")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shanghai_direct_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ShanghaiDirectCollector:
    """Direct collector for Shanghai stocks"""
    
    def __init__(self, data_dir="./qlib_shanghai_data", start_date="20220101", end_date="20250101"):
        self.data_dir = Path(data_dir)
        self.start_date = start_date
        self.end_date = end_date
        
        # Create directories
        self.csv_dir = self.data_dir / "csv_data"
        self.qlib_dir = self.data_dir / "qlib_data"
        
        for dir_path in [self.csv_dir, self.qlib_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Shanghai Direct Collector initialized")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Date range: {start_date} to {end_date}")
    
    def get_shanghai_stocks(self):
        """Get Shanghai stock list"""
        try:
            logger.info("Fetching Shanghai stock list...")
            stock_info = ak.stock_info_a_code_name()
            
            # Filter Shanghai stocks
            shanghai_stocks = stock_info[
                stock_info['code'].str.startswith(('60', '68', '90'))
            ]['code'].tolist()
            
            logger.info(f"Found {len(shanghai_stocks)} Shanghai stocks")
            return shanghai_stocks
            
        except Exception as e:
            logger.error(f"Error fetching stock list: {e}")
            return []
    
    def collect_stock_data(self, symbol, delay=1.5, max_retries=3):
        """Collect data for a single stock with retry logic"""
        for attempt in range(max_retries):
            try:
                # Progressive delay for retries
                actual_delay = delay * (1 + attempt * 0.5)
                time.sleep(actual_delay)
                
                if attempt > 0:
                    logger.info(f"Retry {attempt}/{max_retries-1} for {symbol}")
                
                logger.debug(f"Collecting data for {symbol}...")
                
                data = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=self.start_date,
                    end_date=self.end_date,
                    adjust="qfq"  # Forward adjustment
                )
                
                if data.empty:
                    logger.warning(f"No data for {symbol}")
                    return None
                
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
                
                logger.debug(f"Collected {len(data)} records for {symbol}")
                return data
                
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    # Check if it's a network-related error
                    if any(err in error_msg.lower() for err in ['connection', 'timeout', 'network', 'remote']):
                        logger.warning(f"Network error for {symbol} (attempt {attempt+1}): {error_msg}")
                        # Longer delay for network errors
                        time.sleep(delay * 2)
                        continue
                    else:
                        logger.error(f"Non-network error for {symbol}: {error_msg}")
                        return None
                else:
                    logger.error(f"Failed to collect {symbol} after {max_retries} attempts: {error_msg}")
                    return None
        
        return None
    
    def save_stock_csv(self, symbol, data):
        """Save stock data to CSV"""
        try:
            csv_file = self.csv_dir / f"{symbol}.csv"
            data.to_csv(csv_file, index=False)
            logger.debug(f"Saved {symbol} to {csv_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving {symbol}: {e}")
            return False
    
    def collect_all_stocks(self, limit_stocks=None, delay=1.5, resume_from=None):
        """Collect data for all Shanghai stocks with resume capability"""
        shanghai_stocks = self.get_shanghai_stocks()
        
        if limit_stocks:
            shanghai_stocks = shanghai_stocks[:limit_stocks]
            logger.info(f"Limited to first {limit_stocks} stocks")
        
        # Resume from a specific stock if specified
        start_index = 0
        if resume_from:
            try:
                start_index = shanghai_stocks.index(resume_from)
                logger.info(f"Resuming from stock {resume_from} (index {start_index})")
            except ValueError:
                logger.warning(f"Resume stock {resume_from} not found, starting from beginning")
        
        logger.info(f"Starting collection for {len(shanghai_stocks)} stocks (from index {start_index})...")
        
        successful_count = 0
        failed_count = 0
        failed_stocks = []
        
        # Check for existing files to avoid re-downloading
        existing_files = set()
        if self.csv_dir.exists():
            existing_files = {f.stem for f in self.csv_dir.glob("*.csv")}
            if existing_files:
                logger.info(f"Found {len(existing_files)} existing files, will skip them")
        
        for i, symbol in enumerate(shanghai_stocks[start_index:], start_index + 1):
            # Skip if file already exists
            if symbol in existing_files:
                logger.debug(f"Skipping {symbol} - file already exists")
                successful_count += 1
                continue
            
            logger.info(f"Processing {i}/{len(shanghai_stocks)}: {symbol}")
            
            data = self.collect_stock_data(symbol, delay=delay, max_retries=3)
            
            if data is not None:
                if self.save_stock_csv(symbol, data):
                    successful_count += 1
                    logger.debug(f"✓ Successfully saved {symbol}")
                else:
                    failed_count += 1
                    failed_stocks.append(symbol)
                    logger.warning(f"✗ Failed to save {symbol}")
            else:
                failed_count += 1
                failed_stocks.append(symbol)
                logger.warning(f"✗ Failed to collect {symbol}")
            
            # Progress update every 25 stocks
            if i % 25 == 0:
                success_rate = (successful_count / i) * 100
                logger.info(f"Progress: {i}/{len(shanghai_stocks)} ({success_rate:.1f}% success) - Success: {successful_count}, Failed: {failed_count}")
                
                # Save failed stocks list for potential retry
                if failed_stocks:
                    failed_file = self.data_dir / "failed_stocks.txt"
                    with open(failed_file, 'w') as f:
                        f.write('\n'.join(failed_stocks))
            
            # Adaptive delay based on success rate
            if i > 10:
                recent_success_rate = successful_count / i
                if recent_success_rate < 0.8:  # Less than 80% success
                    delay = min(delay * 1.1, 5.0)  # Increase delay, max 5 seconds
                    logger.info(f"Low success rate ({recent_success_rate:.1%}), increasing delay to {delay:.1f}s")
        
        final_success_rate = (successful_count / len(shanghai_stocks)) * 100
        logger.info(f"Collection completed: {successful_count} successful, {failed_count} failed ({final_success_rate:.1f}% success rate)")
        
        # Save final failed stocks list
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
    
    def retry_failed_stocks(self, delay=2.0):
        """Retry collecting failed stocks"""
        failed_file = self.data_dir / "failed_stocks_final.txt"
        if not failed_file.exists():
            logger.info("No failed stocks file found")
            return True
        
        with open(failed_file, 'r') as f:
            failed_stocks = [line.strip() for line in f if line.strip()]
        
        if not failed_stocks:
            logger.info("No failed stocks to retry")
            return True
        
        logger.info(f"Retrying {len(failed_stocks)} failed stocks with increased delay ({delay}s)...")
        
        successful_count = 0
        still_failed = []
        
        for i, symbol in enumerate(failed_stocks, 1):
            logger.info(f"Retrying {i}/{len(failed_stocks)}: {symbol}")
            
            data = self.collect_stock_data(symbol, delay=delay, max_retries=5)
            
            if data is not None and self.save_stock_csv(symbol, data):
                successful_count += 1
                logger.info(f"✓ Retry successful for {symbol}")
            else:
                still_failed.append(symbol)
                logger.warning(f"✗ Retry failed for {symbol}")
        
        logger.info(f"Retry completed: {successful_count} recovered, {len(still_failed)} still failed")
        
        # Update failed stocks file
        if still_failed:
            with open(failed_file, 'w') as f:
                f.write('\n'.join(still_failed))
        else:
            failed_file.unlink()  # Remove file if no failures
        
        return True
    
    def run_collection(self, limit_stocks=None, delay=1.5, retry_failed=True):
        """Run the complete collection process"""
        logger.info("=" * 60)
        logger.info("SHANGHAI STOCK DIRECT COLLECTION")
        logger.info("=" * 60)
        
        # Step 1: Collect CSV data
        logger.info("Step 1: Collecting stock data...")
        if not self.collect_all_stocks(limit_stocks=limit_stocks, delay=delay):
            logger.error("Stock data collection failed!")
            return False
        
        # Step 1.5: Retry failed stocks if enabled
        if retry_failed:
            logger.info("Step 1.5: Retrying failed stocks...")
            self.retry_failed_stocks(delay=delay * 1.5)
        
        # Step 2: Convert to Qlib format
        logger.info("Step 2: Converting to Qlib format...")
        if not self.convert_to_qlib_format():
            logger.error("Qlib format conversion failed!")
            return False
        
        # Final statistics
        csv_files = list(self.csv_dir.glob("*.csv"))
        logger.info("=" * 60)
        logger.info("COLLECTION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Total CSV files collected: {len(csv_files)}")
        logger.info(f"Qlib data available at: {self.qlib_dir}")
        logger.info("You can now use this data with Qlib by setting:")
        logger.info(f"qlib.init(provider_uri='{self.qlib_dir}', region='cn')")
        
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Direct Shanghai Stock Collection")
    parser.add_argument("--start_date", type=str, default="20220101",
                       help="Start date in YYYYMMDD format")
    parser.add_argument("--end_date", type=str, default="20250101",
                       help="End date in YYYYMMDD format")
    parser.add_argument("--data_dir", type=str, default="./qlib_shanghai_data",
                       help="Data directory")
    parser.add_argument("--limit_stocks", type=int, default=None,
                       help="Limit number of stocks for testing")
    parser.add_argument("--delay", type=float, default=1.5,
                       help="Delay between requests in seconds")
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        datetime.strptime(args.start_date, "%Y%m%d")
        datetime.strptime(args.end_date, "%Y%m%d")
    except ValueError:
        logger.error("Invalid date format. Use YYYYMMDD.")
        return
    
    # Create collector
    collector = ShanghaiDirectCollector(
        data_dir=args.data_dir,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Run collection
    success = collector.run_collection(
        limit_stocks=args.limit_stocks,
        delay=args.delay
    )
    
    if success:
        logger.info("Shanghai stock collection completed successfully!")
    else:
        logger.error("Shanghai stock collection failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()