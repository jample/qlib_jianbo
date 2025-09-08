#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shanghai Stock Market Data Collection Script for Qlib

This script collects all Shanghai stock market data from 2022.01.01 to 2025.01.01
using the existing AkShare collector infrastructure in Qlib.

Usage:
    python collect_shanghai_stocks.py --start_date 20220101 --end_date 20250101
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add the scripts directory to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
QLIB_ROOT = SCRIPT_DIR
COLLECTOR_DIR = QLIB_ROOT / "scripts" / "data_collector" / "akshare"

sys.path.insert(0, str(COLLECTOR_DIR))
sys.path.insert(0, str(QLIB_ROOT / "scripts"))

try:
    import akshare as ak
    import pandas as pd
    from collector import AkShareCollectorCN1d, AkShareNormalizeCN1d
    from data_collector.utils import get_calendar_list
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running in the qlibbase environment with all dependencies installed")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shanghai_stocks_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ShanghaiStockCollector:
    """Specialized collector for Shanghai Stock Exchange stocks"""
    
    def __init__(self, data_dir="./qlib_shanghai_data", start_date="20220101", end_date="20250101"):
        self.data_dir = Path(data_dir)
        self.start_date = start_date
        self.end_date = end_date
        
        # Create directory structure
        self.source_dir = self.data_dir / "source"
        self.normalize_dir = self.data_dir / "normalize" 
        self.qlib_dir = self.data_dir / "qlib_data"
        
        for dir_path in [self.source_dir, self.normalize_dir, self.qlib_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Shanghai Stock Collector initialized")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Date range: {start_date} to {end_date}")
    
    def get_shanghai_stocks(self):
        """Get all Shanghai Stock Exchange listed stocks"""
        try:
            logger.info("Fetching Shanghai stock list...")
            
            # Get all A-share stocks
            stock_info = ak.stock_info_a_code_name()
            
            # Filter for Shanghai stocks (codes starting with 60, 68, 90)
            shanghai_stocks = stock_info[
                stock_info['code'].str.startswith(('60', '68', '90'))
            ]['code'].tolist()
            
            logger.info(f"Found {len(shanghai_stocks)} Shanghai stocks")
            
            # Log some examples
            logger.info(f"Sample Shanghai stocks: {shanghai_stocks[:10]}")
            
            return shanghai_stocks
            
        except Exception as e:
            logger.error(f"Error fetching Shanghai stock list: {e}")
            # Fallback to some major Shanghai stocks
            fallback_stocks = [
                "600000", "600036", "600519", "600276", "600585", "600887", "600009", "600028",
                "600030", "600048", "600050", "600104", "600111", "600150", "600196", "600309",
                "600340", "600362", "600383", "600406", "600438", "600482", "600489", "600498",
                "600547", "600570", "600588", "600606", "600637", "600660", "600690", "600703",
                "600745", "600760", "600795", "600809", "600837", "600848", "600867", "600886",
                "600893", "600900", "600919", "600926", "600958", "600999", "601006", "601012",
                "601066", "601088", "601111", "601138", "601166", "601169", "601186", "601211",
                "601225", "601229", "601238", "601288", "601318", "601328", "601336", "601360",
                "601377", "601390", "601398", "601601", "601628", "601633", "601668", "601669",
                "601688", "601728", "601766", "601788", "601800", "601808", "601818", "601828",
                "601857", "601866", "601872", "601877", "601878", "601881", "601888", "601898",
                "601899", "601901", "601919", "601933", "601939", "601985", "601988", "601989",
                "601991", "601992", "601995", "601997", "601998"
            ]
            logger.info(f"Using fallback list: {len(fallback_stocks)} stocks")
            return fallback_stocks
    
    def collect_data(self, limit_stocks=None, delay=1.0):
        """Collect Shanghai stock data using AkShare collector"""
        try:
            # Get Shanghai stocks
            shanghai_stocks = self.get_shanghai_stocks()
            
            if limit_stocks:
                shanghai_stocks = shanghai_stocks[:limit_stocks]
                logger.info(f"Limited to first {limit_stocks} stocks")
            
            logger.info(f"Starting data collection for {len(shanghai_stocks)} Shanghai stocks")
            
            # Initialize the AkShare collector
            collector = AkShareCollectorCN1d(
                save_dir=self.source_dir,
                start=self.start_date,
                end=self.end_date,
                interval="1d",
                max_workers=4,
                max_collector_count=2,
                delay=delay,
                limit_nums=len(shanghai_stocks),
                adjust="qfq",  # Forward adjustment for backtesting
                data_type="stock"
            )
            
            # Override the instrument list to only include Shanghai stocks
            collector.get_instrument_list = lambda: shanghai_stocks
            
            # Collect the data
            logger.info("Starting data collection...")
            collector.collector_data()
            
            logger.info("Data collection completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
            return False
    
    def normalize_data(self):
        """Normalize collected data for Qlib"""
        try:
            logger.info("Starting data normalization...")
            
            # Initialize normalizer
            normalizer = AkShareNormalizeCN1d(
                source_dir=self.source_dir,
                target_dir=self.normalize_dir,
                max_workers=4,
                date_field_name="date",
                symbol_field_name="symbol"
            )
            
            # Normalize the data
            normalizer.normalize()
            
            logger.info("Data normalization completed!")
            return True
            
        except Exception as e:
            logger.error(f"Error during normalization: {e}")
            return False
    
    def dump_to_qlib_format(self):
        """Convert normalized data to Qlib binary format"""
        try:
            logger.info("Converting to Qlib binary format...")
            
            # Import dump_bin script
            sys.path.insert(0, str(QLIB_ROOT / "scripts"))
            from dump_bin import DumpDataAll
            
            # Dump to binary format
            DumpDataAll(
                csv_path=self.normalize_dir,
                qlib_dir=self.qlib_dir,
                freq="day",
                max_workers=4,
                date_field_name="date",
                file_suffix=".csv",
                symbol_field_name="symbol",
                exclude_fields="",
            ).dump()
            
            logger.info(f"Qlib binary data saved to: {self.qlib_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting to Qlib format: {e}")
            return False
    
    def run_full_pipeline(self, limit_stocks=None, delay=1.0):
        """Run the complete data collection pipeline"""
        logger.info("=" * 60)
        logger.info("SHANGHAI STOCK DATA COLLECTION PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Collect raw data
        logger.info("Step 1: Collecting raw data...")
        if not self.collect_data(limit_stocks=limit_stocks, delay=delay):
            logger.error("Data collection failed!")
            return False
        
        # Step 2: Normalize data
        logger.info("Step 2: Normalizing data...")
        if not self.normalize_data():
            logger.error("Data normalization failed!")
            return False
        
        # Step 3: Convert to Qlib format
        logger.info("Step 3: Converting to Qlib binary format...")
        if not self.dump_to_qlib_format():
            logger.error("Qlib format conversion failed!")
            return False
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Qlib data available at: {self.qlib_dir}")
        logger.info("You can now use this data with Qlib by setting:")
        logger.info(f"qlib.init(provider_uri='{self.qlib_dir}', region='cn')")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Collect Shanghai Stock Market data for Qlib")
    parser.add_argument("--start_date", type=str, default="20220101", 
                       help="Start date in YYYYMMDD format (default: 20220101)")
    parser.add_argument("--end_date", type=str, default="20250101",
                       help="End date in YYYYMMDD format (default: 20250101)")
    parser.add_argument("--data_dir", type=str, default="./qlib_shanghai_data",
                       help="Directory to save data (default: ./qlib_shanghai_data)")
    parser.add_argument("--limit_stocks", type=int, default=None,
                       help="Limit number of stocks for testing (default: all)")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between API requests in seconds (default: 1.0)")
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.start_date, "%Y%m%d")
        datetime.strptime(args.end_date, "%Y%m%d")
    except ValueError:
        logger.error("Invalid date format. Please use YYYYMMDD format.")
        return
    
    # Initialize collector
    collector = ShanghaiStockCollector(
        data_dir=args.data_dir,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Run the pipeline
    success = collector.run_full_pipeline(
        limit_stocks=args.limit_stocks,
        delay=args.delay
    )
    
    if success:
        logger.info("Shanghai stock data collection completed successfully!")
    else:
        logger.error("Shanghai stock data collection failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()