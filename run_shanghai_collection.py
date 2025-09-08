#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shanghai Stock Market Data Collection using Qlib's AkShare Collector

This script uses the existing Qlib AkShare collector to collect Shanghai stock data.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shanghai_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_data_collection(start_date="20220101", end_date="20250101", limit_stocks=None):
    """Run data collection using the existing AkShare collector"""
    
    logger.info("=" * 60)
    logger.info("SHANGHAI STOCK DATA COLLECTION")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Set up directories
    base_dir = Path("./qlib_shanghai_data")
    source_dir = base_dir / "source"
    normalize_dir = base_dir / "normalize"
    qlib_dir = base_dir / "qlib_data"
    
    # Create directories
    for dir_path in [source_dir, normalize_dir, qlib_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Change to the akshare collector directory
    collector_dir = Path("scripts/data_collector/akshare")
    
    try:
        # Step 1: Download stock data
        logger.info("Step 1: Downloading stock data...")
        
        cmd = [
            "python", "collector.py", "download_data",
            "--start", start_date,
            "--end", end_date,
            "--delay", "1.5",
            "--data_type", "stock",
            "--adjust", "qfq",
            "--source_dir", str(source_dir),
            "--max_workers", "4"
        ]
        
        if limit_stocks:
            cmd.extend(["--limit_nums", str(limit_stocks)])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the collector
        result = subprocess.run(
            cmd,
            cwd=collector_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("Data download completed successfully!")
        logger.info(f"Output: {result.stdout}")
        
        # Step 2: Normalize data
        logger.info("Step 2: Normalizing data...")
        
        normalize_cmd = [
            "python", "collector.py", "normalize_data",
            "--source_dir", str(source_dir),
            "--normalize_dir", str(normalize_dir),
            "--region", "CN",
            "--interval", "1d"
        ]
        
        result = subprocess.run(
            normalize_cmd,
            cwd=collector_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("Data normalization completed!")
        logger.info(f"Output: {result.stdout}")
        
        # Step 3: Convert to Qlib format
        logger.info("Step 3: Converting to Qlib binary format...")
        
        dump_cmd = [
            "python", "../dump_bin.py",
            "dump_all",
            "--csv_path", str(normalize_dir),
            "--qlib_dir", str(qlib_dir),
            "--freq", "day",
            "--max_workers", "4"
        ]
        
        result = subprocess.run(
            dump_cmd,
            cwd=collector_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("Qlib format conversion completed!")
        logger.info(f"Output: {result.stdout}")
        
        logger.info("=" * 60)
        logger.info("SUCCESS: All steps completed!")
        logger.info("=" * 60)
        logger.info(f"Qlib data available at: {qlib_dir}")
        logger.info("You can now use this data with Qlib by setting:")
        logger.info(f"qlib.init(provider_uri='{qlib_dir}', region='cn')")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def filter_shanghai_stocks():
    """Create a custom collector that only gets Shanghai stocks"""
    
    logger.info("Creating Shanghai-specific stock filter...")
    
    # Create a custom script that modifies the stock list
    custom_collector_script = """
import sys
from pathlib import Path

# Add the collector path
collector_dir = Path(__file__).parent
sys.path.insert(0, str(collector_dir))
sys.path.insert(0, str(collector_dir.parent))

from collector import AkShareCollectorCN1d
import akshare as ak

# Create a custom collector class for Shanghai stocks
class ShanghaiStockCollector(AkShareCollectorCN1d):
    def get_instrument_list(self):
        # Get all A-share stocks
        stock_list = ak.stock_info_a_code_name()
        
        # Filter for Shanghai stocks (codes starting with 60, 68, 90)
        shanghai_stocks = stock_list[
            stock_list['code'].str.startswith(('60', '68', '90'))
        ]['code'].tolist()
        
        print(f"Found {len(shanghai_stocks)} Shanghai stocks")
        return shanghai_stocks

# Replace the original class
import collector
collector.AkShareCollectorCN1d = ShanghaiStockCollector

# Now run the original collector
if __name__ == "__main__":
    import fire
    from collector import Run
    fire.Fire(Run)
"""
    
    # Write the custom collector
    custom_collector_path = Path("scripts/data_collector/akshare/shanghai_collector.py")
    with open(custom_collector_path, 'w') as f:
        f.write(custom_collector_script)
    
    logger.info(f"Created custom Shanghai collector: {custom_collector_path}")
    return custom_collector_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect Shanghai Stock Market data")
    parser.add_argument("--start_date", type=str, default="20220101", 
                       help="Start date in YYYYMMDD format")
    parser.add_argument("--end_date", type=str, default="20250101",
                       help="End date in YYYYMMDD format")
    parser.add_argument("--limit_stocks", type=int, default=None,
                       help="Limit number of stocks for testing")
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.start_date, "%Y%m%d")
        datetime.strptime(args.end_date, "%Y%m%d")
    except ValueError:
        logger.error("Invalid date format. Please use YYYYMMDD format.")
        return
    
    # Create Shanghai-specific collector
    custom_collector = filter_shanghai_stocks()
    
    # Run the collection process
    success = run_data_collection(
        start_date=args.start_date,
        end_date=args.end_date,
        limit_stocks=args.limit_stocks
    )
    
    if success:
        logger.info("Shanghai stock data collection completed successfully!")
    else:
        logger.error("Shanghai stock data collection failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()