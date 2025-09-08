#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full Shanghai Stock Market Data Collection Script

This script collects all Shanghai stock market data from 2022.01.01 to 2025.01.01
using AkShare and converts it to Qlib format.

Usage:
    python run_full_shanghai_collection.py
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_shanghai_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Full Shanghai Stock Market Data Collection")
    parser.add_argument("--start_date", type=str, default="20220101",
                       help="Start date in YYYYMMDD format (default: 20220101)")
    parser.add_argument("--end_date", type=str, default="20250101",
                       help="End date in YYYYMMDD format (default: 20250101)")
    parser.add_argument("--data_dir", type=str, default="./qlib_shanghai_full_data",
                       help="Data directory (default: ./qlib_shanghai_full_data)")
    parser.add_argument("--delay", type=float, default=1.5,
                       help="Delay between requests in seconds (default: 1.5)")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Maximum number of workers (default: 4)")
    parser.add_argument("--test_mode", action="store_true",
                       help="Run in test mode with limited stocks")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("FULL SHANGHAI STOCK MARKET DATA COLLECTION")
    logger.info("=" * 80)
    logger.info(f"Start date: {args.start_date}")
    logger.info(f"End date: {args.end_date}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Delay: {args.delay} seconds")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Test mode: {args.test_mode}")
    
    # Import the collector
    try:
        from collect_shanghai_direct import ShanghaiDirectCollector
    except ImportError as e:
        logger.error(f"Failed to import collector: {e}")
        sys.exit(1)
    
    # Create collector
    collector = ShanghaiDirectCollector(
        data_dir=args.data_dir,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Set limit for test mode
    limit_stocks = 10 if args.test_mode else None
    
    if args.test_mode:
        logger.info("Running in TEST MODE - collecting only 10 stocks")
    else:
        logger.info("Running in FULL MODE - collecting ALL Shanghai stocks")
        logger.info("This will take several hours to complete...")
        
        # Ask for confirmation
        response = input("Do you want to continue? (y/N): ")
        if response.lower() != 'y':
            logger.info("Collection cancelled by user")
            return
    
    # Run collection
    try:
        success = collector.run_collection(
            limit_stocks=limit_stocks,
            delay=args.delay
        )
        
        if success:
            logger.info("=" * 80)
            logger.info("COLLECTION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            
            # Test the data
            logger.info("Testing the collected data...")
            test_cmd = f"python test_qlib_data.py"
            os.system(f"conda activate qlibbase && cd {args.data_dir} && {test_cmd}")
            
        else:
            logger.error("Collection failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()