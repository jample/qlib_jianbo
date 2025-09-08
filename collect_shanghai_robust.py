#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust Shanghai Stock Collection with Resume Capability

This script provides a more robust collection process that can handle
network errors, interruptions, and resume from where it left off.

Usage:
    python collect_shanghai_robust.py --start_date 20220101 --end_date 20250101
    python collect_shanghai_robust.py --resume  # Resume from last position
"""

import os
import sys
import json
import logging
import argparse
import signal
from pathlib import Path
from datetime import datetime
from collect_shanghai_direct import ShanghaiDirectCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shanghai_robust_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RobustShanghaiCollector(ShanghaiDirectCollector):
    """Enhanced collector with resume capability and better error handling"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_file = self.data_dir / "collection_progress.json"
        self.interrupted = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interruption signals gracefully"""
        logger.info("Received interruption signal, finishing current stock...")
        self.interrupted = True
    
    def save_progress(self, current_index, total_stocks, successful_count, failed_count):
        """Save collection progress"""
        progress = {
            'current_index': current_index,
            'total_stocks': total_stocks,
            'successful_count': successful_count,
            'failed_count': failed_count,
            'timestamp': datetime.now().isoformat(),
            'start_date': self.start_date,
            'end_date': self.end_date
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def load_progress(self):
        """Load previous progress"""
        if not self.progress_file.exists():
            return None
        
        try:
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
            
            # Verify the progress matches current parameters
            if (progress.get('start_date') == self.start_date and 
                progress.get('end_date') == self.end_date):
                return progress
            else:
                logger.warning("Progress file doesn't match current parameters, starting fresh")
                return None
                
        except Exception as e:
            logger.error(f"Error loading progress: {e}")
            return None
    
    def collect_all_stocks_robust(self, limit_stocks=None, delay=1.5, resume=False):
        """Robust collection with resume capability"""
        shanghai_stocks = self.get_shanghai_stocks()
        
        if limit_stocks:
            shanghai_stocks = shanghai_stocks[:limit_stocks]
            logger.info(f"Limited to first {limit_stocks} stocks")
        
        # Load progress if resuming
        start_index = 0
        successful_count = 0
        failed_count = 0
        
        if resume:
            progress = self.load_progress()
            if progress:
                start_index = progress['current_index']
                successful_count = progress['successful_count']
                failed_count = progress['failed_count']
                logger.info(f"Resuming from index {start_index} (Success: {successful_count}, Failed: {failed_count})")
        
        logger.info(f"Starting collection for {len(shanghai_stocks)} stocks (from index {start_index})...")
        
        failed_stocks = []
        
        # Check for existing files
        existing_files = set()
        if self.csv_dir.exists():
            existing_files = {f.stem for f in self.csv_dir.glob("*.csv")}
            if existing_files:
                logger.info(f"Found {len(existing_files)} existing files")
        
        try:
            for i, symbol in enumerate(shanghai_stocks[start_index:], start_index):
                # Check for interruption
                if self.interrupted:
                    logger.info("Collection interrupted by user")
                    break
                
                # Skip if file already exists
                if symbol in existing_files:
                    logger.debug(f"Skipping {symbol} - file already exists")
                    successful_count += 1
                    continue
                
                logger.info(f"Processing {i+1}/{len(shanghai_stocks)}: {symbol}")
                
                # Collect data with retries
                data = self.collect_stock_data(symbol, delay=delay, max_retries=5)
                
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
                
                # Save progress every 10 stocks
                if (i + 1) % 10 == 0:
                    self.save_progress(i + 1, len(shanghai_stocks), successful_count, failed_count)
                
                # Progress update every 25 stocks
                if (i + 1) % 25 == 0:
                    success_rate = (successful_count / (i + 1)) * 100
                    logger.info(f"Progress: {i+1}/{len(shanghai_stocks)} ({success_rate:.1f}% success)")
                    logger.info(f"Success: {successful_count}, Failed: {failed_count}")
                
                # Adaptive delay based on recent failures
                if (i + 1) % 50 == 0 and failed_count > 0:
                    recent_failure_rate = failed_count / (i + 1)
                    if recent_failure_rate > 0.2:  # More than 20% failures
                        delay = min(delay * 1.2, 5.0)
                        logger.info(f"High failure rate, increasing delay to {delay:.1f}s")
        
        except Exception as e:
            logger.error(f"Unexpected error during collection: {e}")
            # Save progress before exiting
            self.save_progress(i, len(shanghai_stocks), successful_count, failed_count)
            raise
        
        # Final progress save
        self.save_progress(len(shanghai_stocks), len(shanghai_stocks), successful_count, failed_count)
        
        final_success_rate = (successful_count / len(shanghai_stocks)) * 100
        logger.info(f"Collection completed: {successful_count} successful, {failed_count} failed ({final_success_rate:.1f}% success rate)")
        
        # Save failed stocks
        if failed_stocks:
            failed_file = self.data_dir / "failed_stocks_final.txt"
            with open(failed_file, 'w') as f:
                f.write('\n'.join(failed_stocks))
            logger.info(f"Failed stocks saved to: {failed_file}")
        
        return successful_count > 0
    
    def run_robust_collection(self, limit_stocks=None, delay=1.5, resume=False, retry_failed=True):
        """Run robust collection process"""
        logger.info("=" * 80)
        logger.info("ROBUST SHANGHAI STOCK COLLECTION")
        logger.info("=" * 80)
        logger.info(f"Resume mode: {resume}")
        logger.info(f"Retry failed: {retry_failed}")
        
        try:
            # Step 1: Collect CSV data
            logger.info("Step 1: Collecting stock data...")
            if not self.collect_all_stocks_robust(limit_stocks=limit_stocks, delay=delay, resume=resume):
                logger.error("Stock data collection failed!")
                return False
            
            # Step 2: Retry failed stocks if enabled
            if retry_failed and not self.interrupted:
                logger.info("Step 2: Retrying failed stocks...")
                self.retry_failed_stocks(delay=delay * 1.5)
            
            # Step 3: Convert to Qlib format
            if not self.interrupted:
                logger.info("Step 3: Converting to Qlib format...")
                if not self.convert_to_qlib_format():
                    logger.error("Qlib format conversion failed!")
                    return False
            
            # Final statistics
            csv_files = list(self.csv_dir.glob("*.csv"))
            logger.info("=" * 80)
            if self.interrupted:
                logger.info("COLLECTION INTERRUPTED BUT PROGRESS SAVED")
                logger.info("You can resume with: python collect_shanghai_robust.py --resume")
            else:
                logger.info("COLLECTION COMPLETED SUCCESSFULLY!")
                # Clean up progress file
                if self.progress_file.exists():
                    self.progress_file.unlink()
            logger.info("=" * 80)
            logger.info(f"Total CSV files collected: {len(csv_files)}")
            logger.info(f"Data directory: {self.data_dir}")
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Collection interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Collection failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description="Robust Shanghai Stock Collection")
    parser.add_argument("--start_date", type=str, default="20220101",
                       help="Start date in YYYYMMDD format")
    parser.add_argument("--end_date", type=str, default="20250101",
                       help="End date in YYYYMMDD format")
    parser.add_argument("--data_dir", type=str, default="./qlib_shanghai_robust_data",
                       help="Data directory")
    parser.add_argument("--limit_stocks", type=int, default=None,
                       help="Limit number of stocks for testing")
    parser.add_argument("--delay", type=float, default=2.0,
                       help="Delay between requests in seconds")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous progress")
    parser.add_argument("--no_retry", action="store_true",
                       help="Don't retry failed stocks")
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        datetime.strptime(args.start_date, "%Y%m%d")
        datetime.strptime(args.end_date, "%Y%m%d")
    except ValueError:
        logger.error("Invalid date format. Use YYYYMMDD.")
        return
    
    # Create collector
    collector = RobustShanghaiCollector(
        data_dir=args.data_dir,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Show collection info
    logger.info(f"Collection parameters:")
    logger.info(f"  Date range: {args.start_date} to {args.end_date}")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Delay: {args.delay} seconds")
    logger.info(f"  Limit stocks: {args.limit_stocks or 'All'}")
    logger.info(f"  Resume: {args.resume}")
    
    if not args.resume and not args.limit_stocks:
        logger.info("This will collect data for ~2,283 Shanghai stocks over 3 years")
        logger.info("Estimated time: 4-8 hours depending on network conditions")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            logger.info("Collection cancelled")
            return
    
    # Run collection
    success = collector.run_robust_collection(
        limit_stocks=args.limit_stocks,
        delay=args.delay,
        resume=args.resume,
        retry_failed=not args.no_retry
    )
    
    if success:
        logger.info("Collection process completed!")
    else:
        logger.error("Collection process failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()