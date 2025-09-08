#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collection Progress Monitor

This script monitors the progress of Shanghai stock data collection.

Usage:
    python monitor_collection.py --data_dir ./qlib_shanghai_robust_data
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def monitor_progress(data_dir):
    """Monitor collection progress"""
    data_path = Path(data_dir)
    progress_file = data_path / "collection_progress.json"
    csv_dir = data_path / "csv_data"
    failed_file = data_path / "failed_stocks_final.txt"
    
    print("=" * 60)
    print("SHANGHAI STOCK COLLECTION PROGRESS MONITOR")
    print("=" * 60)
    
    # Check progress file
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            
            print("📊 Collection Progress:")
            print(f"  Current Index: {progress['current_index']}")
            print(f"  Total Stocks: {progress['total_stocks']}")
            print(f"  Successful: {progress['successful_count']}")
            print(f"  Failed: {progress['failed_count']}")
            
            completion = (progress['current_index'] / progress['total_stocks']) * 100
            success_rate = (progress['successful_count'] / max(progress['current_index'], 1)) * 100
            
            print(f"  Completion: {completion:.1f}%")
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Last Update: {progress['timestamp']}")
            
        except Exception as e:
            print(f"❌ Error reading progress file: {e}")
    else:
        print("📊 No active collection progress found")
    
    # Check CSV files
    if csv_dir.exists():
        csv_files = list(csv_dir.glob("*.csv"))
        print(f"\n📁 CSV Files: {len(csv_files)} files")
        
        if csv_files:
            # Show some sample files
            print("  Sample files:")
            for f in sorted(csv_files)[:5]:
                size_kb = f.stat().st_size / 1024
                print(f"    {f.name} ({size_kb:.1f} KB)")
            
            if len(csv_files) > 5:
                print(f"    ... and {len(csv_files) - 5} more files")
    else:
        print("\n📁 No CSV data directory found")
    
    # Check failed stocks
    if failed_file.exists():
        try:
            with open(failed_file, 'r') as f:
                failed_stocks = [line.strip() for line in f if line.strip()]
            
            print(f"\n❌ Failed Stocks: {len(failed_stocks)}")
            if failed_stocks:
                print("  Sample failed stocks:")
                for stock in failed_stocks[:10]:
                    print(f"    {stock}")
                if len(failed_stocks) > 10:
                    print(f"    ... and {len(failed_stocks) - 10} more")
        except Exception as e:
            print(f"❌ Error reading failed stocks: {e}")
    else:
        print("\n✅ No failed stocks file (good!)")
    
    # Check Qlib data
    qlib_dir = data_path / "qlib_data"
    if qlib_dir.exists():
        features_dir = qlib_dir / "features"
        if features_dir.exists():
            feature_files = list(features_dir.glob("*.bin"))
            print(f"\n🎯 Qlib Binary Files: {len(feature_files)} files")
        else:
            print(f"\n🎯 Qlib data directory exists but no features yet")
    else:
        print(f"\n🎯 No Qlib data directory found")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Monitor Shanghai stock collection progress")
    parser.add_argument("--data_dir", type=str, default="./qlib_shanghai_robust_data",
                       help="Data directory to monitor")
    
    args = parser.parse_args()
    
    monitor_progress(args.data_dir)


if __name__ == "__main__":
    main()