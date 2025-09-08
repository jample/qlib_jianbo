#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example usage of the AkShare collector

This script demonstrates how to use the AkShare collector programmatically
for various data collection scenarios.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
from collector import AkShareCollector, AkShareRun

def example_1_basic_collection():
    """Example 1: Basic data collection"""
    print("=== Example 1: Basic Data Collection ===")
    
    # Create collector
    collector = AkShareCollector(
        save_dir="./example_data",
        start="20250101",
        end="20250601",
        interval="1d",
        delay=1.5,
        limit_nums=10,
        adjust="qfq"
    )
    
    # Collect data
    collector.collector_data()
    print("Basic collection completed!\n")

def example_2_recent_data():
    """Example 2: Collect recent data (last 30 days)"""
    print("=== Example 2: Recent Data Collection ===")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    collector = AkShareCollector(
        save_dir="./recent_data",
        start=start_date.strftime("%Y%m%d"),
        end=end_date.strftime("%Y%m%d"),
        interval="1d",
        delay=1.0,
        limit_nums=5
    )
    
    collector.collector_data()
    print("Recent data collection completed!\n")

def example_3_specific_stocks():
    """Example 3: Collect data for specific stocks"""
    print("=== Example 3: Specific Stocks Collection ===")
    
    # Define specific stocks
    target_stocks = ["000001", "000002", "600000", "600036", "000858"]
    
    collector = AkShareCollector(
        save_dir="./specific_stocks",
        start="20250101",
        end="20250601",
        interval="1d",
        delay=1.0
    )
    
    # Override the get_instrument_list method to return specific stocks
    collector.get_instrument_list = lambda: target_stocks
    
    collector.collector_data()
    print("Specific stocks collection completed!\n")

def example_4_using_runner():
    """Example 4: Using the AkShareRun class"""
    print("=== Example 4: Using AkShareRun ===")
    
    runner = AkShareRun(
        source_dir="./runner_data",
        interval="1d",
        max_workers=1
    )
    
    runner.download_data(
        start="20250101",
        end="20250301",
        delay=1.0,
        limit_nums=5,
        adjust="qfq"
    )
    print("Runner-based collection completed!\n")

def example_5_different_adjustments():
    """Example 5: Different adjustment types"""
    print("=== Example 5: Different Adjustment Types ===")
    
    adjustments = ["qfq", "hfq", ""]
    adjustment_names = ["Forward Adjusted", "Backward Adjusted", "No Adjustment"]
    
    for adj, name in zip(adjustments, adjustment_names):
        print(f"Collecting {name} data...")
        
        collector = AkShareCollector(
            save_dir=f"./adj_data_{adj if adj else 'none'}",
            start="20250101",
            end="20250301",
            interval="1d",
            delay=1.0,
            limit_nums=3,
            adjust=adj
        )
        
        collector.collector_data()
        print(f"{name} collection completed!")
    
    print("All adjustment types completed!\n")

def example_6_data_analysis():
    """Example 6: Basic data analysis after collection"""
    print("=== Example 6: Data Analysis ===")
    
    import pandas as pd
    
    # First collect some data
    collector = AkShareCollector(
        save_dir="./analysis_data",
        start="20250101",
        end="20250601",
        interval="1d",
        delay=1.0,
        limit_nums=3
    )
    
    collector.collector_data()
    
    # Analyze the collected data
    data_dir = Path("./analysis_data")
    csv_files = list(data_dir.glob("*.csv"))
    
    if csv_files:
        print(f"Found {len(csv_files)} data files")
        
        # Load and analyze first file
        sample_file = csv_files[0]
        df = pd.read_csv(sample_file)
        
        print(f"\nAnalyzing {sample_file.name}:")
        print(f"  Records: {len(df)}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Price range: {df['low'].min():.2f} to {df['high'].max():.2f}")
        print(f"  Average volume: {df['volume'].mean():.0f}")
        
        # Calculate returns
        df['daily_return'] = df['close'].pct_change()
        print(f"  Average daily return: {df['daily_return'].mean():.4f}")
        print(f"  Volatility: {df['daily_return'].std():.4f}")
    
    print("Data analysis completed!\n")

def main():
    """Run all examples"""
    print("AkShare Collector Usage Examples")
    print("=" * 50)
    
    examples = [
        example_1_basic_collection,
        example_2_recent_data,
        example_3_specific_stocks,
        example_4_using_runner,
        example_5_different_adjustments,
        example_6_data_analysis
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"Error in example {i}: {e}\n")
    
    print("=" * 50)
    print("All examples completed!")
    print("\nUsage Summary:")
    print("1. Basic collection: AkShareCollector with date range")
    print("2. Recent data: Use datetime calculations")
    print("3. Specific stocks: Override get_instrument_list()")
    print("4. Runner class: Use AkShareRun for command-line style")
    print("5. Adjustments: qfq (forward), hfq (backward), '' (none)")
    print("6. Analysis: Load CSV files with pandas for analysis")

if __name__ == "__main__":
    main()
