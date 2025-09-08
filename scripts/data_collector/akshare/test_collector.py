#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for AkShare collector following fund pattern
"""

import sys
import datetime
from pathlib import Path
import pandas as pd
import akshare as ak

def test_akshare_basic():
    """Test basic AkShare functionality"""
    print("Testing AkShare basic functionality...")
    
    # Test getting stock list
    try:
        stock_list = ak.stock_info_a_code_name()
        print(f"✓ Retrieved {len(stock_list)} stock symbols")
        symbols = stock_list['code'].tolist()[:5]  # Get first 5
        print(f"Sample symbols: {symbols}")
        return symbols
    except Exception as e:
        print(f"✗ Error getting stock list: {e}")
        return []

def test_data_collection(symbols):
    """Test data collection for symbols"""
    print("\nTesting data collection...")
    
    start_date = "20250101"
    end_date = "20250601"
    
    for symbol in symbols:
        try:
            # Get daily data
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            
            if not df.empty:
                print(f"✓ {symbol}: {len(df)} records")
                
                # Show sample data
                print(f"  Columns: {df.columns.tolist()}")
                print(f"  Date range: {df['日期'].min()} to {df['日期'].max()}")
                
                # Normalize column names
                column_mapping = {
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close', 
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'money',
                    '涨跌幅': 'change'
                }
                
                df_norm = df.rename(columns=column_mapping)
                df_norm['date'] = pd.to_datetime(df_norm['date'])
                
                # Save to CSV
                output_dir = Path("test_output")
                output_dir.mkdir(exist_ok=True)
                
                filename = output_dir / f"{symbol}.csv"
                df_norm.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"  Saved to: {filename}")
                
            else:
                print(f"✗ {symbol}: No data")
                
        except Exception as e:
            print(f"✗ {symbol}: Error - {e}")

def test_fund_pattern_structure():
    """Test the fund pattern structure"""
    print("\nTesting fund pattern structure...")
    
    # This would be the structure following fund collector pattern
    class TestAkShareCollector:
        def __init__(self, save_dir, start=None, end=None, interval="1d", adjust="qfq"):
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.start = start
            self.end = end
            self.interval = interval
            self.adjust = adjust
            
        def get_instrument_list(self):
            """Get instrument list like fund collector"""
            try:
                stock_list = ak.stock_info_a_code_name()
                symbols = stock_list['code'].tolist()[:3]  # Limit for test
                print(f"✓ Got {len(symbols)} symbols")
                return symbols
            except Exception as e:
                print(f"✗ Error getting symbols: {e}")
                return []
        
        @staticmethod
        def get_data_from_remote(symbol, interval, start, end, adjust="qfq"):
            """Get data from remote like fund collector"""
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start,
                    end_date=end,
                    adjust=adjust
                )
                
                if not df.empty:
                    # Normalize columns
                    column_mapping = {
                        '日期': 'date',
                        '开盘': 'open',
                        '收盘': 'close', 
                        '最高': 'high',
                        '最低': 'low',
                        '成交量': 'volume',
                        '成交额': 'money',
                        '涨跌幅': 'change'
                    }
                    df = df.rename(columns=column_mapping)
                    df['date'] = pd.to_datetime(df['date'])
                    return df.reset_index(drop=True)
                    
            except Exception as e:
                print(f"Error getting data for {symbol}: {e}")
                
            return pd.DataFrame()
        
        def get_data(self, symbol, interval, start_datetime, end_datetime):
            """Get data method like fund collector"""
            start_str = start_datetime.strftime('%Y%m%d') if hasattr(start_datetime, 'strftime') else start_datetime
            end_str = end_datetime.strftime('%Y%m%d') if hasattr(end_datetime, 'strftime') else end_datetime
            
            return self.get_data_from_remote(symbol, interval, start_str, end_str, self.adjust)
        
        def collector_data(self):
            """Main collection method like fund collector"""
            print("Starting data collection...")
            
            symbols = self.get_instrument_list()
            if not symbols:
                print("No symbols to collect")
                return
            
            for symbol in symbols:
                print(f"Processing {symbol}...")
                
                df = self.get_data(symbol, self.interval, self.start, self.end)
                
                if not df.empty:
                    filename = self.save_dir / f"{symbol}.csv"
                    df.to_csv(filename, index=False, encoding='utf-8-sig')
                    print(f"✓ Saved {len(df)} records to {filename}")
                else:
                    print(f"✗ No data for {symbol}")
    
    # Test the collector
    collector = TestAkShareCollector(
        save_dir="test_fund_pattern",
        start="20250101",
        end="20250601",
        interval="1d",
        adjust="qfq"
    )
    
    collector.collector_data()
    print("✓ Fund pattern test completed")

def main():
    """Main test function"""
    print("AkShare Collector Test (Fund Pattern)")
    print("=" * 50)
    
    # Test 1: Basic AkShare functionality
    symbols = test_akshare_basic()
    
    if symbols:
        # Test 2: Data collection
        test_data_collection(symbols[:3])  # Test with first 3 symbols
        
        # Test 3: Fund pattern structure
        test_fund_pattern_structure()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    
    # Show results
    test_dirs = ["test_output", "test_fund_pattern"]
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            files = list(Path(test_dir).glob("*.csv"))
            print(f"\n{test_dir}: {len(files)} files created")
            for f in files:
                print(f"  - {f.name}")

if __name__ == "__main__":
    main()
