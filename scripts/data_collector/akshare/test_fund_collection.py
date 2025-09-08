#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test fund data collection with specific fund codes
"""

import sys
from pathlib import Path
import pandas as pd
import akshare as ak
from loguru import logger

def test_fund_data_collection():
    """Test fund data collection with known fund codes"""
    
    # Known working fund codes
    test_funds = ['015198', '110022', '161725']
    
    print("Testing fund data collection...")
    
    for fund_code in test_funds:
        try:
            print(f"\n=== Testing fund {fund_code} ===")
            
            # Get fund data
            df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="累计净值走势")
            
            if df.empty:
                print(f"No data for fund {fund_code}")
                continue
                
            print(f"Raw data shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print("Sample data:")
            print(df.head(3))
            
            # Process the data like in the collector
            fund_column_mapping = {
                '净值日期': 'date',
                '单位净值': 'close',
                '累计净值': 'cumulative_nav',
                '日增长率': 'change'
            }
            
            # Check what columns are actually available
            available_mapping = {k: v for k, v in fund_column_mapping.items() if k in df.columns}
            
            # If cumulative_nav is available but not close, use it as close
            if '累计净值' in df.columns and '单位净值' not in df.columns:
                available_mapping['累计净值'] = 'close'
            
            print(f"Available mapping: {available_mapping}")
            
            # Apply mapping
            df_processed = df.rename(columns=available_mapping)
            
            # Ensure date column is datetime
            if 'date' in df_processed.columns:
                df_processed['date'] = pd.to_datetime(df_processed['date'])
            
            # Add required columns for consistency
            if 'close' in df_processed.columns:
                df_processed['open'] = df_processed['close']  # For funds, use close as open
                df_processed['high'] = df_processed['close']  # For funds, use close as high/low
                df_processed['low'] = df_processed['close']
                df_processed['volume'] = 0  # Funds don't have volume
                df_processed['money'] = 0   # Funds don't have money
            
            # Select standard columns
            standard_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'money']
            available_cols = [col for col in standard_cols if col in df_processed.columns]
            
            # Add change and cumulative_nav if available
            if 'change' in df_processed.columns:
                available_cols.append('change')
            if 'cumulative_nav' in df_processed.columns:
                available_cols.append('cumulative_nav')
                
            df_final = df_processed[available_cols].copy()
            
            print(f"Final processed data shape: {df_final.shape}")
            print(f"Final columns: {df_final.columns.tolist()}")
            print("Final sample data:")
            print(df_final.head(3))
            
            # Save to file
            output_file = f"test_fund_{fund_code}.csv"
            df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"Saved to: {output_file}")
            
            # Test successful, break
            print(f"✓ Fund {fund_code} collection successful!")
            return True
            
        except Exception as e:
            print(f"✗ Error with fund {fund_code}: {e}")
            continue
    
    return False

def test_collector_with_specific_funds():
    """Test the collector with specific fund codes"""
    print("\n" + "="*50)
    print("Testing collector with specific fund codes...")
    
    from collector_standalone import AkShareCollectorCN1d
    
    # Create a custom collector that uses specific fund codes
    class TestFundCollector(AkShareCollectorCN1d):
        def _get_fund_list(self):
            # Return specific fund codes for testing
            test_funds = ['015198', '110022', '161725']
            logger.info(f"Using test fund list: {len(test_funds)} funds")
            return test_funds
    
    try:
        collector = TestFundCollector(
            save_dir="./test_fund_output",
            start="2024-01-01",
            end="2025-06-01",
            interval="1d",
            delay=2,
            limit_nums=3,
            data_type="fund"
        )
        
        collector.collector_data()
        print("✓ Collector test completed!")
        
        # Check output files
        output_dir = Path("./test_fund_output")
        if output_dir.exists():
            files = list(output_dir.glob("*.csv"))
            print(f"Generated {len(files)} files:")
            for f in files:
                print(f"  - {f.name}")
                # Show sample content
                df = pd.read_csv(f)
                print(f"    Shape: {df.shape}, Columns: {df.columns.tolist()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Collector test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Fund Data Collection Test")
    print("=" * 50)
    
    # Test 1: Direct fund data collection
    success1 = test_fund_data_collection()
    
    # Test 2: Collector with specific funds
    success2 = test_collector_with_specific_funds()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Direct fund collection: {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"Collector test: {'✓ PASS' if success2 else '✗ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Fund data collection is working.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
