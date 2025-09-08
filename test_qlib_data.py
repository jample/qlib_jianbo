#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify the collected Qlib data works correctly
"""

import sys
from pathlib import Path

def test_qlib_data(data_path="./qlib_shanghai_data/qlib_data"):
    """Test if the collected Qlib data works"""
    
    try:
        import qlib
        from qlib.data import D
        from qlib.constant import REG_CN
        
        print("Testing Qlib data...")
        print(f"Data path: {data_path}")
        
        # Initialize Qlib with our collected data
        qlib.init(provider_uri=data_path, region=REG_CN)
        print("✓ Qlib initialized successfully")
        
        # Test calendar
        calendar = D.calendar(start_time='2024-12-01', end_time='2024-12-10', freq='day')
        print(f"✓ Calendar loaded: {len(calendar)} trading days")
        print(f"  Trading days: {calendar.tolist()}")
        
        # Test instruments
        instruments = D.instruments('all')
        instrument_list = D.list_instruments(instruments=instruments, start_time='2024-12-01', end_time='2024-12-10', as_list=True)
        print(f"✓ Instruments loaded: {len(instrument_list)} stocks")
        print(f"  Sample instruments: {instrument_list[:5]}")
        
        # Test features for a specific stock
        features = D.features([instrument_list[0]], ['$close', '$volume', '$open', '$high', '$low'], 
                            start_time='2024-12-01', end_time='2024-12-10', freq='day')
        print(f"✓ Features loaded for {instrument_list[0]}:")
        print(features)
        
        # Test multiple stocks
        multi_features = D.features(instrument_list[:3], ['$close', '$volume'], 
                                  start_time='2024-12-01', end_time='2024-12-10', freq='day')
        print(f"✓ Multi-stock features loaded:")
        print(multi_features.head(10))
        
        print("\n" + "=" * 50)
        print("✓ ALL QLIB DATA TESTS PASSED!")
        print("The collected Shanghai stock data is working correctly with Qlib.")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"✗ Qlib data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 50)
    print("QLIB DATA VERIFICATION TEST")
    print("=" * 50)
    
    # Test the collected data
    success = test_qlib_data()
    
    if success:
        print("\nThe Shanghai stock data collection was successful!")
        print("You can now use this data for Qlib workflows.")
    else:
        print("\nThere was an issue with the collected data.")


if __name__ == "__main__":
    main()