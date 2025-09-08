#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify the environment and AkShare functionality
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import akshare as ak
        print(f"✓ AkShare version: {ak.__version__}")
    except ImportError as e:
        print(f"✗ AkShare import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas version: {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    # Test qlib import
    try:
        import qlib
        print(f"✓ Qlib version: {qlib.__version__}")
    except ImportError as e:
        print(f"✗ Qlib import failed: {e}")
        return False
    
    return True

def test_akshare_api():
    """Test basic AkShare API functionality"""
    print("\nTesting AkShare API...")
    
    try:
        import akshare as ak
        
        # Test getting stock list
        print("Testing stock list retrieval...")
        stock_list = ak.stock_info_a_code_name()
        print(f"✓ Retrieved {len(stock_list)} stocks")
        
        # Filter Shanghai stocks
        shanghai_stocks = stock_list[
            stock_list['code'].str.startswith(('60', '68', '90'))
        ]['code'].tolist()
        print(f"✓ Found {len(shanghai_stocks)} Shanghai stocks")
        print(f"  Sample: {shanghai_stocks[:5]}")
        
        # Test getting data for one stock
        print("Testing data retrieval for 600000...")
        data = ak.stock_zh_a_hist(
            symbol="600000",
            period="daily", 
            start_date="20241201",
            end_date="20241210",
            adjust="qfq"
        )
        print(f"✓ Retrieved {len(data)} records for 600000")
        print(f"  Columns: {list(data.columns)}")
        
        return True
        
    except Exception as e:
        print(f"✗ AkShare API test failed: {e}")
        return False

def test_qlib_collector():
    """Test if we can import the qlib collector"""
    print("\nTesting Qlib collector import...")
    
    try:
        # Add paths
        SCRIPT_DIR = Path(__file__).resolve().parent
        COLLECTOR_DIR = SCRIPT_DIR / "scripts" / "data_collector" / "akshare"
        sys.path.insert(0, str(COLLECTOR_DIR))
        sys.path.insert(0, str(SCRIPT_DIR / "scripts"))
        
        from collector import AkShareCollectorCN1d
        print("✓ Successfully imported AkShareCollectorCN1d")
        
        return True
        
    except Exception as e:
        print(f"✗ Qlib collector import failed: {e}")
        return False

def main():
    print("=" * 50)
    print("ENVIRONMENT TEST FOR SHANGHAI STOCK COLLECTION")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test AkShare API
    if not test_akshare_api():
        all_tests_passed = False
    
    # Test Qlib collector
    if not test_qlib_collector():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ ALL TESTS PASSED - Environment is ready!")
        print("You can now run: python collect_shanghai_stocks.py")
    else:
        print("✗ SOME TESTS FAILED - Please check the errors above")
    print("=" * 50)

if __name__ == "__main__":
    main()