#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the configurable data processing pipeline

This script tests the pipeline with different configurations to ensure
the data scope parameters work correctly.
"""

import sys
from pathlib import Path
from loguru import logger

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from main_pipeline import MainDataPipeline


def test_pipeline_configurations():
    """Test different pipeline configurations"""
    
    logger.info("Testing configurable data processing pipeline...")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Shanghai Stocks (Default)',
            'config': {
                'data_type': 'stock',
                'exchange_filter': 'shanghai',
                'symbol_filter': ['600'],  # Limit to 600xxx for testing
                'date_range': {
                    'start_date': '2024-01-01',
                    'end_date': '2024-01-31'
                }
            }
        },
        {
            'name': 'All Stocks with Symbol Filter',
            'config': {
                'data_type': 'stock',
                'exchange_filter': 'all',
                'symbol_filter': ['600', '000'],
                'date_range': {
                    'start_date': '2024-01-01',
                    'end_date': '2024-01-31'
                }
            }
        },
        {
            'name': 'Shenzhen Stocks',
            'config': {
                'data_type': 'stock',
                'exchange_filter': 'shenzhen',
                'symbol_filter': ['000'],
                'date_range': {
                    'start_date': '2024-01-01',
                    'end_date': '2024-01-31'
                }
            }
        }
    ]
    
    results = {}
    
    for test_case in test_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {test_case['name']}")
        logger.info(f"{'='*60}")
        
        try:
            # Initialize pipeline with test configuration
            pipeline = MainDataPipeline(test_case['config'])
            
            # Test database connection and info
            db_info = pipeline.extractor.get_database_info()
            logger.info(f"Database info: {db_info}")
            
            # Test symbol extraction
            symbols = pipeline.extractor.get_available_symbols(
                prefix_filter=test_case['config'].get('symbol_filter')
            )
            logger.info(f"Found {len(symbols)} symbols: {symbols[:10]}...")  # Show first 10
            
            # Test data extraction (small sample)
            if symbols:
                sample_data = pipeline.extractor.extract_stock_data(
                    symbols=symbols[:5],  # First 5 symbols
                    start_date=test_case['config']['date_range']['start_date'],
                    end_date=test_case['config']['date_range']['end_date'],
                    limit=100
                )
                logger.info(f"Sample data shape: {sample_data.shape}")
                logger.info(f"Sample data columns: {list(sample_data.columns)}")
                logger.info(f"Sample symbols in data: {sample_data['symbol'].unique()}")
                
                results[test_case['name']] = {
                    'status': 'SUCCESS',
                    'db_info': db_info,
                    'symbols_found': len(symbols),
                    'sample_data_shape': sample_data.shape,
                    'sample_symbols': list(sample_data['symbol'].unique())
                }
            else:
                results[test_case['name']] = {
                    'status': 'NO_DATA',
                    'db_info': db_info,
                    'symbols_found': 0
                }
                
        except Exception as e:
            logger.error(f"Error in test case '{test_case['name']}': {e}")
            results[test_case['name']] = {
                'status': 'ERROR',
                'error': str(e)
            }
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*80}")
    
    for test_name, result in results.items():
        status = result['status']
        logger.info(f"{test_name}: {status}")
        
        if status == 'SUCCESS':
            logger.info(f"  - Symbols found: {result['symbols_found']}")
            logger.info(f"  - Sample data shape: {result['sample_data_shape']}")
            logger.info(f"  - Sample symbols: {result['sample_symbols']}")
        elif status == 'NO_DATA':
            logger.info(f"  - No data found (symbols: {result['symbols_found']})")
        elif status == 'ERROR':
            logger.info(f"  - Error: {result['error']}")
    
    return results


def test_command_line_args():
    """Test command line argument parsing"""
    
    logger.info("\n" + "="*80)
    logger.info("TESTING COMMAND LINE ARGUMENTS")
    logger.info("="*80)
    
    # Test different command line configurations
    test_commands = [
        "--data-type stock --exchange shanghai --symbols 600",
        "--data-type stock --exchange all --symbols 600 000",
        "--data-type fund --db-path fund_data.duckdb",
        "--start-date 2024-01-01 --end-date 2024-01-31 --sequence-length 10"
    ]
    
    logger.info("Example command line usage:")
    for cmd in test_commands:
        logger.info(f"  python main_pipeline.py {cmd}")
    
    logger.info("\nFor full testing, run these commands manually to test argument parsing.")


def main():
    """Main test function"""
    
    logger.info("Starting configurable pipeline tests...")
    
    # Test 1: Different configurations
    config_results = test_pipeline_configurations()
    
    # Test 2: Command line arguments (informational)
    test_command_line_args()
    
    # Final summary
    success_count = sum(1 for result in config_results.values() if result['status'] == 'SUCCESS')
    total_count = len(config_results)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL RESULTS: {success_count}/{total_count} configurations successful")
    logger.info(f"{'='*80}")
    
    if success_count == total_count:
        logger.info("✅ All tests passed! The configurable pipeline is working correctly.")
    else:
        logger.warning(f"⚠️  {total_count - success_count} tests failed. Check the logs above for details.")
    
    return config_results


if __name__ == "__main__":
    main()
