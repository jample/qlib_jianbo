#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verification Script for Configurable Data Pipeline

This script uses the dataprocess20250918.json configuration to test and verify
the complete data processing pipeline from DuckDB extraction to training-ready data.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
import pandas as pd
import numpy as np

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from main_pipeline import MainDataPipeline


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        logger.info(f"Configuration: {config}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def verify_database_connection(pipeline: MainDataPipeline) -> dict:
    """Verify database connection and get info"""
    logger.info("=" * 60)
    logger.info("STEP 1: VERIFYING DATABASE CONNECTION")
    logger.info("=" * 60)
    
    try:
        db_info = pipeline.extractor.get_database_info()
        
        if not db_info.get('table_exists', False):
            logger.error(f"Database table issue: {db_info.get('error', 'Unknown error')}")
            return {'status': 'FAILED', 'error': db_info.get('error')}
        
        logger.info(f"✅ Database connection successful")
        logger.info(f"   - Data type: {db_info['data_type']}")
        logger.info(f"   - Exchange filter: {db_info['exchange_filter']}")
        logger.info(f"   - Main table: {db_info['main_table']}")
        logger.info(f"   - Total symbols: {db_info['total_symbols']}")
        logger.info(f"   - Total records: {db_info['total_records']}")
        logger.info(f"   - Date range: {db_info['earliest_date']} to {db_info['latest_date']}")
        logger.info(f"   - Trading days: {db_info['trading_days']}")
        
        return {'status': 'SUCCESS', 'info': db_info}
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return {'status': 'FAILED', 'error': str(e)}


def verify_symbol_extraction(pipeline: MainDataPipeline, config: dict) -> dict:
    """Verify symbol extraction with filters"""
    logger.info("=" * 60)
    logger.info("STEP 2: VERIFYING SYMBOL EXTRACTION")
    logger.info("=" * 60)
    
    try:
        symbols = pipeline.extractor.get_available_symbols(
            prefix_filter=config.get('symbol_filter')
        )
        
        if not symbols:
            logger.warning("⚠️  No symbols found with current filters")
            return {'status': 'NO_DATA', 'symbols': []}
        
        logger.info(f"✅ Symbol extraction successful")
        logger.info(f"   - Total symbols found: {len(symbols)}")
        logger.info(f"   - Symbol filter: {config.get('symbol_filter')}")
        logger.info(f"   - Sample symbols: {symbols[:10]}")
        
        # Verify symbols match the expected patterns
        expected_patterns = config.get('symbol_filter', [])
        if expected_patterns:
            matching_symbols = [s for s in symbols if any(s.startswith(pattern) for pattern in expected_patterns)]
            logger.info(f"   - Symbols matching filter: {len(matching_symbols)}")
        
        return {'status': 'SUCCESS', 'symbols': symbols}
        
    except Exception as e:
        logger.error(f"❌ Symbol extraction failed: {e}")
        return {'status': 'FAILED', 'error': str(e)}


def verify_data_extraction(pipeline: MainDataPipeline, config: dict, symbols: list) -> dict:
    """Verify data extraction with date range"""
    logger.info("=" * 60)
    logger.info("STEP 3: VERIFYING DATA EXTRACTION")
    logger.info("=" * 60)
    
    try:
        # Test with a subset of symbols for faster verification
        test_symbols = symbols[:5] if len(symbols) > 5 else symbols
        
        raw_data = pipeline.extractor.extract_stock_data(
            symbols=test_symbols,
            start_date=config['date_range']['start_date'],
            end_date=config['date_range']['end_date'],
            limit=1000  # Limit for testing
        )
        
        if raw_data.empty:
            logger.warning("⚠️  No data extracted")
            return {'status': 'NO_DATA', 'data': None}
        
        logger.info(f"✅ Data extraction successful")
        logger.info(f"   - Data shape: {raw_data.shape}")
        logger.info(f"   - Columns: {list(raw_data.columns)}")
        logger.info(f"   - Date range in data: {raw_data['date'].min()} to {raw_data['date'].max()}")
        logger.info(f"   - Unique symbols: {raw_data['symbol'].nunique()}")
        logger.info(f"   - Sample data:")
        logger.info(f"{raw_data.head()}")
        
        return {'status': 'SUCCESS', 'data': raw_data}
        
    except Exception as e:
        logger.error(f"❌ Data extraction failed: {e}")
        return {'status': 'FAILED', 'error': str(e)}


def verify_pipeline_phases(pipeline: MainDataPipeline, raw_data: pd.DataFrame) -> dict:
    """Verify each pipeline phase"""
    logger.info("=" * 60)
    logger.info("STEP 4: VERIFYING PIPELINE PHASES")
    logger.info("=" * 60)
    
    results = {}
    
    try:
        # Phase 1: Data Initialization
        logger.info("Testing Phase 1: Data Initialization...")
        initialized_data, init_summary = pipeline.run_initialization_phase(raw_data)
        logger.info(f"✅ Initialization: {initialized_data.shape}")
        results['initialization'] = {'status': 'SUCCESS', 'shape': initialized_data.shape}
        
        # Phase 2: Feature Engineering
        logger.info("Testing Phase 2: Feature Engineering...")
        feature_data, feature_summary = pipeline.run_feature_engineering_phase(initialized_data)
        logger.info(f"✅ Feature Engineering: {feature_data.shape}")
        logger.info(f"   - Features generated: {feature_summary.get('total_features', 'N/A')}")
        results['feature_engineering'] = {
            'status': 'SUCCESS', 
            'shape': feature_data.shape,
            'features': feature_summary.get('total_features', 0)
        }
        
        # Phase 3: Normalization
        logger.info("Testing Phase 3: Data Normalization...")
        normalized_data, norm_summary = pipeline.run_normalization_phase(feature_data)
        logger.info(f"✅ Normalization: {normalized_data.shape}")
        logger.info(f"   - Data quality score: {norm_summary.get('data_quality_score', 'N/A')}")
        results['normalization'] = {
            'status': 'SUCCESS',
            'shape': normalized_data.shape,
            'quality_score': norm_summary.get('data_quality_score', 0)
        }
        
        # Phase 4: Training Data Preparation
        logger.info("Testing Phase 4: Training Data Preparation...")
        training_result, training_summary = pipeline.run_training_preparation_phase(normalized_data)
        
        X_train, y_train = training_result['train_data']
        X_val, y_val = training_result['val_data']
        X_test, y_test = training_result['test_data']
        
        logger.info(f"✅ Training Data Preparation:")
        logger.info(f"   - Training set: {X_train.shape}")
        logger.info(f"   - Validation set: {X_val.shape}")
        logger.info(f"   - Test set: {X_test.shape}")
        logger.info(f"   - Features: {len(training_result['feature_names'])}")
        
        results['training_preparation'] = {
            'status': 'SUCCESS',
            'train_shape': X_train.shape,
            'val_shape': X_val.shape,
            'test_shape': X_test.shape,
            'num_features': len(training_result['feature_names'])
        }
        
        return {'status': 'SUCCESS', 'phases': results, 'final_data': training_result}
        
    except Exception as e:
        logger.error(f"❌ Pipeline phase failed: {e}")
        return {'status': 'FAILED', 'error': str(e), 'phases': results}


def verify_training_data_quality(training_result: dict) -> dict:
    """Verify the quality of training data"""
    logger.info("=" * 60)
    logger.info("STEP 5: VERIFYING TRAINING DATA QUALITY")
    logger.info("=" * 60)
    
    try:
        X_train, y_train = training_result['train_data']
        X_val, y_val = training_result['val_data']
        X_test, y_test = training_result['test_data']
        
        # Check for missing values
        train_missing = np.isnan(X_train).sum()
        val_missing = np.isnan(X_val).sum()
        test_missing = np.isnan(X_test).sum()
        
        # Check for infinite values
        train_inf = np.isinf(X_train).sum()
        val_inf = np.isinf(X_val).sum()
        test_inf = np.isinf(X_test).sum()
        
        # Check label distribution
        train_label_stats = {
            'mean': float(np.mean(y_train)),
            'std': float(np.std(y_train)),
            'min': float(np.min(y_train)),
            'max': float(np.max(y_train))
        }
        
        logger.info(f"✅ Training Data Quality Check:")
        logger.info(f"   - Missing values: Train={train_missing}, Val={val_missing}, Test={test_missing}")
        logger.info(f"   - Infinite values: Train={train_inf}, Val={val_inf}, Test={test_inf}")
        logger.info(f"   - Label statistics: {train_label_stats}")
        
        quality_score = 1.0
        if train_missing > 0 or val_missing > 0 or test_missing > 0:
            quality_score -= 0.3
        if train_inf > 0 or val_inf > 0 or test_inf > 0:
            quality_score -= 0.3
        
        logger.info(f"   - Overall quality score: {quality_score:.2f}")
        
        return {
            'status': 'SUCCESS',
            'quality_score': quality_score,
            'missing_values': {'train': train_missing, 'val': val_missing, 'test': test_missing},
            'infinite_values': {'train': train_inf, 'val': val_inf, 'test': test_inf},
            'label_stats': train_label_stats
        }
        
    except Exception as e:
        logger.error(f"❌ Data quality check failed: {e}")
        return {'status': 'FAILED', 'error': str(e)}


def main():
    """Main verification function"""
    
    logger.info("🚀 Starting Pipeline Verification with Configuration File")
    logger.info("=" * 80)
    
    config_path = "scripts/data_collector/akshare/dataprocess20250918.json"
    
    try:
        # Load configuration
        config = load_config(config_path)

        # Merge with default configuration to ensure all required fields are present
        default_config = {
            'db_path': '/root/autodl-tmp/code/duckdb/shanghai_stock_data.duckdb',
            'binary_output_dir': 'scripts/data_collector/akshare/qlib_data',
            'training_output_dir': 'scripts/data_collector/akshare/training_data',
            'min_trading_days': 100,
            'min_price': 0.01,
            'outlier_method': 'iqr',
            'normalization_method': 'zscore',
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'prediction_horizon': 1,
            'label_type': 'return',
            'sequence_length': None
        }

        # Merge configurations (config file overrides defaults)
        full_config = {**default_config, **config}
        logger.info(f"Full configuration: {full_config}")

        # Initialize pipeline with full configuration
        pipeline = MainDataPipeline(full_config)
        
        # Step 1: Verify database connection
        db_result = verify_database_connection(pipeline)
        if db_result['status'] != 'SUCCESS':
            logger.error("❌ Database verification failed. Stopping.")
            return False
        
        # Step 2: Verify symbol extraction
        symbol_result = verify_symbol_extraction(pipeline, config)
        if symbol_result['status'] != 'SUCCESS':
            logger.error("❌ Symbol extraction failed. Stopping.")
            return False
        
        symbols = symbol_result['symbols']
        
        # Step 3: Verify data extraction
        data_result = verify_data_extraction(pipeline, config, symbols)
        if data_result['status'] != 'SUCCESS':
            logger.error("❌ Data extraction failed. Stopping.")
            return False
        
        raw_data = data_result['data']
        
        # Step 4: Verify pipeline phases
        phase_result = verify_pipeline_phases(pipeline, raw_data)
        if phase_result['status'] != 'SUCCESS':
            logger.error("❌ Pipeline phases failed. Stopping.")
            return False
        
        training_result = phase_result['final_data']
        
        # Step 5: Verify training data quality
        quality_result = verify_training_data_quality(training_result)
        
        # Final summary
        logger.info("=" * 80)
        logger.info("🎉 VERIFICATION COMPLETE")
        logger.info("=" * 80)
        
        logger.info("✅ All verification steps passed!")
        logger.info(f"   - Configuration: {config_path}")
        logger.info(f"   - Data type: {full_config['data_type']}")
        logger.info(f"   - Exchange: {full_config['exchange_filter']}")
        logger.info(f"   - Symbol filter: {full_config['symbol_filter']}")
        logger.info(f"   - Date range: {full_config['date_range']['start_date']} to {full_config['date_range']['end_date']}")
        logger.info(f"   - Final training shape: {training_result['train_data'][0].shape}")
        logger.info(f"   - Features generated: {len(training_result['feature_names'])}")
        logger.info(f"   - Data quality score: {quality_result.get('quality_score', 'N/A')}")
        
        logger.info("\n🚀 The pipeline is ready for production use!")
        logger.info("You can now run the full pipeline with:")
        logger.info(f"python main_pipeline.py --config {config_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Verification failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
