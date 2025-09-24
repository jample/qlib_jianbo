#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for qlib-based workflow pipeline
"""

import sys
import os
from pathlib import Path
from loguru import logger

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from qlib_workflow_pipeline import QlibWorkflowPipeline


def test_qlib_workflow():
    """Test the complete qlib workflow"""
    
    logger.info("🧪 Testing qlib workflow pipeline...")
    
    try:
        # Initialize pipeline with test configuration
        config_path = "scripts/data_collector/akshare/qlib_config.json"
        pipeline = QlibWorkflowPipeline(config_path=config_path)
        
        # Test individual steps
        logger.info("Testing Step 1: Data conversion...")
        step1_success = pipeline.step1_convert_to_qlib_format()
        if not step1_success:
            logger.error("Step 1 failed")
            return False
        
        logger.info("Testing Step 2: Qlib initialization...")
        step2_success = pipeline.step2_initialize_qlib()
        if not step2_success:
            logger.error("Step 2 failed")
            return False
        
        logger.info("Testing Step 3: Alpha158 handler creation...")
        handler = pipeline.step3_create_alpha158_handler()
        if handler is None:
            logger.error("Step 3 failed")
            return False
        
        logger.info("Testing Step 4: Dataset creation...")
        dataset = pipeline.step4_create_dataset(handler)
        if dataset is None:
            logger.error("Step 4 failed")
            return False
        
        logger.info("Testing Step 5: Model training...")
        model = pipeline.step5_train_model(dataset)
        if model is None:
            logger.error("Step 5 failed")
            return False
        
        logger.info("Testing Step 6: Results saving...")
        step6_success = pipeline.step6_save_results(model, dataset, handler)
        if not step6_success:
            logger.error("Step 6 failed")
            return False
        
        logger.info("✅ All steps completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False


def test_small_dataset():
    """Test with a smaller dataset for quick validation"""
    
    logger.info("🧪 Testing with small dataset...")
    
    try:
        # Create a small test configuration
        small_config = {
            "data_type": "stock",
            "exchange_filter": "shanghai",
            "interval": "1d", 
            "symbol_filter": ["600000", "600036"],  # Just 2 stocks
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-03-31"  # Just 3 months
            },
            "qlib_data_dir": "scripts/data_collector/akshare/qlib_data_test",
            "output_dir": "scripts/data_collector/akshare/qlib_output_test",
            "duckdb_path": "/root/autodl-tmp/code/duckdb/shanghai_stock_data.db",
            "instruments": "all",
            "train_period": ["2024-01-01", "2024-02-29"],
            "valid_period": ["2024-03-01", "2024-03-15"],
            "test_period": ["2024-03-16", "2024-03-31"],
            "model_config": {
                "class": "LGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": {
                    "loss": "mse",
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "num_leaves": 10,
                    "num_threads": 4,
                }
            }
        }
        
        # Save small config
        import json
        small_config_path = "scripts/data_collector/akshare/qlib_config_small.json"
        with open(small_config_path, 'w') as f:
            json.dump(small_config, f, indent=2)
        
        # Run pipeline with small config
        pipeline = QlibWorkflowPipeline(config_path=small_config_path)
        success = pipeline.run_complete_workflow()
        
        if success:
            logger.info("✅ Small dataset test completed successfully!")
            return True
        else:
            logger.error("❌ Small dataset test failed!")
            return False
            
    except Exception as e:
        logger.error(f"Small dataset test failed with error: {e}")
        return False


def main():
    """Main test function"""
    
    logger.info("=" * 60)
    logger.info("QLIB WORKFLOW PIPELINE TESTING")
    logger.info("=" * 60)
    
    # Test 1: Small dataset (quick test)
    logger.info("\n📋 Test 1: Small Dataset Test")
    small_test_success = test_small_dataset()
    
    if small_test_success:
        logger.info("✅ Small dataset test passed!")
        
        # Test 2: Full workflow (if small test passes)
        logger.info("\n📋 Test 2: Full Workflow Test")
        full_test_success = test_qlib_workflow()
        
        if full_test_success:
            logger.info("🎉 All tests passed! Qlib workflow is working correctly.")
            return True
        else:
            logger.error("❌ Full workflow test failed!")
            return False
    else:
        logger.error("❌ Small dataset test failed! Skipping full test.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
