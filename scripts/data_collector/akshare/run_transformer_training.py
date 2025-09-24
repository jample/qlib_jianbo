#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Transformer model training with Shanghai stock data
"""

import sys
import os
from pathlib import Path
from loguru import logger
import argparse

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from qlib_workflow_runner import QlibWorkflowRunner


def main():
    """Main function to run Transformer training"""
    
    parser = argparse.ArgumentParser(description="Run Transformer model training")
    parser.add_argument("--prepare-only", action="store_true", 
                       help="Only prepare data, don't run training")
    parser.add_argument("--config", type=str, 
                       default="scripts/data_collector/akshare/workflow_config_transformer_shanghai.yaml",
                       help="Path to workflow configuration file")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("TRANSFORMER MODEL TRAINING")
    logger.info("=" * 60)
    
    try:
        # Initialize workflow runner
        runner = QlibWorkflowRunner(config_path=args.config)
        
        # Step 1: Prepare data
        logger.info("Step 1: Preparing qlib data...")
        
        # Use small dataset for testing
        data_config = {
            "data_type": "stock",
            "exchange_filter": "shanghai",
            "symbol_filter": ["600000", "600036"],  # Just 2 stocks for testing
            "start_date": "2024-01-01",
            "end_date": "2024-03-31"  # 3 months of data
        }
        
        success = runner.step1_prepare_qlib_data(**data_config)
        
        if not success:
            logger.error("❌ Data preparation failed!")
            return False
        
        logger.info("✅ Data preparation completed!")
        
        if args.prepare_only:
            logger.info("Data preparation only mode - stopping here")
            return True
        
        # Step 2: Run Transformer training
        logger.info("Step 2: Running Transformer model training...")
        
        if not WORKFLOW_AVAILABLE:
            logger.error("❌ Qlib workflow components not available")
            logger.info("Missing dependencies. You can run qrun manually:")
            logger.info(f"  qrun {args.config}")
            return False
        
        success = runner.step2_run_qlib_workflow(experiment_name="transformer_shanghai")
        
        if success:
            logger.info("✅ Transformer training completed successfully!")
            logger.info("Check MLflow UI for training results")
            return True
        else:
            logger.error("❌ Transformer training failed!")
            return False
            
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """Check if required dependencies are available"""
    
    logger.info("Checking dependencies...")
    
    try:
        import torch
        logger.info(f"✓ PyTorch: {torch.__version__}")
    except ImportError:
        logger.error("✗ PyTorch not found - required for Transformer model")
        return False
    
    try:
        import qlib
        logger.info(f"✓ Qlib: {qlib.__version__}")
    except ImportError:
        logger.error("✗ Qlib not found")
        return False
    
    try:
        from qlib.contrib.model.pytorch_transformer_ts import TransformerModel
        logger.info("✓ Transformer model available")
    except ImportError as e:
        logger.error(f"✗ Transformer model not available: {e}")
        return False
    
    return True


if __name__ == "__main__":
    # Check dependencies first
    if not check_dependencies():
        logger.error("❌ Missing required dependencies")
        sys.exit(1)
    
    # Import WORKFLOW_AVAILABLE after dependency check
    try:
        from qlib_workflow_runner import WORKFLOW_AVAILABLE
    except ImportError:
        WORKFLOW_AVAILABLE = False
    
    success = main()
    sys.exit(0 if success else 1)
