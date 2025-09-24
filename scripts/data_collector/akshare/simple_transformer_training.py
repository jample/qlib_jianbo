#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Transformer model training with basic OHLCV features
"""

import sys
import os
import yaml
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np

# Add current directory and repository root to path
sys.path.append(str(Path(__file__).parent))
sys.path.append('/root/mycode/qlibjianbo')

try:
    import qlib
    from qlib import init
    from qlib.contrib.model.pytorch_transformer_ts import TransformerModel
    from qlib.data.dataset import TSDatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.utils import init_instance_by_config
    import torch
    
    DEPENDENCIES_OK = True
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    DEPENDENCIES_OK = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def initialize_qlib(config: dict):
    """Initialize qlib with configuration"""
    qlib_config = config['qlib_init']
    
    logger.info(f"Initializing qlib with provider: {qlib_config['provider_uri']}")
    
    init(
        provider_uri=qlib_config['provider_uri'],
        region=qlib_config['region']
    )
    
    logger.info("✅ Qlib initialized successfully")


def create_simple_dataset(config: dict):
    """Create dataset with basic OHLCV features"""
    
    logger.info("Creating simple dataset with basic features...")
    
    try:
        # Create a simple dataset configuration
        dataset_config = {
            "class": "TSDatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "DataHandlerLP",
                    "module_path": "qlib.data.dataset.handler",
                    "kwargs": {
                        "start_time": "2024-01-01",
                        "end_time": "2024-03-31",
                        "instruments": "all",
                        "fields": ["$open", "$high", "$low", "$close", "$volume"],
                        "label": ["Ref($close, -1)/$close - 1"]
                    }
                },
                "segments": {
                    "train": ["2024-01-01", "2024-02-29"],
                    "valid": ["2024-03-01", "2024-03-15"],
                    "test": ["2024-03-16", "2024-03-31"]
                },
                "step_len": 10
            }
        }
        
        # Create dataset instance
        dataset = init_instance_by_config(dataset_config)
        
        logger.info("✅ Simple dataset created successfully")
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_simple_model():
    """Create simple Transformer model"""
    
    logger.info("Creating simple Transformer model...")
    
    try:
        model_config = {
            "class": "TransformerModel",
            "module_path": "qlib.contrib.model.pytorch_transformer_ts",
            "kwargs": {
                "seed": 0,
                "n_jobs": 2,
                "d_feat": 5,  # 5 basic features: OHLCV
                "d_model": 32,
                "n_heads": 2,
                "num_layers": 2,
                "dropout": 0.1,
                "n_epochs": 5,
                "lr": 0.001,
                "metric": "loss",
                "batch_size": 32,
                "early_stop": 3,
                "loss": "mse",
                "optimizer": "adam"
            }
        }
        
        # Create model instance
        model = init_instance_by_config(model_config)
        
        logger.info("✅ Simple Transformer model created successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_model(model, dataset):
    """Train the model"""
    
    logger.info("Starting simple model training...")
    
    try:
        # Fit the model
        model.fit(dataset)
        
        logger.info("✅ Model training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def evaluate_model(model, dataset):
    """Evaluate the trained model"""
    
    logger.info("Evaluating model...")
    
    try:
        # Get predictions
        predictions = model.predict(dataset)
        
        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Predictions sample:\n{predictions.head()}")
        
        # Basic statistics
        logger.info(f"Prediction statistics:")
        logger.info(f"  Mean: {predictions.mean():.6f}")
        logger.info(f"  Std: {predictions.std():.6f}")
        logger.info(f"  Min: {predictions.min():.6f}")
        logger.info(f"  Max: {predictions.max():.6f}")
        
        logger.info("✅ Model evaluation completed")
        return predictions
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main training function"""
    
    if not DEPENDENCIES_OK:
        logger.error("❌ Missing required dependencies")
        return False
    
    logger.info("=" * 60)
    logger.info("SIMPLE TRANSFORMER MODEL TRAINING")
    logger.info("=" * 60)
    
    try:
        # Initialize qlib
        config = {"qlib_init": {"provider_uri": "scripts/data_collector/akshare/qlib_data", "region": "cn"}}
        initialize_qlib(config)
        
        # Create dataset
        dataset = create_simple_dataset(config)
        if dataset is None:
            return False
        
        # Create model
        model = create_simple_model()
        if model is None:
            return False
        
        # Train model
        success = train_model(model, dataset)
        
        if not success:
            logger.error("❌ Training failed")
            return False
        
        # Evaluate model
        predictions = evaluate_model(model, dataset)
        
        if predictions is not None:
            logger.info("🎉 Simple Transformer training and evaluation completed successfully!")
            return True
        else:
            logger.error("❌ Evaluation failed")
            return False
            
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
