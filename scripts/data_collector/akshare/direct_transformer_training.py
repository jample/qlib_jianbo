#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direct Transformer model training with Shanghai stock data
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
    from qlib.contrib.data.handler import Alpha158
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


def create_dataset(config: dict):
    """Create dataset for training"""
    
    logger.info("Creating dataset...")
    
    dataset_config = config['task']['dataset']
    
    # Create dataset instance
    dataset = init_instance_by_config(dataset_config)
    
    logger.info("✅ Dataset created successfully")
    return dataset


def create_model(config: dict):
    """Create Transformer model"""
    
    logger.info("Creating Transformer model...")
    
    model_config = config['task']['model']
    
    # Create model instance
    model = init_instance_by_config(model_config)
    
    logger.info("✅ Transformer model created successfully")
    return model


def train_model(model, dataset):
    """Train the model"""
    
    logger.info("Starting model training...")
    
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
    logger.info("DIRECT TRANSFORMER MODEL TRAINING")
    logger.info("=" * 60)
    
    config_path = "scripts/data_collector/akshare/workflow_config_transformer_shanghai.yaml"
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        
        # Initialize qlib
        initialize_qlib(config)
        
        # Create dataset
        dataset = create_dataset(config)
        
        # Create model
        model = create_model(config)
        
        # Train model
        success = train_model(model, dataset)
        
        if not success:
            logger.error("❌ Training failed")
            return False
        
        # Evaluate model
        predictions = evaluate_model(model, dataset)
        
        if predictions is not None:
            logger.info("🎉 Training and evaluation completed successfully!")
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
