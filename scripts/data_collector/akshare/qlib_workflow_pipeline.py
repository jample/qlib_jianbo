#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qlib-based Data Processing Workflow

This module implements a complete qlib-native workflow for processing Shanghai stock data:
1. Convert DuckDB data to qlib binary format
2. Use qlib's built-in Alpha158 handler for feature extraction
3. Follow qlib workflow patterns for data preparation and model training
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from loguru import logger
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Qlib imports - handle import path issues
import sys
import os
sys.path.insert(0, '/root/mycode/qlibjianbo')

try:
    import qlib
    from qlib.data import D
    from qlib.constant import REG_CN
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH
    from qlib.utils import init_instance_by_config
except ImportError as e:
    logger.error(f"Failed to import qlib: {e}")
    logger.error("Make sure you're running from the correct environment")
    sys.exit(1)

# Local imports
from duckdb_extractor import DuckDBExtractor


class QlibWorkflowPipeline:
    """Complete qlib-based workflow for Shanghai stock data processing"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize qlib workflow pipeline
        
        Args:
            config_path: Path to JSON configuration file
        """
        self.config = self._load_config(config_path)
        self.qlib_data_dir = Path(self.config.get('qlib_data_dir', 'scripts/data_collector/akshare/qlib_data'))
        self.output_dir = Path(self.config.get('output_dir', 'scripts/data_collector/akshare/qlib_output'))
        
        # Create directories
        self.qlib_data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized qlib workflow pipeline")
        logger.info(f"Qlib data directory: {self.qlib_data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from JSON file or use defaults"""
        default_config = {
            "data_type": "stock",
            "exchange_filter": "shanghai", 
            "interval": "1d",
            "symbol_filter": ["600", "601"],
            "date_range": {
                "start_date": "2022-01-01",
                "end_date": "2024-12-31"
            },
            "qlib_data_dir": "scripts/data_collector/akshare/qlib_data",
            "output_dir": "scripts/data_collector/akshare/qlib_output",
            "duckdb_path": "/root/autodl-tmp/code/duckdb/shanghai_stock_data.db",
            "instruments": "all",  # or specific list like ["600000.SH", "600036.SH"]
            "train_period": ["2022-01-01", "2023-12-31"],
            "valid_period": ["2024-01-01", "2024-06-30"], 
            "test_period": ["2024-07-01", "2024-12-31"],
            "model_config": {
                "class": "LGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": {
                    "loss": "mse",
                    "colsample_bytree": 0.8879,
                    "learning_rate": 0.0421,
                    "subsample": 0.8789,
                    "lambda_l1": 205.6999,
                    "lambda_l2": 580.9768,
                    "max_depth": 8,
                    "num_leaves": 210,
                    "num_threads": 20,
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.info("Using default configuration")
        
        return default_config
    
    def step1_convert_to_qlib_format(self) -> bool:
        """
        Step 1: Convert DuckDB data to qlib binary format
        
        Returns:
            bool: Success status
        """
        try:
            logger.info("Step 1: Converting DuckDB data to qlib binary format...")
            
            # Extract data from DuckDB
            extractor = DuckDBExtractor(
                db_path=self.config['duckdb_path'],
                data_type=self.config['data_type'],
                exchange_filter=self.config['exchange_filter'],
                interval=self.config['interval']
            )
            
            # Get data
            df = extractor.extract_stock_data(
                symbol_filter=self.config['symbol_filter'],
                start_date=self.config['date_range']['start_date'],
                end_date=self.config['date_range']['end_date']
            )
            
            if df.empty:
                logger.error("No data extracted from DuckDB")
                return False
            
            logger.info(f"Extracted {len(df)} records for {df['symbol'].nunique()} symbols")
            
            # Convert to qlib format
            success = self._convert_to_qlib_binary(df)
            
            if success:
                logger.info("✅ Step 1 completed: Data converted to qlib format")
                return True
            else:
                logger.error("❌ Step 1 failed: Data conversion failed")
                return False
                
        except Exception as e:
            logger.error(f"Step 1 failed with error: {e}")
            return False
    
    def _convert_to_qlib_binary(self, df: pd.DataFrame) -> bool:
        """Convert DataFrame to qlib binary format"""
        try:
            # Prepare data in qlib format
            # Qlib expects: datetime index, instrument columns, OHLCV data
            
            # Rename columns to qlib standard
            column_mapping = {
                'open': '$open',
                'high': '$high', 
                'low': '$low',
                'close': '$close',
                'volume': '$volume',
                'amount': '$amount'
            }
            
            df_qlib = df.copy()
            df_qlib = df_qlib.rename(columns=column_mapping)
            
            # Add symbol suffix for Shanghai stocks
            df_qlib['instrument'] = df_qlib['symbol'].apply(lambda x: f"{x}.SH")
            
            # Set datetime index
            df_qlib['date'] = pd.to_datetime(df_qlib['date'])
            df_qlib = df_qlib.set_index(['date', 'instrument'])
            
            # Select only price/volume columns
            price_cols = ['$open', '$high', '$low', '$close', '$volume']
            if '$amount' in df_qlib.columns:
                price_cols.append('$amount')
            
            df_qlib = df_qlib[price_cols]
            
            # Create qlib directory structure
            features_dir = self.qlib_data_dir / "features"
            instruments_dir = self.qlib_data_dir / "instruments" 
            calendars_dir = self.qlib_data_dir / "calendars"
            
            for dir_path in [features_dir, instruments_dir, calendars_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Save instruments list
            instruments = sorted(df_qlib.index.get_level_values('instrument').unique())
            instruments_file = instruments_dir / "all.txt"
            with open(instruments_file, 'w') as f:
                for inst in instruments:
                    f.write(f"{inst}\t{inst}\tstock\n")
            
            # Save calendar
            calendar = sorted(df_qlib.index.get_level_values('date').unique())
            calendar_file = calendars_dir / "day.txt"
            with open(calendar_file, 'w') as f:
                for date in calendar:
                    f.write(f"{date.strftime('%Y-%m-%d')}\n")
            
            # Save features for each instrument
            logger.info("Saving features to qlib binary format...")
            
            for instrument in instruments:
                inst_data = df_qlib.xs(instrument, level='instrument')
                
                # Create instrument directory
                inst_dir = features_dir / instrument.lower()
                inst_dir.mkdir(parents=True, exist_ok=True)
                
                # Save each feature as binary file
                for col in price_cols:
                    if col in inst_data.columns:
                        feature_file = inst_dir / f"{col[1:]}.day.bin"  # Remove $ prefix
                        
                        # Convert to numpy array and save
                        values = inst_data[col].values.astype(np.float32)
                        values.tofile(str(feature_file))
            
            logger.info(f"Saved qlib data for {len(instruments)} instruments")
            return True
            
        except Exception as e:
            logger.error(f"Error converting to qlib binary format: {e}")
            return False
    
    def step2_initialize_qlib(self) -> bool:
        """
        Step 2: Initialize qlib with converted data
        
        Returns:
            bool: Success status
        """
        try:
            logger.info("Step 2: Initializing qlib...")
            
            # Initialize qlib with our data
            qlib.init(
                provider_uri=str(self.qlib_data_dir),
                region=REG_CN,
                expression_cache=None,
                dataset_cache=None,
            )
            
            # Test qlib initialization
            calendar = D.calendar(
                start_time=self.config['date_range']['start_date'],
                end_time=self.config['date_range']['end_date'],
                freq='day'
            )
            
            instruments = D.list_instruments(
                instruments=D.instruments('all'),
                start_time=self.config['date_range']['start_date'],
                end_time=self.config['date_range']['end_date'],
                as_list=True
            )
            
            logger.info(f"✅ Step 2 completed: Qlib initialized successfully")
            logger.info(f"   Calendar: {len(calendar)} trading days")
            logger.info(f"   Instruments: {len(instruments)} stocks")
            
            return True
            
        except Exception as e:
            logger.error(f"Step 2 failed with error: {e}")
            return False

    def step3_create_alpha158_handler(self) -> Optional[Alpha158]:
        """
        Step 3: Create qlib Alpha158 data handler

        Returns:
            Alpha158 handler or None if failed
        """
        try:
            logger.info("Step 3: Creating Alpha158 data handler...")

            # Create Alpha158 handler with our configuration
            handler_config = {
                "instruments": self.config.get('instruments', 'all'),
                "start_time": self.config['date_range']['start_date'],
                "end_time": self.config['date_range']['end_date'],
                "freq": "day",
                "infer_processors": [
                    {"class": "ProcessInf", "kwargs": {}},
                    {"class": "ZScoreNorm", "kwargs": {}},
                    {"class": "Fillna", "kwargs": {}},
                ],
                "learn_processors": [
                    {"class": "DropnaLabel"},
                    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
                ],
                "fit_start_time": self.config['train_period'][0],
                "fit_end_time": self.config['train_period'][1],
            }

            # Initialize Alpha158 handler
            handler = Alpha158(**handler_config)

            # Test handler by fetching some data
            logger.info("Testing Alpha158 handler...")

            # Get feature columns
            feature_cols = handler.get_cols()
            logger.info(f"Alpha158 features: {len(feature_cols)} columns")

            # Fetch a small sample
            sample_data = handler.fetch(
                selector=slice("2024-01-01", "2024-01-10"),
                level="datetime"
            )

            logger.info(f"✅ Step 3 completed: Alpha158 handler created successfully")
            logger.info(f"   Features shape: {sample_data.shape}")
            logger.info(f"   Feature columns: {len(feature_cols)}")

            return handler

        except Exception as e:
            logger.error(f"Step 3 failed with error: {e}")
            return None

    def step4_create_dataset(self, handler: Alpha158) -> Optional[DatasetH]:
        """
        Step 4: Create qlib dataset with train/valid/test splits

        Args:
            handler: Alpha158 data handler

        Returns:
            DatasetH or None if failed
        """
        try:
            logger.info("Step 4: Creating qlib dataset...")

            # Create dataset configuration
            dataset_config = {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": handler,
                    "segments": {
                        "train": self.config['train_period'],
                        "valid": self.config['valid_period'],
                        "test": self.config['test_period'],
                    },
                },
            }

            # Initialize dataset
            dataset = init_instance_by_config(dataset_config)

            # Test dataset
            logger.info("Testing dataset splits...")

            train_data = dataset.prepare("train")
            valid_data = dataset.prepare("valid")
            test_data = dataset.prepare("test")

            logger.info(f"✅ Step 4 completed: Dataset created successfully")
            logger.info(f"   Train data: {train_data[0].shape} features, {train_data[1].shape} labels")
            logger.info(f"   Valid data: {valid_data[0].shape} features, {valid_data[1].shape} labels")
            logger.info(f"   Test data: {test_data[0].shape} features, {test_data[1].shape} labels")

            return dataset

        except Exception as e:
            logger.error(f"Step 4 failed with error: {e}")
            return None

    def step5_train_model(self, dataset: DatasetH) -> Optional[Any]:
        """
        Step 5: Train model using qlib workflow

        Args:
            dataset: Prepared dataset

        Returns:
            Trained model or None if failed
        """
        try:
            logger.info("Step 5: Training model...")

            # Initialize model
            model = init_instance_by_config(self.config['model_config'])

            # Prepare training data
            train_data = dataset.prepare("train")
            valid_data = dataset.prepare("valid")

            # Train model
            logger.info("Starting model training...")
            model.fit(train_data)

            # Validate model
            logger.info("Validating model...")
            valid_pred = model.predict(valid_data)

            # Calculate basic metrics
            valid_labels = valid_data[1]
            correlation = np.corrcoef(valid_pred.flatten(), valid_labels.flatten())[0, 1]

            logger.info(f"✅ Step 5 completed: Model trained successfully")
            logger.info(f"   Validation correlation: {correlation:.4f}")

            return model

        except Exception as e:
            logger.error(f"Step 5 failed with error: {e}")
            return None

    def step6_save_results(self, model: Any, dataset: DatasetH, handler: Alpha158) -> bool:
        """
        Step 6: Save model and results

        Args:
            model: Trained model
            dataset: Dataset
            handler: Data handler

        Returns:
            bool: Success status
        """
        try:
            logger.info("Step 6: Saving results...")

            # Create output directories
            model_dir = self.output_dir / "model"
            results_dir = self.output_dir / "results"

            for dir_path in [model_dir, results_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

            # Save model
            model_path = model_dir / "trained_model.pkl"
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # Save handler
            handler_path = model_dir / "alpha158_handler.pkl"
            handler.to_pickle(str(handler_path), dump_all=True)

            # Generate predictions for all splits
            results = {}
            for split in ['train', 'valid', 'test']:
                data = dataset.prepare(split)
                pred = model.predict(data)

                results[split] = {
                    'predictions': pred,
                    'labels': data[1],
                    'correlation': np.corrcoef(pred.flatten(), data[1].flatten())[0, 1]
                }

            # Save results
            results_path = results_dir / "predictions.pkl"
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)

            # Save summary
            summary = {
                'config': self.config,
                'model_performance': {
                    split: {'correlation': results[split]['correlation']}
                    for split in results.keys()
                },
                'feature_count': len(handler.get_cols()),
                'timestamp': datetime.now().isoformat()
            }

            summary_path = results_dir / "summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"✅ Step 6 completed: Results saved successfully")
            logger.info(f"   Model saved to: {model_path}")
            logger.info(f"   Results saved to: {results_path}")
            logger.info(f"   Summary saved to: {summary_path}")

            return True

        except Exception as e:
            logger.error(f"Step 6 failed with error: {e}")
            return False

    def run_complete_workflow(self) -> bool:
        """
        Run the complete qlib-based workflow

        Returns:
            bool: Success status
        """
        try:
            logger.info("🚀 Starting complete qlib workflow pipeline...")

            # Step 1: Convert data to qlib format
            if not self.step1_convert_to_qlib_format():
                return False

            # Step 2: Initialize qlib
            if not self.step2_initialize_qlib():
                return False

            # Step 3: Create Alpha158 handler
            handler = self.step3_create_alpha158_handler()
            if handler is None:
                return False

            # Step 4: Create dataset
            dataset = self.step4_create_dataset(handler)
            if dataset is None:
                return False

            # Step 5: Train model
            model = self.step5_train_model(dataset)
            if model is None:
                return False

            # Step 6: Save results
            if not self.step6_save_results(model, dataset, handler):
                return False

            logger.info("🎉 Complete qlib workflow pipeline finished successfully!")
            return True

        except Exception as e:
            logger.error(f"Workflow failed with error: {e}")
            return False


def main():
    """Main function to run the qlib workflow pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Qlib-based Data Processing Workflow")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--step", type=str, choices=['1', '2', '3', '4', '5', '6', 'all'],
                       default='all', help="Run specific step or all steps")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = QlibWorkflowPipeline(config_path=args.config)

    # Run specified step(s)
    if args.step == 'all':
        success = pipeline.run_complete_workflow()
    else:
        step_methods = {
            '1': pipeline.step1_convert_to_qlib_format,
            '2': pipeline.step2_initialize_qlib,
            '3': lambda: pipeline.step3_create_alpha158_handler() is not None,
            '4': lambda: pipeline.step4_create_dataset(None) is not None,  # Would need handler
            '5': lambda: pipeline.step5_train_model(None) is not None,     # Would need dataset
            '6': lambda: pipeline.step6_save_results(None, None, None),    # Would need all
        }

        if args.step in ['3', '4', '5', '6']:
            logger.warning(f"Step {args.step} requires previous steps to be completed. Running complete workflow...")
            success = pipeline.run_complete_workflow()
        else:
            success = step_methods[args.step]()

    if success:
        logger.info("✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
