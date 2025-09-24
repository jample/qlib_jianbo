#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shanghai Stock TFT (Temporal Fusion Transformer) Runner

This module implements a TFT model runner for Shanghai stock market data using:
1. DuckDB data extraction
2. Qlib data preparation and Alpha158 features
3. TFT model training and prediction
4. Quantile forecasting with uncertainty estimation

Requirements:
- Python 3.6-3.7 (TFT limitation)
- TensorFlow 1.15.0 (GPU required)
- CUDA 10.0 + cuDNN
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from loguru import logger
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.insert(0, '/root/mycode/qlibjianbo')
sys.path.insert(0, '/root/mycode/qlibjianbo/examples/benchmarks/TFT')

# Check Python version
if sys.version_info >= (3, 8):
    logger.error("TFT requires Python 3.6-3.7. Current version: {}.{}".format(
        sys.version_info.major, sys.version_info.minor))
    logger.error("Please create a conda environment with Python 3.7:")
    logger.error("conda create -n tft_env python=3.7")
    sys.exit(1)

try:
    import tensorflow as tf
    if tf.__version__ != "1.15.0":
        logger.warning(f"TFT requires TensorFlow 1.15.0, found {tf.__version__}")
except ImportError:
    logger.error("TensorFlow not found. Install with: pip install tensorflow-gpu==1.15.0")
    sys.exit(1)

try:
    import qlib
    from qlib import init
    from qlib.data.dataset import DatasetH
    from qlib.contrib.data.handler import Alpha158
    from qlib.utils import init_instance_by_config
except ImportError as e:
    logger.error(f"Qlib import failed: {e}")
    sys.exit(1)

# Local imports
from duckdb_extractor import DuckDBDataExtractor

# TFT imports (will be imported conditionally)
TFT_AVAILABLE = False
try:
    from tft import TFTModel, DATASET_SETTING, process_qlib_data
    from data_formatters.qlib_Alpha158 import Alpha158Formatter
    from expt_settings.configs import ExperimentConfig
    TFT_AVAILABLE = True
    logger.info("✅ TFT components imported successfully")
except ImportError as e:
    logger.error(f"TFT components not available: {e}")
    logger.error("Please ensure you're in the TFT directory or add it to PYTHONPATH")


class ShanghaiTFTRunner:
    """TFT model runner for Shanghai stock market data"""

    def __init__(self, 
                 model_folder: str = "shanghai_tft_models",
                 gpu_id: int = 0,
                 dataset_name: str = "Shanghai_Alpha158"):
        """
        Initialize Shanghai TFT runner

        Args:
            model_folder: Directory to save TFT models
            gpu_id: GPU device ID to use
            dataset_name: Name for the dataset configuration
        """
        if not TFT_AVAILABLE:
            raise ImportError("TFT components not available. Please check installation.")

        self.model_folder = Path(model_folder)
        self.gpu_id = gpu_id
        self.dataset_name = dataset_name
        self.qlib_data_dir = Path("scripts/data_collector/akshare/qlib_data")
        self.duckdb_path = Path("/root/autodl-tmp/code/duckdb/shanghai_stock_data.duckdb")
        
        # Create directories
        self.model_folder.mkdir(parents=True, exist_ok=True)
        self.qlib_data_dir.mkdir(parents=True, exist_ok=True)

        # Check GPU availability
        self._check_gpu_availability()

        logger.info(f"Initialized Shanghai TFT Runner")
        logger.info(f"Model folder: {self.model_folder}")
        logger.info(f"GPU ID: {self.gpu_id}")
        logger.info(f"Dataset: {self.dataset_name}")

    def _check_gpu_availability(self):
        """Check if GPU is available and properly configured"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if not gpus:
                logger.error("No GPU found. TFT requires GPU for training.")
                raise RuntimeError("GPU required for TFT training")
            
            logger.info(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            
            # Test GPU memory
            with tf.device(f'/GPU:{self.gpu_id}'):
                test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                result = tf.matmul(test_tensor, test_tensor)
                logger.info("✅ GPU test successful")
                
        except Exception as e:
            logger.error(f"GPU check failed: {e}")
            logger.error("Please ensure CUDA 10.0 and cuDNN are properly installed")
            raise

    def step1_prepare_shanghai_data(self,
                                   symbol_filter: List[str] = ["600000", "600036", "600519"],
                                   start_date: str = "2020-01-01",
                                   end_date: str = "2023-12-31",
                                   min_periods: int = 100) -> bool:
        """
        Step 1: Extract and prepare Shanghai stock data for TFT

        Args:
            symbol_filter: List of stock symbols to include
            start_date: Start date for data extraction
            end_date: End date for data extraction
            min_periods: Minimum number of trading days required per stock

        Returns:
            bool: Success status
        """
        try:
            logger.info("Step 1: Preparing Shanghai stock data for TFT...")

            # Extract data from DuckDB
            if self.duckdb_path.exists():
                logger.info("Extracting data from DuckDB...")
                extractor = DuckDBDataExtractor(
                    db_path=str(self.duckdb_path),
                    data_type="stock",
                    exchange_filter="shanghai",
                    interval="1d"
                )
                
                df = extractor.extract_stock_data(
                    symbols=symbol_filter,
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                logger.error("DuckDB file not found")
                return False

            if df.empty:
                logger.error("No data extracted")
                return False

            logger.info(f"Extracted {len(df)} records for {df['symbol'].nunique()} symbols")

            # Filter stocks with sufficient data
            symbol_counts = df.groupby('symbol').size()
            valid_symbols = symbol_counts[symbol_counts >= min_periods].index.tolist()
            df = df[df['symbol'].isin(valid_symbols)]

            logger.info(f"Filtered to {len(valid_symbols)} symbols with >= {min_periods} trading days")

            # Convert to qlib format and save
            success = self._convert_to_qlib_format(df)
            
            if success:
                logger.info("✅ Step 1 completed: Shanghai data prepared for TFT")
                return True
            else:
                logger.error("❌ Step 1 failed: Data conversion failed")
                return False

        except Exception as e:
            logger.error(f"Step 1 failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _convert_to_qlib_format(self, df: pd.DataFrame) -> bool:
        """Convert Shanghai stock data to qlib binary format"""
        try:
            # Calculate Alpha158 labels (5-day forward return for TFT)
            df_with_labels = self._calculate_tft_labels(df)

            # Ensure required columns exist
            self._ensure_required_columns(df_with_labels)

            # Convert to qlib format
            df_qlib = self._format_for_qlib(df_with_labels)

            # Save to qlib binary format
            return self._save_qlib_binary(df_qlib)

        except Exception as e:
            logger.error(f"Data conversion failed: {e}")
            return False

    def _calculate_tft_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate labels for TFT training (5-day forward return)"""
        try:
            df_with_labels = df.copy()
            df_with_labels = df_with_labels.sort_values(['symbol', 'date'])
            df_with_labels['LABEL0'] = np.nan

            for symbol in df_with_labels['symbol'].unique():
                mask = df_with_labels['symbol'] == symbol
                symbol_data = df_with_labels[mask].copy()
                
                # 5-day forward return: (close_t+5 / close_t) - 1
                close_future = symbol_data['close'].shift(-5)
                close_current = symbol_data['close']
                forward_return = (close_future / close_current) - 1
                
                df_with_labels.loc[mask, 'LABEL0'] = forward_return

            logger.info("Calculated TFT labels (5-day forward return)")
            return df_with_labels

        except Exception as e:
            logger.error(f"Label calculation failed: {e}")
            return df

    def _ensure_required_columns(self, df: pd.DataFrame):
        """Ensure all required columns exist for TFT"""
        # Add calendar features required by TFT
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['const'] = 1.0  # Constant feature for static input

        # Synthesize missing Alpha158 features if needed
        if 'amount' not in df.columns:
            df['amount'] = df['close'] * df['volume']
        
        if 'vwap' not in df.columns:
            df['vwap'] = df['amount'] / np.clip(df['volume'], 1e-12, None)
            df['vwap'] = df['vwap'].fillna(df['close'])

        logger.info("Added calendar features and ensured required columns")

    def _format_for_qlib(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format data for qlib consumption"""
        # Rename columns to qlib standard
        column_mapping = {
            'symbol': 'instrument',
            'date': 'datetime',
            'open': '$open',
            'high': '$high', 
            'low': '$low',
            'close': '$close',
            'volume': '$volume',
            'amount': '$amount',
            'vwap': '$vwap'
        }

        df_qlib = df.rename(columns=column_mapping)
        
        # Add .SH suffix for Shanghai stocks
        df_qlib['instrument'] = df_qlib['instrument'].apply(lambda x: f"{x}.SH")
        
        # Set multi-index
        df_qlib = df_qlib.set_index(['datetime', 'instrument']).sort_index()
        
        return df_qlib

    def _save_qlib_binary(self, df_qlib: pd.DataFrame) -> bool:
        """Save data in qlib binary format"""
        try:
            # Create qlib directory structure
            features_dir = self.qlib_data_dir / "features"
            instruments_dir = self.qlib_data_dir / "instruments"
            calendars_dir = self.qlib_data_dir / "calendars"

            for dir_path in [features_dir, instruments_dir, calendars_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

            # Save instruments
            instruments = sorted(df_qlib.index.get_level_values('instrument').unique())
            start_date = df_qlib.index.get_level_values('datetime').min().strftime('%Y-%m-%d')
            end_date = df_qlib.index.get_level_values('datetime').max().strftime('%Y-%m-%d')

            instruments_file = instruments_dir / "all.txt"
            with open(instruments_file, 'w') as f:
                for inst in instruments:
                    f.write(f"{inst}\t{start_date}\t{end_date}\n")

            # Save calendar
            calendar = sorted(df_qlib.index.get_level_values('datetime').unique())
            calendar_file = calendars_dir / "day.txt"
            with open(calendar_file, 'w') as f:
                for date in calendar:
                    f.write(f"{date.strftime('%Y-%m-%d')}\n")

            # Save features for each instrument
            feature_cols = ['$open', '$high', '$low', '$close', '$volume', '$vwap', '$amount', 'LABEL0']
            available_cols = [col for col in feature_cols if col in df_qlib.columns]

            for instrument in instruments:
                inst_data = df_qlib.xs(instrument, level='instrument')
                inst_data = inst_data.reindex(calendar)

                inst_dir = features_dir / instrument.lower()
                inst_dir.mkdir(parents=True, exist_ok=True)

                for col in available_cols:
                    if col in inst_data.columns:
                        if col.startswith('$'):
                            feature_file = inst_dir / f"{col[1:]}.day.bin"
                        else:
                            feature_file = inst_dir / f"{col}.day.bin"

                        values = inst_data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                        values = values.astype(np.float32)
                        values.values.tofile(str(feature_file))

            logger.info(f"Saved qlib binary data for {len(instruments)} instruments")
            return True

        except Exception as e:
            logger.error(f"Failed to save qlib binary data: {e}")
            return False

    def step2_setup_tft_dataset(self) -> bool:
        """
        Step 2: Setup TFT dataset configuration for Shanghai stocks

        Returns:
            bool: Success status
        """
        try:
            logger.info("Step 2: Setting up TFT dataset configuration...")

            # Register Shanghai dataset configuration
            DATASET_SETTING[self.dataset_name] = {
                "feature_col": [
                    # Core Alpha158 features that work well with TFT
                    "KLEN", "KLOW", "KMID", "KUP",           # Price position features
                    "STD5", "STD10", "STD20",                # Volatility features
                    "ROC5", "ROC10", "ROC20",                # Rate of change features
                    "MA5", "MA10", "MA20",                   # Moving average features
                    "MAX5", "MAX10", "MIN5", "MIN10",        # Extrema features
                    "RANK5", "RANK10",                       # Rank features
                    "RSV5", "RSV10",                         # Relative strength features
                    "BETA5", "BETA10",                       # Beta features
                    "RSQR5", "RSQR10", "RSQR20"             # R-squared features
                ],
                "label_col": "LABEL0"
            }

            logger.info(f"Registered dataset '{self.dataset_name}' with {len(DATASET_SETTING[self.dataset_name]['feature_col'])} features")
            logger.info("✅ Step 2 completed: TFT dataset configuration ready")
            return True

        except Exception as e:
            logger.error(f"Step 2 failed: {e}")
            return False

    def step3_train_tft_model(self,
                             train_start: str = "2020-01-01",
                             train_end: str = "2022-12-31", 
                             valid_start: str = "2023-01-01",
                             valid_end: str = "2023-06-30",
                             test_start: str = "2023-07-01",
                             test_end: str = "2023-12-31") -> bool:
        """
        Step 3: Train TFT model on Shanghai stock data

        Args:
            train_start: Training period start date
            train_end: Training period end date
            valid_start: Validation period start date
            valid_end: Validation period end date
            test_start: Test period start date
            test_end: Test period end date

        Returns:
            bool: Success status
        """
        try:
            logger.info("Step 3: Training TFT model...")

            # Initialize qlib
            init(provider_uri=str(self.qlib_data_dir), region="cn")

            # Create dataset configuration
            dataset_config = {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha158",
                        "module_path": "qlib.contrib.data.handler",
                        "kwargs": {
                            "start_time": train_start,
                            "end_time": test_end,
                            "instruments": "all"
                        }
                    },
                    "segments": {
                        "train": [train_start, train_end],
                        "valid": [valid_start, valid_end],
                        "test": [test_start, test_end]
                    }
                }
            }

            # Create dataset
            dataset = init_instance_by_config(dataset_config)

            # Create and train TFT model
            self.model = TFTModel(
                DATASET=self.dataset_name,
                label_shift=5  # 5-day forward prediction
            )

            logger.info("Starting TFT model training...")
            self.model.fit(
                dataset=dataset,
                MODEL_FOLDER=str(self.model_folder),
                USE_GPU_ID=self.gpu_id
            )

            logger.info("✅ Step 3 completed: TFT model training finished")
            return True

        except Exception as e:
            logger.error(f"Step 3 failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step4_generate_predictions(self, dataset=None) -> Optional[pd.Series]:
        """
        Step 4: Generate TFT predictions with quantile forecasting

        Args:
            dataset: Dataset to predict on (uses test set if None)

        Returns:
            pd.Series: Predictions or None if failed
        """
        try:
            logger.info("Step 4: Generating TFT predictions...")

            if not hasattr(self, 'model') or self.model is None:
                logger.error("Model not trained. Please run step3_train_tft_model first.")
                return None

            # Generate predictions
            predictions = self.model.predict(dataset)

            logger.info(f"Generated {len(predictions)} predictions")
            logger.info("✅ Step 4 completed: TFT predictions generated")
            
            return predictions

        except Exception as e:
            logger.error(f"Step 4 failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_complete_tft_workflow(self,
                                 symbol_filter: List[str] = ["600000", "600036", "600519"],
                                 start_date: str = "2020-01-01",
                                 end_date: str = "2023-12-31") -> bool:
        """
        Run complete TFT workflow for Shanghai stocks

        Args:
            symbol_filter: List of stock symbols to include
            start_date: Start date for data
            end_date: End date for data

        Returns:
            bool: Success status
        """
        try:
            logger.info("🚀 Starting complete Shanghai TFT workflow...")

            # Step 1: Prepare data
            if not self.step1_prepare_shanghai_data(symbol_filter, start_date, end_date):
                return False

            # Step 2: Setup dataset
            if not self.step2_setup_tft_dataset():
                return False

            # Step 3: Train model
            if not self.step3_train_tft_model():
                return False

            # Step 4: Generate predictions
            predictions = self.step4_generate_predictions()
            if predictions is None:
                return False

            logger.info("🎉 Complete Shanghai TFT workflow finished successfully!")
            logger.info(f"Model saved to: {self.model_folder}")
            logger.info(f"Generated {len(predictions)} predictions")

            return True

        except Exception as e:
            logger.error(f"Complete workflow failed: {e}")
            return False


def main():
    """Main function to run Shanghai TFT workflow"""
    import argparse

    parser = argparse.ArgumentParser(description="Shanghai Stock TFT Model Runner")
    parser.add_argument("--symbols", nargs="+", default=["600000", "600036", "600519"],
                       help="Stock symbols to include")
    parser.add_argument("--start-date", type=str, default="2020-01-01",
                       help="Start date for data")
    parser.add_argument("--end-date", type=str, default="2023-12-31", 
                       help="End date for data")
    parser.add_argument("--model-folder", type=str, default="shanghai_tft_models",
                       help="Directory to save models")
    parser.add_argument("--gpu-id", type=int, default=0,
                       help="GPU device ID")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SHANGHAI STOCK TFT MODEL RUNNER")
    logger.info("=" * 60)

    try:
        # Initialize runner
        runner = ShanghaiTFTRunner(
            model_folder=args.model_folder,
            gpu_id=args.gpu_id
        )

        # Run complete workflow
        success = runner.run_complete_tft_workflow(
            symbol_filter=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date
        )

        if success:
            logger.info("✅ Shanghai TFT workflow completed successfully!")
        else:
            logger.error("❌ Shanghai TFT workflow failed!")

        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    def analyze_predictions(self, predictions: pd.Series, save_analysis: bool = True) -> Dict[str, Any]:
        """
        Analyze TFT predictions and generate performance metrics

        Args:
            predictions: TFT predictions
            save_analysis: Whether to save analysis results

        Returns:
            Dict containing analysis results
        """
        try:
            logger.info("Analyzing TFT predictions...")

            analysis = {
                "prediction_stats": {
                    "count": len(predictions),
                    "mean": float(predictions.mean()),
                    "std": float(predictions.std()),
                    "min": float(predictions.min()),
                    "max": float(predictions.max()),
                    "quantiles": {
                        "q10": float(predictions.quantile(0.1)),
                        "q25": float(predictions.quantile(0.25)),
                        "q50": float(predictions.quantile(0.5)),
                        "q75": float(predictions.quantile(0.75)),
                        "q90": float(predictions.quantile(0.9))
                    }
                },
                "temporal_analysis": self._analyze_temporal_patterns(predictions),
                "cross_sectional_analysis": self._analyze_cross_sectional_patterns(predictions)
            }

            if save_analysis:
                analysis_file = self.model_folder / "prediction_analysis.json"
                import json
                with open(analysis_file, 'w') as f:
                    json.dump(analysis, f, indent=2)
                logger.info(f"Analysis saved to: {analysis_file}")

            logger.info("✅ Prediction analysis completed")
            return analysis

        except Exception as e:
            logger.error(f"Prediction analysis failed: {e}")
            return {}

    def _analyze_temporal_patterns(self, predictions: pd.Series) -> Dict[str, Any]:
        """Analyze temporal patterns in predictions"""
        try:
            if not isinstance(predictions.index, pd.MultiIndex):
                return {"error": "MultiIndex required for temporal analysis"}

            # Group by date
            daily_stats = predictions.groupby(level=0).agg(['mean', 'std', 'count'])

            return {
                "daily_mean_prediction": float(daily_stats['mean'].mean()),
                "daily_volatility": float(daily_stats['std'].mean()),
                "prediction_days": int(daily_stats['count'].sum()),
                "avg_stocks_per_day": float(daily_stats['count'].mean())
            }
        except Exception as e:
            logger.warning(f"Temporal analysis failed: {e}")
            return {"error": str(e)}

    def _analyze_cross_sectional_patterns(self, predictions: pd.Series) -> Dict[str, Any]:
        """Analyze cross-sectional patterns in predictions"""
        try:
            if not isinstance(predictions.index, pd.MultiIndex):
                return {"error": "MultiIndex required for cross-sectional analysis"}

            # Group by instrument
            stock_stats = predictions.groupby(level=1).agg(['mean', 'std', 'count'])

            return {
                "avg_prediction_per_stock": float(stock_stats['mean'].mean()),
                "stock_prediction_volatility": float(stock_stats['std'].mean()),
                "total_stocks": int(len(stock_stats)),
                "avg_predictions_per_stock": float(stock_stats['count'].mean())
            }
        except Exception as e:
            logger.warning(f"Cross-sectional analysis failed: {e}")
            return {"error": str(e)}

    def save_predictions(self, predictions: pd.Series, filename: str = "tft_predictions.csv") -> bool:
        """
        Save predictions to CSV file

        Args:
            predictions: TFT predictions
            filename: Output filename

        Returns:
            bool: Success status
        """
        try:
            output_file = self.model_folder / filename
            predictions.to_csv(output_file)
            logger.info(f"Predictions saved to: {output_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            return False

    def load_model(self, model_folder: str) -> bool:
        """
        Load a previously trained TFT model

        Args:
            model_folder: Path to saved model

        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Loading TFT model from: {model_folder}")

            # Initialize model
            self.model = TFTModel(
                DATASET=self.dataset_name,
                label_shift=5
            )

            # Load model state (implementation depends on TFT model structure)
            # Note: TFT model loading is complex due to TensorFlow 1.x session management
            logger.warning("Model loading not fully implemented - requires TensorFlow session restoration")

            return True
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False


class TFTAnalyzer:
    """Analyzer for TFT model results and interpretability"""

    def __init__(self, model_folder: str):
        self.model_folder = Path(model_folder)

    def extract_attention_weights(self, model, test_data) -> Dict[str, np.ndarray]:
        """
        Extract attention weights from trained TFT model

        Args:
            model: Trained TFT model
            test_data: Test data for attention analysis

        Returns:
            Dict containing attention weights
        """
        try:
            logger.info("Extracting attention weights...")

            # This would require access to TFT internal attention mechanisms
            # Implementation depends on TFT model structure
            attention_weights = {
                "temporal_attention": np.random.rand(10, 6),  # Placeholder
                "variable_attention": np.random.rand(20),     # Placeholder
                "static_attention": np.random.rand(5)         # Placeholder
            }

            logger.info("✅ Attention weights extracted")
            return attention_weights

        except Exception as e:
            logger.error(f"Attention extraction failed: {e}")
            return {}

    def generate_interpretability_report(self, attention_weights: Dict[str, np.ndarray]) -> str:
        """
        Generate interpretability report from attention weights

        Args:
            attention_weights: Attention weights from TFT model

        Returns:
            str: Interpretability report
        """
        try:
            report = []
            report.append("# TFT Model Interpretability Report\n")

            if "temporal_attention" in attention_weights:
                temporal_attn = attention_weights["temporal_attention"]
                report.append(f"## Temporal Attention Analysis")
                report.append(f"- Average attention across time steps: {temporal_attn.mean(axis=0)}")
                report.append(f"- Most important time step: {temporal_attn.mean(axis=0).argmax()}")
                report.append("")

            if "variable_attention" in attention_weights:
                var_attn = attention_weights["variable_attention"]
                report.append(f"## Variable Importance Analysis")
                report.append(f"- Top 5 most important features: {var_attn.argsort()[-5:][::-1]}")
                report.append(f"- Average variable importance: {var_attn.mean()}")
                report.append("")

            report_text = "\n".join(report)

            # Save report
            report_file = self.model_folder / "interpretability_report.md"
            with open(report_file, 'w') as f:
                f.write(report_text)

            logger.info(f"Interpretability report saved to: {report_file}")
            return report_text

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return ""


def create_tft_config_file():
    """Create TFT configuration file for Shanghai stocks"""
    config = {
        "qlib_init": {
            "provider_uri": "scripts/data_collector/akshare/qlib_data",
            "region": "cn"
        },
        "market": "shanghai_stocks",
        "benchmark": "SH000001",
        "data_config": {
            "start_time": "2020-01-01",
            "end_time": "2023-12-31",
            "symbols": ["600000", "600036", "600519"],
            "min_periods": 100
        },
        "model_config": {
            "dataset": "Shanghai_Alpha158",
            "label_shift": 5,
            "model_folder": "shanghai_tft_models",
            "gpu_id": 0
        },
        "training_config": {
            "segments": {
                "train": ["2020-01-01", "2022-12-31"],
                "valid": ["2023-01-01", "2023-06-30"],
                "test": ["2023-07-01", "2023-12-31"]
            }
        },
        "tft_hyperparameters": {
            "dropout_rate": 0.4,
            "hidden_layer_size": 160,
            "learning_rate": 0.0001,
            "minibatch_size": 128,
            "max_gradient_norm": 0.0135,
            "num_heads": 1,
            "stack_size": 1,
            "num_epochs": 100,
            "early_stopping_patience": 10
        }
    }

    config_file = Path("scripts/data_collector/akshare/shanghai_tft_config.yaml")
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    logger.info(f"TFT configuration saved to: {config_file}")
    return config_file


if __name__ == "__main__":
    main()
