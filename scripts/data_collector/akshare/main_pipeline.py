#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Data Processing Pipeline

This script orchestrates the complete pipeline from DuckDB extraction to 
training-ready data with proper logging and error handling.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from duckdb_extractor import DuckDBDataExtractor
from binary_converter import BinaryDataConverter
from data_initializer import DataInitializer
from alpha158_calculator import Alpha158Calculator
from data_normalizer import DataNormalizer
from training_data_prep import TrainingDataPreparator


class MainDataPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: dict = None):
        """
        Initialize the main pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.setup_logging()
        
        # Initialize components
        self.extractor = DuckDBDataExtractor(
            db_path=self.config['db_path'],
            data_type=self.config['data_type'],
            exchange_filter=self.config['exchange_filter'],
            interval=self.config['interval']
        )
        self.binary_converter = BinaryDataConverter(self.config['binary_output_dir'])
        self.data_initializer = DataInitializer(
            min_trading_days=self.config['min_trading_days'],
            min_price=self.config['min_price']
        )
        self.alpha_calculator = Alpha158Calculator()
        self.normalizer = DataNormalizer(
            outlier_method=self.config['outlier_method'],
            normalization_method=self.config['normalization_method']
        )
        self.training_prep = TrainingDataPreparator(
            train_ratio=self.config['train_ratio'],
            val_ratio=self.config['val_ratio'],
            test_ratio=self.config['test_ratio'],
            prediction_horizon=self.config['prediction_horizon']
        )
        
        logger.info("Main pipeline initialized successfully")
    
    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            # Data source configuration
            'db_path': '/root/autodl-tmp/code/duckdb/shanghai_stock_data.duckdb',
            'data_type': 'stock',  # 'stock' or 'fund'
            'exchange_filter': 'shanghai',  # 'shanghai', 'shenzhen', 'all', or None
            'interval': '1d',  # '1d', '1w', '1m' (currently only 1d supported)

            # Output directories
            'binary_output_dir': 'scripts/data_collector/akshare/qlib_data',
            'training_output_dir': 'scripts/data_collector/akshare/training_data',

            # Data filtering
            'min_trading_days': 100,
            'min_price': 0.01,
            'symbol_filter': None,  # None for all symbols, or list of prefixes like ['600', '601']
            'date_range': {
                'start_date': '2022-01-01',
                'end_date': '2024-12-31'
            },

            # Processing configuration
            'outlier_method': 'iqr',
            'normalization_method': 'zscore',
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'prediction_horizon': 1,
            'label_type': 'return',
            'sequence_length': None,  # None for tabular data, set to int for sequence data
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path('scripts/data_collector/akshare/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logger.remove()  # Remove default handler
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG"
        )
        
        logger.info(f"Logging setup completed. Log file: {log_file}")
    
    def run_extraction_phase(self) -> tuple:
        """Run data extraction phase"""
        try:
            logger.info("=" * 60)
            logger.info("PHASE 1: DATA EXTRACTION")
            logger.info("=" * 60)
            
            # Get database info
            db_info = self.extractor.get_database_info()
            logger.info(f"Database info: {db_info}")
            
            # Get symbols to process
            if self.config['symbol_filter']:
                symbols = self.extractor.get_available_symbols(self.config['symbol_filter'])
            else:
                symbols = self.extractor.get_available_symbols()
            
            logger.info(f"Processing {len(symbols)} symbols")
            
            # Extract data
            raw_data = self.extractor.extract_stock_data(
                symbols=symbols,
                start_date=self.config['date_range']['start_date'],
                end_date=self.config['date_range']['end_date']
            )
            
            # Validate and clean
            clean_data = self.extractor.validate_and_clean_data(raw_data)
            
            # Get summary
            summary = self.extractor.get_data_summary(clean_data)
            
            logger.info("Data extraction phase completed successfully")
            return clean_data, summary
            
        except Exception as e:
            logger.error(f"Error in extraction phase: {e}")
            raise
    
    def run_initialization_phase(self, raw_data) -> tuple:
        """Run data initialization phase"""
        try:
            logger.info("=" * 60)
            logger.info("PHASE 2: DATA INITIALIZATION")
            logger.info("=" * 60)
            
            # Initialize data
            initialized_data, validation_summary = self.data_initializer.initialize_data(raw_data)
            
            logger.info("Data initialization phase completed successfully")
            return initialized_data, validation_summary
            
        except Exception as e:
            logger.error(f"Error in initialization phase: {e}")
            raise
    
    def run_feature_engineering_phase(self, initialized_data) -> tuple:
        """Run feature engineering phase"""
        try:
            logger.info("=" * 60)
            logger.info("PHASE 3: FEATURE ENGINEERING (ALPHA 158)")
            logger.info("=" * 60)
            
            # Calculate Alpha 158 features
            feature_data = self.alpha_calculator.calculate_alpha158_features(initialized_data)
            
            # Get feature summary
            feature_summary = self.alpha_calculator.get_feature_summary(feature_data)
            
            logger.info("Feature engineering phase completed successfully")
            return feature_data, feature_summary
            
        except Exception as e:
            logger.error(f"Error in feature engineering phase: {e}")
            raise
    
    def run_normalization_phase(self, feature_data) -> tuple:
        """Run data normalization phase"""
        try:
            logger.info("=" * 60)
            logger.info("PHASE 4: DATA NORMALIZATION")
            logger.info("=" * 60)
            
            # Clean and normalize data
            normalized_data, validation_results = self.normalizer.clean_and_normalize_pipeline(feature_data)
            
            # Save normalization parameters
            norm_params_file = Path(self.config['training_output_dir']) / 'normalization_params.pkl'
            norm_params_file.parent.mkdir(parents=True, exist_ok=True)
            self.normalizer.save_normalization_params(str(norm_params_file))
            
            logger.info("Data normalization phase completed successfully")
            return normalized_data, validation_results
            
        except Exception as e:
            logger.error(f"Error in normalization phase: {e}")
            raise
    
    def run_training_preparation_phase(self, normalized_data) -> dict:
        """Run training data preparation phase"""
        try:
            logger.info("=" * 60)
            logger.info("PHASE 5: TRAINING DATA PREPARATION")
            logger.info("=" * 60)
            
            # Prepare training dataset
            training_result = self.training_prep.prepare_complete_training_dataset(
                df=normalized_data,
                label_type=self.config['label_type'],
                sequence_length=self.config['sequence_length'],
                output_dir=self.config['training_output_dir']
            )
            
            # Get data summary
            data_summary = self.training_prep.get_data_summary(training_result)
            
            logger.info("Training data preparation phase completed successfully")
            return training_result, data_summary
            
        except Exception as e:
            logger.error(f"Error in training preparation phase: {e}")
            raise
    
    def run_binary_conversion_phase(self, clean_data) -> dict:
        """Run binary conversion phase (optional)"""
        try:
            logger.info("=" * 60)
            logger.info("PHASE: BINARY CONVERSION (QLIB FORMAT)")
            logger.info("=" * 60)
            
            # Convert to qlib binary format
            qlib_data_path = self.binary_converter.convert_dataframe_to_qlib_format(clean_data)
            
            # Get conversion summary
            conversion_summary = self.binary_converter.get_conversion_summary()
            
            logger.info("Binary conversion phase completed successfully")
            return qlib_data_path, conversion_summary
            
        except Exception as e:
            logger.error(f"Error in binary conversion phase: {e}")
            raise

    def run_complete_pipeline(self, convert_to_binary: bool = True) -> dict:
        """
        Run the complete data processing pipeline

        Args:
            convert_to_binary: Whether to convert data to qlib binary format

        Returns:
            Dictionary with all results and summaries
        """
        try:
            start_time = datetime.now()
            logger.info("=" * 80)
            logger.info("STARTING COMPLETE DATA PROCESSING PIPELINE")
            logger.info("=" * 80)
            logger.info(f"Pipeline configuration: {self.config}")

            results = {}

            # Phase 1: Data Extraction
            raw_data, extraction_summary = self.run_extraction_phase()
            results['extraction'] = {
                'data': raw_data,
                'summary': extraction_summary
            }

            # Phase 2: Data Initialization
            initialized_data, init_summary = self.run_initialization_phase(raw_data)
            results['initialization'] = {
                'data': initialized_data,
                'summary': init_summary
            }

            # Phase 3: Feature Engineering
            feature_data, feature_summary = self.run_feature_engineering_phase(initialized_data)
            results['feature_engineering'] = {
                'data': feature_data,
                'summary': feature_summary
            }

            # Phase 4: Data Normalization
            normalized_data, norm_summary = self.run_normalization_phase(feature_data)
            results['normalization'] = {
                'data': normalized_data,
                'summary': norm_summary
            }

            # Phase 5: Training Data Preparation
            training_result, training_summary = self.run_training_preparation_phase(normalized_data)
            results['training_preparation'] = {
                'result': training_result,
                'summary': training_summary
            }

            # Optional: Binary Conversion
            if convert_to_binary:
                binary_path, binary_summary = self.run_binary_conversion_phase(raw_data)
                results['binary_conversion'] = {
                    'path': binary_path,
                    'summary': binary_summary
                }

            # Calculate total runtime
            end_time = datetime.now()
            runtime = end_time - start_time

            # Final summary
            final_summary = {
                'pipeline_completed': True,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_runtime': str(runtime),
                'phases_completed': list(results.keys()),
                'final_data_shape': training_result['metadata']['train_shape'],
                'total_features': training_result['metadata']['num_features'],
                'output_files': training_result.get('file_paths', {})
            }

            results['final_summary'] = final_summary

            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Total runtime: {runtime}")
            logger.info(f"Final training data shape: {training_result['metadata']['train_shape']}")
            logger.info(f"Total features generated: {training_result['metadata']['num_features']}")
            logger.info(f"Output directory: {self.config['training_output_dir']}")

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def save_pipeline_results(self, results: dict, output_file: str = None) -> str:
        """Save pipeline results to file"""
        try:
            if output_file is None:
                output_dir = Path(self.config['training_output_dir'])
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # Prepare serializable results (exclude large DataFrames)
            serializable_results = {}
            for phase, data in results.items():
                if phase == 'final_summary':
                    serializable_results[phase] = data
                else:
                    serializable_results[phase] = {
                        'summary': data.get('summary', {}),
                        'metadata': data.get('result', {}).get('metadata', {}) if 'result' in data else {}
                    }

            import json
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)

            logger.info(f"Pipeline results saved to: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Error saving pipeline results: {e}")
            raise


def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(
        description='Configurable Stock/Fund Data Processing Pipeline - Extract, process, and prepare data for ML training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process Shanghai stocks (default)
  python main_pipeline.py --data-type stock --exchange shanghai

  # Process all stocks with specific symbols
  python main_pipeline.py --data-type stock --exchange all --symbols 600 601

  # Process funds
  python main_pipeline.py --data-type fund --db-path fund_data.duckdb

  # Create sequence data for LSTM
  python main_pipeline.py --sequence-length 20 --label-type return

  # Custom date range
  python main_pipeline.py --start-date 2023-01-01 --end-date 2024-06-30
        """
    )

    # Configuration
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')

    # Data source configuration
    parser.add_argument('--db-path', type=str, default='/root/autodl-tmp/code/duckdb/shanghai_stock_data.duckdb',
                       help='Path to DuckDB database file')
    parser.add_argument('--data-type', type=str, default='stock', choices=['stock', 'fund'],
                       help='Type of data to process (stock or fund)')
    parser.add_argument('--exchange', type=str, choices=['shanghai', 'shenzhen', 'all'],
                       help='Exchange filter (shanghai, shenzhen, all)')
    parser.add_argument('--interval', type=str, default='1d', choices=['1d', '1w', '1m'],
                       help='Data interval (currently only 1d supported)')

    # Output configuration
    parser.add_argument('--output-dir', type=str, default='scripts/data_collector/akshare/training_data',
                       help='Output directory for training data')

    # Data filtering
    parser.add_argument('--symbols', type=str, nargs='+', help='Symbol prefixes to filter (e.g., 600 601)')
    parser.add_argument('--start-date', type=str, default='2022-01-01', help='Start date for data extraction')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date for data extraction')

    # Model configuration
    parser.add_argument('--label-type', type=str, default='return', choices=['return', 'direction', 'volatility'],
                       help='Type of labels to create')
    parser.add_argument('--sequence-length', type=int, help='Sequence length for time series models')

    # Processing options
    parser.add_argument('--no-binary', action='store_true', help='Skip binary conversion')

    args = parser.parse_args()

    try:
        # Load configuration
        if args.config:
            import json
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            config = {}

        # Set default configuration values
        default_config = {
            'db_path': '/root/autodl-tmp/code/duckdb/shanghai_stock_data.duckdb',
            'data_type': 'stock',
            'exchange_filter': 'shanghai',
            'interval': '1d',
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
            'sequence_length': None,
            'symbol_filter': ['600', '601'],
            'date_range': {'start_date': '2022-01-01', 'end_date': '2024-12-31'}
        }

        # Merge with loaded config (loaded config takes precedence)
        full_config = {**default_config, **config}

        # Override config with command line arguments (command line takes highest precedence)
        if args.db_path:
            full_config['db_path'] = args.db_path
        if args.data_type:
            full_config['data_type'] = args.data_type
        if args.exchange:
            full_config['exchange_filter'] = args.exchange
        if args.interval:
            full_config['interval'] = args.interval
        if args.output_dir:
            full_config['training_output_dir'] = args.output_dir
        if args.symbols:
            full_config['symbol_filter'] = args.symbols
        if args.start_date:
            full_config['date_range'] = full_config.get('date_range', {})
            full_config['date_range']['start_date'] = args.start_date
        if args.end_date:
            full_config['date_range'] = full_config.get('date_range', {})
            full_config['date_range']['end_date'] = args.end_date
        if args.label_type:
            full_config['label_type'] = args.label_type
        if args.sequence_length:
            full_config['sequence_length'] = args.sequence_length

        # Initialize and run pipeline
        pipeline = MainDataPipeline(full_config)
        results = pipeline.run_complete_pipeline(convert_to_binary=not args.no_binary)

        # Save results
        results_file = pipeline.save_pipeline_results(results)

        print(f"\n{'='*80}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"Results saved to: {results_file}")
        print(f"Training data directory: {full_config.get('training_output_dir', 'N/A')}")
        print(f"Total runtime: {results['final_summary']['total_runtime']}")
        print(f"Final data shape: {results['final_summary']['final_data_shape']}")
        print(f"Total features: {results['final_summary']['total_features']}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
