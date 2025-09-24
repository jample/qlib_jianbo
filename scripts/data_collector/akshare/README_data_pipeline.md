# Configurable Stock/Fund Data Processing Pipeline

This comprehensive data processing pipeline transforms raw stock or fund market data from DuckDB into training-ready datasets for machine learning models. The pipeline supports configurable data scope (stock/fund, exchange filters, time ranges, intervals) and includes data extraction, feature engineering with Alpha 158 features, normalization, and preparation for both tabular and sequence-based models.

## Overview

The pipeline consists of 7 main components:

1. **DuckDB Data Extractor** - Extracts and validates stock data from DuckDB
2. **Binary Data Converter** - Converts data to qlib-compatible binary format
3. **Data Initializer** - Prepares panel data structure and handles missing values
4. **Alpha 158 Feature Calculator** - Generates comprehensive technical analysis features
5. **Data Normalizer** - Performs outlier detection, cleaning, and normalization
6. **Training Data Preparator** - Creates train/validation/test splits with labels
7. **Main Pipeline** - Orchestrates the complete workflow

## Quick Start

### Basic Usage

```bash
# Run the complete pipeline with default settings (Shanghai stocks)
python main_pipeline.py

# Process different data types and exchanges
python main_pipeline.py --data-type stock --exchange shanghai
python main_pipeline.py --data-type fund --db-path fund_data.duckdb
python main_pipeline.py --data-type stock --exchange shenzhen

# Specify custom database and output paths
python main_pipeline.py --db-path /path/to/your/database.duckdb --output-dir /path/to/output

# Filter specific symbols (e.g., only 600xxx and 601xxx stocks)
python main_pipeline.py --symbols 600 601 --exchange shanghai

# Process all exchanges with symbol filter
python main_pipeline.py --exchange all --symbols 000 002 300

# Create sequence data for LSTM models
python main_pipeline.py --sequence-length 20 --label-type return

# Custom time range and interval
python main_pipeline.py --start-date 2023-01-01 --end-date 2024-06-30 --interval 1d

# Skip binary conversion (faster processing)
python main_pipeline.py --no-binary
```

### Configuration File

Create a JSON configuration file for advanced settings:

```json
{
  "db_path": "scripts/data_collector/akshare/source/shanghai_stock_data.duckdb",
  "data_type": "stock",
  "exchange_filter": "shanghai",
  "interval": "1d",
  "training_output_dir": "scripts/data_collector/akshare/training_data",
  "min_trading_days": 100,
  "outlier_method": "iqr",
  "normalization_method": "zscore",
  "train_ratio": 0.7,
  "val_ratio": 0.15,
  "test_ratio": 0.15,
  "prediction_horizon": 1,
  "label_type": "return",
  "sequence_length": null,
  "symbol_filter": ["600", "601", "603"],
  "date_range": {
    "start_date": "2022-01-01",
    "end_date": "2024-12-31"
  }
}
```

Then run: `python main_pipeline.py --config config.json`

## Data Scope Configuration

The pipeline supports flexible data scope configuration to work with different data types and sources:

### Data Types
- **`stock`**: Stock market data (default)
- **`fund`**: Fund/ETF data

### Exchange Filters
- **`shanghai`**: Shanghai Stock Exchange (6xxxxx codes: 600xxx, 601xxx, 603xxx, 605xxx, 688xxx, 689xxx)
- **`shenzhen`**: Shenzhen Stock Exchange (000xxx, 001xxx, 002xxx, 003xxx, 300xxx codes)
- **`all`**: All exchanges
- **`None`**: No exchange filtering

### Data Intervals
- **`1d`**: Daily data (currently supported)
- **`1w`**: Weekly data (planned)
- **`1m`**: Monthly data (planned)

### Symbol Filtering
- **Prefix-based**: Filter by symbol prefixes (e.g., `['600', '601']` for specific Shanghai stock types)
- **Exchange-aware**: Automatically applies exchange-specific symbol patterns
- **Combined filtering**: Exchange filter + prefix filter for precise control

### Database Table Mapping
The pipeline automatically maps to appropriate database tables based on configuration:
- **Stock data**: `stock_data` table
- **Fund data**: `fund_data` table
- **Metadata**: `{data_type}_update_metadata` table
- **Symbols cache**: `{exchange}_{data_type}s` table

### Configuration Examples

```bash
# Shanghai stocks only (default)
python main_pipeline.py --data-type stock --exchange shanghai

# Shenzhen stocks with specific prefixes
python main_pipeline.py --data-type stock --exchange shenzhen --symbols 000 002

# All stocks from all exchanges
python main_pipeline.py --data-type stock --exchange all

# Fund data processing
python main_pipeline.py --data-type fund --db-path fund_database.duckdb

# Custom symbol filtering without exchange restriction
python main_pipeline.py --exchange all --symbols 600 000 300
```

## Pipeline Phases

### Phase 1: Data Extraction
- Connects to configurable DuckDB database
- Extracts data based on configured scope (stock/fund, exchange, symbols)
- Applies exchange-specific symbol filtering (Shanghai: 6xxxxx, Shenzhen: 000xxx/300xxx)
- Validates data integrity (price relationships, missing values)
- Filters symbols with insufficient trading history

### Phase 2: Data Initialization
- Creates proper panel data structure (symbols × dates)
- Handles missing values with forward/backward fill
- Validates data completeness and consistency
- Prepares data for feature engineering

### Phase 3: Feature Engineering (Alpha 158)
- **Return Features**: Various return calculations (1d, 5d, 10d, 20d, 60d)
- **Moving Averages**: Simple and exponential moving averages
- **Volatility Features**: Rolling standard deviations and ATR
- **Momentum Features**: RSI, price position, rank-based features
- **Volume Features**: Volume ratios, OBV, volume-price trend
- **Correlation Features**: Price-volume correlations
- **Technical Indicators**: MACD, Bollinger Bands, Stochastic, CCI
- **Cross-sectional Features**: Industry and market relative features

### Phase 4: Data Normalization
- **Outlier Detection**: IQR, Z-score, or quantile-based methods
- **Outlier Handling**: Clipping, winsorization, or removal
- **Missing Value Imputation**: Forward fill, backward fill, mean, or median
- **Normalization**: Z-score, robust, min-max, quantile, or rank-based
- **Cross-sectional Normalization**: Normalize across stocks at each time point

### Phase 5: Training Data Preparation
- **Label Creation**: Return, direction, or volatility labels
- **Time-based Splits**: Chronological train/validation/test splits
- **Feature Matrix**: Tabular data for traditional ML models
- **Sequence Data**: Time series data for LSTM/GRU models
- **Data Export**: NumPy arrays with metadata

### Phase 6: Binary Conversion (Optional)
- Converts data to qlib binary format for fast loading
- Creates proper directory structure (features/, calendars/, instruments/)
- Optimized for quantitative research workflows

## Output Structure

```
training_data/
├── X_train.npy              # Training features
├── y_train.npy              # Training labels
├── X_val.npy                # Validation features
├── y_val.npy                # Validation labels
├── X_test.npy               # Test features
├── y_test.npy               # Test labels
├── feature_names.txt        # List of feature names
├── metadata.json            # Dataset metadata
├── normalization_params.pkl # Normalization parameters
└── pipeline_results_*.json  # Complete pipeline results

qlib_data/                   # Binary format (if enabled)
├── features/
├── calendars/
└── instruments/

logs/                        # Pipeline logs
└── pipeline_*.log
```

## Feature Categories

The Alpha 158 feature set includes:

- **Price Features** (40 features): Returns, moving averages, price ratios
- **Volatility Features** (20 features): Rolling volatilities, ATR, relative volatility
- **Momentum Features** (30 features): RSI, momentum indicators, rank features
- **Volume Features** (25 features): Volume ratios, OBV, volume-price relationships
- **Correlation Features** (15 features): Price-volume correlations over different periods
- **Technical Features** (20 features): MACD, Bollinger Bands, Stochastic oscillators
- **Cross-sectional Features** (8 features): Industry and market relative metrics

## Data Quality Assurance

- **Validation Checks**: Price relationships, volume consistency, date continuity
- **Outlier Detection**: Multiple methods with configurable thresholds
- **Missing Value Handling**: Symbol-specific imputation strategies
- **Data Integrity**: Comprehensive logging and error handling
- **Quality Scoring**: Automated data quality assessment

## Performance Optimization

- **Parallel Processing**: Vectorized operations using pandas/numpy
- **Memory Efficiency**: Chunked processing for large datasets
- **Caching**: Intermediate results saved for pipeline resumption
- **Binary Format**: Fast loading with qlib binary format

## Usage Examples

### Example 1: Basic Pipeline for Traditional ML
```python
from main_pipeline import MainDataPipeline

# Initialize with default configuration
pipeline = MainDataPipeline()

# Run complete pipeline
results = pipeline.run_complete_pipeline()

# Access training data
X_train, y_train = results['training_preparation']['result']['train_data']
print(f"Training data shape: {X_train.shape}")
```

### Example 2: LSTM Sequence Data
```python
config = {
    'sequence_length': 20,
    'label_type': 'return',
    'normalization_method': 'robust'
}

pipeline = MainDataPipeline(config)
results = pipeline.run_complete_pipeline()

# Access sequence data
X_train, y_train = results['training_preparation']['result']['train_data']
print(f"Sequence data shape: {X_train.shape}")  # (samples, timesteps, features)
```

### Example 3: Custom Symbol Filtering
```python
config = {
    'symbol_filter': ['600', '601'],  # Only 600xxx and 601xxx stocks
    'date_range': {
        'start_date': '2023-01-01',
        'end_date': '2024-06-30'
    }
}

pipeline = MainDataPipeline(config)
results = pipeline.run_complete_pipeline()
```

## Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- duckdb >= 0.8.0
- loguru >= 0.6.0

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce date range or use symbol filtering
2. **Missing Data**: Check database integrity and date ranges
3. **Feature Calculation Errors**: Ensure sufficient historical data
4. **Normalization Issues**: Check for constant features or extreme outliers

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

The pipeline provides comprehensive logging and timing information for each phase, helping identify bottlenecks and optimize performance.

## License

This pipeline is designed for quantitative research and educational purposes. Please ensure compliance with data usage policies and regulations.
