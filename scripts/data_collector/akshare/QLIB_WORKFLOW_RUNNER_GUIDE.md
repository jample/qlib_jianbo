# Qlib Workflow Runner - Complete User Guide

## 📋 Overview

The `qlib_workflow_runner.py` is a comprehensive workflow runner for Microsoft's Qlib quantitative investment platform, specifically designed for Shanghai stock market data. It provides a complete pipeline from data preparation to model training with support for both LightGBM and Transformer models.

## 🏗️ Architecture & Design

### Core Components

1. **QlibWorkflowRunner**: Main orchestrator class
2. **Alpha158Enhanced**: Enhanced Alpha158 feature handler
3. **DuckDBDataExtractor**: Data extraction from DuckDB
4. **Configuration System**: YAML-based workflow configurations

### Key Design Principles

- **Standard Qlib Workflow**: Follows qlib's official patterns and conventions
- **Binary Data Format**: Uses qlib's optimized binary storage for fast I/O
- **Modular Design**: Separates data preparation, model training, and evaluation
- **Multi-Model Support**: Supports both traditional ML (LightGBM) and deep learning (Transformer) models
- **Production Ready**: Includes error handling, logging, and validation

## 🚀 Quick Start

### Basic Usage

```python
from qlib_workflow_runner import QlibWorkflowRunner

# Initialize runner with LightGBM (default)
runner = QlibWorkflowRunner(model_type="lgb")

# Run complete workflow
success = runner.run_complete_workflow()
```

### Transformer Model Usage

```python
# Initialize runner with Transformer model
runner = QlibWorkflowRunner(model_type="transformer")

# Run transformer training workflow
success = runner.run_transformer_training()
```

## 📊 Model Types & Configurations

### 1. LightGBM Model (Default)

**Configuration Files:**
- `workflow_config_shanghai_alpha158.yaml` - Full Alpha158 features
- `workflow_config_shanghai_simple.yaml` - Basic OHLCV features

**Features:**
- 158 technical indicators (Alpha158)
- Gradient boosting algorithm
- Fast training and inference
- Good interpretability

**Usage:**
```python
runner = QlibWorkflowRunner(
    config_path="workflow_config_shanghai_alpha158.yaml",
    model_type="lgb"
)
```

### 2. Transformer Model

**Configuration Files:**
- `workflow_config_shanghai_simple_transformer.yaml` - Basic features for Transformer
- `workflow_config_transformer_shanghai.yaml` - Advanced Transformer config

**Features:**
- Deep learning architecture
- Sequence modeling capabilities
- Attention mechanisms
- Time series pattern recognition

**Usage:**
```python
runner = QlibWorkflowRunner(
    config_path="workflow_config_shanghai_simple_transformer.yaml",
    model_type="transformer"
)
```

## 🔧 Detailed API Reference

### QlibWorkflowRunner Class

#### Constructor

```python
QlibWorkflowRunner(
    config_path: str = None,
    model_type: str = "lgb"
)
```

**Parameters:**
- `config_path`: Path to YAML workflow configuration file
- `model_type`: Model type ("lgb" or "transformer")

**Auto-Configuration Logic:**
- If `config_path` is None, automatically selects appropriate config based on `model_type`
- LightGBM: Uses `workflow_config_shanghai_simple.yaml`
- Transformer: Uses `workflow_config_shanghai_simple_transformer.yaml`

#### Core Methods

##### step1_prepare_qlib_data()

Converts DuckDB data to qlib binary format.

```python
step1_prepare_qlib_data(
    data_type: str = "stock",
    exchange_filter: str = "shanghai", 
    symbol_filter: List[str] = ["600", "601"],
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31"
) -> bool
```

**Parameters:**
- `data_type`: Type of data ("stock" or "fund")
- `exchange_filter`: Exchange filter ("shanghai")
- `symbol_filter`: List of symbol prefixes or exact symbols
- `start_date`: Start date for data extraction
- `end_date`: End date for data extraction

**Process:**
1. Extracts data from DuckDB or CSV files
2. Calculates Alpha158 labels (2-day forward return)
3. Converts to qlib binary format
4. Creates qlib directory structure (features/, instruments/, calendars/)
5. Validates data quality and filters invalid instruments

##### step2_run_qlib_workflow()

Runs the qlib workflow using standard qlib mechanisms.

```python
step2_run_qlib_workflow(
    experiment_name: str = "shanghai_alpha158"
) -> bool
```

**Parameters:**
- `experiment_name`: Name for the experiment (used in MLflow tracking)

**Process:**
1. Initializes qlib with data directory
2. Loads YAML configuration
3. Creates dataset and model instances
4. Trains model with error handling
5. Saves model and evaluation metrics

#### High-Level Workflow Methods

##### run_complete_workflow()

Runs the complete workflow for LightGBM models.

```python
run_complete_workflow(
    data_config: Optional[Dict] = None,
    experiment_name: str = "shanghai_alpha158"
) -> bool
```

##### run_transformer_training()

Specialized workflow for Transformer models.

```python
run_transformer_training(
    data_config: Optional[Dict] = None,
    experiment_name: str = "shanghai_transformer"
) -> bool
```

**Default Data Configuration:**
```python
{
    "data_type": "stock",
    "exchange_filter": "shanghai", 
    "symbol_filter": ["600000", "600036"],  # Specific symbols for testing
    "start_date": "2024-01-01",
    "end_date": "2024-03-31"  # Shorter range for faster training
}
```

## 📁 File Structure & Dependencies

### Directory Structure
```
scripts/data_collector/akshare/
├── qlib_workflow_runner.py              # Main runner
├── duckdb_extractor.py                  # Data extraction
├── alpha158_enhanced.py                 # Enhanced Alpha158 handler
├── workflow_config_*.yaml               # Configuration files
├── qlib_data/                           # Qlib binary data
│   ├── features/                        # Feature files per instrument
│   ├── instruments/                     # Instrument lists
│   └── calendars/                       # Trading calendars
├── source/                              # Raw CSV data (fallback)
└── models/                              # Trained models
```

### Related Python Classes

#### DuckDBDataExtractor
Located in `duckdb_extractor.py`

**Purpose:** Extracts Shanghai stock data from DuckDB database

**Key Methods:**
- `extract_stock_data()`: Extracts time series data with filtering
- `get_available_symbols()`: Lists available stock symbols
- `validate_data_quality()`: Validates extracted data

#### Alpha158Enhanced  
Located in `alpha158_enhanced.py`

**Purpose:** Enhanced Alpha158 handler with rolling features

**Features:**
- Enables correlation operations (CORR, CORD, RSQR)
- Supports volume-weighted moving averages (WVMA)
- Includes volatility and residual features (VSTD, RESI)

## 🎯 Usage Patterns

### Pattern 1: Quick LightGBM Training

```python
from qlib_workflow_runner import QlibWorkflowRunner

# Simple LightGBM training with default settings
runner = QlibWorkflowRunner()
success = runner.run_complete_workflow()

if success:
    print("✅ Training completed successfully!")
else:
    print("❌ Training failed!")
```

### Pattern 2: Custom Data Configuration

```python
# Custom data configuration
data_config = {
    "data_type": "stock",
    "exchange_filter": "shanghai",
    "symbol_filter": ["600000", "600036", "600519"],  # Specific stocks
    "start_date": "2023-01-01", 
    "end_date": "2024-06-30"
}

runner = QlibWorkflowRunner(model_type="lgb")
success = runner.run_complete_workflow(
    data_config=data_config,
    experiment_name="custom_lgb_experiment"
)
```

### Pattern 3: Transformer Model Training

```python
# Transformer model with optimized configuration
runner = QlibWorkflowRunner(
    config_path="workflow_config_shanghai_simple_transformer.yaml",
    model_type="transformer"
)

# Use smaller dataset for faster training
data_config = {
    "symbol_filter": ["600000", "600036"],  # Just 2 stocks
    "start_date": "2024-01-01",
    "end_date": "2024-03-31"  # 3 months of data
}

success = runner.run_transformer_training(
    data_config=data_config,
    experiment_name="transformer_test"
)
```

### Pattern 4: Step-by-Step Execution

```python
runner = QlibWorkflowRunner(model_type="lgb")

# Step 1: Prepare data only
data_prepared = runner.step1_prepare_qlib_data(
    symbol_filter=["600000", "600036"],
    start_date="2024-01-01", 
    end_date="2024-03-31"
)

if data_prepared:
    # Step 2: Run training
    training_success = runner.step2_run_qlib_workflow(
        experiment_name="step_by_step_training"
    )
```

### Pattern 5: Command Line Usage

```bash
# Basic usage
python qlib_workflow_runner.py --model lgb

# Custom configuration
python qlib_workflow_runner.py \
    --model transformer \
    --config workflow_config_shanghai_simple_transformer.yaml \
    --experiment-name my_transformer \
    --symbols 600000 600036 \
    --start-date 2024-01-01 \
    --end-date 2024-03-31

# Prepare data only (for use with qrun)
python qlib_workflow_runner.py --prepare-only
```

### Pattern 6: Integration with qrun

```bash
# First prepare data
python qlib_workflow_runner.py --prepare-only

# Then use qlib's qrun command
qrun workflow_config_shanghai_alpha158.yaml
```

## ⚙️ Configuration Files Explained

### LightGBM Configuration (workflow_config_shanghai_alpha158.yaml)

**Key Sections:**
- `qlib_init`: Qlib initialization settings
- `data_handler_config`: Data processing configuration  
- `task.model`: LightGBM model parameters
- `task.dataset`: Dataset configuration with Alpha158 handler
- `task.record`: Experiment tracking and analysis

**Model Parameters:**
```yaml
model:
    class: LGBModel
    kwargs:
        loss: mse
        learning_rate: 0.0421
        max_depth: 8
        num_leaves: 210
        # ... other hyperparameters
```

### Transformer Configuration (workflow_config_shanghai_simple_transformer.yaml)

**Key Differences:**
- Uses `TSDatasetH` for time series data
- Transformer-specific hyperparameters
- Sequence length configuration (`step_len: 20`)

**Model Parameters:**
```yaml
model:
    class: TransformerModel
    kwargs:
        d_feat: 20          # Feature dimension
        d_model: 64         # Model dimension
        n_heads: 4          # Attention heads
        num_layers: 3       # Transformer layers
        n_epochs: 10        # Training epochs
        batch_size: 800     # Batch size
```

## 🔍 Advanced Features

### 1. Automatic Dependency Management

The runner automatically checks and handles dependencies:

```python
def _check_dependencies(self) -> bool:
    """Check and install required dependencies for the model type"""
    if self.model_type == "transformer":
        # Checks for PyTorch and installs if missing
        # Pre-loads Transformer models to suppress warnings
```

### 2. Data Quality Validation

Comprehensive data validation during conversion:

```python
def _convert_to_qlib_binary(self, df: pd.DataFrame) -> bool:
    # Validates data quality
    # Filters instruments with insufficient data
    # Handles missing values appropriately
    # Ensures consistent data format
```

### 3. Error Handling & Recovery

Robust error handling for common issues:

- **CUDA Availability**: Automatically falls back to CPU for Transformer training
- **Missing Dependencies**: Provides clear installation instructions
- **Data Issues**: Validates and filters problematic data
- **Configuration Errors**: Provides helpful error messages

### 4. Performance Monitoring

Built-in IC (Information Coefficient) evaluation:

```python
def _compute_ic_rankic(self, preds: pd.Series, labels: pd.Series) -> Dict[str, Any]:
    # Computes daily IC and Rank IC
    # Provides comprehensive performance metrics
    # Saves evaluation results to JSON
```

### 5. Experiment Tracking

Integration with MLflow for experiment tracking:

- Automatic experiment naming
- Model versioning
- Performance metrics logging
- Reproducible results

## 🚨 Common Issues & Solutions

### Issue 1: PyTorch Not Found (Transformer Models)

**Error:** `ImportError: No module named 'torch'`

**Solution:**
```bash
pip install torch torchvision
```

Or let the runner auto-install:
```python
runner = QlibWorkflowRunner(model_type="transformer")
# Will automatically install PyTorch if missing
```

### Issue 2: Insufficient Data

**Error:** `No valid data found` or `Insufficient data quality`

**Solution:**
- Use specific stock symbols instead of prefixes
- Reduce date range for testing
- Check data availability in DuckDB

```python
# Use specific symbols that are known to have data
data_config = {
    "symbol_filter": ["600000", "600036"],  # Specific symbols
    "start_date": "2024-01-01",
    "end_date": "2024-03-31"  # Shorter range
}
```

### Issue 3: Memory Issues with Large Datasets

**Solution:**
- Limit number of symbols for initial testing
- Use shorter date ranges
- Increase system memory or use cloud instances

### Issue 4: Configuration File Not Found

**Error:** `FileNotFoundError: workflow_config_*.yaml`

**Solution:**
- Ensure you're running from the correct directory
- Use absolute paths for configuration files
- Check file permissions

## 📈 Performance Optimization Tips

### 1. Data Preparation Optimization

- **Use DuckDB**: Much faster than CSV files
- **Filter Early**: Use specific symbols rather than broad prefixes
- **Reasonable Date Ranges**: Start with 3-6 months for testing

### 2. Model Training Optimization

**LightGBM:**
- Adjust `num_threads` based on CPU cores
- Use `early_stopping` to prevent overfitting
- Tune hyperparameters with smaller datasets first

**Transformer:**
- Start with smaller models (`d_model=32`, `num_layers=2`)
- Use GPU if available
- Reduce `batch_size` if memory is limited

### 3. System Optimization

- **Memory**: Ensure sufficient RAM (8GB+ recommended)
- **Storage**: Use SSD for faster I/O
- **CPU**: Multi-core processors benefit LightGBM training

## 🔄 Integration Patterns

### With Jupyter Notebooks

```python
# In Jupyter notebook
import sys
sys.path.append('/path/to/qlibjianbo/scripts/data_collector/akshare')

from qlib_workflow_runner import QlibWorkflowRunner

runner = QlibWorkflowRunner(model_type="lgb")
success = runner.run_complete_workflow()

# Analyze results
if success:
    # Load and analyze model results
    # Visualize performance metrics
    # Generate reports
```

### With Production Pipelines

```python
# Production pipeline integration
class ProductionPipeline:
    def __init__(self):
        self.runner = QlibWorkflowRunner(model_type="lgb")
    
    def daily_retrain(self):
        # Update data configuration for latest data
        data_config = {
            "start_date": "2023-01-01",
            "end_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        success = self.runner.run_complete_workflow(
            data_config=data_config,
            experiment_name=f"daily_retrain_{datetime.now().strftime('%Y%m%d')}"
        )
        
        return success
```

### With MLOps Platforms

```python
# MLflow integration example
import mlflow

with mlflow.start_run():
    runner = QlibWorkflowRunner(model_type="transformer")
    
    # Log parameters
    mlflow.log_param("model_type", "transformer")
    mlflow.log_param("data_range", "2024-01-01 to 2024-03-31")
    
    # Run training
    success = runner.run_transformer_training()
    
    # Log results
    mlflow.log_metric("training_success", 1 if success else 0)
```

## 📚 Additional Resources

### Related Files
- `demo_qlib_alpha158.py`: Demonstrates Alpha158 features
- `README_qlib_workflow.md`: Detailed workflow documentation
- `test_qlib_workflow.py`: Unit tests and examples

### External Documentation
- [Qlib Official Documentation](https://qlib.readthedocs.io/)
- [Alpha158 Paper](https://arxiv.org/abs/2101.02118)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)

### Community Resources
- [Qlib GitHub Repository](https://github.com/microsoft/qlib)
- [Qlib Examples](https://github.com/microsoft/qlib/tree/main/examples)
- [Quantitative Finance with Python](https://github.com/topics/quantitative-finance)

## 🤖 Transformer Model Deep Dive

### Architecture Details

The Transformer model in qlib uses a time series-specific architecture:

**Key Components:**
- **Positional Encoding**: Handles temporal relationships
- **Multi-Head Attention**: Captures feature interactions
- **Feed-Forward Networks**: Non-linear transformations
- **Layer Normalization**: Stabilizes training

### Transformer-Specific Configuration

#### Model Hyperparameters Explained

```yaml
model:
    class: TransformerModel
    kwargs:
        d_feat: 20          # Input feature dimension (number of features)
        d_model: 64         # Hidden dimension of the model
        n_heads: 4          # Number of attention heads (d_model % n_heads == 0)
        num_layers: 3       # Number of transformer layers
        dropout: 0.1        # Dropout rate for regularization
        n_epochs: 10        # Training epochs
        lr: 0.0001         # Learning rate
        batch_size: 800     # Batch size for training
        early_stop: 5       # Early stopping patience
        loss: mse          # Loss function (mse, mae)
        optimizer: adam     # Optimizer (adam, sgd)
        device: auto       # Device (auto, cpu, cuda)
```

#### Dataset Configuration for Transformers

```yaml
dataset:
    class: TSDatasetH      # Time Series Dataset Handler
    kwargs:
        step_len: 20       # Sequence length (lookback window)
        segments:
            train: [2024-01-01, 2024-02-29]
            valid: [2024-03-01, 2024-03-15]
            test: [2024-03-16, 2024-03-31]
```

### Transformer Training Best Practices

#### 1. Feature Selection for Transformers

**Recommended Features:**
```python
# Basic momentum and trend features work well
features = [
    "KLEN", "KLOW", "KMID", "KUP",     # Price position features
    "STD5", "STD10", "STD20",          # Volatility features
    "ROC5", "ROC10", "ROC20",          # Rate of change
    "MA5", "MA10", "MA20",             # Moving averages
    "RSV5", "RSV10",                   # Relative strength
    "BETA5", "BETA10"                  # Beta coefficients
]
```

#### 2. Sequence Length Optimization

```python
# Shorter sequences for limited data
step_len: 10    # For datasets < 100 days

# Medium sequences for normal use
step_len: 20    # For datasets 100-500 days

# Longer sequences for large datasets
step_len: 30    # For datasets > 500 days
```

#### 3. Training Strategies

**Progressive Training:**
```python
# Start with simple configuration
runner = QlibWorkflowRunner(
    config_path="workflow_config_transformer_simple.yaml",
    model_type="transformer"
)

# Small dataset for initial testing
data_config = {
    "symbol_filter": ["600000"],  # Single stock
    "start_date": "2024-01-01",
    "end_date": "2024-02-29"      # 2 months
}

success = runner.run_transformer_training(data_config)
```

**Scaling Up:**
```python
# After successful simple training, scale up
data_config = {
    "symbol_filter": ["600000", "600036", "600519"],  # Multiple stocks
    "start_date": "2023-01-01",
    "end_date": "2024-03-31"      # Longer period
}
```

### Transformer vs LightGBM Comparison

| Aspect | Transformer | LightGBM |
|--------|-------------|----------|
| **Training Speed** | Slower (GPU recommended) | Faster (CPU efficient) |
| **Memory Usage** | Higher | Lower |
| **Data Requirements** | More data needed | Works with less data |
| **Feature Engineering** | Less manual engineering | Benefits from feature engineering |
| **Interpretability** | Lower (black box) | Higher (feature importance) |
| **Sequence Modeling** | Excellent | Limited |
| **Hyperparameter Tuning** | More complex | Simpler |

## 🔧 Advanced Customization

### Custom Feature Engineering

#### Creating Custom Alpha158 Handler

```python
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL

class CustomAlpha158(Alpha158):
    """Custom Alpha158 with domain-specific features"""

    def get_feature_config(self):
        # Add custom features for Shanghai market
        config = {
            "kbar": {},
            "price": {
                "windows": [0, 1, 2, 3, 4],
                "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]
            },
            "volume": {
                "windows": [0, 1, 2, 3, 4]
            },
            "rolling": {
                "windows": [5, 10, 20, 30, 60],
                "include": ["ROC", "MA", "STD", "BETA", "RSQR"],
                "exclude": ["CORR", "CORD"]  # Exclude problematic features
            }
        }
        return Alpha158DL.get_feature_config(config)
```

#### Using Custom Handler

```python
# Modify configuration to use custom handler
config = {
    "task": {
        "dataset": {
            "kwargs": {
                "handler": {
                    "class": "CustomAlpha158",
                    "module_path": "your_module_path"
                }
            }
        }
    }
}
```

### Custom Data Processing Pipeline

```python
class CustomQlibWorkflowRunner(QlibWorkflowRunner):
    """Extended runner with custom data processing"""

    def custom_data_preprocessing(self, df):
        """Add custom preprocessing steps"""
        # Remove outliers
        df = self._remove_outliers(df)

        # Add custom technical indicators
        df = self._add_custom_indicators(df)

        # Industry-specific adjustments
        df = self._apply_industry_adjustments(df)

        return df

    def _remove_outliers(self, df):
        """Remove statistical outliers"""
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                df = df[(df[col] >= Q1) & (df[col] <= Q3)]
        return df
```

### Multi-Model Ensemble

```python
class EnsembleRunner:
    """Run multiple models and ensemble results"""

    def __init__(self):
        self.lgb_runner = QlibWorkflowRunner(model_type="lgb")
        self.transformer_runner = QlibWorkflowRunner(model_type="transformer")

    def run_ensemble_training(self, data_config):
        """Train both models and create ensemble"""

        # Train LightGBM
        lgb_success = self.lgb_runner.run_complete_workflow(
            data_config=data_config,
            experiment_name="ensemble_lgb"
        )

        # Train Transformer
        transformer_success = self.transformer_runner.run_transformer_training(
            data_config=data_config,
            experiment_name="ensemble_transformer"
        )

        if lgb_success and transformer_success:
            # Create ensemble predictions
            return self._create_ensemble()

        return False

    def _create_ensemble(self):
        """Combine predictions from multiple models"""
        # Load models and create weighted ensemble
        # Implementation depends on specific requirements
        pass
```

## 🎯 Production Deployment Patterns

### 1. Batch Training Pipeline

```python
import schedule
import time
from datetime import datetime, timedelta

class ProductionTrainingPipeline:
    def __init__(self):
        self.runner = QlibWorkflowRunner(model_type="lgb")

    def weekly_retrain(self):
        """Weekly model retraining"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data

        data_config = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "symbol_filter": ["600", "601"]  # Shanghai A-shares
        }

        experiment_name = f"weekly_retrain_{end_date.strftime('%Y%m%d')}"

        success = self.runner.run_complete_workflow(
            data_config=data_config,
            experiment_name=experiment_name
        )

        if success:
            self._deploy_model(experiment_name)
        else:
            self._alert_failure(experiment_name)

    def _deploy_model(self, experiment_name):
        """Deploy trained model to production"""
        # Implementation for model deployment
        pass

    def _alert_failure(self, experiment_name):
        """Send alert on training failure"""
        # Implementation for failure alerts
        pass

# Schedule weekly retraining
pipeline = ProductionTrainingPipeline()
schedule.every().sunday.at("02:00").do(pipeline.weekly_retrain)

while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

### 2. Real-time Prediction Service

```python
from flask import Flask, request, jsonify
import pickle
import pandas as pd

class PredictionService:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Initialize qlib for data access
        import qlib
        qlib.init(provider_uri="qlib_data", region="cn")

    def predict(self, symbols, date):
        """Make predictions for given symbols and date"""
        try:
            # Fetch latest data
            data = self._fetch_latest_data(symbols, date)

            # Make predictions
            predictions = self.model.predict(data)

            return {
                "status": "success",
                "predictions": predictions.to_dict(),
                "timestamp": date
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def _fetch_latest_data(self, symbols, date):
        """Fetch latest data for prediction"""
        # Implementation for data fetching
        pass

# Flask API
app = Flask(__name__)
service = PredictionService("models/production_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symbols = data.get('symbols', [])
    date = data.get('date')

    result = service.predict(symbols, date)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 3. Model Monitoring and Validation

```python
class ModelMonitor:
    """Monitor model performance in production"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.performance_threshold = 0.02  # Minimum IC threshold

    def validate_model_performance(self, test_data):
        """Validate model performance on recent data"""

        # Load model
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)

        # Make predictions
        predictions = model.predict(test_data)

        # Calculate performance metrics
        ic_metrics = self._calculate_ic_metrics(predictions, test_data)

        # Check if performance is acceptable
        if ic_metrics['ic_mean'] < self.performance_threshold:
            self._trigger_retraining()
            return False

        return True

    def _calculate_ic_metrics(self, predictions, test_data):
        """Calculate IC metrics for validation"""
        # Implementation for IC calculation
        pass

    def _trigger_retraining(self):
        """Trigger model retraining when performance degrades"""
        # Implementation for triggering retraining
        pass
```

## 📊 Performance Analysis and Debugging

### Model Performance Analysis

```python
def analyze_model_performance(experiment_name):
    """Analyze trained model performance"""

    # Load experiment results
    import mlflow

    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    # Analyze metrics
    for _, run in runs.iterrows():
        print(f"Run ID: {run['run_id']}")
        print(f"IC Mean: {run['metrics.ic_mean']:.4f}")
        print(f"Rank IC Mean: {run['metrics.rank_ic_mean']:.4f}")
        print(f"Training Time: {run['metrics.training_time']:.2f}s")
        print("-" * 40)

# Usage
analyze_model_performance("shanghai_transformer")
```

### Debugging Common Issues

#### Debug Data Quality Issues

```python
def debug_data_quality(runner):
    """Debug data quality issues"""

    # Check data availability
    extractor = runner.duckdb_extractor
    symbols = extractor.get_available_symbols()
    print(f"Available symbols: {len(symbols)}")

    # Check date ranges
    for symbol in symbols[:5]:  # Check first 5 symbols
        data = extractor.extract_stock_data([symbol])
        if not data.empty:
            print(f"{symbol}: {data['date'].min()} to {data['date'].max()}")
        else:
            print(f"{symbol}: No data available")

# Usage
runner = QlibWorkflowRunner()
debug_data_quality(runner)
```

#### Debug Model Training Issues

```python
def debug_training_issues(config_path):
    """Debug model training issues"""

    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check configuration
    print("Configuration Analysis:")
    print(f"Model: {config['task']['model']['class']}")
    print(f"Dataset: {config['task']['dataset']['class']}")

    # Check data handler
    handler_config = config['task']['dataset']['kwargs']['handler']
    print(f"Handler: {handler_config['class']}")

    # Check segments
    segments = config['task']['dataset']['kwargs']['segments']
    for segment, dates in segments.items():
        print(f"{segment}: {dates[0]} to {dates[1]}")

# Usage
debug_training_issues("workflow_config_shanghai_simple_transformer.yaml")
```

## 🚀 Optimization and Scaling

### Performance Optimization

#### Memory Optimization

```python
# For large datasets, use data streaming
class StreamingQlibRunner(QlibWorkflowRunner):
    """Memory-efficient runner for large datasets"""

    def _convert_to_qlib_binary_streaming(self, df):
        """Convert data in chunks to reduce memory usage"""

        chunk_size = 10000  # Process 10k records at a time

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            self._process_chunk(chunk)

    def _process_chunk(self, chunk):
        """Process individual data chunk"""
        # Implementation for chunk processing
        pass
```

#### GPU Acceleration for Transformers

```python
# Optimize Transformer training for GPU
def optimize_transformer_config():
    """Return GPU-optimized Transformer configuration"""

    return {
        "model": {
            "kwargs": {
                "device": "cuda",           # Use GPU
                "batch_size": 1024,         # Larger batch size for GPU
                "n_epochs": 20,             # More epochs with GPU speed
                "lr": 0.001,               # Higher learning rate
                "num_workers": 4            # Parallel data loading
            }
        }
    }
```

### Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

class DistributedTransformerRunner(QlibWorkflowRunner):
    """Distributed training for Transformer models"""

    def __init__(self, rank, world_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank
        self.world_size = world_size

        # Initialize distributed training
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def _create_distributed_model(self, model):
        """Wrap model for distributed training"""
        model = model.to(f'cuda:{self.rank}')
        return DistributedDataParallel(model, device_ids=[self.rank])
```

---

*This comprehensive guide covers all aspects of using `qlib_workflow_runner.py` for both basic and advanced use cases. The runner provides a robust foundation for quantitative finance research and production deployment with Shanghai stock market data.*
