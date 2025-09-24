# Shanghai Stock TFT (Temporal Fusion Transformer) Implementation

## 📋 Overview

This implementation provides a complete pipeline for training Temporal Fusion Transformer (TFT) models on Shanghai stock market data using Microsoft's Qlib framework. TFT is a state-of-the-art deep learning model for multi-horizon time series forecasting with built-in interpretability.

## 🏗️ Architecture

```
Shanghai TFT Pipeline
├── Data Extraction (DuckDB) → Data Preparation (Qlib) → TFT Training → Predictions
├── shanghai_tft_runner.py          # Main TFT runner implementation
├── shanghai_tft_config.yaml        # Configuration file
├── example_tft_usage.py            # Usage examples
├── TFT_QLIB_GUIDE.md               # Comprehensive guide
└── Models & Results                # Trained models and outputs
```

## 🚀 Quick Start

### Prerequisites

**⚠️ Important System Requirements:**
- **Python 3.6-3.7** (TFT limitation - will not work with Python 3.8+)
- **GPU with CUDA 10.0** (CPU not supported)
- **TensorFlow 1.15.0** (specific version required)

### Installation

```bash
# Create Python 3.7 environment
conda create -n tft_env python=3.7
conda activate tft_env

# Install CUDA 10.0 and cuDNN
conda install anaconda cudatoolkit=10.0
conda install cudnn

# Install TensorFlow 1.15
pip install tensorflow-gpu==1.15.0

# Install other dependencies
pip install pandas==1.1.0 scikit-learn matplotlib seaborn
pip install pyqlib loguru pyyaml
```

### Basic Usage

```python
from shanghai_tft_runner import ShanghaiTFTRunner

# Initialize runner
runner = ShanghaiTFTRunner(
    model_folder="my_tft_models",
    gpu_id=0
)

# Run complete TFT workflow
success = runner.run_complete_tft_workflow(
    symbol_filter=["600000", "600036", "600519"],
    start_date="2020-01-01",
    end_date="2023-12-31"
)
```

### Command Line Usage

```bash
# Basic training
python shanghai_tft_runner.py --symbols 600000 600036 600519

# Custom date range
python shanghai_tft_runner.py \
    --symbols 600000 600036 600519 \
    --start-date 2021-01-01 \
    --end-date 2023-12-31 \
    --model-folder custom_tft_models

# Run examples
python example_tft_usage.py --example 1  # Basic training
python example_tft_usage.py --all        # All examples
```

## 📊 Key Features

### 1. Multi-Horizon Forecasting
- Predict stock returns 1, 3, 5, or 10 days ahead
- Quantile predictions (P10, P50, P90) for uncertainty estimation
- Configurable prediction horizons

### 2. Advanced Feature Engineering
- **Alpha158 Features**: 25+ technical indicators optimized for TFT
- **Calendar Features**: Day of week, month, year
- **Static Features**: Stock-specific constant features
- **Temporal Features**: Historical price and volume data

### 3. Built-in Interpretability
- **Attention Weights**: Understand which time steps are important
- **Variable Importance**: Identify most predictive features
- **Temporal Patterns**: Analyze prediction patterns over time

### 4. Production-Ready Pipeline
- **Data Quality Validation**: Automatic outlier detection and handling
- **Model Checkpointing**: Save and resume training
- **Performance Monitoring**: Comprehensive evaluation metrics
- **Scalable Architecture**: Handle multiple stocks efficiently

## 🔧 Configuration

### Model Configuration (shanghai_tft_config.yaml)

```yaml
# Key TFT hyperparameters
tft_hyperparameters:
  dropout_rate: 0.4              # Regularization
  hidden_layer_size: 160         # Model capacity
  learning_rate: 0.0001          # Training speed
  num_heads: 1                   # Attention heads
  num_epochs: 100                # Training epochs
  total_time_steps: 12           # Sequence length (6+6)
  num_encoder_steps: 6           # Historical lookback

# Data configuration
data_config:
  symbols: ["600000", "600036", "600519"]
  start_time: "2020-01-01"
  end_time: "2023-12-31"
  min_periods: 100               # Min trading days per stock
```

### Feature Configuration

```yaml
# Selected Alpha158 features for TFT
selected_features:
  # Price position features
  - "KLEN"   # (close - low) / (high - low)
  - "KLOW"   # (close - low) / (open - low)
  - "KMID"   # (close - open) / (high - low)
  
  # Volatility features  
  - "STD5"   # 5-day standard deviation
  - "STD10"  # 10-day standard deviation
  - "STD20"  # 20-day standard deviation
  
  # Momentum features
  - "ROC5"   # 5-day rate of change
  - "ROC10"  # 10-day rate of change
  - "ROC20"  # 20-day rate of change
```

## 📈 Usage Examples

### Example 1: Basic Training

```python
from shanghai_tft_runner import ShanghaiTFTRunner

runner = ShanghaiTFTRunner()
success = runner.run_complete_tft_workflow(
    symbol_filter=["600000", "600036"],
    start_date="2022-01-01",
    end_date="2023-06-30"
)
```

### Example 2: Step-by-Step Training

```python
runner = ShanghaiTFTRunner(model_folder="step_by_step_models")

# Step 1: Prepare data
runner.step1_prepare_shanghai_data(
    symbol_filter=["600000", "600036", "600519"],
    start_date="2021-01-01",
    end_date="2023-12-31"
)

# Step 2: Setup dataset
runner.step2_setup_tft_dataset()

# Step 3: Train model
runner.step3_train_tft_model(
    train_start="2021-01-01",
    train_end="2022-12-31",
    valid_start="2023-01-01",
    valid_end="2023-06-30",
    test_start="2023-07-01", 
    test_end="2023-12-31"
)

# Step 4: Generate predictions
predictions = runner.step4_generate_predictions()

# Step 5: Analyze results
analysis = runner.analyze_predictions(predictions)
```

### Example 3: Multi-Horizon Forecasting

```python
horizons = [1, 3, 5, 10]  # Different prediction horizons

for horizon in horizons:
    runner = ShanghaiTFTRunner(
        model_folder=f"tft_horizon_{horizon}d"
    )
    
    # Train model for specific horizon
    success = runner.run_complete_tft_workflow(
        symbol_filter=["600000", "600036"],
        start_date="2022-01-01",
        end_date="2023-06-30"
    )
    
    if success:
        predictions = runner.step4_generate_predictions()
        print(f"{horizon}-day predictions: {len(predictions)}")
```

## 🔍 Advanced Features

### Prediction Analysis

```python
# Generate detailed analysis
analysis = runner.analyze_predictions(predictions, save_analysis=True)

# Access prediction statistics
stats = analysis['prediction_stats']
print(f"Mean prediction: {stats['mean']:.4f}")
print(f"Prediction std: {stats['std']:.4f}")
print(f"Quantiles: {stats['quantiles']}")

# Temporal analysis
temporal = analysis['temporal_analysis']
print(f"Daily mean: {temporal['daily_mean_prediction']}")
print(f"Prediction days: {temporal['prediction_days']}")
```

### Interpretability Analysis

```python
from shanghai_tft_runner import TFTAnalyzer

# Initialize analyzer
analyzer = TFTAnalyzer(model_folder="my_tft_models")

# Extract attention weights
attention_weights = analyzer.extract_attention_weights(model, test_data)

# Generate interpretability report
report = analyzer.generate_interpretability_report(attention_weights)
print(report)
```

### Configuration-Based Training

```python
import yaml

# Load configuration
with open("shanghai_tft_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Use configuration parameters
runner = ShanghaiTFTRunner(
    model_folder=config['model_config']['model_folder'],
    gpu_id=config['model_config']['gpu_id']
)

# Train with config parameters
success = runner.run_complete_tft_workflow(
    symbol_filter=config['data_config']['symbols'],
    start_date=config['data_config']['start_time'],
    end_date=config['data_config']['end_time']
)
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Python Version Error
```
Error: TFT requires Python 3.6-3.7
Solution: conda create -n tft_env python=3.7
```

#### 2. GPU Not Found
```
Error: No GPU found. TFT requires GPU for training.
Solution: Install CUDA 10.0 and verify GPU availability
```

#### 3. TensorFlow Version Issues
```
Error: TensorFlow version incompatible
Solution: pip install tensorflow-gpu==1.15.0
```

#### 4. Memory Issues
```
Error: Out of GPU memory
Solution: Reduce batch size in config:
  minibatch_size: 64  # Reduce from 128
  hidden_layer_size: 80  # Reduce from 160
```

### Performance Optimization

```python
# Optimized configuration for limited resources
optimized_config = {
    'tft_hyperparameters': {
        'hidden_layer_size': 80,    # Smaller model
        'minibatch_size': 64,       # Smaller batches
        'num_epochs': 50,           # Fewer epochs
        'early_stopping_patience': 5
    }
}
```

## 📁 File Structure

```
scripts/data_collector/akshare/
├── shanghai_tft_runner.py          # Main TFT implementation
├── shanghai_tft_config.yaml        # Configuration file
├── example_tft_usage.py            # Usage examples
├── TFT_QLIB_GUIDE.md               # Comprehensive guide
├── README_TFT_SHANGHAI.md          # This file
├── qlib_data/                      # Qlib binary data
├── shanghai_tft_models/            # Trained models
└── source/                         # Raw data (DuckDB/CSV)
```

## 🎯 Best Practices

### 1. Data Preparation
- **Start Small**: Begin with 2-3 stocks for testing
- **Quality Check**: Ensure sufficient trading days per stock (100+ recommended)
- **Date Ranges**: Use reasonable date ranges (2-3 years for training)

### 2. Model Training
- **GPU Monitoring**: Monitor GPU memory usage during training
- **Early Stopping**: Use validation loss for early stopping
- **Checkpointing**: Save model checkpoints regularly

### 3. Evaluation
- **Multiple Metrics**: Use IC, Rank IC, MSE, and quantile losses
- **Out-of-Sample**: Always test on unseen data
- **Uncertainty**: Leverage quantile predictions for risk assessment

### 4. Production
- **Model Versioning**: Track model versions and hyperparameters
- **Performance Monitoring**: Monitor prediction quality over time
- **Retraining**: Establish regular retraining schedules

## 📚 Additional Resources

- **TFT_QLIB_GUIDE.md**: Comprehensive TFT guide with theory and implementation details
- **example_tft_usage.py**: 5 complete usage examples
- **shanghai_tft_config.yaml**: Full configuration reference
- **Original TFT Paper**: [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the comprehensive guide in `TFT_QLIB_GUIDE.md`
3. Run the examples in `example_tft_usage.py`
4. Verify system requirements (Python 3.6-3.7, CUDA 10.0, TensorFlow 1.15)

---

*This implementation provides a production-ready TFT pipeline for Shanghai stock market prediction with comprehensive documentation, examples, and best practices.*
