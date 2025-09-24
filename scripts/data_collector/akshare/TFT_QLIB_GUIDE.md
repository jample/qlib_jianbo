# Temporal Fusion Transformer (TFT) with Qlib - Complete Guide

## 📋 Overview

The Temporal Fusion Transformer (TFT) is a state-of-the-art deep learning model for multi-horizon time series forecasting, developed by Google Research. This guide explains how to use TFT with Microsoft's Qlib framework for stock price prediction using Shanghai stock market data.

## 🏗️ TFT Architecture

### Key Components

1. **Variable Selection Networks**: Automatically selects relevant features
2. **Gated Residual Networks**: Provides non-linear processing with skip connections  
3. **Multi-Head Attention**: Captures temporal relationships across different time scales
4. **Quantile Forecasting**: Provides prediction intervals (P10, P50, P90)
5. **Interpretability**: Built-in attention weights for feature importance analysis

### TFT vs Standard Transformer

| Feature | TFT | Standard Transformer |
|---------|-----|---------------------|
| **Variable Selection** | Automatic feature selection | Manual feature engineering |
| **Temporal Patterns** | Multi-scale temporal modeling | Single-scale attention |
| **Uncertainty** | Quantile predictions (P10, P50, P90) | Point predictions |
| **Interpretability** | Attention weights + variable importance | Limited interpretability |
| **Input Types** | Static, known, observed inputs | Homogeneous inputs |

## 🎯 TFT Input Types

TFT categorizes inputs into three types:

### 1. Static Inputs
- **Definition**: Features that don't change over time for each entity
- **Examples**: Stock sector, market cap category, listing exchange
- **Usage**: `InputTypes.STATIC_INPUT`

### 2. Known Inputs  
- **Definition**: Future values are known at prediction time
- **Examples**: Calendar features (day of week, month, holidays), planned events
- **Usage**: `InputTypes.KNOWN_INPUT`

### 3. Observed Inputs
- **Definition**: Historical values only, future values unknown
- **Examples**: Stock prices, volume, technical indicators
- **Usage**: `InputTypes.OBSERVED_INPUT`

## 🔧 Qlib TFT Implementation Structure

### Core Files

```
examples/benchmarks/TFT/
├── tft.py                           # Main TFT model wrapper for Qlib
├── workflow_config_tft_Alpha158.yaml # Qlib workflow configuration
├── data_formatters/
│   ├── base.py                      # Base data formatter classes
│   └── qlib_Alpha158.py            # Alpha158-specific formatter
├── libs/
│   ├── tft_model.py                # Core TFT implementation
│   ├── utils.py                    # Utility functions
│   └── hyperparam_opt.py           # Hyperparameter optimization
└── expt_settings/
    └── configs.py                  # Experiment configurations
```

### Key Classes

#### 1. TFTModel (tft.py)
- **Purpose**: Qlib wrapper for TFT model
- **Inherits**: `ModelFT` (Qlib's base model class)
- **Key Methods**:
  - `fit()`: Train the TFT model
  - `predict()`: Generate predictions with quantiles
  - `_prepare_data()`: Convert Qlib data to TFT format

#### 2. Alpha158Formatter (data_formatters/qlib_Alpha158.py)
- **Purpose**: Formats Alpha158 features for TFT consumption
- **Key Methods**:
  - `set_scalers()`: Calibrate feature scaling
  - `transform_inputs()`: Apply scaling and preprocessing
  - `format_predictions()`: Convert predictions back to original scale

#### 3. TemporalFusionTransformer (libs/tft_model.py)
- **Purpose**: Core TFT implementation
- **Key Components**:
  - Variable selection networks
  - Gated residual networks
  - Multi-head attention mechanisms
  - Quantile loss functions

## 📊 Data Preparation for TFT

### Alpha158 Feature Mapping

TFT uses a subset of Alpha158 features optimized for temporal modeling:

```python
DATASET_SETTING = {
    "Alpha158": {
        "feature_col": [
            "RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10",
            "CORR5", "CORD5", "CORR10", "ROC60", "RESI10", 
            "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5",
            "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"
        ],
        "label_col": "LABEL0"
    }
}
```

### Data Transformation Pipeline

1. **Feature Selection**: Extract relevant Alpha158 features
2. **Label Shifting**: Apply temporal shift for prediction horizon
3. **Calendar Features**: Add day_of_week, month, year
4. **Scaling**: StandardScaler for numerical features
5. **Categorical Encoding**: LabelEncoder for categorical features

### Data Format Requirements

```python
# Required columns for TFT
columns = [
    'instrument',     # Entity identifier (CATEGORICAL, ID)
    'date',          # Time identifier (DATE, TIME) 
    'LABEL0',        # Target variable (REAL_VALUED, TARGET)
    'day_of_week',   # Calendar feature (CATEGORICAL, KNOWN_INPUT)
    'month',         # Calendar feature (CATEGORICAL, KNOWN_INPUT)
    'year',          # Calendar feature (CATEGORICAL, KNOWN_INPUT)
    # Alpha158 features (REAL_VALUED, OBSERVED_INPUT)
    'RESI5', 'WVMA5', 'RSQR5', ...
    'const'          # Constant feature (CATEGORICAL, STATIC_INPUT)
]
```

## ⚙️ Configuration and Hyperparameters

### Model Hyperparameters

```yaml
# Key TFT hyperparameters
model_params:
    dropout_rate: 0.4              # Dropout for regularization
    hidden_layer_size: 160         # Hidden layer size
    learning_rate: 0.0001          # Learning rate
    minibatch_size: 128            # Batch size
    max_gradient_norm: 0.0135      # Gradient clipping
    num_heads: 1                   # Attention heads
    stack_size: 1                  # Number of TFT blocks

# Fixed parameters
fixed_params:
    total_time_steps: 12           # Total sequence length (6 + 6)
    num_encoder_steps: 6           # Historical steps
    num_epochs: 100                # Training epochs
    early_stopping_patience: 10    # Early stopping patience
```

### Workflow Configuration

```yaml
task:
    model:
        class: TFTModel
        module_path: tft
    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: Alpha158
                module_path: qlib.contrib.data.handler
            segments:
                train: [2008-01-01, 2014-12-31]
                valid: [2015-01-01, 2016-12-31] 
                test: [2017-01-01, 2020-08-01]
```

## 🚀 Usage Patterns

### Pattern 1: Basic TFT Training

```python
import qlib
from qlib.utils import init_instance_by_config
import yaml

# Initialize qlib
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

# Load configuration
with open("workflow_config_tft_Alpha158.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Create dataset
dataset = init_instance_by_config(config['task']['dataset'])

# Create and train TFT model
model = TFTModel(DATASET="Alpha158", label_shift=5)
model.fit(dataset, MODEL_FOLDER="tft_models", USE_GPU_ID=0)

# Make predictions
predictions = model.predict(dataset)
```

### Pattern 2: Custom Feature Configuration

```python
# Define custom feature set
CUSTOM_FEATURES = [
    "KLEN", "KLOW", "KMID", "KUP",     # Price position
    "STD5", "STD10", "STD20",          # Volatility
    "ROC5", "ROC10", "ROC20",          # Momentum
    "MA5", "MA10", "MA20"              # Moving averages
]

# Update dataset configuration
DATASET_SETTING["Custom"] = {
    "feature_col": CUSTOM_FEATURES,
    "label_col": "LABEL0"
}

# Train with custom features
model = TFTModel(DATASET="Custom", label_shift=5)
```

### Pattern 3: Multi-Horizon Forecasting

```python
# Configure for different prediction horizons
horizons = [1, 3, 5, 10]  # 1-day, 3-day, 5-day, 10-day ahead

for horizon in horizons:
    model = TFTModel(
        DATASET="Alpha158", 
        label_shift=horizon
    )
    model.fit(dataset, MODEL_FOLDER=f"tft_models_h{horizon}")
    
    # Get quantile predictions
    predictions = model.predict(dataset)
    # predictions contains P10, P50, P90 forecasts
```

## 🔍 Advanced Features

### 1. Quantile Predictions

TFT provides uncertainty estimates through quantile forecasting:

```python
# After prediction, access different quantiles
output_map = model.model.predict(test_data, return_targets=True)

# Extract quantile predictions
p10_forecast = model.data_formatter.format_predictions(output_map["p10"])  # Lower bound
p50_forecast = model.data_formatter.format_predictions(output_map["p50"])  # Median
p90_forecast = model.data_formatter.format_predictions(output_map["p90"])  # Upper bound

# Combine for final prediction
final_prediction = (p50_forecast + p90_forecast) / 2
```

### 2. Attention Analysis

```python
# Extract attention weights for interpretability
attention_weights = model.model.get_attention_weights(test_data)

# Analyze temporal attention patterns
temporal_attention = attention_weights['temporal_attention']
variable_attention = attention_weights['variable_attention']

# Visualize most important time steps and features
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(temporal_attention.mean(axis=0))
plt.title('Temporal Attention Weights')
plt.xlabel('Time Steps')

plt.subplot(1, 2, 2)
plt.barh(range(len(FEATURES)), variable_attention.mean(axis=0))
plt.title('Variable Importance')
plt.yticks(range(len(FEATURES)), FEATURES)
```

### 3. Hyperparameter Optimization

```python
from libs.hyperparam_opt import HyperparamOptManager

# Define search space
param_ranges = {
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'hidden_layer_size': [80, 120, 160, 200, 240],
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
    'num_heads': [1, 2, 4, 8],
    'stack_size': [1, 2, 3]
}

# Run hyperparameter optimization
opt_manager = HyperparamOptManager(
    param_ranges=param_ranges,
    fixed_params=fixed_params,
    num_trials=50
)

best_params = opt_manager.optimize(train_data, valid_data)
```

## 🛠️ System Requirements

### Hardware Requirements

- **GPU**: CUDA-compatible GPU (required, CPU not supported)
- **Memory**: 8GB+ RAM, 4GB+ GPU memory
- **Storage**: 10GB+ for model checkpoints and data

### Software Requirements

```bash
# Core requirements
tensorflow-gpu==1.15.0  # Specific version required
pandas==1.1.0
numpy>=1.18.0
scikit-learn>=0.23.0

# CUDA requirements
cudatoolkit=10.0
cudnn

# Python version
python>=3.6,<3.8  # TFT only supports Python 3.6-3.7
```

### Installation Steps

```bash
# Create conda environment
conda create -n tft_env python=3.7
conda activate tft_env

# Install CUDA (if not already installed)
conda install anaconda cudatoolkit=10.0
conda install cudnn

# Install TensorFlow GPU
pip install tensorflow-gpu==1.15.0

# Install other requirements
pip install pandas==1.1.0 scikit-learn matplotlib seaborn

# Install qlib
pip install pyqlib
```

## ⚠️ Important Limitations

### 1. Python Version Constraint
- **Only supports Python 3.6-3.7**
- TensorFlow 1.15 compatibility issues with newer Python versions

### 2. GPU Requirement
- **Must run on GPU** - CPU execution will raise errors
- Requires CUDA 10.0 specifically

### 3. Memory Requirements
- Large memory footprint due to attention mechanisms
- May need to reduce batch size for limited GPU memory

### 4. TensorFlow Version Lock
- **Must use TensorFlow 1.15** - not compatible with TF 2.x
- Legacy TensorFlow session management

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Version Mismatch
```bash
# Error: CUDA version not compatible
# Solution: Install specific CUDA version
conda install anaconda cudatoolkit=10.0
conda install cudnn
```

#### 2. Python Version Issues
```bash
# Error: TensorFlow 1.15 not available for Python 3.8+
# Solution: Use Python 3.7
conda create -n tft_env python=3.7
```

#### 3. GPU Memory Issues
```python
# Error: Out of GPU memory
# Solution: Reduce batch size
model_params = {
    'minibatch_size': 64,  # Reduce from 128
    'hidden_layer_size': 80  # Reduce from 160
}
```

#### 4. Data Format Issues
```python
# Error: Column definition mismatch
# Solution: Ensure all required columns are present
required_columns = ['instrument', 'date', 'LABEL0', 'day_of_week', 'month', 'year', 'const']
missing_columns = set(required_columns) - set(df.columns)
if missing_columns:
    print(f"Missing columns: {missing_columns}")
```

## 📈 Performance Optimization

### 1. Data Preprocessing Optimization

```python
# Efficient data loading
def optimize_data_loading(df):
    # Use categorical dtypes for memory efficiency
    df['instrument'] = df['instrument'].astype('category')
    df['day_of_week'] = df['day_of_week'].astype('category')
    df['month'] = df['month'].astype('category')
    
    # Use float32 instead of float64
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    return df
```

### 2. Training Optimization

```python
# Optimized training parameters
optimized_params = {
    'num_epochs': 50,              # Reduce epochs with early stopping
    'early_stopping_patience': 5,  # Aggressive early stopping
    'minibatch_size': 256,         # Larger batch size if memory allows
    'multiprocessing_workers': 4,   # Parallel data loading
    'learning_rate': 0.001         # Higher learning rate for faster convergence
}
```

### 3. Model Size Optimization

```python
# Smaller model for faster training
compact_params = {
    'hidden_layer_size': 80,    # Reduce from 160
    'num_heads': 1,             # Single attention head
    'stack_size': 1,            # Single TFT block
    'dropout_rate': 0.3         # Moderate regularization
}
```

## 🎯 Best Practices

### 1. Data Preparation
- **Feature Selection**: Use domain knowledge to select relevant Alpha158 features
- **Data Quality**: Ensure no missing values in critical columns
- **Temporal Consistency**: Maintain consistent time intervals
- **Scaling**: Always apply proper scaling to numerical features

### 2. Model Training
- **Start Small**: Begin with smaller models and scale up
- **Monitor Overfitting**: Use validation loss for early stopping
- **GPU Utilization**: Monitor GPU memory usage during training
- **Checkpointing**: Save model checkpoints regularly

### 3. Evaluation
- **Multiple Metrics**: Use IC, Rank IC, and quantile losses
- **Out-of-Sample Testing**: Always test on unseen data
- **Uncertainty Analysis**: Leverage quantile predictions for risk assessment
- **Attention Analysis**: Use attention weights for model interpretation

### 4. Production Deployment
- **Model Versioning**: Track model versions and parameters
- **Inference Optimization**: Optimize for prediction speed
- **Monitoring**: Monitor model performance degradation
- **Retraining**: Establish retraining schedules

---

*This guide provides comprehensive coverage of using TFT with Qlib for stock price prediction. The combination of TFT's advanced temporal modeling capabilities with Qlib's quantitative finance framework offers powerful tools for financial forecasting.*
