# Qlib Label Configuration and Objective Functions

## Overview

In Qlib, **labels define the objective function for training**. The label represents what you want to predict - typically future stock returns that can be used for ranking stocks or making trading decisions.

## 📊 Core Concepts

### 1. Label as Objective Function
Labels in Qlib define:
- **What to predict**: Future price movements, returns, volatility, etc.
- **Time horizon**: How far ahead to predict (1 day, 2 days, etc.)
- **Calculation method**: Formula for computing the target value

### 2. Label Expression Syntax
Qlib uses expression-based labels with operators like:
- `Ref($close, -2)`: Close price 2 days ago
- `$close`: Current close price
- `/`, `-`, `+`, `*`: Mathematical operations
- `Log()`, `Abs()`, `Rank()`: Functions

## 🔧 Implementation Methods

### Method 1: YAML Configuration Files

```yaml
# Example: workflow_config_lightgbm_Alpha158.yaml
task:
    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: Alpha158
                module_path: qlib.contrib.data.handler
                kwargs:
                    # Default label from Alpha158 class
                    # OR override with custom label:
                    label: ["Ref($close, -2) / Ref($close, -1) - 1"]
```

### Method 2: Python Code - Using Handler Classes

```python
# qlib/contrib/data/handler.py
class Alpha158(DataHandlerLP):
    def get_label_config(self):
        # Returns: (expressions, column_names)
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

class Alpha158vwap(Alpha158):
    def get_label_config(self):
        # Using VWAP instead of close price
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]
```

### Method 3: Python Code - Direct Configuration

```python
# examples/workflow_by_code.py
from qlib.tests.config import CSI300_GBDT_TASK
from qlib.utils import init_instance_by_config

# CSI300_GBDT_TASK contains default label configuration
model = init_instance_by_config(CSI300_GBDT_TASK["model"])
dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])

# Train model with the configured labels
model.fit(dataset)
```

## 📋 Label Examples and Use Cases

### 1. Single-Day Return Prediction (Most Common)
```python
# Predict next day return
label = ["Ref($close, -2) / Ref($close, -1) - 1"]
# Interpretation: (tomorrow_close / today_close) - 1
```

### 2. Multi-Day Return Prediction
```python
# Predict 2-day ahead return  
label = ["Ref($close, -3) / Ref($close, -1) - 1"]
# Interpretation: (day_after_tomorrow_close / today_close) - 1
```

### 3. Multi-Label Training (Multiple Horizons)
```python
# From TCTS example - predict multiple time horizons
label = [
    "Ref($close, -2) / Ref($close, -1) - 1",  # 1-day ahead
    "Ref($close, -3) / Ref($close, -1) - 1",  # 2-day ahead  
    "Ref($close, -4) / Ref($close, -1) - 1",  # 3-day ahead
]
```

### 4. VWAP-Based Labels
```python
# Use Volume Weighted Average Price
label = ["Ref($vwap, -2) / Ref($vwap, -1) - 1"]
```

### 5. Log Returns
```python
# Logarithmic returns (more stable)
label = ["Log(Ref($close, -2) / Ref($close, -1))"]
```

### 6. Absolute Return Labels  
```python
# Absolute returns (for volatility prediction)
label = ["Abs(Ref($close, -2) / Ref($close, -1) - 1)"]
```

## 🏗️ Complete Working Examples

### Example 1: Custom Handler with Custom Label

```python
# custom_handler.py
from qlib.contrib.data.handler import Alpha158

class CustomReturnHandler(Alpha158):
    def get_label_config(self):
        # 5-day forward return
        return ["Ref($close, -6) / Ref($close, -1) - 1"], ["RETURN_5D"]

# Usage in workflow
task_config = {
    "model": {
        "class": "LGBModel", 
        "module_path": "qlib.contrib.model.gbdt"
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset", 
        "kwargs": {
            "handler": {
                "class": "CustomReturnHandler",
                "module_path": "custom_handler",
                "kwargs": {
                    "instruments": "csi300",
                    "start_time": "2010-01-01",
                    "end_time": "2020-01-01"
                }
            }
        }
    }
}
```

### Example 2: Runtime Label Override

```python
# Override label at runtime
import qlib
from qlib.contrib.data.handler import Alpha158

# Initialize Qlib
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

# Create handler with custom label
handler = Alpha158(
    instruments="csi300",
    start_time="2010-01-01", 
    end_time="2020-01-01",
    # Override default label
    label=["Ref($close, -3) / Ref($close, -1) - 1"]  # 2-day ahead return
)

# Create dataset
from qlib.data.dataset import DatasetH
dataset = DatasetH(
    handler=handler,
    segments={
        "train": ("2010-01-01", "2016-12-31"),
        "valid": ("2017-01-01", "2018-12-31"), 
        "test": ("2019-01-01", "2020-01-01")
    }
)

# Train model
from qlib.contrib.model.gbdt import LGBModel
model = LGBModel()
model.fit(dataset)
```

### Example 3: Multiple Objective Functions (Multi-Task Learning)

```python
# Multi-task learning with multiple labels
class MultiTaskHandler(Alpha158):
    def get_label_config(self):
        return [
            "Ref($close, -2) / Ref($close, -1) - 1",      # Short-term return
            "Ref($close, -6) / Ref($close, -1) - 1",      # Medium-term return  
            "Abs(Ref($close, -2) / Ref($close, -1) - 1)"   # Volatility
        ], ["RETURN_1D", "RETURN_5D", "VOLATILITY_1D"]
```

## 🎯 Label Processing Pipeline

### Data Flow:
1. **Raw Data**: OHLCV price data
2. **Label Calculation**: Apply expression to compute target values
3. **Preprocessing**: Normalization, ranking, outlier handling
4. **Training**: Model learns to predict label values
5. **Prediction**: Generate scores for ranking stocks

### Label Processors:
```python
# Common label preprocessing steps
learn_processors = [
    {"class": "DropnaLabel"},           # Remove NaN labels
    {"class": "CSRankNorm",             # Cross-sectional ranking
     "kwargs": {"fields_group": "label"}}
]
```

## ⚙️ Advanced Label Configurations

### 1. Conditional Labels
```python
# Binary classification labels
label = ["If(Ref($close, -2) / Ref($close, -1) > 1.02, 1, 0)"]  # 1 if return > 2%
```

### 2. Sector-Relative Labels
```python
# Return relative to sector average
label = ["(Ref($close, -2) / Ref($close, -1) - 1) - Mean(Ref($close, -2) / Ref($close, -1) - 1)"]
```

### 3. Risk-Adjusted Labels  
```python
# Sharpe ratio style labels
label = ["(Ref($close, -2) / Ref($close, -1) - 1) / Std(Ref($close, -2) / Ref($close, -1) - 1, 20)"]
```

## 📊 Model Training with Labels

### Complete Training Example:
```python
import qlib
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH

# 1. Initialize
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

# 2. Define objective function via label
custom_label = ["Ref($close, -2) / Ref($close, -1) - 1"]

# 3. Create data handler
handler = Alpha158(
    instruments="csi300",
    start_time="2010-01-01",
    end_time="2020-01-01", 
    label=custom_label  # This is your objective function!
)

# 4. Create dataset with train/valid/test splits
dataset = DatasetH(
    handler=handler,
    segments={
        "train": ("2010-01-01", "2016-12-31"),  # Training period
        "valid": ("2017-01-01", "2018-12-31"),  # Validation period
        "test": ("2019-01-01", "2020-01-01")    # Test period
    }
)

# 5. Initialize model
model = LGBModel(
    loss="mse",  # Mean Squared Error loss matches regression label
    learning_rate=0.1,
    max_depth=8,
    num_leaves=210
)

# 6. Train model (optimizes to predict the label values)
model.fit(dataset)

# 7. Make predictions
predictions = model.predict(dataset)

# 8. The predictions can now be used for:
# - Stock ranking and selection
# - Portfolio construction
# - Trading signal generation
```

## 🔍 Key Insights

### 1. Label = Objective Function
- The label expression **directly defines what the model optimizes for**
- Different labels lead to different trading strategies
- Choose labels that align with your investment goals

### 2. Time Horizon Matters
- `Ref($close, -2)`: Next day prediction (high frequency trading)
- `Ref($close, -6)`: 5-day prediction (swing trading)  
- `Ref($close, -21)`: 20-day prediction (position trading)

### 3. Label Engineering Best Practices
- **Normalization**: Use relative returns rather than absolute prices
- **Stability**: Avoid labels with extreme outliers
- **Liquidity**: Ensure sufficient trading data for label calculation
- **Look-ahead Bias**: Be careful with future information leakage

### 4. Common Label Patterns
- **Return Prediction**: `Ref($close, -n) / Ref($close, -1) - 1`
- **Ranking Signals**: Cross-sectional ranking of returns
- **Binary Classification**: Threshold-based up/down labels
- **Multi-task**: Multiple time horizons or metrics

## 📁 Related Files Reference

- **Handler Definitions**: `qlib/contrib/data/handler.py` 
- **Configuration Examples**: `examples/benchmarks/*/workflow_config_*.yaml`
- **Data Loaders**: `qlib/contrib/data/loader.py`
- **Workflow Examples**: `examples/workflow_by_code.py`
- **Test Configurations**: `qlib/tests/config.py`

The label configuration is the **heart of your quantitative strategy** - it defines exactly what patterns the model will learn to recognize and predict!