# Qlib Standard Workflow for Shanghai Stock Data

This document describes the **proper qlib workflow** for processing Shanghai stock market data using Microsoft's Qlib quantitative investment platform with **standard qlib mechanisms**.

## 🎯 Overview

This implementation follows **qlib's standard workflow patterns**:

- **YAML Configuration**: Uses qlib's standard YAML config format (like `workflow_config_lightgbm_Alpha158_csi500.yaml`)
- **qrun Compatibility**: Compatible with qlib's `qrun` command
- **Native Alpha158 Handler**: Uses `qlib.contrib.data.handler.Alpha158`
- **Qlib Binary Format**: Converts data to qlib's optimized binary storage
- **Standard Task Structure**: Follows qlib's task/model/dataset/record structure

## 🏗️ Architecture

```
DuckDB Data → Qlib Binary Format → Alpha158 Handler → Dataset → Model Training
     ↓              ↓                    ↓             ↓           ↓
  Raw OHLCV    Binary Files        158 Features    Train/Val/Test  LightGBM
```

## 📁 Files Structure

```
scripts/data_collector/akshare/
├── workflow_config_shanghai_alpha158.yaml  # Standard qlib YAML config
├── qlib_workflow_runner.py                 # Qlib workflow runner
├── demo_qlib_alpha158.py                   # Alpha158 feature demonstration
├── README_qlib_workflow.md                 # This documentation
├── qlib_data/                              # Qlib binary data directory
│   ├── features/                           # Binary feature files per instrument
│   ├── instruments/                        # Stock instrument lists
│   └── calendars/                          # Trading calendars
└── mlruns/                                 # MLflow experiment tracking
    └── shanghai_alpha158/                  # Experiment results
```

## 🔧 Standard Qlib Configuration

### **YAML Configuration Structure**

The workflow uses qlib's standard YAML format (similar to `workflow_config_lightgbm_Alpha158_csi500.yaml`):

```yaml
qlib_init:
    provider_uri: "scripts/data_collector/akshare/qlib_data"
    region: cn

data_handler_config: &data_handler_config
    start_time: 2022-01-01
    end_time: 2024-12-31
    fit_start_time: 2022-01-01
    fit_end_time: 2023-12-31
    instruments: all
    infer_processors:
        - class: ProcessInf
        - class: ZScoreNorm
        - class: Fillna
    learn_processors:
        - class: DropnaLabel
        - class: CSZScoreNorm

task:
    model:
        class: LGBModel
        module_path: qlib.contrib.model.gbdt
    dataset:
        class: DatasetH
        kwargs:
            handler:
                class: Alpha158
                module_path: qlib.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: [2022-01-01, 2023-12-31]
                valid: [2024-01-01, 2024-06-30]
                test: [2024-07-01, 2024-12-31]
```

## 🔧 Alpha158 Features Logic

### **Qlib's Alpha158 Implementation**

The workflow uses qlib's native `Alpha158DL.get_feature_config()` which generates **158 features**:

#### **1. K-bar Features (9 features)**
```python
"KMID": "($close-$open)/$open",           # Mid price ratio
"KLEN": "($high-$low)/$open",             # Length ratio  
"KMID2": "($close-$open)/($high-$low+1e-12)", # Mid/Length ratio
"KUP": "($high-Greater($open, $close))/$open", # Upper shadow
"KLOW": "(Less($open, $close)-$low)/$open",    # Lower shadow
# ... and 4 more variations
```

#### **2. Price Features (20 features)**
```python
# Normalized prices for windows [0,1,2,3,4]
"OPEN0": "$open/$close",     # Current open/close
"HIGH1": "Ref($high,1)/$close", # Previous high/current close
# ... for OPEN, HIGH, LOW, CLOSE, VWAP
```

#### **3. Volume Features (5 features)**
```python
"VOLUME0": "$volume/($volume+1e-12)",      # Current volume
"VOLUME1": "Ref($volume,1)/($volume+1e-12)", # Previous volume
# ... for windows [0,1,2,3,4]
```

#### **4. Rolling Features (124 features)**
```python
# For windows [5,10,20,30,60] and operators [ROC,MA,STD,RANK,MAX,MIN]
"ROC5": "($close/Ref($close,5)-1)",       # 5-day return
"MA10": "Mean($close,10)/$close",         # 10-day MA ratio
"STD20": "Std($close,20)/$close",         # 20-day volatility
"RANK30": "Rank($close,30)",              # 30-day rank
# ... total 124 combinations
```

### **Feature Configuration**

The Alpha158 features are configured in `qlib_config.json`:

```json
{
  "alpha158_config": {
    "kbar": {},                           # Enable K-bar features
    "price": {
      "windows": [0, 1, 2, 3, 4],        # Price lookback windows
      "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]
    },
    "volume": {
      "windows": [0, 1, 2, 3, 4]         # Volume lookback windows  
    },
    "rolling": {
      "windows": [5, 10, 20, 30, 60],    # Rolling windows
      "include": ["ROC", "MA", "STD", "RANK", "MAX", "MIN"],
      "exclude": []                       # Operators to exclude
    }
  }
}
```

## 🚀 Usage

### **Method 1: Standard qrun Command (Recommended)**

```bash
# Step 1: Prepare data for qlib
cd /root/mycode/qlibjianbo
python scripts/data_collector/akshare/qlib_workflow_runner.py --prepare-only

# Step 2: Run qlib workflow using standard qrun
qrun scripts/data_collector/akshare/workflow_config_shanghai_alpha158.yaml
```

### **Method 2: Integrated Workflow Runner**

```bash
# Run complete workflow (data preparation + qlib workflow)
python scripts/data_collector/akshare/qlib_workflow_runner.py

# Custom parameters
python scripts/data_collector/akshare/qlib_workflow_runner.py \
    --symbols 600000 600036 600519 \
    --start-date 2023-01-01 \
    --end-date 2024-12-31 \
    --experiment-name my_shanghai_experiment
```

### **Method 3: Demo and Testing**

```bash
# Show Alpha158 feature logic
python scripts/data_collector/akshare/demo_qlib_alpha158.py

# Show complete 184 features
python scripts/data_collector/akshare/show_full_alpha158.py
```

## 📊 Workflow Steps

### **Step 1: Data Preparation**
- Extracts data from DuckDB using configurable parameters
- Converts to qlib binary format with proper directory structure
- Creates instruments list (`all.txt`) and trading calendar (`day.txt`)
- Saves OHLCV data as binary files per instrument

### **Step 2: Qlib Workflow Execution**
- Uses qlib's standard `workflow()` function or `qrun` command
- Initializes qlib with converted binary data
- Creates Alpha158 data handler with 184 features
- Applies data processors (normalization, cleaning, missing value handling)
- Creates train/validation/test dataset splits
- Trains LightGBM model on Alpha158 features
- Records results using qlib's recording system

## 🔄 Qlib Workflow Components

### **1. qlib_init Section**
```yaml
qlib_init:
    provider_uri: "scripts/data_collector/akshare/qlib_data"  # Our binary data
    region: cn                                               # China region
```

### **2. Data Handler Configuration**
```yaml
data_handler_config: &data_handler_config
    start_time: 2022-01-01
    end_time: 2024-12-31
    instruments: all                    # Use all instruments in our data
    infer_processors:                   # Feature processing
        - class: ProcessInf             # Handle infinite values
        - class: ZScoreNorm            # Z-score normalization
        - class: Fillna                # Fill missing values
    learn_processors:                   # Label processing
        - class: DropnaLabel           # Drop samples with NaN labels
        - class: CSZScoreNorm          # Cross-sectional Z-score for labels
```

### **3. Task Configuration**
```yaml
task:
    model:                             # LightGBM model
        class: LGBModel
        module_path: qlib.contrib.model.gbdt
    dataset:                           # Dataset with Alpha158 handler
        class: DatasetH
        kwargs:
            handler:
                class: Alpha158        # Native qlib Alpha158 handler
                module_path: qlib.contrib.data.handler
            segments:                  # Train/valid/test splits
                train: [2022-01-01, 2023-12-31]
                valid: [2024-01-01, 2024-06-30]
                test: [2024-07-01, 2024-12-31]
```

## 🔍 Key Advantages

### **1. Native Qlib Integration**
- Uses qlib's optimized Alpha158 implementation
- Leverages qlib's data processing pipeline
- Compatible with qlib's model zoo

### **2. Performance Optimized**
- Binary data format for fast I/O
- Vectorized feature calculations
- Efficient memory usage

### **3. Configurable & Extensible**
- JSON-based configuration
- Modular step execution
- Easy to customize features

### **4. Production Ready**
- Follows qlib workflow patterns
- Comprehensive error handling
- Detailed logging and monitoring

## 📈 Expected Output

After successful execution:

```
qlib_output/
├── model/
│   ├── trained_model.pkl       # LightGBM model
│   └── alpha158_handler.pkl    # Data handler
└── results/
    ├── predictions.pkl         # All predictions
    └── summary.json           # Performance metrics
```

**Performance Metrics:**
- Train/Valid/Test correlations
- Feature importance scores
- Model configuration details
- Processing timestamps

## 🛠️ Troubleshooting

### **Common Issues:**

1. **DuckDB Connection Error**
   - Check database path in config
   - Verify data exists for specified date range

2. **Qlib Initialization Failed**
   - Ensure binary data conversion completed
   - Check qlib_data directory structure

3. **Memory Issues**
   - Reduce date range or symbol count
   - Adjust model parameters

4. **Feature Calculation Slow**
   - Qlib's implementation is optimized
   - Much faster than custom implementation

## 🔄 Integration with Existing Pipeline

This qlib workflow can replace the custom pipeline:

```python
# Old approach (custom implementation)
from alpha158_calculator import Alpha158Calculator
calculator = Alpha158Calculator()
features = calculator.calculate_features(df)

# New approach (qlib native)
from qlib.contrib.data.handler import Alpha158
handler = Alpha158(**config)
features = handler.fetch(col_set="feature")
```

The qlib approach provides the same 158 features but with better performance, reliability, and integration with the broader qlib ecosystem.
