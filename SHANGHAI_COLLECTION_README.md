# Shanghai Stock Market Data Collection for Qlib

This project provides comprehensive tools to collect Shanghai Stock Exchange data from 2022.01.01 to 2025.01.01 using AkShare and convert it to Qlib format.

## 🚀 Quick Start

### Prerequisites

1. **Python Environment**: Use the `qlibbase` conda environment
2. **Required Packages**: All necessary packages are already installed in `qlibbase`
   - akshare (1.17.44)
   - qlib (0.9.8.dev3)
   - pandas, numpy, etc.

### Test Collection (Recommended First)

```bash
# Activate the environment
conda activate qlibbase

# Test with a small sample (3 stocks, recent dates)
python collect_shanghai_direct.py --start_date 20241201 --end_date 20241210 --limit_stocks 3 --delay 2.0

# Verify the test data works
python test_qlib_data.py
```

### Full Collection

```bash
# Run full collection (all Shanghai stocks, 2022-2025)
python run_full_shanghai_collection.py

# Or with custom parameters
python collect_shanghai_direct.py --start_date 20220101 --end_date 20250101 --delay 1.5
```

## 📁 Project Structure

```
├── collect_shanghai_direct.py      # Main collection script
├── run_full_shanghai_collection.py # Full collection runner
├── test_environment.py             # Environment verification
├── test_qlib_data.py              # Data verification
├── SHANGHAI_COLLECTION_README.md   # This file
└── qlib_shanghai_data/             # Generated data directory
    ├── csv_data/                   # Raw CSV files
    ├── qlib_data/                  # Qlib binary format
    │   ├── calendars/
    │   ├── features/
    │   └── instruments/
    └── logs/
```

## 🛠️ Scripts Overview

### 1. `collect_shanghai_direct.py`

**Main collection script** that:
- Fetches all Shanghai Stock Exchange stocks (codes starting with 60, 68, 90)
- Collects daily OHLCV data with forward adjustment (qfq)
- Saves data in CSV format
- Converts to Qlib binary format
- Includes comprehensive error handling and logging

**Usage:**
```bash
python collect_shanghai_direct.py [OPTIONS]

Options:
  --start_date YYYYMMDD    Start date (default: 20220101)
  --end_date YYYYMMDD      End date (default: 20250101)  
  --data_dir PATH          Data directory (default: ./qlib_shanghai_data)
  --limit_stocks N         Limit stocks for testing (default: all)
  --delay SECONDS          Delay between requests (default: 1.5)
```

### 2. `test_environment.py`

**Environment verification** that checks:
- Package imports (akshare, qlib, pandas, numpy)
- AkShare API connectivity
- Shanghai stock list retrieval
- Sample data collection
- Qlib collector imports

### 3. `test_qlib_data.py`

**Data verification** that tests:
- Qlib initialization with collected data
- Calendar loading
- Instrument listing
- Feature extraction
- Multi-stock queries

### 4. `run_full_shanghai_collection.py`

**Full collection runner** with:
- Progress tracking
- User confirmation for full collection
- Test mode option
- Automatic data verification

## 📊 Data Specifications

### Stock Coverage
- **Exchange**: Shanghai Stock Exchange (SSE)
- **Stock Types**: A-shares with codes starting with:
  - `60xxxx`: Main board stocks
  - `68xxxx`: STAR Market (科创板)
  - `90xxxx`: B-shares
- **Total Stocks**: ~2,283 stocks (as of 2024)

### Data Fields
- **Date**: Trading date
- **OHLCV**: Open, High, Low, Close, Volume
- **Amount**: Trading amount
- **Change**: Price change percentage
- **Turnover**: Turnover rate
- **Adjustment**: Forward adjustment (qfq) applied

### Date Range
- **Start**: 2022-01-01
- **End**: 2025-01-01
- **Frequency**: Daily
- **Trading Days**: ~730 trading days

## ⚙️ Configuration

### Rate Limiting
- **Default Delay**: 1.5 seconds between requests
- **Recommended**: 1.0-2.0 seconds to avoid rate limiting
- **API Limits**: AkShare has built-in rate limiting

### Performance Settings
- **Max Workers**: 4 (for Qlib conversion)
- **Memory Usage**: ~2-4GB during processing
- **Disk Space**: ~500MB-1GB for full dataset

### Error Handling
- **Network Errors**: Automatic retry with exponential backoff
- **Missing Data**: Graceful handling of delisted/suspended stocks
- **Data Validation**: Comprehensive checks for data quality
- **Logging**: Detailed logs for debugging

## 🚀 Usage Examples

### Basic Collection
```bash
# Collect recent data for testing
python collect_shanghai_direct.py --start_date 20241101 --end_date 20241130 --limit_stocks 10

# Full historical collection
python collect_shanghai_direct.py --start_date 20220101 --end_date 20250101
```

### Using Collected Data with Qlib
```python
import qlib
from qlib.data import D
from qlib.constant import REG_CN

# Initialize Qlib with collected data
qlib.init(provider_uri='./qlib_shanghai_data/qlib_data', region=REG_CN)

# Get stock data
data = D.features(['600000'], ['$close', '$volume'], 
                 start_time='2022-01-01', end_time='2024-12-31', freq='day')
print(data.head())

# Get all Shanghai stocks
instruments = D.list_instruments(D.instruments('all'), as_list=True)
print(f"Total stocks: {len(instruments)}")
```

### Integration with Qlib Workflows
```python
# Use with Qlib models
from qlib.workflow import R
from qlib.utils import init_instance_by_config

# Model configuration
model_config = {
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
        "num_threads": 20
    }
}

# Initialize and train model
model = init_instance_by_config(model_config)
# ... rest of workflow
```

## 📈 Performance Expectations

### Collection Time
- **Test (10 stocks, 1 month)**: ~30 seconds
- **Sample (100 stocks, 1 year)**: ~10 minutes  
- **Full (2283 stocks, 3 years)**: ~4-6 hours

### Data Size
- **Per Stock**: ~1-2KB per trading day
- **Full Dataset**: ~500MB-1GB total
- **Qlib Binary**: ~50% smaller than CSV

### Success Rate
- **Typical**: 95-98% success rate
- **Common Issues**: Delisted stocks, network timeouts
- **Retry Logic**: Automatic retry for failed requests

## 🔧 Troubleshooting

### Common Issues

1. **Network Errors**
   ```
   Error: Connection timeout
   Solution: Increase delay, check internet connection
   ```

2. **Rate Limiting**
   ```
   Error: Too many requests
   Solution: Increase --delay parameter to 2.0 or higher
   ```

3. **Missing Data**
   ```
   Warning: No data for stock XXXXXX
   Cause: Stock may be delisted or suspended
   Action: This is normal, script continues with other stocks
   ```

4. **Memory Issues**
   ```
   Error: Out of memory
   Solution: Process in smaller batches, increase system memory
   ```

### Debug Mode
```bash
# Enable debug logging
export PYTHONPATH=$PYTHONPATH:$(pwd)
python collect_shanghai_direct.py --limit_stocks 5 --delay 3.0
```

### Data Validation
```bash
# Verify environment
python test_environment.py

# Verify collected data
python test_qlib_data.py

# Check data completeness
ls -la qlib_shanghai_data/csv_data/ | wc -l
```

## 📋 Best Practices

### Before Collection
1. **Test Environment**: Run `test_environment.py`
2. **Small Test**: Collect 5-10 stocks first
3. **Check Disk Space**: Ensure 2-3GB free space
4. **Stable Network**: Use reliable internet connection

### During Collection
1. **Monitor Progress**: Check logs regularly
2. **Don't Interrupt**: Let the process complete
3. **Network Stability**: Avoid network-intensive tasks
4. **System Resources**: Monitor CPU/memory usage

### After Collection
1. **Verify Data**: Run `test_qlib_data.py`
2. **Check Completeness**: Verify expected number of stocks
3. **Backup Data**: Copy to secure location
4. **Document**: Note any issues or missing stocks

## 🤝 Contributing

To improve the collection scripts:

1. **Error Handling**: Add more robust error recovery
2. **Performance**: Optimize for faster collection
3. **Features**: Add support for more data types
4. **Documentation**: Improve usage examples

## 📄 License

This project follows the same license as the parent Qlib project.

## 🆘 Support

For issues:

1. **Check Logs**: Review detailed log files
2. **Test Environment**: Run verification scripts
3. **AkShare Issues**: Check AkShare documentation
4. **Qlib Issues**: Refer to Qlib documentation

---

**Happy Data Collecting! 📊🚀**

*Last Updated: 2025-09-08*