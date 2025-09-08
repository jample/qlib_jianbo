# AkShare Data Collection for 2025

This directory contains comprehensive tools and examples for using AkShare to collect Chinese financial data in 2025. AkShare is a powerful Python library that provides access to various Chinese financial data sources.

## 🚀 Quick Start

### Installation

```bash
# Install AkShare
pip install akshare

# Install additional dependencies
pip install pandas numpy loguru fire schedule
```

### Basic Usage

```python
import akshare as ak

# Get stock data
stock_data = ak.stock_zh_a_hist(symbol="000001", period="daily", 
                               start_date="20240101", end_date="20251231", adjust="qfq")

# Get real-time data
realtime_data = ak.stock_zh_a_spot_em()

# Get fund data
fund_data = ak.fund_portfolio_hold_em(symbol="015198", date="2025")
```

## 📁 File Structure

```
aksharedata/
├── README.md                 # This file
├── config.json              # Configuration file
├── data_updater.py          # Main data collection script
├── usage_examples.py        # Comprehensive usage examples
├── akshare_2025_guide.ipynb # Jupyter notebook guide
└── test.ipynb              # Your original test notebook
```

## 🛠️ Main Components

### 1. Data Updater (`data_updater.py`)

A comprehensive data collection and updating system with features:

- **Multi-threaded data collection**
- **Automated daily updates**
- **Historical data backfilling**
- **Data validation and quality checks**
- **Configurable settings**
- **Error handling and logging**

#### Usage Examples:

```bash
# Daily update
python data_updater.py --mode daily --symbols 000001,000002,600000

# Backfill historical data
python data_updater.py --mode backfill --start-date 20240101 --end-date 20241231

# Real-time data collection
python data_updater.py --mode realtime

# Use custom configuration
python data_updater.py --mode daily --config config.json --limit 50
```

### 2. Usage Examples (`usage_examples.py`)

Comprehensive examples covering:

- Stock data collection (daily, intraday, real-time)
- Index data and constituents
- Fund data and holdings
- Economic indicators
- Sector and industry data
- Financial statements
- Data analysis techniques
- Export and integration methods

### 3. Configuration (`config.json`)

Customizable settings for:

- Default stock symbols
- Major indices
- Fund codes
- API rate limiting
- Data validation options
- Output formats

### 4. Jupyter Notebook Guide (`akshare_2025_guide.ipynb`)

Interactive guide with step-by-step examples for:

- Getting started with AkShare
- Data collection best practices
- Integration with Qlib
- Advanced usage patterns

## 📊 Data Types Supported

### Stock Data
- **Daily OHLCV data** with multiple adjustment types
- **Intraday data** (1-minute, 5-minute, etc.)
- **Real-time quotes** and market data
- **Financial statements** (balance sheet, income statement, cash flow)
- **Corporate actions** and dividends

### Index Data
- **Major indices**: CSI300, CSI500, SSE50, ChiNext, STAR50
- **Index constituents** and weights
- **Sector indices** and thematic indices

### Fund Data
- **Mutual fund net values** and performance
- **Fund holdings** and portfolio composition
- **Fund rankings** and ratings

### Economic Data
- **Macroeconomic indicators**: GDP, CPI, PMI
- **Monetary policy**: Interest rates, money supply
- **Government data**: Fiscal data, trade statistics

### Market Data
- **Sector performance** and industry classification
- **Concept boards** and thematic investing
- **Market sentiment** indicators

## 🔧 Advanced Features

### Data Collection Strategies

1. **Incremental Updates**: Only collect new data since last update
2. **Batch Processing**: Handle large datasets efficiently
3. **Error Recovery**: Robust error handling and retry mechanisms
4. **Rate Limiting**: Respect API limits to avoid blocking

### Data Quality Assurance

1. **Validation Checks**: Verify data completeness and accuracy
2. **Anomaly Detection**: Identify and flag unusual data points
3. **Data Cleaning**: Handle missing values and outliers
4. **Format Standardization**: Consistent data formats across sources

### Integration Options

1. **Qlib Integration**: Direct compatibility with Qlib data format
2. **Database Storage**: Support for various database backends
3. **Cloud Storage**: Integration with cloud storage services
4. **Real-time Streaming**: Live data feeds for trading systems

## 📈 Best Practices for 2025

### 1. Rate Limiting
```python
import time
time.sleep(1.5)  # 1.5 second delay between requests
```

### 2. Error Handling
```python
try:
    data = ak.stock_zh_a_hist(symbol="000001", ...)
    if data.empty:
        logger.warning("No data received")
except Exception as e:
    logger.error(f"API error: {e}")
```

### 3. Data Validation
```python
def validate_stock_data(df):
    required_cols = ['日期', '开盘', '收盘', '最高', '最低', '成交量']
    return all(col in df.columns for col in required_cols)
```

### 4. Concurrent Processing
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(collect_data, symbol) for symbol in symbols]
```

### 5. Data Caching
```python
import pickle
from pathlib import Path

def cache_data(data, filename):
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    with open(cache_dir / filename, 'wb') as f:
        pickle.dump(data, f)
```

## 🔄 Automated Data Collection

### Daily Update Schedule

```python
import schedule
import time

def daily_job():
    updater = AkShareDataUpdater()
    updater.daily_update()

schedule.every().day.at("18:00").do(daily_job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Systemd Service (Linux)

Create `/etc/systemd/system/akshare-updater.service`:

```ini
[Unit]
Description=AkShare Data Updater
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/aksharedata
ExecStart=/usr/bin/python3 data_updater.py --mode daily
Restart=always

[Install]
WantedBy=multi-user.target
```

## 🚨 Common Issues and Solutions

### 1. Network Errors
- **Issue**: Connection timeouts or network errors
- **Solution**: Implement retry logic with exponential backoff

### 2. Rate Limiting
- **Issue**: Too many requests error
- **Solution**: Increase delay between requests, use connection pooling

### 3. Data Format Changes
- **Issue**: AkShare API changes column names or formats
- **Solution**: Regular testing and flexible column mapping

### 4. Memory Issues
- **Issue**: Out of memory with large datasets
- **Solution**: Process data in chunks, use generators

## 📚 Additional Resources

### Official Documentation
- [AkShare Official Docs](https://akshare.akfamily.xyz/)
- [AkShare GitHub](https://github.com/akfamily/akshare)

### Community Resources
- [AkShare Examples](https://github.com/akfamily/akshare/tree/master/example)
- [Qlib Documentation](https://qlib.readthedocs.io/)

### Data Sources
- East Money (东方财富)
- Sina Finance (新浪财经)
- Tencent Finance (腾讯财经)
- Wind (万得)
- Choice (东方财富Choice)

## 🤝 Contributing

To contribute improvements:

1. Fork the repository
2. Create a feature branch
3. Add comprehensive error handling
4. Include unit tests
5. Update documentation
6. Submit a pull request

## 📄 License

This project follows the same license as the parent Qlib project.

## 🆘 Support

For issues and questions:

1. Check the [AkShare FAQ](https://akshare.akfamily.xyz/introduction.html#faq)
2. Review the usage examples in this directory
3. Check the logs for detailed error messages
4. Open an issue with detailed reproduction steps

---

**Happy Data Collecting! 📊🚀**
