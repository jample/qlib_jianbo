# AkShare Data Collector for Qlib

This data collector integrates AkShare with Qlib to provide comprehensive Chinese financial data collection capabilities.

## Features

- **Stock Data**: Daily and intraday stock prices with multiple adjustment types
- **Index Data**: Major market indices (CSI300, CSI500, SSE50, etc.)
- **Fund Data**: Mutual fund net values and portfolio holdings
- **Economic Data**: Macroeconomic indicators and financial statements
- **Real-time Data**: Live market data and quotes
- **Qlib Integration**: Seamless integration with Qlib data format

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Verify AkShare installation:
```python
import akshare as ak
print(ak.__version__)
```

## Usage

### Basic Data Collection

```bash
# Collect daily stock data
python collector.py download_data --start 20240101 --end 20241231 --interval 1d --delay 1

# Collect 1-minute data (recent data only)
python collector.py download_data --start 20241201 --end 20241231 --interval 1min --delay 1

# Limit number of stocks
python collector.py download_data --start 20240101 --end 20241231 --limit_nums 100
```

### Advanced Usage

```python
from collector import AkShareCollector, AkShareNormalize
from pathlib import Path

# Initialize collector
collector = AkShareCollector(
    save_dir="./akshare_data",
    start="20240101",
    end="20241231", 
    interval="1d",
    adjust="qfq",  # Forward adjustment
    delay=1
)

# Collect data
collector.collector_data()

# Normalize data
normalizer = AkShareNormalize()
# ... normalization process
```

### Data Types and Adjustments

**Adjustment Types:**
- `qfq`: Forward adjustment (前复权) - Recommended for backtesting
- `hfq`: Backward adjustment (后复权) 
- `""`: No adjustment (不复权)

**Intervals:**
- `1d`: Daily data
- `1min`: 1-minute intraday data (recent data only)

## Data Format

The collector outputs data in Qlib-compatible format:

| Column | Description |
|--------|-------------|
| date   | Trading date |
| open   | Opening price |
| close  | Closing price |
| high   | Highest price |
| low    | Lowest price |
| volume | Trading volume |
| money  | Trading amount |
| change | Price change percentage |
| symbol | Stock symbol |

## Integration with Qlib

After collecting data, you can use it with Qlib:

```python
import qlib
from qlib.data import D

# Initialize Qlib with collected data
qlib.init(provider_uri="./akshare_data", region="cn")

# Use the data
df = D.features(D.instruments("all"), ["$close", "$volume"], freq="day")
```

## AkShare 2025 Usage Examples

### 1. Stock Data Collection

```python
import akshare as ak

# Get stock list
stocks = ak.stock_info_a_code_name()

# Get daily data with forward adjustment
data = ak.stock_zh_a_hist(symbol="000001", period="daily", 
                         start_date="20240101", end_date="20251231", adjust="qfq")

# Get real-time data
realtime = ak.stock_zh_a_spot_em()
```

### 2. Index Data

```python
# Major indices
csi300 = ak.index_zh_a_hist(symbol="sh000300", period="daily", 
                           start_date="20240101", end_date="20251231")

# Index constituents
csi300_stocks = ak.index_stock_cons_csindex(symbol="000300")
```

### 3. Fund Data

```python
# Fund list
funds = ak.fund_name_em()

# Fund net value history
fund_hist = ak.fund_open_fund_info_em(fund="015198", indicator="累计净值走势")

# Fund holdings
holdings = ak.fund_portfolio_hold_em(symbol="015198", date="2025")
```

### 4. Economic Indicators

```python
# GDP data
gdp = ak.macro_china_gdp()

# CPI data  
cpi = ak.macro_china_cpi()

# Money supply
money_supply = ak.macro_china_money_supply()
```

### 5. Sector and Industry Data

```python
# Industry sectors
industries = ak.stock_board_industry_name_em()

# Concept boards
concepts = ak.stock_board_concept_name_em()

# Stocks in banking sector
bank_stocks = ak.stock_board_industry_cons_em(symbol="银行")
```

### 6. Financial Statements

```python
# Balance sheet
balance_sheet = ak.stock_balance_sheet_by_report_em(symbol="000001")

# Income statement
income_statement = ak.stock_profit_sheet_by_report_em(symbol="000001")

# Cash flow statement
cash_flow = ak.stock_cash_flow_sheet_by_report_em(symbol="000001")
```

## Best Practices for 2025

1. **Rate Limiting**: Use appropriate delays (1-2 seconds) between requests
2. **Error Handling**: Implement robust error handling for network issues
3. **Data Validation**: Validate data quality and completeness
4. **Incremental Updates**: Update only recent data to avoid redundant requests
5. **Caching**: Cache frequently accessed data to reduce API calls
6. **Monitoring**: Monitor data collection success rates and errors

## Troubleshooting

### Common Issues

1. **Network Errors**: Check internet connection and firewall settings
2. **Rate Limiting**: Increase delay between requests
3. **Data Missing**: Some stocks may be delisted or suspended
4. **Format Changes**: AkShare API may change, update collector accordingly

### Error Handling

The collector includes comprehensive error handling:
- Network timeouts and retries
- Data validation and cleaning
- Graceful handling of missing data
- Detailed logging for debugging

## Performance Tips

1. **Parallel Processing**: Use multiple workers for large datasets
2. **Batch Processing**: Process data in batches to manage memory
3. **Selective Collection**: Collect only required symbols and date ranges
4. **Data Compression**: Compress stored data to save space

## Contributing

To contribute improvements:
1. Follow the existing code structure
2. Add comprehensive error handling
3. Include unit tests
4. Update documentation

## License

This collector follows the same license as the parent Qlib project.
