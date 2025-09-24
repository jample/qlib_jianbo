# AkShare Data Collector (collector_standalone.py)

## Overview

`collector_standalone.py` is a comprehensive, rate-limited data collector for Chinese stock market data using the AkShare library. It follows enterprise-grade design patterns with robust error handling, incremental updates, resume capabilities, and efficient DuckDB storage.

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Database Schema](#database-schema)
- [Commands Reference](#commands-reference)
- [Usage Examples](#usage-examples)
- [Logic Flow](#logic-flow)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Key Features

### 🚀 **Core Capabilities**
- **Shanghai Stock Exchange Focus**: Specialized for Shanghai stocks (6xxxxx codes)
- **Rate Limiting**: Built-in 12-second minimum delays to prevent API blocking
- **Resume Capability**: Automatically resumes interrupted downloads
- **Incremental Updates**: Only downloads missing data ranges
- **Dual Storage**: Both CSV files and DuckDB database storage
- **Intelligent Retry Logic**: 3-attempt retry with permanent failure marking
- **Status Tracking**: Active/NA status management for stocks

### 📊 **Data Management**
- **Smart Filtering**: Excludes delisted and suspended stocks
- **Metadata Tracking**: Comprehensive update history and statistics
- **Transaction Safety**: Database operations with rollback on errors
- **Duplicate Prevention**: Handles data conflicts intelligently
- **Cache Management**: Symbol caching with automatic refresh

## Architecture

### Class Hierarchy
```
AkShareCollector (Abstract Base)
├── AkShareCollectorCN (China Market)
    └── AkShareCollectorCN1d (Daily Data)
        
Run (Command Interface)
├── download_data()
├── update_data()
├── query_data()
├── refresh_cache()
├── clean_cache()
├── show_cache_info()
└── reset_failed_stocks()
```

### Core Components

1. **Data Collector**: Handles API calls and data retrieval
2. **Database Manager**: DuckDB operations and schema management
3. **Cache System**: Stock symbol caching and validation
4. **Retry Engine**: Intelligent failure handling and retry logic
5. **Progress Tracker**: Real-time download progress and statistics

## Database Schema

### Primary Tables

#### `stock_data` - Main Data Storage
```sql
CREATE TABLE stock_data (
    symbol VARCHAR NOT NULL,           -- Stock code (e.g., '600000')
    date DATE NOT NULL,               -- Trading date
    open DECIMAL(10,2),               -- Opening price
    high DECIMAL(10,2),               -- Highest price
    low DECIMAL(10,2),                -- Lowest price
    close DECIMAL(10,2),              -- Closing price
    volume BIGINT,                    -- Trading volume
    amount DECIMAL(18,2),             -- Trading amount
    amplitude DECIMAL(8,4),           -- Price amplitude
    change_percent DECIMAL(8,4),      -- Change percentage
    change_amount DECIMAL(10,2),      -- Change amount
    turnover_rate DECIMAL(8,4),       -- Turnover rate
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date)
);
```

#### `stock_update_metadata` - Update Tracking
```sql
CREATE TABLE stock_update_metadata (
    symbol VARCHAR PRIMARY KEY,
    name VARCHAR,                     -- Stock name
    first_date DATE,                  -- First data date
    last_date DATE,                   -- Last data date
    total_records INTEGER DEFAULT 0,  -- Total record count
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR DEFAULT 'active'
);
```

#### `shanghai_stocks` - Symbol Cache
```sql
CREATE TABLE shanghai_stocks (
    symbol VARCHAR PRIMARY KEY,
    name VARCHAR,
    cached_date TIMESTAMP,
    exchange VARCHAR,
    status VARCHAR DEFAULT 'active',  -- 'active' or 'NA'
    retry_count INTEGER DEFAULT 0,
    last_attempt_date TIMESTAMP,
    failure_reason VARCHAR
);
```

### Database Files
- **Main Database**: `source/shanghai_stock_data.duckdb`
- **Symbol Cache**: `source/stock_symbols_cache.duckdb`
- **CSV Backup**: `source/{symbol}.csv` files

## Commands Reference

### 1. download_data - Primary Data Collection

**Purpose**: Download stock data with automatic resume capability

**Syntax**:
```bash
python collector_standalone.py download_data [OPTIONS]
```

**Key Parameters**:
- `--start`: Start date (YYYY-MM-DD)
- `--end`: End date (YYYY-MM-DD)
- `--data_type`: Data type ("stock" or "fund")
- `--shanghai_only`: Filter Shanghai stocks only (True/False)
- `--delay`: Delay between requests (minimum 12 seconds)
- `--force`: Force complete redownload (True/False)
- `--limit_nums`: Limit number of stocks for testing

**Resume Logic**:
- Without `--force`: Automatically detects existing data and resumes
- With `--force=True`: Ignores existing data, starts fresh

### 2. update_data - Incremental Updates

**Purpose**: Update existing data with only missing dates

**Syntax**:
```bash
python collector_standalone.py update_data [OPTIONS]
```

**Behavior**:
- Always uses incremental mode
- Calculates missing date ranges per stock
- Skips stocks that are up to date

### 3. query_data - Data Retrieval

**Purpose**: Query stored data from DuckDB

**Syntax**:
```bash
python collector_standalone.py query_data [OPTIONS]
```

**Parameters**:
- `--symbol`: Specific stock symbol
- `--start_date`: Query start date
- `--end_date`: Query end date
- `--limit`: Maximum records to return

### 4. Cache Management Commands

#### refresh_cache - Update Symbol Cache
```bash
python collector_standalone.py refresh_cache
```

#### clean_cache - Remove Inactive Stocks
```bash
python collector_standalone.py clean_cache
```

#### show_cache_info - Cache Statistics
```bash
python collector_standalone.py show_cache_info
```

#### reset_failed_stocks - Reset NA Status
```bash
python collector_standalone.py reset_failed_stocks [--symbols SYMBOLS]
```

## Usage Examples

### 🚀 **Getting Started**

#### 1. First-Time Full Download
```bash
# Download all Shanghai stocks from 2022 to current date
python collector_standalone.py download_data \
    --start 2022-01-01 \
    --end 2025-12-31 \
    --data_type stock \
    --shanghai_only True \
    --delay 12
```

#### 2. Test with Limited Stocks
```bash
# Test with first 10 stocks only
python collector_standalone.py download_data \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --data_type stock \
    --shanghai_only True \
    --delay 12 \
    --limit_nums 10
```

### 🔄 **Resume Operations**

#### 3. Resume Interrupted Download
```bash
# Automatically resumes from where you left off
python collector_standalone.py download_data \
    --start 2022-01-01 \
    --end 2025-12-31 \
    --data_type stock \
    --shanghai_only True \
    --delay 12
```

#### 4. Force Complete Restart
```bash
# Ignore existing data, start fresh
python collector_standalone.py download_data \
    --start 2022-01-01 \
    --end 2025-12-31 \
    --data_type stock \
    --shanghai_only True \
    --delay 12 \
    --force True
```

### 📈 **Data Updates**

#### 5. Daily Incremental Update
```bash
# Update to current date (run daily)
python collector_standalone.py update_data \
    --end $(date +%Y-%m-%d)
```

#### 6. Specific Date Range Update
```bash
# Update specific missing period
python collector_standalone.py update_data \
    --start 2024-06-01 \
    --end 2024-12-31
```

### 🔍 **Data Queries**

#### 7. Query Specific Stock
```bash
# Get latest 100 records for 600000
python collector_standalone.py query_data \
    --symbol 600000 \
    --limit 100
```

#### 8. Query Date Range
```bash
# Get all stocks for specific month
python collector_standalone.py query_data \
    --start_date 2024-01-01 \
    --end_date 2024-01-31 \
    --limit 1000
```

#### 9. Database Statistics
```bash
# Show overall database statistics
python collector_standalone.py query_data --limit 1
```

### 🔧 **Maintenance Operations**

#### 10. Check Cache Status
```bash
# View symbol cache information
python collector_standalone.py show_cache_info
```

#### 11. Refresh Symbol Cache
```bash
# Update stock symbol list from AkShare
python collector_standalone.py refresh_cache
```

#### 12. Reset Failed Stocks
```bash
# Reset all failed stocks for retry
python collector_standalone.py reset_failed_stocks

# Reset specific stocks
python collector_standalone.py reset_failed_stocks \
    --symbols "600001,600002,600003"
```

#### 13. Clean Inactive Stocks
```bash
# Remove delisted stocks from cache
python collector_standalone.py clean_cache
```

## Logic Flow

### Download Process Flow

```
1. Initialize Collector
   ├── Load configuration
   ├── Set rate limiting (12s minimum)
   └── Initialize DuckDB connections

2. Check Existing Data (if not --force)
   ├── Query database for existing records
   ├── If data exists: Switch to incremental mode
   └── If no data: Use full download mode

3. Symbol Management
   ├── Load Shanghai stock symbols from cache
   ├── Filter out delisted/suspended stocks
   ├── Exclude stocks marked as 'NA'
   └── Apply limit_nums if specified

4. For Each Stock:
   ├── Calculate incremental date range
   │   ├── Get last update date from metadata
   │   ├── Compare with requested range
   │   └── Determine: full/incremental/up_to_date/gap_fill
   │
   ├── Download Data (with retry logic)
   │   ├── Attempt 1: Call AkShare API
   │   ├── Attempt 2: Retry if failed
   │   ├── Attempt 3: Final attempt
   │   └── Mark as 'NA' if all attempts fail
   │
   └── Store Data
       ├── Save to CSV file
       ├── Store in DuckDB (transactional)
       └── Update metadata

5. Final Summary
   ├── Success/failure statistics
   ├── Failed stocks list
   └── Performance metrics
```

### Incremental Date Calculation Logic

```python
def _calculate_incremental_date_range(symbol, start, end):
    last_date = get_last_update_date(symbol)
    
    if last_date is None:
        return start, end, "full"          # No data exists
    
    if last_date >= end:
        return None, None, "up_to_date"    # Already current
    
    if last_date < start:
        return start, end, "gap_fill"      # Gap in data
    
    # Normal incremental: next day after last update
    return last_date + 1_day, end, "incremental"
```

### Retry Logic Flow

```
For each API call:
1. Check stock status
   ├── If status == 'NA': Skip (marked as failed)
   └── If status == 'active': Proceed

2. Retry Loop (max 3 attempts)
   ├── Call AkShare API
   ├── Apply 12-second rate limiting
   ├── Check if data returned
   │   ├── Success: Return data
   │   └── Failure: Continue to next attempt
   └── Update retry count

3. After All Attempts Failed
   ├── Mark status as 'NA'
   ├── Record failure reason
   ├── Update retry count
   └── Log permanent failure
```

## Error Handling

### Common Error Types

1. **API Rate Limiting**
   - **Symptom**: Frequent empty responses
   - **Handling**: 12-second minimum delays
   - **Solution**: Automatic rate limiting enforcement

2. **Network Timeouts**
   - **Symptom**: Connection errors
   - **Handling**: 3-attempt retry logic
   - **Solution**: Exponential backoff between attempts

3. **Invalid Stock Symbols**
   - **Symptom**: Consistent empty responses
   - **Handling**: Mark as 'NA' after 3 failures
   - **Solution**: Symbol cache filtering

4. **Database Conflicts**
   - **Symptom**: Transaction errors
   - **Handling**: Rollback on errors
   - **Solution**: DELETE + INSERT pattern for updates

5. **Date Range Issues**
   - **Symptom**: No data for valid stocks
   - **Handling**: Skip non-trading days automatically
   - **Solution**: AkShare handles market holidays

### Status Management

- **active**: Stock is working normally
- **NA**: Stock failed 3 times, permanently skipped
- **Retry logic**: Reset NA status using `reset_failed_stocks`

## Best Practices

### 🎯 **Production Usage**

1. **Start Small**: Test with `--limit_nums 10` first
2. **Use Appropriate End Dates**: Don't use past dates as end dates
3. **Monitor Progress**: Watch logs for patterns in failures
4. **Regular Maintenance**: Refresh cache weekly, reset failed stocks monthly
5. **Backup Strategy**: CSV files serve as backup to DuckDB

### ⚡ **Performance Optimization**

1. **Rate Limiting**: Keep 12-second delays to avoid blocks
2. **Incremental Updates**: Use `update_data` for daily maintenance
3. **Resume Capability**: Let system auto-resume interrupted downloads
4. **Database Maintenance**: Regular VACUUM and ANALYZE operations

### 🔒 **Data Integrity**

1. **Transaction Safety**: All database operations are transactional
2. **Duplicate Handling**: Primary key constraints prevent duplicates  
3. **Metadata Consistency**: Automatic metadata updates with each insert
4. **Backup Verification**: Compare CSV and DuckDB data periodically

## Troubleshooting

### Common Issues and Solutions

#### 1. "No data returned" for Valid Stocks
**Cause**: Date range issues (e.g., holidays, weekends, future dates)
**Solution**: 
```bash
# Check what dates are actually available
python collector_standalone.py query_data --symbol 600000 --limit 5

# Use proper end date
python collector_standalone.py download_data --end 2025-12-31
```

#### 2. High Failure Rate
**Cause**: API rate limiting or network issues
**Solution**:
```bash
# Increase delay
python collector_standalone.py download_data --delay 15

# Reset failed stocks and retry
python collector_standalone.py reset_failed_stocks
```

#### 3. Database Corruption
**Cause**: Interrupted transactions or disk issues
**Solution**:
```bash
# Check database integrity
python collector_standalone.py query_data --limit 1

# If corrupted, restart with --force
python collector_standalone.py download_data --force True
```

#### 4. Symbol Cache Issues
**Cause**: Outdated or corrupted cache
**Solution**:
```bash
# Refresh symbol cache
python collector_standalone.py refresh_cache

# Clean inactive stocks
python collector_standalone.py clean_cache
```

#### 5. Resume Not Working
**Cause**: Wrong end date or force parameter
**Solution**:
```bash
# Ensure end date is in future
python collector_standalone.py download_data --end 2025-12-31

# Don't use --force unless needed
python collector_standalone.py download_data  # (no --force)
```

### Monitoring Commands

```bash
# Check overall progress
python collector_standalone.py query_data --limit 1

# Check specific stock status
python collector_standalone.py query_data --symbol 600000 --limit 5

# View cache statistics
python collector_standalone.py show_cache_info

# Monitor failed stocks
grep "marked as NA" logfile.log | wc -l
```

## Environment Setup

### Prerequisites

```bash
# Activate qlib environment
conda activate qlib

# Required packages
pip install akshare duckdb pandas tqdm fire loguru
```

### Directory Structure

```
scripts/data_collector/akshare/
├── collector_standalone.py          # Main collector
├── source/                           # Data storage
│   ├── shanghai_stock_data.duckdb   # Main database
│   ├── stock_symbols_cache.duckdb   # Symbol cache
│   ├── {symbol}.csv                 # CSV backups
│   └── failed_stocks.txt            # Failed stocks list
└── README_collector_standalone.md   # This documentation
```

## Performance Expectations

### Typical Performance Metrics

- **Rate**: ~300 stocks/hour (12-second delays)
- **Full Shanghai Collection**: ~8-10 hours for 2300+ stocks
- **Incremental Update**: 1-3 hours depending on missing data
- **Database Size**: ~1-2 GB for 3 years of daily data
- **Success Rate**: 95-98% (excluding permanently delisted stocks)

### Resource Usage

- **Memory**: 100-500 MB during operation
- **Disk**: 2x data size (CSV + DuckDB storage)
- **Network**: Minimal (single API calls per stock)
- **CPU**: Low (I/O bound workload)

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log files for error patterns  
3. Test with small datasets first (`--limit_nums 10`)
4. Use cache management commands for maintenance

**Happy Data Collecting!** 🚀📈