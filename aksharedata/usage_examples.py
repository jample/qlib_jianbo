#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AkShare Usage Examples for 2025

This script demonstrates various ways to use AkShare for data collection in 2025.
It covers all major data types and provides practical examples for quantitative research.
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print(f"AkShare Version: {ak.__version__}")
print(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
print("=" * 60)

def example_1_stock_data():
    """Example 1: Stock Data Collection"""
    print("\n1. STOCK DATA COLLECTION")
    print("-" * 30)
    
    # Get stock list
    stock_list = ak.stock_info_a_code_name()
    print(f"Total A-share stocks: {len(stock_list)}")
    print(stock_list.head())
    
    # Get daily stock data with different adjustments
    symbol = "000001"  # 平安银行
    
    # Forward adjusted (recommended for backtesting)
    stock_qfq = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                   start_date="20240101", end_date="20251231", adjust="qfq")
    print(f"\nForward adjusted data for {symbol}: {stock_qfq.shape}")
    print(stock_qfq.tail(3))
    
    # Real-time data
    try:
        realtime = ak.stock_zh_a_spot_em()
        print(f"\nReal-time data: {realtime.shape}")
        print("Top 5 by market cap:")
        print(realtime.head())
    except Exception as e:
        print(f"Error getting real-time data: {e}")

def example_2_index_data():
    """Example 2: Index Data Collection"""
    print("\n2. INDEX DATA COLLECTION")
    print("-" * 30)
    
    indices = {
        'CSI300': 'sh000300',
        'CSI500': 'sh000905', 
        'SSE50': 'sh000016',
        'ChiNext': 'sz399006'
    }
    
    for name, code in indices.items():
        try:
            data = ak.index_zh_a_hist(symbol=code, period="daily", 
                                    start_date="20240101", end_date="20251231")
            print(f"{name} ({code}): {len(data)} records")
            if not data.empty:
                latest = data.iloc[-1]
                print(f"  Latest close: {latest['收盘']:.2f}")
        except Exception as e:
            print(f"Error getting {name}: {e}")
    
    # Index constituents
    try:
        csi300_stocks = ak.index_stock_cons_csindex(symbol="000300")
        print(f"\nCSI300 constituents: {len(csi300_stocks)}")
        print(csi300_stocks.head())
    except Exception as e:
        print(f"Error getting CSI300 constituents: {e}")

def example_3_fund_data():
    """Example 3: Fund Data Collection"""
    print("\n3. FUND DATA COLLECTION")
    print("-" * 30)
    
    # Fund list
    fund_list = ak.fund_name_em()
    print(f"Total funds: {len(fund_list)}")
    
    # Example fund
    fund_code = "015198"
    
    # Fund net value history
    try:
        fund_hist = ak.fund_open_fund_info_em(fund=fund_code, indicator="累计净值走势")
        print(f"\nFund {fund_code} history: {fund_hist.shape}")
        print(fund_hist.tail(3))
    except Exception as e:
        print(f"Error getting fund history: {e}")
    
    # Fund holdings
    try:
        holdings_2025 = ak.fund_portfolio_hold_em(symbol=fund_code, date="2025")
        print(f"\nFund {fund_code} holdings 2025: {len(holdings_2025)} positions")
        if not holdings_2025.empty:
            print("Top 5 holdings:")
            print(holdings_2025.head())
    except Exception as e:
        print(f"Error getting fund holdings: {e}")

def example_4_economic_data():
    """Example 4: Economic Indicators"""
    print("\n4. ECONOMIC INDICATORS")
    print("-" * 30)
    
    indicators = {
        'GDP': lambda: ak.macro_china_gdp(),
        'CPI': lambda: ak.macro_china_cpi(),
        'Money Supply': lambda: ak.macro_china_money_supply(),
        'LPR': lambda: ak.macro_china_lpr()
    }
    
    for name, func in indicators.items():
        try:
            data = func()
            print(f"{name}: {len(data)} records")
            if not data.empty:
                print(f"  Latest: {data.iloc[-1].to_dict()}")
        except Exception as e:
            print(f"Error getting {name}: {e}")

def example_5_sector_data():
    """Example 5: Sector and Industry Data"""
    print("\n5. SECTOR AND INDUSTRY DATA")
    print("-" * 30)
    
    # Industry sectors
    try:
        industries = ak.stock_board_industry_name_em()
        print(f"Industry sectors: {len(industries)}")
        print(industries.head())
    except Exception as e:
        print(f"Error getting industries: {e}")
    
    # Hot sectors
    hot_sectors = ['银行', '半导体', '新能源汽车', '人工智能']
    
    for sector in hot_sectors:
        try:
            stocks = ak.stock_board_industry_cons_em(symbol=sector)
            print(f"\n{sector} sector: {len(stocks)} stocks")
            if not stocks.empty:
                print(f"  Top stock: {stocks.iloc[0]['名称']} ({stocks.iloc[0]['代码']})")
        except Exception as e:
            print(f"Error getting {sector} stocks: {e}")

def example_6_financial_statements():
    """Example 6: Financial Statements"""
    print("\n6. FINANCIAL STATEMENTS")
    print("-" * 30)
    
    symbol = "000001"  # 平安银行
    
    statements = {
        'Balance Sheet': lambda: ak.stock_balance_sheet_by_report_em(symbol=symbol),
        'Income Statement': lambda: ak.stock_profit_sheet_by_report_em(symbol=symbol),
        'Cash Flow': lambda: ak.stock_cash_flow_sheet_by_report_em(symbol=symbol)
    }
    
    for name, func in statements.items():
        try:
            data = func()
            print(f"{name} for {symbol}: {data.shape}")
            if not data.empty:
                print(f"  Columns: {len(data.columns)}")
                print(f"  Periods: {len(data)}")
        except Exception as e:
            print(f"Error getting {name}: {e}")

def example_7_data_analysis():
    """Example 7: Basic Data Analysis"""
    print("\n7. BASIC DATA ANALYSIS")
    print("-" * 30)
    
    # Analyze CSI300 performance
    try:
        csi300 = ak.index_zh_a_hist(symbol="sh000300", period="daily", 
                                   start_date="20240101", end_date="20251231")
        
        if not csi300.empty:
            # Calculate returns
            csi300['daily_return'] = csi300['收盘'].pct_change()
            csi300['cumulative_return'] = (1 + csi300['daily_return']).cumprod() - 1
            
            print(f"CSI300 Analysis:")
            print(f"  Total return: {csi300['cumulative_return'].iloc[-1]:.2%}")
            print(f"  Volatility: {csi300['daily_return'].std() * np.sqrt(252):.2%}")
            print(f"  Max drawdown: {(csi300['cumulative_return'] - csi300['cumulative_return'].cummax()).min():.2%}")
            
            # Best and worst days
            best_day = csi300.loc[csi300['daily_return'].idxmax()]
            worst_day = csi300.loc[csi300['daily_return'].idxmin()]
            
            print(f"  Best day: {best_day['日期']} (+{best_day['daily_return']:.2%})")
            print(f"  Worst day: {worst_day['日期']} ({worst_day['daily_return']:.2%})")
            
    except Exception as e:
        print(f"Error in analysis: {e}")

def example_8_data_export():
    """Example 8: Data Export and Integration"""
    print("\n8. DATA EXPORT AND INTEGRATION")
    print("-" * 30)
    
    # Collect sample data
    symbols = ['000001', '000002', '600000']
    all_data = []
    
    for symbol in symbols:
        try:
            data = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                    start_date="20241201", end_date="20241231", adjust="qfq")
            if not data.empty:
                data['symbol'] = symbol
                all_data.append(data)
                print(f"Collected {len(data)} records for {symbol}")
        except Exception as e:
            print(f"Error collecting {symbol}: {e}")
    
    if all_data:
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Normalize column names for qlib compatibility
        column_mapping = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close', 
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'money',
            '涨跌幅': 'change'
        }
        
        combined_data = combined_data.rename(columns=column_mapping)
        combined_data['date'] = pd.to_datetime(combined_data['date'])
        
        print(f"\nCombined dataset: {combined_data.shape}")
        print("Columns:", combined_data.columns.tolist())
        
        # Save to CSV
        output_file = f"akshare_sample_data_{datetime.now().strftime('%Y%m%d')}.csv"
        combined_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Data saved to: {output_file}")

def example_9_automated_collection():
    """Example 9: Automated Data Collection Strategy"""
    print("\n9. AUTOMATED COLLECTION STRATEGY")
    print("-" * 30)
    
    from data_updater import AkShareDataUpdater
    
    # Initialize updater
    updater = AkShareDataUpdater(data_dir="./sample_data")
    
    # Daily update example
    print("Performing sample daily update...")
    sample_symbols = ['000001', '000002', '600000']
    updater.daily_update(symbols=sample_symbols, days_back=5)
    
    print("Sample automated collection completed")

def main():
    """Run all examples"""
    print("AKSHARE 2025 USAGE EXAMPLES")
    print("=" * 60)
    
    examples = [
        example_1_stock_data,
        example_2_index_data, 
        example_3_fund_data,
        example_4_economic_data,
        example_5_sector_data,
        example_6_financial_statements,
        example_7_data_analysis,
        example_8_data_export,
        # example_9_automated_collection  # Commented out to avoid file operations
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\nError in example {i}: {e}")
        
        if i < len(examples):
            print("\n" + "=" * 60)
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("\nKey Takeaways for 2025:")
    print("1. Use 'qfq' adjustment for backtesting")
    print("2. Implement proper rate limiting (1-2 seconds delay)")
    print("3. Handle errors gracefully with try-except blocks")
    print("4. Validate data quality before using")
    print("5. Save data in standardized formats for reuse")
    print("6. Use concurrent processing for large datasets")
    print("7. Monitor API changes and update code accordingly")

if __name__ == "__main__":
    main()
