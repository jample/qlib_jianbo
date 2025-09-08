#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete Fund Data Workflow: Collection → Qlib → Model Training

This script provides a complete workflow for:
1. Downloading fund data using AkShare
2. Converting to qlib format
3. Preparing data for model training
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import akshare as ak

# Add qlib path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

class FundDataWorkflow:
    """Complete fund data workflow from collection to model training"""
    
    def __init__(self, data_dir="./fund_workflow_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.raw_dir = self.data_dir / "raw"
        self.qlib_dir = self.data_dir / "qlib_format"
        self.features_dir = self.data_dir / "features"
        
        for dir_path in [self.raw_dir, self.qlib_dir, self.features_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Verified fund codes that work well (removed problematic ones)
        self.verified_funds = [
            "015198",  # 华夏中证500ETF联接A
            "110022",  # 易方达消费行业股票
            "161725",  # 招商中证白酒指数(LOF)A
            "000001",  # 华夏成长混合
            "519066",  # 汇添富蓝筹稳健混合A
            "110011",  # 易方达中小盘混合
            "519674",  # 银河创新成长混合
            "110020",  # 易方达沪深300ETF联接A
            "161017",  # 富国中证500指数(LOF)
            "000831",  # 工银医疗保健行业股票A
            "519736",  # 交银新成长混合
            "000248",  # 汇添富中证主要消费ETF联接A
            "519068",  # 汇添富成长焦点混合
            "110003",  # 易方达上证50指数A
            "000478",  # 建信中证红利潜力指数A
            "519772",  # 交银成长30混合
            "000596",  # 前海开源中证军工指数A
            "519697",  # 交银优势行业混合
            "110027",  # 易方达安心回报债券A
            "519018",  # 汇添富均衡增长混合
        ]
        
        logger.info(f"Fund workflow initialized with {len(self.verified_funds)} verified funds")
    
    def _validate_fund_code(self, fund_code):
        """Validate if a fund code is accessible via AkShare API"""
        try:
            # Try to get basic fund info first (lighter API call)
            df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="累计净值走势")
            return not df.empty
        except Exception as e:
            # Check for specific error patterns that indicate invalid fund codes
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['data_acworthtrend', 'referenceerror', 'syntaxerror', 'unexpected token']):
                return False
            # For other errors, assume it might be temporary
            return True

    def step1_collect_fund_data(self, start_date="2025-01-01", end_date="2025-06-01", delay=1):
        """Step 1: Collect fund data from AkShare with improved error handling"""
        logger.info(f"Step 1: Collecting fund data from {start_date} to {end_date}")

        successful_funds = []
        failed_funds = []

        # First, validate fund codes
        logger.info("Validating fund codes...")
        valid_funds = []
        for fund_code in self.verified_funds:
            try:
                if self._validate_fund_code(fund_code):
                    valid_funds.append(fund_code)
                else:
                    logger.warning(f"Fund {fund_code} failed validation, skipping")
                    failed_funds.append(fund_code)
            except Exception as e:
                logger.warning(f"Error validating fund {fund_code}: {e}")
                failed_funds.append(fund_code)

        logger.info(f"Validated {len(valid_funds)} out of {len(self.verified_funds)} funds")

        # Now collect data for valid funds
        for i, fund_code in enumerate(valid_funds):
            try:
                logger.info(f"Processing fund {fund_code} ({i+1}/{len(valid_funds)})")

                # Get fund data with retry mechanism
                df = self._get_fund_data_with_retry(fund_code, max_retries=2)

                if df is None or df.empty:
                    logger.warning(f"No data retrieved for fund {fund_code}")
                    failed_funds.append(fund_code)
                    continue

                # Process the data
                df_processed = self._process_fund_data(df, fund_code, start_date, end_date)

                if not df_processed.empty:
                    # Save raw data
                    output_file = self.raw_dir / f"{fund_code}.csv"
                    df_processed.to_csv(output_file, index=False, encoding='utf-8-sig')
                    successful_funds.append(fund_code)
                    logger.info(f"✓ Saved {len(df_processed)} records for {fund_code}")
                else:
                    failed_funds.append(fund_code)
                    logger.warning(f"✗ No data in date range for {fund_code}")

                # Rate limiting
                if delay > 0:
                    import time
                    time.sleep(delay)

            except Exception as e:
                logger.error(f"✗ Error processing {fund_code}: {e}")
                failed_funds.append(fund_code)
                continue

        logger.info(f"Step 1 completed: {len(successful_funds)} successful, {len(failed_funds)} failed")
        if failed_funds:
            logger.info(f"Failed funds: {failed_funds}")

        return successful_funds, failed_funds

    def _get_fund_data_with_retry(self, fund_code, max_retries=2):
        """Get fund data with retry mechanism"""
        import time

        for attempt in range(max_retries + 1):
            try:
                df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="累计净值走势")
                if not df.empty:
                    return df
                else:
                    logger.warning(f"Empty data for {fund_code}, attempt {attempt + 1}")

            except Exception as e:
                error_str = str(e).lower()

                # Check for permanent errors (invalid fund codes)
                if any(keyword in error_str for keyword in ['data_acworthtrend', 'referenceerror', 'syntaxerror']):
                    logger.error(f"Permanent error for {fund_code}: {e}")
                    return None

                # For temporary errors, retry
                logger.warning(f"Temporary error for {fund_code}, attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    time.sleep(1)  # Wait before retry
                    continue
                else:
                    logger.error(f"Max retries exceeded for {fund_code}")
                    return None

        return None
    
    def _process_fund_data(self, df, fund_code, start_date, end_date):
        """Process raw fund data to standard format"""
        # Column mapping
        column_mapping = {
            '净值日期': 'date',
            '单位净值': 'close',
            '累计净值': 'cumulative_nav',
            '日增长率': 'change'
        }
        
        # Apply available mappings
        available_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        
        # Use cumulative_nav as close if close not available
        if '累计净值' in df.columns and '单位净值' not in df.columns:
            available_mapping['累计净值'] = 'close'
        
        df = df.rename(columns=available_mapping)
        
        # Ensure date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        
        # Add required columns for qlib compatibility
        if 'close' in df.columns:
            df['open'] = df['close']  # For funds, use close as open
            df['high'] = df['close']  # For funds, use close as high/low
            df['low'] = df['close']
            df['volume'] = 0  # Funds don't have volume
            df['money'] = 0   # Funds don't have money
        
        # Add symbol column
        df['symbol'] = fund_code
        
        # Select final columns
        final_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'money', 'symbol']
        if 'change' in df.columns:
            final_cols.append('change')
        if 'cumulative_nav' in df.columns:
            final_cols.append('cumulative_nav')
        
        available_final_cols = [col for col in final_cols if col in df.columns]
        df = df[available_final_cols].copy()
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def step2_convert_to_qlib_format(self):
        """Step 2: Convert raw data to qlib format"""
        logger.info("Step 2: Converting to qlib format")
        
        raw_files = list(self.raw_dir.glob("*.csv"))
        if not raw_files:
            logger.error("No raw data files found. Run step1 first.")
            return []
        
        converted_funds = []
        
        for raw_file in raw_files:
            try:
                fund_code = raw_file.stem
                df = pd.read_csv(raw_file)
                
                if df.empty:
                    continue
                
                # Ensure date is datetime
                df['date'] = pd.to_datetime(df['date'])
                
                # Sort by date
                df = df.sort_values('date').reset_index(drop=True)
                
                # Calculate additional features for qlib
                if 'close' in df.columns:
                    # Calculate returns
                    df['return'] = df['close'].pct_change()
                    df['return_1d'] = df['return']
                    df['return_5d'] = df['close'].pct_change(5)
                    df['return_10d'] = df['close'].pct_change(10)
                    
                    # Calculate moving averages
                    df['ma_5'] = df['close'].rolling(5).mean()
                    df['ma_10'] = df['close'].rolling(10).mean()
                    df['ma_20'] = df['close'].rolling(20).mean()
                    
                    # Calculate volatility
                    df['volatility_5d'] = df['return'].rolling(5).std()
                    df['volatility_10d'] = df['return'].rolling(10).std()
                    
                    # Calculate relative strength
                    df['rsi_14'] = self._calculate_rsi(df['close'], 14)
                
                # Remove NaN values
                df = df.dropna()
                
                if not df.empty:
                    # Save qlib format data
                    qlib_file = self.qlib_dir / f"{fund_code}.csv"
                    df.to_csv(qlib_file, index=False, encoding='utf-8-sig')
                    converted_funds.append(fund_code)
                    logger.info(f"✓ Converted {fund_code}: {len(df)} records")
                
            except Exception as e:
                logger.error(f"✗ Error converting {raw_file.name}: {e}")
                continue
        
        logger.info(f"Step 2 completed: {len(converted_funds)} funds converted")
        return converted_funds
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def step3_prepare_features_for_training(self):
        """Step 3: Prepare features for model training"""
        logger.info("Step 3: Preparing features for model training")
        
        qlib_files = list(self.qlib_dir.glob("*.csv"))
        if not qlib_files:
            logger.error("No qlib format files found. Run step2 first.")
            return None
        
        # Combine all fund data
        all_data = []
        
        for qlib_file in qlib_files:
            try:
                df = pd.read_csv(qlib_file)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.error(f"Error reading {qlib_file.name}: {e}")
                continue
        
        if not all_data:
            logger.error("No valid data found")
            return None
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        
        # Sort by symbol and date
        combined_df = combined_df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Prepare feature matrix
        feature_columns = [
            'return_1d', 'return_5d', 'return_10d',
            'ma_5', 'ma_10', 'ma_20',
            'volatility_5d', 'volatility_10d',
            'rsi_14'
        ]
        
        # Select available feature columns
        available_features = [col for col in feature_columns if col in combined_df.columns]
        
        # Create feature matrix
        feature_df = combined_df[['date', 'symbol', 'close'] + available_features].copy()
        
        # Create target variable (next day return)
        feature_df = feature_df.sort_values(['symbol', 'date'])
        feature_df['target'] = feature_df.groupby('symbol')['return_1d'].shift(-1)
        
        # Remove NaN values
        feature_df = feature_df.dropna()
        
        # Save feature data
        feature_file = self.features_dir / "fund_features.csv"
        feature_df.to_csv(feature_file, index=False, encoding='utf-8-sig')
        
        # Save summary statistics
        summary_file = self.features_dir / "feature_summary.csv"
        summary_stats = feature_df[available_features + ['target']].describe()
        summary_stats.to_csv(summary_file, encoding='utf-8-sig')
        
        logger.info(f"Step 3 completed: {len(feature_df)} samples with {len(available_features)} features")
        logger.info(f"Features: {available_features}")
        logger.info(f"Data saved to: {feature_file}")
        
        return feature_df, available_features
    
    def run_complete_workflow(self, start_date="2025-01-01", end_date="2025-06-01", delay=1):
        """Run the complete workflow"""
        logger.info("Starting complete fund data workflow")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Step 1: Collect data
        successful_funds, failed_funds = self.step1_collect_fund_data(start_date, end_date, delay)
        
        if not successful_funds:
            logger.error("No data collected. Workflow stopped.")
            return None
        
        # Step 2: Convert to qlib format
        converted_funds = self.step2_convert_to_qlib_format()
        
        if not converted_funds:
            logger.error("No data converted. Workflow stopped.")
            return None
        
        # Step 3: Prepare features
        result = self.step3_prepare_features_for_training()
        
        if result is None:
            logger.error("Feature preparation failed.")
            return None
        
        feature_df, available_features = result
        
        logger.info("=" * 60)
        logger.info("WORKFLOW COMPLETED SUCCESSFULLY!")
        logger.info(f"✓ Collected data for {len(successful_funds)} funds")
        logger.info(f"✓ Converted {len(converted_funds)} funds to qlib format")
        logger.info(f"✓ Prepared {len(feature_df)} samples for training")
        logger.info(f"✓ Features: {available_features}")
        logger.info(f"✓ Data directory: {self.data_dir}")
        logger.info("=" * 60)
        
        return {
            'feature_data': feature_df,
            'features': available_features,
            'successful_funds': successful_funds,
            'data_dir': self.data_dir
        }


def main():
    """Main function to run the workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fund Data Workflow")
    parser.add_argument("--start", default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-06-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests")
    parser.add_argument("--data_dir", default="./fund_workflow_data", help="Data directory")
    
    args = parser.parse_args()
    
    # Create workflow
    workflow = FundDataWorkflow(data_dir=args.data_dir)
    
    # Run complete workflow
    result = workflow.run_complete_workflow(
        start_date=args.start,
        end_date=args.end,
        delay=args.delay
    )
    
    if result:
        print("\n🎉 Workflow completed successfully!")
        print(f"📁 Data saved to: {result['data_dir']}")
        print(f"📊 Features: {result['features']}")
        print(f"💾 Training data shape: {result['feature_data'].shape}")
    else:
        print("\n❌ Workflow failed. Check the logs above.")


if __name__ == "__main__":
    main()
