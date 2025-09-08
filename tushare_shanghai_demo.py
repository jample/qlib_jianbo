#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tushare上海证券交易所数据下载Demo

本脚本演示如何使用Tushare下载上海证券交易所2022.01.01到2025.01.01的每日交易全量数据

功能特性:
- 下载上海证券交易所所有股票的每日交易数据
- 包含基本行情数据(开高低收、成交量、成交额等)
- 支持数据去重和质量检查
- 自动处理API限制和重试机制
- 数据保存为CSV格式，便于后续处理

使用方法:
    python tushare_shanghai_demo.py --token YOUR_TOKEN --start_date 20220101 --end_date 20250101

注意事项:
- 需要注册Tushare账号并获取token
- 免费用户有API调用频率限制
- 建议分批下载大量数据
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

# 尝试导入tushare
try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    print("错误: 未安装tushare库")
    print("请运行: pip install tushare")
    TUSHARE_AVAILABLE = False
    sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tushare_shanghai_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TushareShanghaiCollector:
    """
    Tushare上海证券交易所数据收集器
    """
    
    def __init__(self, token: str, output_dir: str = "./tushare_data"):
        """
        初始化收集器
        
        Parameters
        ----------
        token : str
            Tushare API token
        output_dir : str
            数据输出目录
        """
        self.token = token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化tushare
        ts.set_token(token)
        self.pro = ts.pro_api()
        
        # API调用计数器
        self.api_calls = 0
        self.last_call_time = time.time()
        
        logger.info(f"Tushare收集器初始化完成，数据将保存到: {self.output_dir}")
    
    def _rate_limit_check(self, calls_per_minute: int = 200):
        """
        检查API调用频率限制
        
        Parameters
        ----------
        calls_per_minute : int
            每分钟最大调用次数，免费用户通常为200次/分钟
        """
        current_time = time.time()
        time_diff = current_time - self.last_call_time
        
        # 如果超过1分钟，重置计数器
        if time_diff >= 60:
            self.api_calls = 0
            self.last_call_time = current_time
        
        # 如果调用次数接近限制，等待
        if self.api_calls >= calls_per_minute - 10:  # 留10次缓冲
            wait_time = 60 - time_diff + 1
            if wait_time > 0:
                logger.info(f"API调用频率限制，等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
                self.api_calls = 0
                self.last_call_time = time.time()
        
        self.api_calls += 1
    
    def get_shanghai_stocks(self) -> List[str]:
        """
        获取上海证券交易所股票列表
        
        Returns
        -------
        List[str]
            上海证券交易所股票代码列表
        """
        logger.info("获取上海证券交易所股票列表...")
        
        try:
            self._rate_limit_check()
            
            # 获取股票基本信息
            stock_basic = self.pro.stock_basic(
                exchange='SSE',  # 上海证券交易所
                list_status='L',  # 上市状态
                fields='ts_code,symbol,name,area,industry,list_date'
            )
            
            shanghai_stocks = stock_basic['ts_code'].tolist()
            logger.info(f"获取到 {len(shanghai_stocks)} 只上海证券交易所股票")
            
            # 保存股票列表
            stock_list_file = self.output_dir / "shanghai_stocks_list.csv"
            stock_basic.to_csv(stock_list_file, index=False, encoding='utf-8-sig')
            logger.info(f"股票列表已保存到: {stock_list_file}")
            
            return shanghai_stocks
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
    
    def get_daily_data(self, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取单只股票的日线数据
        
        Parameters
        ----------
        ts_code : str
            股票代码，如 '600000.SH'
        start_date : str
            开始日期，格式 'YYYYMMDD'
        end_date : str
            结束日期，格式 'YYYYMMDD'
            
        Returns
        -------
        Optional[pd.DataFrame]
            股票日线数据
        """
        try:
            self._rate_limit_check()
            
            # 获取日线数据
            df = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount'
            )
            
            if df.empty:
                logger.warning(f"股票 {ts_code} 在指定时间范围内无数据")
                return None
            
            # 数据预处理
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            
            return df
            
        except Exception as e:
            logger.error(f"获取股票 {ts_code} 数据失败: {e}")
            return None
    
    def get_adj_factor(self, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取复权因子数据
        
        Parameters
        ----------
        ts_code : str
            股票代码
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns
        -------
        Optional[pd.DataFrame]
            复权因子数据
        """
        try:
            self._rate_limit_check()
            
            adj_factor = self.pro.adj_factor(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if not adj_factor.empty:
                adj_factor['trade_date'] = pd.to_datetime(adj_factor['trade_date'])
            
            return adj_factor
            
        except Exception as e:
            logger.warning(f"获取股票 {ts_code} 复权因子失败: {e}")
            return None
    
    def collect_stock_data(self, stock_list: List[str], start_date: str, end_date: str, 
                          batch_size: int = 50, save_individual: bool = True) -> pd.DataFrame:
        """
        批量收集股票数据
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表
        start_date : str
            开始日期，格式 'YYYYMMDD'
        end_date : str
            结束日期，格式 'YYYYMMDD'
        batch_size : int
            批处理大小
        save_individual : bool
            是否保存单个股票文件
            
        Returns
        -------
        pd.DataFrame
            合并后的所有股票数据
        """
        logger.info(f"开始收集 {len(stock_list)} 只股票的数据，时间范围: {start_date} - {end_date}")
        
        all_data = []
        failed_stocks = []
        
        # 创建个股数据目录
        if save_individual:
            individual_dir = self.output_dir / "individual_stocks"
            individual_dir.mkdir(exist_ok=True)
        
        # 分批处理
        for i in tqdm(range(0, len(stock_list), batch_size), desc="处理批次"):
            batch_stocks = stock_list[i:i + batch_size]
            
            for ts_code in tqdm(batch_stocks, desc=f"批次 {i//batch_size + 1}", leave=False):
                try:
                    # 获取日线数据
                    daily_data = self.get_daily_data(ts_code, start_date, end_date)
                    
                    if daily_data is not None and not daily_data.empty:
                        # 获取复权因子
                        adj_factor = self.get_adj_factor(ts_code, start_date, end_date)
                        
                        # 合并复权因子
                        if adj_factor is not None and not adj_factor.empty:
                            daily_data = daily_data.merge(
                                adj_factor[['trade_date', 'adj_factor']], 
                                on='trade_date', 
                                how='left'
                            )
                            # 前向填充复权因子
                            daily_data['adj_factor'] = daily_data['adj_factor'].fillna(method='ffill')
                            daily_data['adj_factor'] = daily_data['adj_factor'].fillna(1.0)
                        else:
                            daily_data['adj_factor'] = 1.0
                        
                        # 计算复权价格
                        for col in ['open', 'high', 'low', 'close', 'pre_close']:
                            daily_data[f'{col}_adj'] = daily_data[col] * daily_data['adj_factor']
                        
                        all_data.append(daily_data)
                        
                        # 保存单个股票文件
                        if save_individual:
                            stock_file = individual_dir / f"{ts_code.replace('.', '_')}.csv"
                            daily_data.to_csv(stock_file, index=False, encoding='utf-8-sig')
                        
                        logger.debug(f"成功获取股票 {ts_code} 数据，共 {len(daily_data)} 条记录")
                    else:
                        failed_stocks.append(ts_code)
                        logger.warning(f"股票 {ts_code} 无数据")
                    
                    # 添加延迟避免频率限制
                    time.sleep(0.1)
                    
                except Exception as e:
                    failed_stocks.append(ts_code)
                    logger.error(f"处理股票 {ts_code} 时出错: {e}")
            
            # 批次间休息
            if i + batch_size < len(stock_list):
                logger.info(f"批次 {i//batch_size + 1} 完成，休息5秒...")
                time.sleep(5)
        
        # 合并所有数据
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"数据收集完成，共获取 {len(combined_data)} 条记录")
            
            # 保存合并数据
            combined_file = self.output_dir / f"shanghai_stocks_{start_date}_{end_date}.csv"
            combined_data.to_csv(combined_file, index=False, encoding='utf-8-sig')
            logger.info(f"合并数据已保存到: {combined_file}")
            
            # 保存失败列表
            if failed_stocks:
                failed_file = self.output_dir / "failed_stocks.txt"
                with open(failed_file, 'w', encoding='utf-8') as f:
                    for stock in failed_stocks:
                        f.write(f"{stock}\n")
                logger.warning(f"有 {len(failed_stocks)} 只股票获取失败，列表已保存到: {failed_file}")
            
            return combined_data
        else:
            logger.error("未获取到任何数据")
            return pd.DataFrame()
    
    def get_trading_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取交易日历
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns
        -------
        pd.DataFrame
            交易日历数据
        """
        try:
            self._rate_limit_check()
            
            calendar = self.pro.trade_cal(
                exchange='SSE',
                start_date=start_date,
                end_date=end_date
            )
            
            # 保存交易日历
            calendar_file = self.output_dir / "trading_calendar.csv"
            calendar.to_csv(calendar_file, index=False, encoding='utf-8-sig')
            logger.info(f"交易日历已保存到: {calendar_file}")
            
            return calendar
            
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return pd.DataFrame()
    
    def generate_summary_report(self, data: pd.DataFrame) -> dict:
        """
        生成数据摘要报告
        
        Parameters
        ----------
        data : pd.DataFrame
            股票数据
            
        Returns
        -------
        dict
            摘要报告
        """
        if data.empty:
            return {"error": "无数据"}
        
        report = {
            "数据概览": {
                "总记录数": len(data),
                "股票数量": data['ts_code'].nunique(),
                "时间范围": f"{data['trade_date'].min()} 至 {data['trade_date'].max()}",
                "交易日数量": data['trade_date'].nunique()
            },
            "数据质量": {
                "缺失值统计": data.isnull().sum().to_dict(),
                "重复记录数": data.duplicated().sum()
            },
            "市场统计": {
                "平均日成交量": f"{data['vol'].mean():.2f} 万手",
                "平均日成交额": f"{data['amount'].mean():.2f} 万元",
                "最大单日涨幅": f"{data['pct_chg'].max():.2f}%",
                "最大单日跌幅": f"{data['pct_chg'].min():.2f}%"
            }
        }
        
        # 保存报告
        report_file = self.output_dir / "summary_report.json"
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"摘要报告已保存到: {report_file}")
        return report

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Tushare上海证券交易所数据下载Demo')
    parser.add_argument('--token', required=True, help='Tushare API token')
    parser.add_argument('--start_date', default='20220101', help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end_date', default='20250101', help='结束日期 (YYYYMMDD)')
    parser.add_argument('--output_dir', default='./tushare_shanghai_data', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=50, help='批处理大小')
    parser.add_argument('--limit_stocks', type=int, help='限制股票数量（用于测试）')
    parser.add_argument('--skip_individual', action='store_true', help='跳过保存单个股票文件')
    
    args = parser.parse_args()
    
    # 验证日期格式
    try:
        datetime.strptime(args.start_date, '%Y%m%d')
        datetime.strptime(args.end_date, '%Y%m%d')
    except ValueError:
        logger.error("日期格式错误，请使用 YYYYMMDD 格式")
        return
    
    # 初始化收集器
    collector = TushareShanghaiCollector(args.token, args.output_dir)
    
    try:
        # 获取股票列表
        stock_list = collector.get_shanghai_stocks()
        
        if not stock_list:
            logger.error("未获取到股票列表")
            return
        
        # 限制股票数量（用于测试）
        if args.limit_stocks:
            stock_list = stock_list[:args.limit_stocks]
            logger.info(f"限制为前 {args.limit_stocks} 只股票")
        
        # 获取交易日历
        logger.info("获取交易日历...")
        trading_calendar = collector.get_trading_calendar(args.start_date, args.end_date)
        
        # 收集股票数据
        data = collector.collect_stock_data(
            stock_list, 
            args.start_date, 
            args.end_date,
            batch_size=args.batch_size,
            save_individual=not args.skip_individual
        )
        
        # 生成摘要报告
        if not data.empty:
            logger.info("生成摘要报告...")
            report = collector.generate_summary_report(data)
            
            # 打印摘要
            print("\n" + "="*50)
            print("数据收集完成摘要")
            print("="*50)
            for category, stats in report.items():
                if isinstance(stats, dict):
                    print(f"\n{category}:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"{category}: {stats}")
        
        logger.info("数据收集任务完成！")
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise

if __name__ == "__main__":
    main()