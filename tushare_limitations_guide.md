# Tushare使用限制和实际要求指南

## 1. 账户类型和权限限制

### 免费用户限制
- **API调用频率**: 200次/分钟
- **每日调用上限**: 2000次/天
- **数据获取范围**: 最近3年的数据
- **并发限制**: 单线程访问
- **数据延迟**: 实时数据延迟15分钟

### 付费用户权限
| 套餐类型 | 月费(元) | 调用频率 | 每日上限 | 历史数据 | 实时数据 |
|---------|---------|----------|----------|----------|----------|
| 基础版 | 300 | 500次/分钟 | 10000次/天 | 10年 | 延迟3分钟 |
| 专业版 | 1200 | 1000次/分钟 | 50000次/天 | 全部 | 实时 |
| 机构版 | 5000+ | 2000次/分钟 | 200000次/天 | 全部 | 实时 |

## 2. 数据获取限制

### 时间范围限制
```python
# 免费用户示例
start_date = '20220101'  # 最早只能获取3年前数据
end_date = '20250101'    # 当前日期

# 付费用户可获取更早数据
start_date = '19900101'  # 专业版可获取全部历史数据
```

### 单次请求限制
- **日线数据**: 单次最多获取5000条记录
- **分钟数据**: 单次最多获取8000条记录
- **tick数据**: 单次最多获取10000条记录

### 字段限制
```python
# 基础字段（免费用户可用）
basic_fields = 'ts_code,trade_date,open,high,low,close,vol,amount'

# 高级字段（需要付费）
advanced_fields = 'turnover_rate,volume_ratio,pe,pb,ps,dv_ratio'
```

## 3. API调用频率控制

### 推荐的频率控制策略
```python
import time
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, calls_per_minute=180):  # 留20次缓冲
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        now = datetime.now()
        # 移除1分钟前的调用记录
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < timedelta(minutes=1)]
        
        # 如果调用次数接近限制，等待
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0]).total_seconds() + 1
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.calls = []
        
        self.calls.append(now)
```

## 4. 数据完整性考虑

### 停牌和退市股票
- 停牌期间无交易数据
- 退市股票历史数据可能不完整
- 新股上市前无历史数据

### 数据质量检查
```python
def validate_data_quality(df):
    """
    数据质量检查
    """
    issues = []
    
    # 检查缺失值
    missing_data = df.isnull().sum()
    if missing_data.any():
        issues.append(f"存在缺失值: {missing_data[missing_data > 0].to_dict()}")
    
    # 检查异常价格
    if (df['high'] < df['low']).any():
        issues.append("存在最高价低于最低价的异常数据")
    
    # 检查成交量异常
    if (df['vol'] < 0).any():
        issues.append("存在负成交量")
    
    return issues
```

## 5. 网络和稳定性考虑

### 网络重试机制
```python
import requests
from functools import wraps

def retry_on_failure(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"第{attempt + 1}次尝试失败: {e}，{delay}秒后重试...")
                    time.sleep(delay * (2 ** attempt))  # 指数退避
            return None
        return wrapper
    return decorator
```

### 服务器维护时间
- **日常维护**: 每日凌晨2:00-4:00可能服务不稳定
- **周末维护**: 周六晚上可能有系统升级
- **节假日**: 交易所休市期间数据更新延迟

## 6. 成本估算

### 上海证券交易所全量数据成本估算

假设上海证券交易所约1800只股票，时间范围2022-2025年（约3年）：

```python
# 成本计算
stocks_count = 1800
trading_days_per_year = 250
years = 3
total_records = stocks_count * trading_days_per_year * years
print(f"预计总记录数: {total_records:,}")

# API调用次数估算（考虑分批获取）
records_per_call = 5000  # 单次最多获取记录数
api_calls_needed = (total_records // records_per_call) + 1
print(f"预计API调用次数: {api_calls_needed:,}")

# 免费用户需要天数
free_daily_limit = 2000
days_needed_free = api_calls_needed // free_daily_limit + 1
print(f"免费用户需要天数: {days_needed_free}")

# 付费用户成本
if api_calls_needed > 10000:
    print("建议使用专业版套餐 (1200元/月)")
else:
    print("基础版套餐即可满足需求 (300元/月)")
```

**预计结果**:
- 总记录数: 1,350,000条
- API调用次数: 约270次
- 免费用户: 1天内可完成
- 付费建议: 基础版即可

## 7. 优化建议

### 数据获取策略
1. **分批获取**: 避免单次请求过大
2. **增量更新**: 只获取新增数据
3. **本地缓存**: 避免重复请求
4. **并行处理**: 付费用户可考虑多线程

### 存储优化
```python
# 使用parquet格式存储，压缩比更高
import pandas as pd

# 保存为parquet格式
df.to_parquet('shanghai_stocks.parquet', compression='snappy')

# 按日期分区存储
for date in df['trade_date'].unique():
    date_data = df[df['trade_date'] == date]
    date_data.to_parquet(f'data/date={date}/data.parquet')
```

### 错误处理
```python
# 常见错误及处理
ERROR_CODES = {
    '40001': '缺少权限',
    '40002': 'API调用频率超限',
    '40003': '数据不存在',
    '40004': '参数错误',
    '50001': '服务器内部错误'
}

def handle_tushare_error(error_code, error_msg):
    if error_code == '40002':
        print("API调用频率超限，等待60秒...")
        time.sleep(60)
        return True  # 可重试
    elif error_code == '40001':
        print("权限不足，请检查账户类型")
        return False  # 不可重试
    else:
        print(f"未知错误: {error_code} - {error_msg}")
        return False
```

## 8. 实际使用建议

### 对于个人用户
1. **注册免费账户**: 先测试基本功能
2. **小批量测试**: 先获取少量数据验证
3. **评估需求**: 根据实际需求决定是否升级

### 对于机构用户
1. **直接购买专业版**: 避免频率限制
2. **建立数据管道**: 自动化数据更新
3. **数据质量监控**: 建立数据质量检查机制

### 替代方案
如果Tushare成本过高，可考虑：
1. **AkShare**: 免费，但数据源不如Tushare稳定
2. **Wind API**: 机构级数据，成本更高但质量更好
3. **自建爬虫**: 技术要求高，维护成本大

## 9. 法律和合规考虑

- **数据使用协议**: 严格遵守Tushare用户协议
- **商业用途**: 商业使用需要相应授权
- **数据分发**: 不得向第三方分发原始数据
- **频率限制**: 不得使用技术手段绕过API限制

## 10. 总结

对于下载上海证券交易所2022-2025年全量数据：

**免费用户**:
- ✅ 可以完成任务
- ⚠️ 需要控制调用频率
- ⚠️ 数据范围可能受限

**付费用户**:
- ✅ 更稳定的服务
- ✅ 更完整的历史数据
- ✅ 更高的调用频率
- ❌ 需要付费成本

**推荐方案**: 先用免费账户测试，确认需求后再考虑升级付费版本。