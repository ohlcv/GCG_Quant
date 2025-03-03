# GCG_Quant工具模块快速使用指南

本指南提供GCG_Quant项目中新增工具模块的常用功能和使用示例，帮助开发者快速掌握这些工具的使用方法。

## 1. 配置工具 (config_utils.py)

### 加载和保存配置

```python
from utils.config_utils import load_yaml_config, save_yaml_config

# 加载配置文件
config = load_yaml_config("config.yaml")

# 修改配置
config["database"]["host"] = "new_host"

# 保存配置
save_yaml_config(config, "config.yaml")
```

### 环境变量覆盖配置

```python
from utils.config_utils import apply_env_override

# 设置环境变量 (通常在启动脚本或操作系统中设置)
# export GCG_DATABASE_HOST=localhost
# export GCG_DATABASE_PORT=5432

# 应用环境变量覆盖
config = apply_env_override(config, prefix="GCG_")
```

### 配置验证

```python
from utils.config_utils import validate_config

# 定义配置模式
schema = {
    "database": {
        "type": "dict",
        "required": True,
        "schema": {
            "host": {"type": "string", "required": True},
            "port": {"type": "int", "required": True, "min": 1, "max": 65535}
        }
    }
}

# 验证配置
errors = validate_config(config, schema)
if errors:
    print("配置错误:", errors)
else:
    print("配置有效")
```

### 获取和设置配置值

```python
from utils.config_utils import get_config_value, set_config_value

# 获取嵌套配置值
db_host = get_config_value(config, "database.host", default="localhost")

# 设置嵌套配置值
config = set_config_value(config, "database.port", 5432)
```

## 2. 字符串工具 (string_utils.py)

### 交易对符号处理

```python
from utils.string_utils import normalize_symbol, split_symbol

# 标准化交易对符号
symbol = normalize_symbol("BTCUSDT")  # 返回 "BTC/USDT"

# 分割交易对符号
base, quote = split_symbol("BTC/USDT")  # 返回 ("BTC", "USDT")
```

### 数值格式化

```python
from utils.string_utils import format_price, format_quantity, format_percentage

# 格式化价格 (自动处理小数位数)
price_str = format_price(1234.56)  # 返回 "1234.56"
small_price_str = format_price(0.00012345)  # 返回 "0.0001235"

# 格式化数量
qty_str = format_quantity(1234.5)  # 返回 "1234.5"
int_qty_str = format_quantity(1234.0)  # 返回 "1234"

# 格式化百分比
pct_str = format_percentage(0.1234)  # 返回 "+12.34%"
```

### 表格格式化

```python
from utils.string_utils import format_table

# 创建数据表格
data = [
    {"symbol": "BTC/USDT", "price": 50000, "change": 0.05},
    {"symbol": "ETH/USDT", "price": 4000, "change": -0.02}
]

# 格式化为ASCII表格
table = format_table(data)
print(table)
```

### 字符串工具

```python
from utils.string_utils import truncate_string, mask_sensitive_data, string_similarity

# 截断长字符串
truncated = truncate_string("这是一个很长的字符串", max_length=10)  # 返回 "这是一个很长..."

# 遮蔽敏感数据
masked = mask_sensitive_data("api_secret_key")  # 返回 "ap**********ey"

# 计算字符串相似度
similarity = string_similarity("BTC/USDT", "BTCUSDT")  # 返回相似度 (0.0-1.0)
```

## 3. 数学工具 (math_utils.py)

### 价格舍入和精度控制

```python
from utils.math_utils import round_to_tick_size, round_decimal_places, truncate_decimal_places

# 舍入到指定刻度大小
price = round_to_tick_size(1234.567, tick_size=0.01)  # 返回 1234.57

# 舍入到指定小数位数
price = round_decimal_places(1234.567, decimal_places=2)  # 返回 1234.57

# 截断到指定小数位数 (不进行四舍五入)
price = truncate_decimal_places(1234.567, decimal_places=2)  # 返回 1234.56
```

### 技术指标计算

```python
from utils.math_utils import calculate_sma, calculate_ema, calculate_rsi, calculate_macd, calculate_bollinger_bands

# 简单移动平均线
prices = [100, 110, 120, 130, 140]
sma = calculate_sma(prices, window=3)  # 返回 [110.0, 120.0, 130.0]

# 指数移动平均线
ema = calculate_ema(prices, window=3)  # 返回 EMA值列表

# RSI 计算
rsi = calculate_rsi(prices, window=14)  # 返回 RSI值列表

# MACD 计算
macd_line, signal_line, histogram = calculate_macd(prices)  # 返回 MACD组件

# 布林带计算
middle, upper, lower = calculate_bollinger_bands(prices, window=20, num_std=2.0)
```

### 风险和性能评估

```python
from utils.math_utils import calculate_sharpe_ratio, calculate_max_drawdown, calculate_kelly_criterion

# 计算夏普比率
returns = [0.01, -0.005, 0.02, 0.01, -0.01]
sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)

# 计算最大回撤
prices = [100, 110, 105, 95, 100, 105, 110, 95]
max_dd, peak_idx, trough_idx = calculate_max_drawdown(prices)  # 返回 (回撤率, 峰值索引, 谷值索引)

# 计算凯利公式
kelly = calculate_kelly_criterion(win_rate=0.6, win_loss_ratio=2.0)  # 返回建议仓位比例
```

### 数据处理

```python
from utils.math_utils import normalize_data, calculate_z_score, exponential_smoothing

# 数据归一化
data = [10, 20, 30, 40, 50]
normalized = normalize_data(data, method='minmax')  # 返回 [0.0, 0.25, 0.5, 0.75, 1.0]

# 计算Z分数
z_score = calculate_z_score(42, mean=30, std_dev=10)  # 返回 1.2

# 指数平滑
smoothed = exponential_smoothing(data, alpha=0.3)  # 返回平滑后的数据
```

## 4. 文件工具 (file_utils.py)

### 文件和目录操作

```python
from utils.file_utils import ensure_directory, copy_file, move_file, delete_file, list_files

# 确保目录存在
ensure_directory("data/history")

# 复制文件
copy_file("source.txt", "destination.txt", overwrite=True)

# 移动文件
move_file("old_path.txt", "new_path.txt")

# 删除文件
delete_file("temp.txt")

# 列出目录中的文件
files = list_files("data", pattern="*.csv", recursive=True)
```

### 文件读写

```python
from utils.file_utils import read_text_file, write_text_file, read_json_file, write_json_file

# 读取文本文件
content = read_text_file("config.txt", encoding="utf-8")

# 写入文本文件
write_text_file("log.txt", "程序已启动", append=True)

# 读取JSON文件
data = read_json_file("config.json")

# 写入JSON文件
write_json_file("output.json", {"status": "success", "data": [1, 2, 3]})
```

### CSV文件处理

```python
from utils.file_utils import read_csv_file, write_csv_file

# 读取CSV文件
rows = read_csv_file("data.csv", delimiter=",", has_header=True)

# 写入CSV文件
data = [
    {"name": "BTC", "price": 50000},
    {"name": "ETH", "price": 4000}
]
write_csv_file("prices.csv", data, fieldnames=["name", "price"])
```

### 文件查找和信息

```python
from utils.file_utils import find_files, get_file_info, get_file_hash

# 查找文件
csv_files = find_files("data", extension="csv", recursive=True)

# 获取文件信息
info = get_file_info("data.csv")
print(f"文件大小: {info['formatted_size']}, 修改时间: {info['modified']}")

# 计算文件哈希值
md5_hash = get_file_hash("data.bin", hash_type="md5")
```

### 压缩和解压缩

```python
from utils.file_utils import compress_files, extract_zip, compress_gzip, extract_gzip

# 压缩文件
compress_files(["file1.txt", "file2.txt"], "archive.zip")

# 解压ZIP文件
extract_zip("archive.zip", "extracted_files")

# GZIP压缩
compress_gzip("large_file.txt")  # 创建 large_file.txt.gz

# GZIP解压
extract_gzip("large_file.txt.gz")
```

## 5. 实际应用示例

### 配置驱动的应用程序

```python
from utils.config_utils import load_yaml_config, validate_config, apply_env_override
from utils.file_utils import ensure_directory

# 加载配置
config = load_yaml_config("config.yaml")

# 应用环境变量覆盖
config = apply_env_override(config, prefix="GCG_")

# 验证配置
schema = {...}  # 配置模式
errors = validate_config(config, schema)
if errors:
    for error in errors:
        print(f"配置错误: {error}")
    exit(1)

# 使用配置
db_config = config["database"]
data_dir = config["data_storage"]["data_dir"]

# 确保必要的目录存在
ensure_directory(data_dir)
```

### 交易对数据处理

```python
from utils.string_utils import normalize_symbol, format_price, format_percentage
from utils.math_utils import calculate_sma, calculate_rsi

# 数据处理
symbol = normalize_symbol("BTCUSDT")  # 转为 "BTC/USDT"
prices = [50000, 49500, 51000, 52000, 51500]

# 计算指标
sma = calculate_sma(prices, window=3)
rsi = calculate_rsi(prices)

# 格式化输出
formatted_data = [
    {"symbol": symbol, 
     "price": format_price(prices[-1]), 
     "change": format_percentage((prices[-1] - prices[0]) / prices[0])}
]
```

### 数据保存和加载

```python
from utils.file_utils import write_json_file, read_json_file, ensure_directory
import time

# 创建数据目录
data_dir = "data/market"
ensure_directory(data_dir)

# 保存市场数据
market_data = {...}  # 市场数据
timestamp = int(time.time())
file_path = f"{data_dir}/market_{timestamp}.json"
write_json_file(file_path, market_data)

# 加载历史数据
history_files = list_files(data_dir, pattern="market_*.json")
history_data = []
for file in history_files:
    data = read_json_file(file)
if data:
        history_data.append(data)
```

### 数据分析与统计

```python
from utils.math_utils import calculate_standard_deviation, calculate_correlation, calculate_percentile

# 价格数据
btc_prices = [50000, 49500, 51000, 52000, 51500]
eth_prices = [4000, 3950, 4100, 4200, 4150]

# 计算基本统计量
btc_std_dev = calculate_standard_deviation(btc_prices)
correlation = calculate_correlation(btc_prices, eth_prices)

# 计算百分位数
btc_median = calculate_percentile(btc_prices, 50)  # 中位数
btc_p90 = calculate_percentile(btc_prices, 90)     # 90百分位数

# 分析结果
print(f"BTC价格标准差: {btc_std_dev}")
print(f"BTC/ETH相关系数: {correlation}")
print(f"BTC价格中位数: {btc_median}")
print(f"BTC价格90百分位: {btc_p90}")
```

### 工具组合应用

```python
from utils.config_utils import load_yaml_config
from utils.string_utils import format_table
from utils.math_utils import calculate_max_drawdown, calculate_sharpe_ratio
from utils.file_utils import read_csv_file, write_json_file

# 加载配置
config = load_yaml_config("backtest_config.yaml")

# 读取历史数据
data = read_csv_file(config["data_file"])
prices = [float(row["close"]) for row in data]
dates = [row["date"] for row in data]

# 计算收益率
returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

# 计算性能指标
max_dd, peak_idx, trough_idx = calculate_max_drawdown(prices)
sharpe = calculate_sharpe_ratio(returns, risk_free_rate=config["risk_free_rate"])

# 生成报告数据
report = {
    "symbol": config["symbol"],
    "period": f"{dates[0]} to {dates[-1]}",
    "initial_price": prices[0],
    "final_price": prices[-1],
    "return": (prices[-1] - prices[0]) / prices[0],
    "max_drawdown": max_dd,
    "max_drawdown_period": f"{dates[peak_idx]} to {dates[trough_idx]}",
    "sharpe_ratio": sharpe
}

# 保存结果
write_json_file("backtest_report.json", report)

# 显示报告表格
table_data = [{k: v for k, v in report.items()}]
report_table = format_table(table_data)
print(report_table)
```

## 6. 最佳实践和注意事项

### 错误处理

工具函数设计了一套一致的错误处理机制，对于文件操作等可能失败的函数，通常返回布尔值或None表示成功或失败。建议采用以下模式处理潜在错误：

```python
from utils.file_utils import read_json_file

data = read_json_file("config.json")
if data is None:
    # 处理读取失败的情况
    print("读取配置文件失败，使用默认配置")
    data = default_config
```

### 配置管理

配置管理是系统的核心，建议遵循以下最佳实践：

1. **分层配置**：将配置分为默认配置、用户配置和环境变量三层
2. **配置验证**：始终在使用前验证配置
3. **敏感信息保护**：敏感信息（如API密钥）应通过环境变量传入，而不是存储在配置文件中

```python
from utils.config_utils import load_yaml_config, apply_env_override, validate_config

# 默认配置
default_config = {...}

# 加载用户配置并合并
user_config = load_yaml_config("config.yaml")
if user_config:
    config = deep_merge(default_config, user_config)
else:
    config = default_config

# 应用环境变量覆盖
config = apply_env_override(config, prefix="GCG_")

# 验证配置
errors = validate_config(config, schema)
```

### 类型安全和性能优化

工具函数使用类型注解，可以配合类型检查工具提高代码可靠性：

```python
# 安装类型检查工具
# pip install mypy

# 运行类型检查
# mypy utils/math_utils.py
```

对于数学计算和文件处理等性能敏感的操作，考虑以下优化策略：

1. **批量处理**：使用批量操作而不是多次单独调用
2. **惰性加载**：只在需要时加载数据
3. **结果缓存**：缓存重复计算的结果

```python
# 批量文件操作示例
from utils.file_utils import list_files, read_json_file

# 批量读取文件
files = list_files("data", pattern="*.json")
data_batch = []
for file in files:
    data = read_json_file(file)
    if data:
        data_batch.append(data)

# 一次性处理数据
process_data_batch(data_batch)
```

### 日志与调试

工具函数大多使用print输出错误信息，建议在生产环境中将这些输出重定向到日志系统：

```python
import sys
from utils.logger import setup_logger

# 创建日志记录器
logger = setup_logger("utils")

# 捕获标准输出和错误
class LogCapture:
    def __init__(self, logger):
        self.logger = logger
    
    def write(self, msg):
        if msg and not msg.isspace():
            self.logger.info(msg)
    
    def flush(self):
        pass

# 重定向工具函数的输出到日志
sys.stdout = LogCapture(logger)
sys.stderr = LogCapture(logger)
```

## 7. 集成与扩展建议

### 与现有代码集成

将工具模块集成到现有代码中，可以逐步替换旧的实现：

```python
# 旧代码
import os
import json

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

# 新代码
from utils.config_utils import load_json_config

def load_config(config_file):
    return load_json_config(config_file)
```

### 扩展工具模块

可以通过子类化或添加新函数来扩展工具模块的功能：

```python
from utils.math_utils import calculate_sma

def calculate_weighted_sma(data, window, weights=None):
    """
    计算加权简单移动平均
    
    Args:
        data: 数据列表
        window: 窗口大小
        weights: 权重列表，长度应等于window
        
    Returns:
        List[float]: 加权SMA值列表
    """
    if weights is None:
        # 默认等权重，即普通SMA
        return calculate_sma(data, window)
    
    # 确保权重长度正确
    if len(weights) != window:
        raise ValueError(f"权重列表长度应为{window}")
    
    # 计算加权SMA
    result = []
    for i in range(len(data) - window + 1):
        weighted_sum = sum(data[i+j] * weights[j] for j in range(window))
        weight_sum = sum(weights)
        result.append(weighted_sum / weight_sum)
    
    return result
```

### 创建工具服务

对于频繁使用的功能，可以考虑创建服务类来管理状态和缓存：

```python
class TechnicalIndicatorService:
    def __init__(self):
        self.cache = {}
    
    def get_sma(self, data, window):
        """获取SMA，带缓存"""
        cache_key = f"sma_{window}_{hash(tuple(data))}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = calculate_sma(data, window)
        self.cache[cache_key] = result
        return result
    
    def get_ema(self, data, window):
        """获取EMA，带缓存"""
        cache_key = f"ema_{window}_{hash(tuple(data))}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = calculate_ema(data, window)
        self.cache[cache_key] = result
        return result
    
    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()
```

## 8. 常见问题解答

### Q: 如何处理工具函数中的异常？

A: 大多数工具函数都内部捕获异常并返回None或False，但您也可以选择自己处理异常：

```python
try:
    data = read_json_file("config.json")
    # 处理数据
except Exception as e:
    print(f"读取文件时发生错误: {str(e)}")
    # 处理错误
```

### Q: 工具模块是否线程安全？

A: 大多数函数是纯函数，不维护内部状态，因此是线程安全的。但在并发环境中使用文件操作时，仍需注意文件锁定和竞态条件。

### Q: 如何处理大文件？

A: 对于大文件，建议使用流式处理而不是一次性加载：

```python
def process_large_csv(file_path, batch_size=1000):
    """分批处理大CSV文件"""
    import csv
    
    results = []
    with open(file_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        batch = []
        
        for row in reader:
            batch.append(row)
            
            if len(batch) >= batch_size:
                # 处理一批数据
                process_batch(batch)
                results.extend(batch)
                batch = []
        
        # 处理最后一批
        if batch:
            process_batch(batch)
            results.extend(batch)
    
    return results
```

### Q: 配置文件中的路径如何处理？

A: 建议使用相对路径并在运行时解析：

```python
from utils.config_utils import load_yaml_config
from utils.file_utils import get_absolute_path
import os

# 获取应用根目录
app_root = os.path.dirname(os.path.abspath(__file__))

# 加载配置
config = load_yaml_config("config.yaml")

# 解析相对路径
data_dir = config.get("data_dir", "data")
absolute_data_dir = get_absolute_path(os.path.join(app_root, data_dir))
```

## 9. 总结

本指南涵盖了GCG_Quant工具模块的主要功能和使用方法，从基本操作到高级应用。这些工具模块设计符合Python最佳实践，注重可读性、可维护性和性能。

通过合理使用这些工具模块，可以：

1. 大幅减少重复代码
2. 提高代码质量和可靠性
3. 加速开发流程
4. 确保系统行为一致性

建议按照实际需求逐步集成这些工具，并根据项目特点进行必要的调整和扩展。对于特定领域的功能，可以在这些基础工具之上构建更专业化的模块。

如有任何问题或建议，请随时咨询开发团队。
