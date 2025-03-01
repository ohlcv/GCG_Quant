# constants.py - 系统常量定义

"""
文件说明：
    这个文件定义了GCG_Quant系统中使用的各种常量和枚举值。
    集中定义常量可以提高代码一致性，避免硬编码和潜在错误。
    常量使用全大写命名，表示它们不应该被修改。

学习目标：
    1. 了解Python中常量的定义和使用方法
    2. 学习集中管理常量的最佳实践
    3. 掌握命名规范和组织常量的技巧
"""

# 版本信息
VERSION = "0.1.0"
BUILD_DATE = "2025-03-01"

# 时间周期定义
TIMEFRAME_1M = "1m"
TIMEFRAME_5M = "5m"
TIMEFRAME_15M = "15m"
TIMEFRAME_1H = "1h"
TIMEFRAME_4H = "4h"
TIMEFRAME_1D = "1d"

# 所有支持的时间周期列表
SUPPORTED_TIMEFRAMES = [
    TIMEFRAME_1M,
    TIMEFRAME_5M,
    TIMEFRAME_15M,
    TIMEFRAME_1H,
    TIMEFRAME_4H,
    TIMEFRAME_1D,
]

# 支持的交易所列表
SUPPORTED_EXCHANGES = [
    "binance",
    "okex",
    "huobi",
    "bybit",
    "kucoin",
    "coinbase",
    "bitstamp",
]

# 数据类型
DATA_TYPE_TICK = "tick"
DATA_TYPE_KLINE = "kline"

# 交易方向
TRADE_SIDE_BUY = "buy"
TRADE_SIDE_SELL = "sell"

# 数据源类型
SOURCE_TYPE_EXCHANGE = "exchange"
SOURCE_TYPE_FILE = "file"

# 数据库类型
DB_TYPE_SQLITE = "sqlite"
DB_TYPE_TIMESCALEDB = "timescaledb"

# Redis键前缀
REDIS_KEY_TICK = "tick"
REDIS_KEY_KLINE = "kline"
REDIS_KEY_LATEST = "latest"

# 默认批量处理大小
DEFAULT_BATCH_SIZE = 1000

# 日志级别
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"
LOG_LEVEL_CRITICAL = "CRITICAL"

# 文件扩展名
FILE_EXT_CSV = ".csv"
FILE_EXT_JSON = ".json"

# 默认配置文件路径
DEFAULT_CONFIG_FILE = "config.yaml"
# 默认批量处理大小（已在其他文件使用）
DEFAULT_BATCH_SIZE = 1000
