### 3.3 Redis管理器 (redis_manager.py) (续)

```python
    def _get_expire_seconds(self, timeframe: str) -> int:
        """
        根据时间周期获取过期时间（秒）
        
        Args:
            timeframe: 时间周期
            
        Returns:
            int: 过期时间（秒）
        """
        # 根据时间周期设置不同的过期时间
        if timeframe == '1m':
            return 3600 * 24  # 1天
        elif timeframe == '5m':
            return 3600 * 24 * 3  # 3天
        elif timeframe == '15m':
            return 3600 * 24 * 7  # 7天
        elif timeframe == '1h':
            return 3600 * 24 * 30  # 30天
        elif timeframe == '4h':
            return 3600 * 24 * 90  # 90天
        elif timeframe == '1d':
            return 3600 * 24 * 365  # 365天
        else:
            return 3600 * 24 * 7  # 默认7天
```

### 4. 日志模块设计

#### 4.1 日志工具 (logger.py)

```python
import os
import sys
from datetime import datetime
from loguru import logger
import json

def get_logger(name, level="INFO"):
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别，默认为INFO
        
    Returns:
        Logger: 日志记录器
    """
    # 确保日志目录存在
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取当前日期
    today = datetime.now().strftime("%Y%m%d")
    
    # 构建日志文件路径
    log_file = os.path.join(log_dir, f"{today}_{name}.log")
    
    # 移除所有处理器
    logger.remove()
    
    # 添加标准输出处理器
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        filter=lambda record: record["extra"].get("name", "") == name,
        level=level,
        colorize=True,
    )
    
    # 添加文件处理器
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        filter=lambda record: record["extra"].get("name", "") == name,
        level=level,
        rotation="00:00",  # 每天午夜轮换日志
        compression="zip",  # 压缩旧日志
        retention="30 days",  # 保留30天的日志
    )
    
    # 创建带有名称的上下文日志记录器
    named_logger = logger.bind(name=name)
    
    return named_logger

# 配置JSON序列化日志格式，便于ELK等工具处理
def setup_json_logger(name, level="INFO"):
    """
    设置JSON格式的日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别，默认为INFO
        
    Returns:
        Logger: 日志记录器
    """
    # 确保日志目录存在
    log_dir = os.path.join(os.getcwd(), "logs", "json")
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取当前日期
    today = datetime.now().strftime("%Y%m%d")
    
    # 构建日志文件路径
    log_file = os.path.join(log_dir, f"{today}_{name}.json")
    
    # 自定义JSON格式化处理器
    def json_formatter(record):
        log_data = {
            "timestamp": record["time"].strftime("%Y-%m-%d %H:%M:%S.%f"),
            "level": record["level"].name,
            "name": record["extra"].get("name", ""),
            "module": record["module"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
        }
        
        # 添加异常信息（如果有）
        if record["exception"]:
            log_data["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback,
            }
        
        return json.dumps(log_data)
    
    # 添加JSON文件处理器
    logger.add(
        log_file,
        format=json_formatter,
        filter=lambda record: record["extra"].get("name", "") == name,
        level=level,
        rotation="00:00",  # 每天午夜轮换日志
        compression="zip",  # 压缩旧日志
        retention="30 days",  # 保留30天的日志
    )
    
    # 创建带有名称的上下文日志记录器
    named_logger = logger.bind(name=name)
    
    return named_logger
```

### 5. 配置管理模块设计

#### 5.1 配置设置 (settings.py)

```python
import os
import yaml
from typing import Dict, Any, Optional

# 默认配置
DEFAULT_CONFIG = {
    # 数据采集配置
    "data_collector": {
        "exchange": {
            "exchange_id": "binance",
            "api_key": "",
            "secret": "",
            "timeout": 30000,
        },
        "file_import": {
            "data_dir": "./data/raw"
        },
        "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
    },
    
    # 数据存储配置
    "data_storage": {
        "timescale": {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "postgres",
            "database": "gcg_quant"
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": ""
        }
    },
    
    # 日志配置
    "logging": {
        "level": "INFO",
        "json_format": False,
        "retention_days": 30
    }
}

def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置，如果未指定配置文件，则使用默认配置
    
    Args:
        config_file: 配置文件路径，默认为None
        
    Returns:
        Dict[str, Any]: 配置字典
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                
            # 递归合并配置
            merge_configs(config, user_config)
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
    
    return config

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并配置字典
    
    Args:
        base: 基础配置
        override: 覆盖配置
        
    Returns:
        Dict[str, Any]: 合并后的配置
    """
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            merge_configs(base[key], value)
        else:
            base[key] = value
    
    return base

def save_config(config: Dict[str, Any], config_file: str) -> bool:
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        config_file: 配置文件路径
        
    Returns:
        bool: 保存是否成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return True
    except Exception as e:
        print(f"保存配置文件失败: {str(e)}")
        return False
```

#### 5.2 常量定义 (constants.py)

```python
# 版本信息
VERSION = "0.1.0"
BUILD_DATE = "2025-02-29"

# 时间周期定义
TIMEFRAME_1M = "1m"
TIMEFRAME_5M = "5m"
TIMEFRAME_15M = "15m"
TIMEFRAME_1H = "1h"
TIMEFRAME_4H = "4h"
TIMEFRAME_1D = "1d"

# 支持的交易所列表
SUPPORTED_EXCHANGES = [
    "binance",
    "okex",
    "huobi",
    "bybit",
    "kucoin",
    "coinbase",
    "bitstamp"
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

# Redis键前缀
REDIS_KEY_TICK = "tick"
REDIS_KEY_KLINE = "kline"
REDIS_KEY_LATEST = "latest"
```

### 6. 主程序设计 (main.py)

```python
import asyncio
import argparse
import os
from typing import Dict, Any, List

from src.config.settings import load_config
from src.utils.logger import get_logger
from src.data_collector.exchange_collector import ExchangeCollector
from src.data_collector.file_importer import FileImporter
from src.data_storage.timescale_manager import TimescaleManager
from src.data_storage.redis_manager import RedisManager

# 主日志记录器
logger = get_logger("main")

class GCGQuant:
    """GCG_Quant主程序类"""
    
    def __init__(self, config_file: str = None):
        """
        初始化GCG_Quant
        
        Args:
            config_file: 配置文件路径，默认为None
        """
        # 加载配置
        self.config = load_config(config_file)
        logger.info(f"GCG_Quant初始化，加载配置: {config_file or '默认配置'}")
        
        # 初始化组件（延迟加载）
        self.exchange_collector = None
        self.file_importer = None
        self.timescale_manager = None
        self.redis_manager = None
    
    async def initialize(self) -> bool:
        """
        初始化所有组件
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("初始化GCG_Quant组件")
            
            # 初始化数据存储组件
            self.timescale_manager = TimescaleManager(self.config["data_storage"]["timescale"])
            self.redis_manager = RedisManager(self.config["data_storage"]["redis"])
            
            # 连接数据库
            timescale_connected = await self.timescale_manager.connect()
            redis_connected = await self.redis_manager.connect()
            
            if not timescale_connected or not redis_connected:
                logger.error("数据库连接失败")
                return False
            
            # 初始化数据采集组件
            self.exchange_collector = ExchangeCollector(self.config["data_collector"]["exchange"])
            self.file_importer = FileImporter(self.config["data_collector"]["file_import"])
            
            # 注册数据回调
            await self._register_callbacks()
            
            logger.info("GCG_Quant组件初始化成功")
            return True
        except Exception as e:
            logger.error(f"GCG_Quant初始化失败: {str(e)}")
            return False
    
    async def _register_callbacks(self):
        """注册数据回调函数"""
        # 注册Tick数据回调
        async def on_tick_data(tick_data):
            # 保存到Redis和TimescaleDB
            await self.redis_manager.save_tick_data(tick_data)
            await self.timescale_manager.save_tick_data(tick_data)
        
        # 注册K线数据回调
        async def on_kline_data(kline_data):
            # 保存到Redis和TimescaleDB
            await self.redis_manager.save_kline_data(kline_data)
            await self.timescale_manager.save_kline_data(kline_data)
        
        # 获取交易品种和时间周期
        symbols = self.config["data_collector"]["symbols"]
        timeframes = self.config["data_collector"]["timeframes"]
        
        # 设置订阅
        await self.exchange_collector.start()
        await self.exchange_collector.subscribe_tick(symbols, on_tick_data)
        
        for timeframe in timeframes:
            await self.exchange_collector.subscribe_kline(symbols, timeframe, on_kline_data)
    
    async def start(self) -> bool:
        """
        启动GCG_Quant
        
        Returns:
            bool: 启动是否成功
        """
        try:
            logger.info("启动GCG_Quant")
            
            # 初始化组件
            initialized = await self.initialize()
            if not initialized:
                logger.error("GCG_Quant组件初始化失败，无法启动")
                return False
            
            logger.info("GCG_Quant启动成功")
            
            # 保持运行状态
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("接收到中断信号，停止GCG_Quant")
            await self.stop()
        except Exception as e:
            logger.error(f"GCG_Quant运行时发生错误: {str(e)}")
            await self.stop()
            return False
    
    async def stop(self) -> bool:
        """
        停止GCG_Quant
        
        Returns:
            bool: 停止是否成功
        """
        try:
            logger.info("停止GCG_Quant")
            
            # 停止各组件
            if self.exchange_collector:
                await self.exchange_collector.stop()
            
            if self.timescale_manager:
                await self.timescale_manager.disconnect()
            
            if self.redis_manager:
                await self.redis_manager.disconnect()
            
            logger.info("GCG_Quant已停止")
            return True
        except Exception as e:
            logger.error(f"停止GCG_Quant时发生错误: {str(e)}")
            return False

async def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="GCG_Quant量化交易系统")
    parser.add_argument("-c", "--config", help="配置文件路径", default="config.yaml")
    args = parser.parse_args()
    
    # 创建并启动GCG_Quant
    gcg_quant = GCGQuant(args.config)
    await gcg_quant.start()

if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())
```

### 7. 测试设计

#### 7.1 数据采集测试 (test_collector.py)

```python
import asyncio
import unittest
from datetime import datetime, timedelta
import os
import json

from src.config.settings import load_config
from src.data_collector.exchange_collector import ExchangeCollector
from src.data_collector.file_importer import FileImporter

class TestExchangeCollector(unittest.TestCase):
    """交易所数据采集器测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试前准备"""
        config = load_config()
        cls.collector = ExchangeCollector(config["data_collector"]["exchange"])
        
        # 使用异步测试辅助函数
        cls.loop = asyncio.get_event_loop()
        cls.loop.run_until_complete(cls.collector.connect())
    
    @classmethod
    def tearDownClass(cls):
        """测试后清理"""
        cls.loop.run_until_complete(cls.collector.disconnect())
    
    def test_fetch_tick_data(self):
        """测试获取Tick数据"""
        # 选择一个测试交易对
        symbol = "BTC/USDT"
        
        # 执行异步测试
        result = self.loop.run_until_complete(self.collector.fetch_tick_data(symbol))
        
        # 验证结果
        self.assertIsInstance(result, list)
        if result:  # 如果有数据
            self.assertEqual(result[0]["symbol"], symbol)
            self.assertIn("price", result[0])
            self.assertIn("amount", result[0])
            self.assertIn("timestamp", result[0])
    
    def test_fetch_kline_data(self):
        """测试获取K线数据"""
        # 选择一个测试交易对和时间周期
        symbol = "BTC/USDT"
        timeframe = "1h"
        
        # 获取过去24小时的数据
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        
        # 执行异步测试
        result = self.loop.run_until_complete(
            self.collector.fetch_kline_data(symbol, timeframe, start_time, end_time)
        )
        
        # 验证结果
        self.assertIsInstance(result, list)
        if result:  # 如果有数据
            self.assertEqual(result[0]["symbol"], symbol)
            self.assertEqual(result[0]["timeframe"], timeframe)
            self.assertIn("open", result[0])
            self.assertIn("high", result[0])
            self.assertIn("low", result[0])
            self.assertIn("close", result[0])
            self.assertIn("volume", result[0])

class TestFileImporter(unittest.TestCase):
    """文件数据导入器测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试前准备"""
        # 创建测试数据目录
        cls.test_data_dir = "test_data"
        os.makedirs(cls.test_data_dir, exist_ok=True)
        
        # 创建测试配置
        cls.config = {"data_dir": cls.test_data_dir}
        cls.importer = FileImporter(cls.config)
        
        # 创建测试数据文件
        cls._create_test_data()
        
        # 使用异步测试辅助函数
        cls.loop = asyncio.get_event_loop()
        cls.loop.run_until_complete(cls.importer.connect())
    
    @classmethod
    def tearDownClass(cls):
        """测试后清理"""
        cls.loop.run_until_complete(cls.importer.disconnect())
        
        # 清理测试数据文件
        for file in os.listdir(cls.test_data_dir):
            os.remove(os.path.join(cls.test_data_dir, file))
        os.rmdir(cls.test_data_dir)
    
    @classmethod
    def _create_test_data(cls):
        """创建测试数据文件"""
        # 创建Tick测试数据
        tick_data = []
        for i in range(100):
            tick_data.append({
                "symbol": "BTC/USDT",
                "timestamp": int((datetime.now() - timedelta(minutes=i)).timestamp() * 1000),
                "datetime": (datetime.now() - timedelta(minutes=i)).isoformat(),
                "price": 40000 + i * 10,
                "amount": 0.1 + i * 0.01,
                "side": "buy" if i % 2 == 0 else "sell",
                "source": "test"
            })
        
        # 创建K线测试数据
        kline_data = []
        for i in range(24):
            kline_data.append({
                "symbol": "BTC/USDT",
                "timestamp": int((datetime.now() - timedelta(hours=i)).timestamp() * 1000),
                "datetime": (datetime.now() - timedelta(hours=i)).isoformat(),
                "timeframe": "1h",
                "open": 40000 + i * 100,
                "high": 40100 + i * 100,
                "low": 39900 + i * 100,
                "close": 40050 + i * 100,
                "volume": 10 + i,
                "source": "test"
            })
        
        # 写入CSV文件
        cls._write_csv(os.path.join(cls.test_data_dir, "tick_BTC_USDT.csv"), tick_data)
        cls._write_csv(os.path.join(cls.test_data_dir, "kline_BTC_USDT_1h.csv"), kline_data)
    
    @classmethod
    def _write_csv(cls, file_path, data):
        """写入CSV文件"""
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    
    def test_fetch_tick_data(self):
        """测试从文件获取Tick数据"""
        # 选择一个测试交易对
        symbol = "BTC/USDT"
        
        # 执行异步测试
        result = self.loop.run_until_complete(self.importer.fetch_tick_data(symbol))
        
        # 验证结果
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]["symbol"], symbol)
    
    def test_fetch_kline_data(self):
        """测试从文件获取K线数据"""
        # 选择一个测试交易对和时间周期
        symbol = "BTC/USDT"
        timeframe = "1h"
        
        # 执行异步测试
        result = self.loop.run_until_complete(
            self.importer.fetch_kline_data(symbol, timeframe)
        )
        
        # 验证结果
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]["symbol"], symbol)
        self.assertEqual(result[0]["timeframe"], timeframe)

if __name__ == "__main__":
    unittest.main()
```

#### 7.2 数据存储测试 (test_storage.py)

```python
import asyncio
import unittest
from datetime import datetime, timedelta
import os
import random

from src.config.settings import load_config
from src.data_storage.timescale_manager import TimescaleManager
from src.data_storage.redis_manager import RedisManager
from src.data_storage.models import TickData, KlineData

class TestTimescaleManager(unittest.TestCase):
    """TimescaleDB管理器测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试前准备"""
        config = load_config()
        cls.manager = TimescaleManager(config["data_storage"]["timescale"])
        
        # 使用异步测试辅助函数
        cls.loop = asyncio.get_event_loop()
        cls.loop.run_until_complete(cls.manager.connect())
        
        # 生成测试数据
        cls.test_data = cls._generate_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """测试后清理"""
        cls.loop.run_until_complete(cls._cleanup_test_data())
        cls.loop.run_until_complete(cls.manager.disconnect())
    
    @classmethod
    def _generate_test_data(cls):
        """生成测试数据"""
        # 生成Tick测试数据
        tick_data = []
        for i in range(10):
            tick = TickData(
                symbol="TEST/USDT",
                timestamp=int((datetime.now() - timedelta(minutes=i)).timestamp() * 1000),
                datetime=datetime.now() - timedelta(minutes=i),
                price=1000 + random.random() * 100,
                amount=1 + random.random(),
                side="buy" if i % 2 == 0 else "sell",
                source="test",
                trade_id=f"test_{i}"
            )
            tick_data.append(tick)
        
        # 生成K线测试数据
        kline_data = []
        for i in range(5):
            kline = KlineData(
                symbol="TEST/USDT",
                timestamp=int((datetime.now() - timedelta(hours=i)).timestamp() * 1000),
                datetime=datetime.now() - timedelta(hours=i),
                timeframe="1h",
                open=1000 + random.random() * 100,
                high=1100 + random.random() * 100,
                low=900 + random.random() * 100,
                close=1050 + random.random() * 100,
                volume=10 + random.random() * 5,
                source="test"
            )
            kline_data.append(kline)
        
        return {"tick": tick_data, "kline": kline_data}
    
    @classmethod
    async def _cleanup_test_data(cls):
        """清理测试数据"""
        async with cls.manager.pool.acquire() as conn:
            await conn.execute("DELETE FROM tick_data WHERE symbol = 'TEST/USDT'")
            await conn.execute("DELETE FROM kline_data WHERE symbol = 'TEST/USDT'")
    
    def test_save_and_query_tick_data(self):
        """测试保存和查询Tick数据"""
        # 保存测试数据
        success = self.loop.run_until_complete(
            self.manager.save_tick_data(self.test_data["tick"])
        )
        self.assertTrue(success)
        
        # 查询数据
        result = self.loop.run_until_complete(
            self.manager.query_tick_data("TEST/USDT")
        )
        
        # 验证结果
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), len(self.test_data["tick"]))
        self.assertEqual(result[0].symbol, "TEST/USDT")
    
    def test_save_and_query_kline_data(self):
        """测试保存和查询K线数据"""
        # 保存测试数据
        success = self.loop.run_until_complete(
            self.manager.save_kline_data(self.test_data["kline"])
        )
        self.assertTrue(success)
        
        # 查询数据
        result = self.loop.run_until_complete(
            self.manager.query_kline_data("TEST/USDT", "1h")
        )
        
        # 验证结果
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), len(self.test_data["kline"]))
        self.assertEqual(result[0].symbol, "TEST/USDT")
        self.assertEqual(result[0].timeframe, "1h")

class TestRedisManager(unittest.TestCase):
    """Redis管理器测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试前准备"""
        config = load_config()
        cls.manager = RedisManager(config["data_storage"]["redis"])
        
        # 使用异步测试辅助函数
        cls.loop = asyncio.get_event_loop()
        cls.loop.run_until_complete(cls.manager.connect())
        
        # 生成测试数据
        cls.test_data = cls._generate_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """测试后清理"""
        cls.loop.run_until_complete(cls._cleanup_test_data())
        cls.loop.run_until_complete(cls.manager.disconnect())
    
    @classmethod
    def _generate_test_data(cls):
        """生成测试数据"""
        # 生成Tick测试数据
        tick_data = []
        for i in range(5):
            tick = TickData(
                symbol="TEST/USDT",
                timestamp=int((datetime.now() - timedelta(seconds=i)).timestamp() * 1000),
                datetime=datetime.now() - timedelta(seconds=i),
                price=1000 + random.random() * 100,
                amount=1 + random.random(),
                side="buy" if i % 2 == 0 else "sell",
                source="test",
                trade_id=f"test_{i}"
            )
            tick_data.append(tick)