# test_collector.py - 数据采集组件测试

"""
文件说明：
    这个文件包含了对数据采集组件的单元测试，主要测试ExchangeCollector和FileImporter类。
    使用Python标准库中的unittest模块和异步测试技术，验证数据采集功能的正确性。
    测试用例覆盖了连接管理、数据获取和订阅功能，使用mock技术模拟交易所和文件数据。

学习目标：
    1. 了解如何编写异步单元测试
    2. 学习如何使用mock技术模拟外部依赖
    3. 掌握测试覆盖率和测试用例设计
    4. 理解测试驱动开发(TDD)的基本思想
"""

import unittest
import asyncio
import os
import json
import tempfile
from unittest import mock
from datetime import datetime, timedelta
import logging

# 禁用测试过程中的日志输出
logging.disable(logging.CRITICAL)

# 导入被测试的模块
from src.data_collector.exchange_collector import ExchangeCollector
from src.data_collector.file_importer import FileImporter
from src.data_storage.models import TickData, KlineData
from src.config.constants import SUPPORTED_TIMEFRAMES


class TestExchangeCollector(unittest.TestCase):
    """
    ExchangeCollector测试类

    学习点：
    - 使用unittest框架编写测试用例
    - 异步测试方法的实现
    - 为外部依赖创建mock对象
    """

    def setUp(self):
        """
        测试前的准备工作

        学习点：
        - setUp方法在每个测试用例执行前运行
        - 创建测试配置和mock对象
        """
        # 测试配置
        self.config = {
            "exchange_id": "binance",
            "api_key": "test_api_key",
            "secret": "test_secret",
            "timeout": 10000,
            "use_websocket": False,
        }

        # 创建mock对象
        self.mock_exchange_patcher = mock.patch("ccxt.async_support.binance")
        self.mock_exchange = self.mock_exchange_patcher.start()

        # 创建测试实例
        self.collector = ExchangeCollector(self.config)
        self.collector.exchange = self.mock_exchange

    def tearDown(self):
        """
        测试后的清理工作

        学习点：
        - tearDown方法在每个测试用例执行后运行
        - 确保资源被正确释放
        """
        # 停止所有mock
        self.run_async(self.importer.disconnect())  # 清理异步资源
        self.temp_dir.cleanup()
        self.mock_exchange_patcher.stop()

    def run_async(self, coroutine):
        """
        运行异步方法的辅助函数

        Args:
            coroutine: 异步协程对象

        Returns:
            任何异步方法的返回值

        学习点：
        - 在同步测试框架中运行异步代码
        - 使用事件循环执行协程
        """
        return asyncio.get_event_loop().run_until_complete(coroutine)

    def test_init(self):
        """
        测试初始化方法

        学习点：
        - 检查实例属性是否正确初始化
        - 使用assertEqual等断言方法验证结果
        """
        collector = ExchangeCollector(self.config)
        self.assertEqual(collector.exchange_id, "binance")
        self.assertEqual(collector.api_key, "test_api_key")
        self.assertEqual(collector.secret, "test_secret")
        self.assertEqual(collector.timeout, 10000)
        self.assertEqual(collector.use_websocket, False)

    def test_connect(self):
        """
        测试连接方法

        学习点：
        - 测试异步连接方法
        - 验证是否正确调用了交易所API
        """
        # 设置mock行为
        self.mock_exchange.load_markets.return_value = None

        # 运行测试
        result = self.run_async(self.collector.connect())

        # 验证结果
        self.assertTrue(result)
        self.mock_exchange.load_markets.assert_called_once()

    def test_disconnect(self):
        """
        测试断开连接方法

        学习点：
        - 测试资源释放
        - 验证断开连接的行为
        """
        # 设置mock行为
        self.mock_exchange.close.return_value = None

        # 运行测试
        result = self.run_async(self.collector.disconnect())

        # 验证结果
        self.assertTrue(result)
        self.mock_exchange.close.assert_called_once()

    def test_fetch_tick_data(self):
        """
        测试获取Tick数据方法

        学习点：
        - 模拟API响应数据
        - 验证数据转换和处理逻辑
        """
        # 模拟交易所API的响应
        mock_trades = [
            {
                "id": "1",
                "timestamp": 1614556800000,  # 2021-03-01 00:00:00
                "datetime": "2021-03-01T00:00:00Z",
                "symbol": "BTC/USDT",
                "price": 50000.0,
                "amount": 1.0,
                "side": "buy",
            },
            {
                "id": "2",
                "timestamp": 1614556860000,  # 2021-03-01 00:01:00
                "datetime": "2021-03-01T00:01:00Z",
                "symbol": "BTC/USDT",
                "price": 50100.0,
                "amount": 0.5,
                "side": "sell",
            },
        ]
        self.mock_exchange.fetch_trades.return_value = mock_trades

        # 运行测试
        result = self.run_async(self.collector.fetch_tick_data("BTC/USDT"))

        # 验证结果
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["price"], 50000.0)
        self.assertEqual(result[1]["price"], 50100.0)
        self.mock_exchange.fetch_trades.assert_called_once()

    def test_fetch_kline_data(self):
        """
        测试获取K线数据方法

        学习点：
        - 模拟OHLCV数据
        - 验证K线数据处理
        """
        # 模拟交易所API的响应
        mock_ohlcv = [
            [
                1614556800000,
                50000.0,
                50100.0,
                49900.0,
                50050.0,
                100.0,
            ],  # [timestamp, open, high, low, close, volume]
            [1614560400000, 50050.0, 50200.0, 50000.0, 50150.0, 150.0],
        ]
        self.mock_exchange.fetch_ohlcv.return_value = mock_ohlcv

        # 运行测试
        result = self.run_async(self.collector.fetch_kline_data("BTC/USDT", "1h"))

        # 验证结果
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["open"], 50000.0)
        self.assertEqual(result[0]["high"], 50100.0)
        self.assertEqual(result[0]["low"], 49900.0)
        self.assertEqual(result[0]["close"], 50050.0)
        self.assertEqual(result[0]["volume"], 100.0)
        self.mock_exchange.fetch_ohlcv.assert_called_once()

    def test_subscribe_tick(self):
        """
        测试订阅Tick数据方法

        学习点：
        - 测试回调函数机制
        - 验证异步任务创建
        """

        # 模拟回调函数
        async def mock_callback(data):
            pass

        # 使用补丁替换_poll_tick_data方法
        with mock.patch.object(self.collector, "_poll_tick_data", return_value=None):
            # 运行测试
            result = self.run_async(
                self.collector.subscribe_tick(["BTC/USDT"], mock_callback)
            )

            # 验证结果
            self.assertTrue(result)
            self.assertIn("BTC/USDT", self.collector.tick_subscriptions)

    def test_subscribe_kline(self):
        """
        测试订阅K线数据方法

        学习点：
        - 测试多参数订阅
        - 验证订阅数据结构
        """

        # 模拟回调函数
        async def mock_callback(data):
            pass

        # 使用补丁替换_poll_kline_data方法
        with mock.patch.object(self.collector, "_poll_kline_data", return_value=None):
            # 运行测试
            result = self.run_async(
                self.collector.subscribe_kline(["BTC/USDT"], "1h", mock_callback)
            )

            # 验证结果
            self.assertTrue(result)
            self.assertIn("1h", self.collector.kline_subscriptions)
            self.assertIn("BTC/USDT", self.collector.kline_subscriptions["1h"])

    def test_format_symbol(self):
        """
        测试交易品种符号格式化方法

        学习点：
        - 测试辅助方法
        - 验证多种输入情况
        """
        # 测试已经是正确格式的情况
        self.assertEqual(self.collector._format_symbol("BTC/USDT"), "BTC/USDT")

        # 测试需要格式化的情况
        self.assertEqual(self.collector._format_symbol("BTCUSDT"), "BTC/USDT")
        self.assertEqual(self.collector._format_symbol("ETHBTC"), "ETH/BTC")

    def test_get_poll_interval(self):
        """
        测试获取轮询间隔方法

        学习点：
        - 测试逻辑分支
        - 验证不同输入的输出结果
        """
        # 测试不同时间周期
        self.assertEqual(self.collector._get_poll_interval("1m"), 10)
        self.assertEqual(self.collector._get_poll_interval("5m"), 30)
        self.assertEqual(self.collector._get_poll_interval("15m"), 60)
        self.assertEqual(self.collector._get_poll_interval("1h"), 120)
        self.assertEqual(self.collector._get_poll_interval("4h"), 300)
        self.assertEqual(self.collector._get_poll_interval("1d"), 600)

        # 测试未知时间周期
        self.assertEqual(self.collector._get_poll_interval("unknown"), 60)


class TestFileImporter(unittest.TestCase):
    """
    FileImporter测试类

    学习点：
    - 测试文件I/O操作
    - 使用临时文件进行测试
    - 异步文件操作的测试
    """

    def setUp(self):
        """
        测试前的准备工作

        学习点：
        - 创建临时目录
        - 准备测试数据文件
        """
        # 创建临时目录
        self.temp_dir = tempfile.TemporaryDirectory()

        # 测试配置
        self.config = {"data_dir": self.temp_dir.name}

        # 创建测试实例
        self.importer = FileImporter(self.config)

        # 创建测试数据文件
        self.create_test_files()

    def tearDown(self):
        """
        测试后的清理工作

        学习点：
        - 清理临时文件
        - 确保资源释放
        """
        # 清理临时目录
        self.temp_dir.cleanup()

    def run_async(self, coroutine):
        """
        运行异步方法的辅助函数

        Args:
            coroutine: 异步协程对象

        Returns:
            任何异步方法的返回值
        """
        return asyncio.get_event_loop().run_until_complete(coroutine)

    def create_test_files(self):
        """
        创建测试数据文件

        学习点：
        - 生成测试数据
        - 写入CSV和JSON文件
        """
        # 创建Tick数据CSV文件
        tick_csv_path = os.path.join(self.temp_dir.name, "BTCUSDT_tick_20250101.csv")
        with open(tick_csv_path, "w", newline="") as f:
            f.write("time,price,amount,side,id\n")
            f.write("2025-01-01T00:00:00,50000.0,1.0,buy,1\n")
            f.write("2025-01-01T00:01:00,50100.0,0.5,sell,2\n")

        # 创建Tick数据JSON文件
        tick_json_path = os.path.join(self.temp_dir.name, "ETHUSDT_tick_20250101.json")
        tick_data = [
            {
                "time": "2025-01-01T00:00:00",
                "price": 3000.0,
                "amount": 10.0,
                "side": "buy",
                "id": "1",
            },
            {
                "time": "2025-01-01T00:01:00",
                "price": 3010.0,
                "amount": 5.0,
                "side": "sell",
                "id": "2",
            },
        ]
        with open(tick_json_path, "w") as f:
            json.dump(tick_data, f)

        # 创建K线数据CSV文件
        kline_csv_path = os.path.join(
            self.temp_dir.name, "BTCUSDT_kline_1h_20250101.csv"
        )
        with open(kline_csv_path, "w", newline="") as f:
            f.write("time,open,high,low,close,volume\n")
            f.write("2025-01-01T00:00:00,50000.0,50100.0,49900.0,50050.0,100.0\n")
            f.write("2025-01-01T01:00:00,50050.0,50200.0,50000.0,50150.0,150.0\n")

        # 创建K线数据JSON文件
        kline_json_path = os.path.join(
            self.temp_dir.name, "ETHUSDT_kline_1h_20250101.json"
        )
        kline_data = [
            {
                "time": "2025-01-01T00:00:00",
                "open": 3000.0,
                "high": 3050.0,
                "low": 2950.0,
                "close": 3025.0,
                "volume": 1000.0,
            },
            {
                "time": "2025-01-01T01:00:00",
                "open": 3025.0,
                "high": 3075.0,
                "low": 3000.0,
                "close": 3050.0,
                "volume": 1500.0,
            },
        ]
        with open(kline_json_path, "w") as f:
            json.dump(kline_data, f)

    def test_init(self):
        """
        测试初始化方法

        学习点：
        - 检查实例属性是否正确初始化
        - 验证配置传递
        """
        importer = FileImporter(self.config)
        self.assertEqual(importer.data_dir, self.temp_dir.name)
        self.assertEqual(importer.file_cache, {"tick": {}, "kline": {}})

    def test_connect(self):
        """
        测试连接方法（扫描文件）

        学习点：
        - 测试文件扫描逻辑
        - 验证文件缓存结构
        """
        # 运行测试
        result = self.run_async(self.importer.connect())

        # 验证结果
        self.assertTrue(result)
        self.assertIn("BTCUSDT", self.importer.file_cache["tick"])
        self.assertIn("ETHUSDT", self.importer.file_cache["tick"])

    def test_disconnect(self):
        """
        测试断开连接方法（清理缓存）

        学习点：
        - 测试资源释放
        - 验证缓存清理
        """
        # 先连接以填充缓存
        self.run_async(self.importer.connect())

        # 然后断开连接
        result = self.run_async(self.importer.disconnect())

        # 验证结果
        self.assertTrue(result)
        self.assertEqual(self.importer.file_cache, {"tick": {}, "kline": {}})

    def test_fetch_tick_data(self):
        """
        测试获取Tick数据方法

        学习点：
        - 测试文件读取和解析
        - 验证数据过滤和转换
        """
        # 先连接以扫描文件
        self.run_async(self.importer.connect())

        # 运行测试
        result = self.run_async(self.importer.fetch_tick_data("BTCUSDT"))

        # 验证结果
        self.assertGreaterEqual(len(result), 2)
        self.assertIn("price", result[0])
        self.assertIn("amount", result[0])
        self.assertIn("side", result[0])

    def test_fetch_kline_data(self):
        """
        测试获取K线数据方法

        学习点：
        - 测试不同时间周期的数据读取
        - 验证OHLCV数据解析
        """
        # 先连接以扫描文件
        self.run_async(self.importer.connect())

        # 运行测试
        result = self.run_async(self.importer.fetch_kline_data("BTCUSDT", "1h"))

        # 验证结果
        self.assertGreaterEqual(len(result), 2)
        self.assertIn("open", result[0])
        self.assertIn("high", result[0])
        self.assertIn("low", result[0])
        self.assertIn("close", result[0])
        self.assertIn("volume", result[0])

    def test_subscribe_methods(self):
        """
        测试订阅方法（应返回False）

        学习点：
        - 测试不支持的功能
        - 验证接口完整性
        """

        # 模拟回调函数
        async def mock_callback(data):
            pass

        # 测试订阅Tick数据
        result = self.run_async(
            self.importer.subscribe_tick(["BTCUSDT"], mock_callback)
        )
        self.assertFalse(result)

        # 测试订阅K线数据
        result = self.run_async(
            self.importer.subscribe_kline(["BTCUSDT"], "1h", mock_callback)
        )
        self.assertFalse(result)

    def test_extract_symbol_from_filename(self):
        """
        测试从文件名提取交易品种

        学习点：
        - 测试字符串处理
        - 验证不同文件命名格式
        """
        # 测试不同格式的文件名
        self.assertEqual(
            self.importer._extract_symbol_from_filename("BTCUSDT_tick_20250101.csv"),
            "BTCUSDT",
        )
        self.assertEqual(
            self.importer._extract_symbol_from_filename("tick_BTCUSDT_20250101.csv"),
            "BTCUSDT",
        )

        # 测试无法提取的情况
        self.assertIsNone(
            self.importer._extract_symbol_from_filename("invalid_file.csv")
        )

    def test_extract_symbol_and_timeframe(self):
        """
        测试从文件名提取交易品种和时间周期

        学习点：
        - 测试多值提取
        - 验证不同格式的处理
        """
        # 测试不同格式的文件名
        symbol, timeframe = self.importer._extract_symbol_and_timeframe(
            "BTCUSDT_kline_1h_20250101.csv"
        )
        self.assertEqual(symbol, "BTCUSDT")
        self.assertEqual(timeframe, "1h")

        symbol, timeframe = self.importer._extract_symbol_and_timeframe(
            "kline_BTCUSDT_1h_20250101.csv"
        )
        self.assertEqual(symbol, "BTCUSDT")
        self.assertEqual(timeframe, "1h")

        # 测试无法提取的情况
        symbol, timeframe = self.importer._extract_symbol_and_timeframe(
            "invalid_file.csv"
        )
        self.assertIsNone(symbol)
        self.assertIsNone(timeframe)

    def test_connect_failure(self):
        self.mock_exchange.load_markets.side_effect = Exception("Connection failed")
        result = self.run_async(self.collector.connect())
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
