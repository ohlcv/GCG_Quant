# test_storage.py - 数据存储组件测试

"""
文件说明：
    这个文件包含了对数据存储组件的单元测试，主要测试SQLiteManager、TimescaleManager和RedisManager类。
    使用Python标准库中的unittest模块和异步测试技术，验证数据存储和查询功能的正确性。
    测试用例覆盖了数据库连接、表创建、数据存储和查询等关键功能，采用mock技术模拟数据库和Redis连接。

学习目标：
    1. 了解如何编写数据库相关的单元测试
    2. 学习如何模拟数据库和缓存操作
    3. 掌握异步测试的高级技巧
    4. 理解测试隔离和环境准备的重要性
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
from src.data_storage.sqlite_manager import SQLiteManager
from src.data_storage.timescale_manager import TimescaleManager
from src.data_storage.redis_manager import RedisManager
from src.data_storage.models import TickData, KlineData


class TestSQLiteManager(unittest.TestCase):
    """
    SQLiteManager测试类

    学习点：
    - 临时数据库的创建和测试
    - 异步数据库操作的测试
    - 事务和回滚的测试技巧
    """

    def setUp(self):
        """
        测试前的准备工作

        学习点：
        - 使用临时文件作为测试数据库
        - 避免测试影响实际数据
        """
        # 创建临时文件作为数据库
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_file.close()

        # 测试配置
        self.config = {"db_file": self.temp_file.name}

        # 创建测试实例
        self.db_manager = SQLiteManager(self.config)

        # 创建测试数据
        self.create_test_data()

    def tearDown(self):
        """
        测试后的清理工作

        学习点：
        - 清理临时数据库
        - 确保资源释放
        """
        # 清理临时数据库文件
        os.unlink(self.temp_file.name)

    def run_async(self, coroutine):
        """
        运行异步方法的辅助函数

        Args:
            coroutine: 异步协程对象

        Returns:
            任何异步方法的返回值
        """
        return asyncio.get_event_loop().run_until_complete(coroutine)

    def create_test_data(self):
        """
        创建测试数据

        学习点：
        - 准备测试所需的数据对象
        - 测试数据的多样性
        """
        # 创建Tick数据
        self.tick_data = TickData(
            symbol="BTC/USDT",
            timestamp=int(datetime.now().timestamp() * 1000),
            datetime=datetime.now(),
            price=50000.0,
            amount=1.0,
            side="buy",
            source="test",
            trade_id="1",
        )

        # 创建K线数据
        self.kline_data = KlineData(
            symbol="BTC/USDT",
            timestamp=int(datetime.now().timestamp() * 1000),
            datetime=datetime.now(),
            timeframe="1h",
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
            source="test",
        )

    def test_init(self):
        """
        测试初始化方法

        学习点：
        - 检查实例属性是否正确初始化
        - 验证配置传递
        """
        db_manager = SQLiteManager(self.config)
        self.assertEqual(db_manager.db_file, self.temp_file.name)
        self.assertIsNone(db_manager.connection)

    def test_connect_and_disconnect(self):
        """
        测试连接和断开连接方法

        学习点：
        - 测试异步数据库连接
        - 验证连接状态和资源管理
        """
        # 测试连接
        connect_result = self.run_async(self.db_manager.connect())
        self.assertTrue(connect_result)
        self.assertIsNotNone(self.db_manager.connection)

        # 测试断开连接
        disconnect_result = self.run_async(self.db_manager.disconnect())
        self.assertTrue(disconnect_result)
        self.assertIsNone(self.db_manager.connection)

    def test_init_tables(self):
        """
        测试初始化表方法

        学习点：
        - 测试数据库表创建
        - 验证表结构和索引
        """
        # 首先连接数据库
        self.run_async(self.db_manager.connect())

        # 测试初始化表
        init_result = self.run_async(self.db_manager.init_tables())
        self.assertTrue(init_result)

        # 验证表是否创建成功
        # 这里可以使用SQL查询验证表结构

        # 断开连接
        self.run_async(self.db_manager.disconnect())

    def test_save_and_query_tick_data(self):
        """
        测试保存和查询Tick数据方法

        学习点：
        - 测试数据写入和读取
        - 验证查询过滤和结果排序
        """
        # 首先连接数据库
        self.run_async(self.db_manager.connect())

        # 保存测试数据
        save_result = self.run_async(self.db_manager.save_tick_data(self.tick_data))
        self.assertTrue(save_result)

        # 查询数据
        query_result = self.run_async(self.db_manager.query_tick_data("BTC/USDT"))

        # 验证查询结果
        self.assertGreaterEqual(len(query_result), 1)
        self.assertEqual(query_result[0].symbol, "BTC/USDT")
        self.assertEqual(query_result[0].price, 50000.0)

        # 断开连接
        self.run_async(self.db_manager.disconnect())

    def test_save_and_query_kline_data(self):
        """
        测试保存和查询K线数据方法

        学习点：
        - 测试多字段数据的保存和查询
        - 验证时间范围过滤
        """
        # 首先连接数据库
        self.run_async(self.db_manager.connect())

        # 保存测试数据
        save_result = self.run_async(self.db_manager.save_kline_data(self.kline_data))
        self.assertTrue(save_result)

        # 查询数据
        query_result = self.run_async(
            self.db_manager.query_kline_data("BTC/USDT", "1h")
        )

        # 验证查询结果
        self.assertGreaterEqual(len(query_result), 1)
        self.assertEqual(query_result[0].symbol, "BTC/USDT")
        self.assertEqual(query_result[0].timeframe, "1h")
        self.assertEqual(query_result[0].open, 50000.0)
        self.assertEqual(query_result[0].high, 50100.0)
        self.assertEqual(query_result[0].low, 49900.0)
        self.assertEqual(query_result[0].close, 50050.0)
        self.assertEqual(query_result[0].volume, 100.0)

        # 断开连接
        self.run_async(self.db_manager.disconnect())

    def test_batch_save(self):
        """
        测试批量保存方法

        学习点：
        - 测试批量数据操作
        - 验证性能优化措施
        """
        # 首先连接数据库
        self.run_async(self.db_manager.connect())

        # 创建批量测试数据
        tick_data_list = []
        for i in range(10):
            tick = TickData(
                symbol="BTC/USDT",
                timestamp=int(
                    (datetime.now() + timedelta(minutes=i)).timestamp() * 1000
                ),
                datetime=datetime.now() + timedelta(minutes=i),
                price=50000.0 + i * 10,
                amount=1.0,
                side="buy",
                source="test",
                trade_id=str(i + 1),
            )
            tick_data_list.append(tick)

        # 批量保存
        save_result = self.run_async(self.db_manager.save_tick_data(tick_data_list))
        self.assertTrue(save_result)

        # 查询数据
        query_result = self.run_async(self.db_manager.query_tick_data("BTC/USDT"))

        # 验证查询结果
        self.assertGreaterEqual(len(query_result), 10)

        # 断开连接
        self.run_async(self.db_manager.disconnect())

    def test_time_range_query(self):
        """
        测试时间范围查询

        学习点：
        - 测试条件查询
        - 验证时间过滤功能
        """
        # 首先连接数据库
        self.run_async(self.db_manager.connect())

        # 创建具有不同时间的测试数据
        now = datetime.now()
        tick1 = TickData(
            symbol="BTC/USDT",
            timestamp=int((now - timedelta(hours=2)).timestamp() * 1000),
            datetime=now - timedelta(hours=2),
            price=50000.0,
            amount=1.0,
            side="buy",
            source="test",
            trade_id="1",
        )
        tick2 = TickData(
            symbol="BTC/USDT",
            timestamp=int((now - timedelta(hours=1)).timestamp() * 1000),
            datetime=now - timedelta(hours=1),
            price=50100.0,
            amount=0.5,
            side="sell",
            source="test",
            trade_id="2",
        )
        tick3 = TickData(
            symbol="BTC/USDT",
            timestamp=int(now.timestamp() * 1000),
            datetime=now,
            price=50200.0,
            amount=0.2,
            side="buy",
            source="test",
            trade_id="3",
        )

        # 保存测试数据
        self.run_async(self.db_manager.save_tick_data([tick1, tick2, tick3]))

        # 使用时间范围查询
        start_time = now - timedelta(hours=1, minutes=30)
        end_time = now - timedelta(minutes=30)
        query_result = self.run_async(
            self.db_manager.query_tick_data("BTC/USDT", start_time, end_time)
        )

        # 验证查询结果
        self.assertEqual(len(query_result), 1)
        self.assertEqual(query_result[0].trade_id, "2")

        # 断开连接
        self.run_async(self.db_manager.disconnect())


class TestTimescaleManager(unittest.TestCase):
    """
    TimescaleManager测试类

    学习点：
    - 模拟PostgreSQL/TimescaleDB连接
    - 测试时间序列数据库特性
    - 高级查询和聚合功能测试
    """

    def setUp(self):
        """
        测试前的准备工作

        学习点：
        - 模拟数据库连接
        - 准备测试环境
        """
        # 测试配置
        self.config = {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "postgres",
            "database": "test_gcg_quant",
        }

        # 创建测试实例
        self.db_manager = TimescaleManager(self.config)

        # 模拟连接池
        self.mock_pool_patcher = mock.patch("asyncpg.create_pool")
        self.mock_pool = self.mock_pool_patcher.start()

        # 创建测试数据
        self.create_test_data()

    def tearDown(self):
        """
        测试后的清理工作

        学习点：
        - 清理模拟对象
        - 释放资源
        """
        # 停止模拟
        self.mock_pool_patcher.stop()

    def run_async(self, coroutine):
        """
        运行异步方法的辅助函数

        Args:
            coroutine: 异步协程对象

        Returns:
            任何异步方法的返回值
        """
        return asyncio.get_event_loop().run_until_complete(coroutine)

    def create_test_data(self):
        """
        创建测试数据

        学习点：
        - 准备测试所需的数据对象
        - 模拟数据的多样性
        """
        # 创建Tick数据
        self.tick_data = TickData(
            symbol="BTC/USDT",
            timestamp=int(datetime.now().timestamp() * 1000),
            datetime=datetime.now(),
            price=50000.0,
            amount=1.0,
            side="buy",
            source="test",
            trade_id="1",
        )

        # 创建K线数据
        self.kline_data = KlineData(
            symbol="BTC/USDT",
            timestamp=int(datetime.now().timestamp() * 1000),
            datetime=datetime.now(),
            timeframe="1h",
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
            source="test",
        )

    def test_init(self):
        """
        测试初始化方法

        学习点：
        - 检查实例属性是否正确初始化
        - 验证配置传递
        """
        db_manager = TimescaleManager(self.config)
        self.assertEqual(db_manager.host, "localhost")
        self.assertEqual(db_manager.port, 5432)
        self.assertEqual(db_manager.user, "postgres")
        self.assertEqual(db_manager.password, "postgres")
        self.assertEqual(db_manager.database, "test_gcg_quant")
        self.assertIsNone(db_manager.pool)

    def test_connect_and_disconnect(self):
        """
        测试连接和断开连接方法

        学习点：
        - 测试异步数据库连接池
        - 验证连接状态和资源管理
        """
        # 设置mock行为
        mock_pool = mock.MagicMock()
        self.mock_pool.return_value = mock_pool

        # 模拟连接
        connect_result = self.run_async(self.db_manager.connect())
        self.assertTrue(connect_result)
        self.assertEqual(self.db_manager.pool, mock_pool)

        # 模拟断开连接
        disconnect_result = self.run_async(self.db_manager.disconnect())
        self.assertTrue(disconnect_result)

    def test_init_tables(self):
        """
        测试初始化表方法

        学习点：
        - 测试TimescaleDB特有的表创建
        - 验证超表(hypertable)创建
        """
        # 设置mock行为
        mock_pool = mock.MagicMock()
        mock_conn = mock.MagicMock()
        mock_conn.execute = mock.AsyncMock()
        mock_conn.fetchval = mock.AsyncMock(return_value=None)
        mock_pool.acquire = mock.AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        self.mock_pool.return_value = mock_pool
        self.db_manager.pool = mock_pool

        # 测试初始化表
        init_result = self.run_async(self.db_manager.init_tables())
        self.assertTrue(init_result)

        # 验证是否执行了必要的SQL语句
        self.assertTrue(mock_conn.execute.called)

    def test_save_tick_data(self):
        """
        测试保存Tick数据方法

        学习点：
        - 测试批量数据写入
        - 验证事务管理
        """
        # 设置mock行为
        mock_pool = mock.MagicMock()
        mock_conn = mock.MagicMock()
        mock_conn.executemany = mock.AsyncMock()
        mock_conn.transaction = mock.MagicMock()
        mock_conn.transaction.return_value.__aenter__ = mock.AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = mock.AsyncMock()
        mock_pool.acquire = mock.AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        self.mock_pool.return_value = mock_pool
        self.db_manager.pool = mock_pool

        # 测试保存Tick数据
        save_result = self.run_async(self.db_manager.save_tick_data(self.tick_data))
        self.assertTrue(save_result)

        # 验证是否执行了必要的SQL语句
        self.assertTrue(mock_conn.executemany.called)

    def test_save_kline_data(self):
        """
        测试保存K线数据方法

        学习点：
        - 测试OHLCV数据写入
        - 验证参数绑定
        """
        # 设置mock行为
        mock_pool = mock.MagicMock()
        mock_conn = mock.MagicMock()
        mock_conn.executemany = mock.AsyncMock()
        mock_conn.transaction = mock.MagicMock()
        mock_conn.transaction.return_value.__aenter__ = mock.AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = mock.AsyncMock()
        mock_pool.acquire = mock.AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        self.mock_pool.return_value = mock_pool
        self.db_manager.pool = mock_pool

        # 测试保存K线数据
        save_result = self.run_async(self.db_manager.save_kline_data(self.kline_data))
        self.assertTrue(save_result)

        # 验证是否执行了必要的SQL语句
        self.assertTrue(mock_conn.executemany.called)

    def test_query_tick_data(self):
        """
        测试查询Tick数据方法

        学习点：
        - 测试时间范围查询
        - 验证结果转换
        """
        # 设置mock行为
        mock_pool = mock.MagicMock()
        mock_conn = mock.MagicMock()
        mock_conn.fetch = mock.AsyncMock()

        # 模拟查询结果
        mock_result = [
            {
                "symbol": "BTC/USDT",
                "time": datetime.now(),
                "price": 50000.0,
                "amount": 1.0,
                "side": "buy",
                "source": "test",
                "trade_id": "1",
            }
        ]
        mock_conn.fetch.return_value = mock_result

        mock_pool.acquire = mock.AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        self.mock_pool.return_value = mock_pool
        self.db_manager.pool = mock_pool

        # 测试查询Tick数据
        query_result = self.run_async(self.db_manager.query_tick_data("BTC/USDT"))

        # 验证查询结果
        self.assertEqual(len(query_result), 1)
        self.assertEqual(query_result[0].symbol, "BTC/USDT")
        self.assertEqual(query_result[0].price, 50000.0)

        # 验证是否执行了必要的SQL语句
        self.assertTrue(mock_conn.fetch.called)

    def test_query_kline_data(self):
        """
        测试查询K线数据方法

        学习点：
        - 测试多条件查询
        - 验证对象映射
        """
        # 设置mock行为
        mock_pool = mock.MagicMock()
        mock_conn = mock.MagicMock()
        mock_conn.fetch = mock.AsyncMock()

        # 模拟查询结果
        mock_result = [
            {
                "symbol": "BTC/USDT",
                "time": datetime.now(),
                "timeframe": "1h",
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.0,
                "source": "test",
            }
        ]
        mock_conn.fetch.return_value = mock_result

        mock_pool.acquire = mock.AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        self.mock_pool.return_value = mock_pool
        self.db_manager.pool = mock_pool

        # 测试查询K线数据
        query_result = self.run_async(
            self.db_manager.query_kline_data("BTC/USDT", "1h")
        )

        # 验证查询结果
        self.assertEqual(len(query_result), 1)
        self.assertEqual(query_result[0].symbol, "BTC/USDT")
        self.assertEqual(query_result[0].timeframe, "1h")

        # 验证是否执行了必要的SQL语句
        self.assertTrue(mock_conn.fetch.called)


class TestRedisManager(unittest.TestCase):
    """
    RedisManager测试类

    学习点：
    - 模拟Redis连接和操作
    - 测试发布/订阅功能
    - 验证缓存机制
    """

    def setUp(self):
        """
        测试前的准备工作

        学习点：
        - 模拟Redis连接
        - 准备测试环境
        """
        # 测试配置
        self.config = {"host": "localhost", "port": 6379, "db": 0, "password": ""}

        # 创建测试实例
        self.redis_manager = RedisManager(self.config, use_redis=True)

        # 模拟Redis连接
        self.mock_redis_patcher = mock.patch("aioredis.from_url")
        self.mock_redis = self.mock_redis_patcher.start()

        # 创建测试数据
        self.create_test_data()

    def tearDown(self):
        """
        测试后的清理工作

        学习点：
        - 清理模拟对象
        - 释放资源
        """
        # 停止模拟
        self.mock_redis_patcher.stop()

    def run_async(self, coroutine):
        """
        运行异步方法的辅助函数

        Args:
            coroutine: 异步协程对象

        Returns:
            任何异步方法的返回值
        """
        return asyncio.get_event_loop().run_until_complete(coroutine)

    def create_test_data(self):
        """
        创建测试数据

        学习点：
        - 准备测试所需的数据对象
        - 模拟数据的多样性
        """
        # 创建Tick数据
        self.tick_data = TickData(
            symbol="BTC/USDT",
            timestamp=int(datetime.now().timestamp() * 1000),
            datetime=datetime.now(),
            price=50000.0,
            amount=1.0,
            side="buy",
            source="test",
            trade_id="1",
        )

        # 创建K线数据
        self.kline_data = KlineData(
            symbol="BTC/USDT",
            timestamp=int(datetime.now().timestamp() * 1000),
            datetime=datetime.now(),
            timeframe="1h",
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
            source="test",
        )

    def test_init(self):
        """
        测试初始化方法

        学习点：
        - 检查实例属性是否正确初始化
        - 验证配置传递
        """
        redis_manager = RedisManager(self.config, use_redis=True)
        self.assertEqual(redis_manager.host, "localhost")
        self.assertEqual(redis_manager.port, 6379)
        self.assertEqual(redis_manager.db, 0)
        self.assertEqual(redis_manager.password, "")
        self.assertTrue(redis_manager.use_redis)
        self.assertIsNone(redis_manager.redis)

    def test_connect_and_disconnect(self):
        """
        测试连接和断开连接方法

        学习点：
        - 测试Redis连接和断开
        - 验证连接状态
        """
        # 设置mock行为
        mock_redis_client = mock.MagicMock()
        mock_redis_client.close = mock.AsyncMock()
        self.mock_redis.return_value = mock_redis_client

        # 测试连接
        connect_result = self.run_async(self.redis_manager.connect())
        self.assertTrue(connect_result)
        self.assertEqual(self.redis_manager.redis, mock_redis_client)

        # 测试断开连接
        disconnect_result = self.run_async(self.redis_manager.disconnect())
        self.assertTrue(disconnect_result)
        self.assertIsNone(self.redis_manager.redis)
        mock_redis_client.close.assert_called_once()

    def test_disabled_redis(self):
        """
        测试禁用Redis功能

        学习点：
        - 测试功能开关
        - 验证禁用状态下的行为
        """
        # 创建禁用Redis的实例
        disabled_redis = RedisManager(self.config, use_redis=False)

        # 测试连接（应直接返回True，不进行实际连接）
        connect_result = self.run_async(disabled_redis.connect())
        self.assertTrue(connect_result)
        self.assertIsNone(disabled_redis.redis)

        # 测试保存数据（应直接返回True，不进行实际操作）
        save_result = self.run_async(disabled_redis.save_tick_data(self.tick_data))
        self.assertTrue(save_result)

        # 测试获取数据（应返回None）
        get_result = self.run_async(disabled_redis.get_latest_tick("BTC/USDT"))
        self.assertIsNone(get_result)

    def test_save_tick_data(self):
        """
        测试保存Tick数据方法

        学习点：
        - 测试Redis流水线(pipeline)操作
        - 验证数据序列化和存储
        """
        # 设置mock行为
        mock_redis_client = mock.MagicMock()
        mock_pipeline = mock.MagicMock()
        mock_pipeline.hset = mock.MagicMock()
        mock_pipeline.expire = mock.MagicMock()
        mock_pipeline.publish = mock.MagicMock()
        mock_pipeline.execute = mock.AsyncMock()
        mock_redis_client.pipeline.return_value = mock_pipeline

        self.mock_redis.return_value = mock_redis_client
        self.redis_manager.redis = mock_redis_client

        # 测试保存单个Tick数据
        save_result = self.run_async(self.redis_manager.save_tick_data(self.tick_data))
        self.assertTrue(save_result)

        # 验证是否执行了必要的Redis操作
        self.assertTrue(mock_pipeline.hset.called)
        self.assertTrue(mock_pipeline.expire.called)
        self.assertTrue(mock_pipeline.publish.called)
        self.assertTrue(mock_pipeline.execute.called)

        # 测试保存多个Tick数据
        save_result = self.run_async(
            self.redis_manager.save_tick_data([self.tick_data, self.tick_data])
        )
        self.assertTrue(save_result)

    def test_save_kline_data(self):
        """
        测试保存K线数据方法

        学习点：
        - 测试不同数据类型的存储
        - 验证键命名规则
        """
        # 设置mock行为
        mock_redis_client = mock.MagicMock()
        mock_pipeline = mock.MagicMock()
        mock_pipeline.set = mock.MagicMock()
        mock_pipeline.publish = mock.MagicMock()
        mock_pipeline.execute = mock.AsyncMock()
        mock_redis_client.pipeline.return_value = mock_pipeline

        self.mock_redis.return_value = mock_redis_client
        self.redis_manager.redis = mock_redis_client

        # 测试保存单个K线数据
        save_result = self.run_async(
            self.redis_manager.save_kline_data(self.kline_data)
        )
        self.assertTrue(save_result)

        # 验证是否执行了必要的Redis操作
        self.assertTrue(mock_pipeline.set.called)
        self.assertTrue(mock_pipeline.publish.called)
        self.assertTrue(mock_pipeline.execute.called)

        # 测试保存多个K线数据
        save_result = self.run_async(
            self.redis_manager.save_kline_data([self.kline_data, self.kline_data])
        )
        self.assertTrue(save_result)

    def test_get_latest_tick(self):
        """
        测试获取最新Tick数据方法

        学习点：
        - 测试数据检索
        - 验证数据反序列化
        """
        # 设置mock行为
        mock_redis_client = mock.MagicMock()

        # 模拟Redis响应
        tick_dict = self.tick_data.to_dict()
        mock_redis_client.get = mock.AsyncMock(return_value=json.dumps(tick_dict))

        self.mock_redis.return_value = mock_redis_client
        self.redis_manager.redis = mock_redis_client

        # 测试获取最新Tick数据
        result = self.run_async(self.redis_manager.get_latest_tick("BTC/USDT"))

        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(result.symbol, "BTC/USDT")
        self.assertEqual(result.price, 50000.0)

        # 验证是否执行了必要的Redis操作
        mock_redis_client.get.assert_called_once()

    def test_get_latest_kline(self):
        """
        测试获取最新K线数据方法

        学习点：
        - 测试不同类型数据的检索
        - 验证复杂对象的反序列化
        """
        # 设置mock行为
        mock_redis_client = mock.MagicMock()

        # 模拟Redis响应
        kline_dict = self.kline_data.to_dict()
        mock_redis_client.get = mock.AsyncMock(return_value=json.dumps(kline_dict))

        self.mock_redis.return_value = mock_redis_client
        self.redis_manager.redis = mock_redis_client

        # 测试获取最新K线数据
        result = self.run_async(self.redis_manager.get_latest_kline("BTC/USDT", "1h"))

        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(result.symbol, "BTC/USDT")
        self.assertEqual(result.timeframe, "1h")
        self.assertEqual(result.open, 50000.0)

        # 验证是否执行了必要的Redis操作
        mock_redis_client.get.assert_called_once()

    def test_subscribe_tick(self):
        """
        测试订阅Tick数据方法

        学习点：
        - 测试Redis发布/订阅功能
        - 验证订阅回调机制
        """
        # 设置mock行为
        mock_redis_client = mock.MagicMock()
        mock_pubsub = mock.MagicMock()
        mock_pubsub.psubscribe = mock.MagicMock()
        mock_redis_client.pubsub = mock.MagicMock(return_value=mock_pubsub)

        self.mock_redis.return_value = mock_redis_client
        self.redis_manager.redis = mock_redis_client
        self.redis_manager.pubsub = mock_pubsub

        # 模拟回调函数
        async def mock_callback(data):
            pass

        # 测试订阅Tick数据
        with mock.patch("asyncio.create_task") as mock_task:
            subscribe_result = self.run_async(
                self.redis_manager.subscribe_tick(["BTC/USDT"], mock_callback)
            )
            self.assertTrue(subscribe_result)

            # 验证是否执行了必要的Redis操作
            self.assertTrue(mock_pubsub.psubscribe.called)
            self.assertTrue(mock_task.called)

    def test_subscribe_kline(self):
        """
        测试订阅K线数据方法

        学习点：
        - 测试不同频道的订阅
        - 验证多频道处理
        """
        # 设置mock行为
        mock_redis_client = mock.MagicMock()
        mock_pubsub = mock.MagicMock()
        mock_pubsub.psubscribe = mock.MagicMock()
        mock_redis_client.pubsub = mock.MagicMock(return_value=mock_pubsub)

        self.mock_redis.return_value = mock_redis_client
        self.redis_manager.redis = mock_redis_client
        self.redis_manager.pubsub = mock_pubsub

        # 模拟回调函数
        async def mock_callback(data):
            pass

        # 测试订阅K线数据
        with mock.patch("asyncio.create_task") as mock_task:
            subscribe_result = self.run_async(
                self.redis_manager.subscribe_kline(["BTC/USDT"], "1h", mock_callback)
            )
            self.assertTrue(subscribe_result)

            # 验证是否执行了必要的Redis操作
            self.assertTrue(mock_pubsub.psubscribe.called)
            self.assertTrue(mock_task.called)

    def test_error_handling(self):
        """
        测试错误处理

        学习点：
        - 测试异常处理
        - 验证错误恢复机制
        """
        # 设置mock行为引发异常
        mock_redis_client = mock.MagicMock()
        mock_redis_client.get = mock.AsyncMock(side_effect=Exception("Redis error"))

        self.mock_redis.return_value = mock_redis_client
        self.redis_manager.redis = mock_redis_client

        # 测试错误处理
        result = self.run_async(self.redis_manager.get_latest_tick("BTC/USDT"))

        # 验证结果（应返回None而不是引发异常）
        self.assertIsNone(result)

    def test_large_batch_save(self):
        self.run_async(self.db_manager.connect())
        large_batch = [self.tick_data] * 1000  # 超大批量
        save_result = self.run_async(
            self.db_manager.save_tick_data(large_batch, batch_size=500)
        )
        self.assertTrue(save_result)
        query_result = self.run_async(self.db_manager.query_tick_data("BTC/USDT"))
        self.assertEqual(len(query_result), 1000)
        self.run_async(self.db_manager.disconnect())

    def test_tick_expiry(self):
        mock_redis_client = mock.MagicMock()
        mock_pipeline = mock.MagicMock()
        mock_pipeline.set = mock.MagicMock()
        mock_pipeline.execute = mock.AsyncMock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        self.redis_manager.redis = mock_redis_client
        save_result = self.run_async(self.redis_manager.save_tick_data(self.tick_data))
        self.assertTrue(save_result)
        mock_pipeline.expire.assert_called_with(3600)  # 验证 1 小时过期


if __name__ == "__main__":
    unittest.main()
