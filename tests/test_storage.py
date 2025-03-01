# test_storage.py - 数据存储组件测试 (修复后版本)

"""
文件说明：
    这个文件包含了对数据存储组件的单元测试，主要测试 SQLiteManager、TimescaleManager 和 RedisManager 类。
    使用 Python 标准库中的 unittest 模块和异步测试技术，验证数据存储和查询功能的正确性。
    测试用例覆盖了数据库连接、表创建、数据存储和查询等关键功能，采用 mock 技术模拟数据库和 Redis 连接。

学习目标：
    1. 了解如何编写数据库相关的单元测试
    2. 学习如何模拟数据库和缓存操作
    3. 掌握异步测试的高级技巧
    4. 理解测试隔离和环境准备的重要性

注意事项：
    运行此测试可能会产生"coroutine 'AsyncMockMixin._execute_mock_call' was never awaited"警告。
    这些警告是测试环境中使用AsyncMock模拟Redis异步方法的副作用，不影响测试结果和生产代码。
    
    警告产生的原因：
    1. 在模拟Redis异步方法(如pubsub.psubscribe())时，Python创建了协程对象
    2. 这些协程在测试环境中没有被等待，因此产生警告
    3. 这是unittest.mock库与异步代码交互时的已知限制

    在生产环境中，所有协程都会被正确地等待，不会出现这些警告。
    这些警告不代表应用代码设计有问题，只是测试框架的局限性。

可能的改进方案：
    - 使用pytest和pytest-asyncio可能会减少这类警告
    - 设置警告过滤器可以屏蔽这些警告: warnings.filterwarnings("ignore", message="coroutine.*was never awaited")
    - 对于高级用户，可以考虑使用asynctest库代替标准unittest
"""

import unittest
import asyncio
import os
import json
import tempfile
import warnings
from unittest.mock import MagicMock, AsyncMock, patch, ANY
from datetime import datetime, timedelta
import logging

# 可选：屏蔽协程未等待的警告
# 取消下面这行的注释可以屏蔽警告
warnings.filterwarnings("ignore", message="coroutine.*was never awaited")

# 禁用测试过程中的日志输出，避免干扰测试结果
logging.disable(logging.CRITICAL)

# 导入被测试的模块
from src.data_storage.sqlite_manager import SQLiteManager
from src.data_storage.timescale_manager import TimescaleManager
from src.data_storage.redis_manager import RedisManager
from src.data_storage.models import TickData, KlineData


class TestSQLiteManager(unittest.TestCase):
    """SQLiteManager 测试类"""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_file.close()
        self.config = {"db_file": self.temp_file.name}
        self.db_manager = SQLiteManager(self.config)
        self.create_test_data()

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def run_async(self, coroutine):
        return asyncio.get_event_loop().run_until_complete(coroutine)

    def create_test_data(self):
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
        """测试初始化方法"""
        db_manager = SQLiteManager(self.config)
        self.assertEqual(db_manager.db_file, self.temp_file.name)
        self.assertIsNone(db_manager.connection)

    def test_connect_and_disconnect(self):
        """测试连接和断开连接方法"""
        connect_result = self.run_async(self.db_manager.connect())
        self.assertTrue(connect_result)
        self.assertIsNotNone(self.db_manager.connection)
        disconnect_result = self.run_async(self.db_manager.disconnect())
        self.assertTrue(disconnect_result)
        self.assertIsNone(self.db_manager.connection)

    def test_init_tables(self):
        """测试初始化表方法"""
        self.run_async(self.db_manager.connect())
        init_result = self.run_async(self.db_manager.init_tables())
        self.assertTrue(init_result)
        self.run_async(self.db_manager.disconnect())

    def test_save_and_query_tick_data(self):
        """测试保存和查询 Tick 数据"""
        self.run_async(self.db_manager.connect())
        save_result = self.run_async(self.db_manager.save_tick_data(self.tick_data))
        self.assertTrue(save_result)
        query_result = self.run_async(self.db_manager.query_tick_data("BTC/USDT"))
        self.assertGreaterEqual(len(query_result), 1)
        self.assertEqual(query_result[0].symbol, "BTC/USDT")
        self.assertEqual(query_result[0].price, 50000.0)
        self.run_async(self.db_manager.disconnect())

    def test_save_and_query_kline_data(self):
        """测试保存和查询 K 线数据"""
        self.run_async(self.db_manager.connect())
        save_result = self.run_async(self.db_manager.save_kline_data(self.kline_data))
        self.assertTrue(save_result)
        query_result = self.run_async(
            self.db_manager.query_kline_data("BTC/USDT", "1h")
        )
        self.assertGreaterEqual(len(query_result), 1)
        self.assertEqual(query_result[0].symbol, "BTC/USDT")
        self.assertEqual(query_result[0].timeframe, "1h")
        self.assertEqual(query_result[0].open, 50000.0)
        self.run_async(self.db_manager.disconnect())

    def test_batch_save(self):
        """测试批量保存方法"""
        self.run_async(self.db_manager.connect())
        tick_data_list = [
            TickData(
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
            for i in range(10)
        ]
        save_result = self.run_async(self.db_manager.save_tick_data(tick_data_list))
        self.assertTrue(save_result)
        query_result = self.run_async(self.db_manager.query_tick_data("BTC/USDT"))
        self.assertGreaterEqual(len(query_result), 10)
        self.run_async(self.db_manager.disconnect())

    def test_time_range_query(self):
        """测试时间范围查询"""
        self.run_async(self.db_manager.connect())
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
        self.run_async(self.db_manager.save_tick_data([tick1, tick2, tick3]))
        start_time = now - timedelta(hours=1, minutes=30)
        end_time = now - timedelta(minutes=30)
        query_result = self.run_async(
            self.db_manager.query_tick_data("BTC/USDT", start_time, end_time)
        )
        self.assertEqual(len(query_result), 1)
        self.assertEqual(query_result[0].trade_id, "2")
        self.run_async(self.db_manager.disconnect())


class TestTimescaleManager(unittest.TestCase):
    """TimescaleManager 测试类"""

    def setUp(self):
        self.config = {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "postgres",
            "database": "test_gcg_quant",
        }
        self.db_manager = TimescaleManager(self.config)
        self.mock_pool = MagicMock()
        self.mock_conn = MagicMock()
        # Set up necessary methods as AsyncMock for await support
        self.mock_conn.execute = AsyncMock()
        self.mock_conn.fetchval = AsyncMock(
            return_value=None
        )  # Default return None for hypertable check
        self.mock_conn.executemany = AsyncMock()
        self.mock_conn.fetch = AsyncMock()
        async_context = AsyncMock()
        async_context.__aenter__ = AsyncMock(return_value=self.mock_conn)
        async_context.__aexit__ = AsyncMock(return_value=None)
        self.mock_pool.acquire.return_value = async_context
        self.mock_pool.close = AsyncMock(return_value=None)
        transaction_context = AsyncMock()
        transaction_context.__aenter__ = AsyncMock(return_value=self.mock_conn)
        transaction_context.__aexit__ = AsyncMock(return_value=None)
        self.mock_conn.transaction = MagicMock(return_value=transaction_context)
        self.db_manager.pool = self.mock_pool
        self.create_test_data()

    def tearDown(self):
        pass

    def run_async(self, coroutine):
        return asyncio.get_event_loop().run_until_complete(coroutine)

    def create_test_data(self):
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
        """测试初始化方法"""
        db_manager = TimescaleManager(self.config)
        self.assertEqual(db_manager.host, "localhost")
        self.assertEqual(db_manager.port, 5432)
        self.assertEqual(db_manager.user, "postgres")
        self.assertEqual(db_manager.password, "postgres")
        self.assertEqual(db_manager.database, "test_gcg_quant")
        self.assertIsNone(db_manager.pool)

    def test_connect_and_disconnect(self):
        """测试连接和断开连接方法"""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = self.mock_pool
            connect_result = self.run_async(self.db_manager.connect())
            self.assertTrue(connect_result)
            self.assertEqual(self.db_manager.pool, self.mock_pool)
            disconnect_result = self.run_async(self.db_manager.disconnect())
            self.assertTrue(disconnect_result)
            self.assertIsNone(self.db_manager.pool)

    def test_init_tables(self):
        """测试初始化表方法"""
        init_result = self.run_async(self.db_manager.init_tables())
        self.assertTrue(init_result)
        self.mock_conn.execute.assert_any_call(
            """
                CREATE TABLE IF NOT EXISTS tick_data (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    price DOUBLE PRECISION NOT NULL,
                    amount DOUBLE PRECISION NOT NULL,
                    side TEXT NOT NULL,
                    source TEXT NOT NULL,
                    trade_id TEXT
                );
            """
        )
        self.mock_conn.execute.assert_any_call(
            """
                CREATE TABLE IF NOT EXISTS kline_data (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open DOUBLE PRECISION NOT NULL,
                    high DOUBLE PRECISION NOT NULL,
                    low DOUBLE PRECISION NOT NULL,
                    close DOUBLE PRECISION NOT NULL,
                    volume DOUBLE PRECISION NOT NULL,
                    source TEXT NOT NULL
                );
            """
        )

    def test_save_tick_data(self):
        """测试保存 Tick 数据方法"""
        save_result = self.run_async(self.db_manager.save_tick_data(self.tick_data))
        self.assertTrue(save_result)
        self.mock_conn.executemany.assert_called()

    def test_save_kline_data(self):
        """测试保存 K 线数据方法"""
        save_result = self.run_async(self.db_manager.save_kline_data(self.kline_data))
        self.assertTrue(save_result)
        self.mock_conn.executemany.assert_called()

    def test_query_tick_data(self):
        """测试查询 Tick 数据方法"""
        self.mock_conn.fetch.return_value = [
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
        query_result = self.run_async(self.db_manager.query_tick_data("BTC/USDT"))
        self.assertEqual(len(query_result), 1)
        self.assertEqual(query_result[0].symbol, "BTC/USDT")
        self.mock_conn.fetch.assert_called()

    def test_query_kline_data(self):
        """测试查询 K 线数据方法"""
        self.mock_conn.fetch.return_value = [
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
        query_result = self.run_async(
            self.db_manager.query_kline_data("BTC/USDT", "1h")
        )
        self.assertEqual(len(query_result), 1)
        self.assertEqual(query_result[0].symbol, "BTC/USDT")
        self.mock_conn.fetch.assert_called()


class TestRedisManager(unittest.TestCase):
    """RedisManager 测试类"""

    def setUp(self):
        self.config = {"host": "localhost", "port": 6379, "db": 0, "password": ""}
        self.redis_manager = RedisManager(self.config, use_redis=True)

        # 创建一个真正的MagicMock对象作为Redis实例
        self.mock_redis = MagicMock()

        # 为Redis的close方法创建一个AsyncMock
        self.mock_redis.close = AsyncMock()

        # 为Redis的get方法创建一个AsyncMock
        self.mock_redis.get = AsyncMock()

        # 创建一个MagicMock作为Pipeline
        self.mock_pipeline = MagicMock()

        # Pipeline的方法返回self以支持链式调用
        self.mock_pipeline.hset.return_value = self.mock_pipeline
        self.mock_pipeline.expire.return_value = self.mock_pipeline
        self.mock_pipeline.publish.return_value = self.mock_pipeline
        self.mock_pipeline.set.return_value = self.mock_pipeline

        # 设置execute为AsyncMock，以便能够await它
        self.mock_pipeline.execute = AsyncMock()

        # 确保Redis的pipeline方法返回我们配置的mock_pipeline
        self.mock_redis.pipeline.return_value = self.mock_pipeline

        # 创建一个MagicMock作为PubSub
        self.mock_pubsub = MagicMock()

        # 设置PubSub的异步方法
        self.mock_pubsub.psubscribe = AsyncMock()
        self.mock_pubsub.unsubscribe = AsyncMock()
        self.mock_pubsub.run = AsyncMock()

        # 确保Redis的pubsub方法返回我们配置的mock_pubsub
        self.mock_redis.pubsub.return_value = self.mock_pubsub

        # 设置redis_manager实例的redis和pubsub属性
        self.redis_manager.redis = self.mock_redis
        self.redis_manager.pubsub = self.mock_pubsub

        self.create_test_data()

    def tearDown(self):
        pass

    def run_async(self, coroutine):
        return asyncio.get_event_loop().run_until_complete(coroutine)

    def create_test_data(self):
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
        """测试初始化方法"""
        redis_manager = RedisManager(self.config, use_redis=True)
        self.assertEqual(redis_manager.host, "localhost")
        self.assertEqual(redis_manager.port, 6379)
        self.assertEqual(redis_manager.db, 0)
        self.assertTrue(redis_manager.use_redis)
        self.assertIsNone(redis_manager.redis)

    def test_connect_and_disconnect(self):
        """测试连接和断开连接方法"""
        with patch(
            "redis.asyncio.Redis.from_url", new_callable=AsyncMock
        ) as mock_from_url:
            mock_redis = MagicMock()
            mock_redis.close = AsyncMock()
            mock_pubsub = MagicMock()
            mock_redis.pubsub.return_value = mock_pubsub
            mock_from_url.return_value = mock_redis

            # 此处保存旧的redis和pubsub，以便恢复
            old_redis = self.redis_manager.redis
            old_pubsub = self.redis_manager.pubsub

            # 设置为None以便测试connect
            self.redis_manager.redis = None
            self.redis_manager.pubsub = None

            connect_result = self.run_async(self.redis_manager.connect())
            self.assertTrue(connect_result)

            # 断开连接
            disconnect_result = self.run_async(self.redis_manager.disconnect())
            self.assertTrue(disconnect_result)
            self.assertIsNone(self.redis_manager.redis)

            # 恢复为原来的mock以继续其他测试
            self.redis_manager.redis = old_redis
            self.redis_manager.pubsub = old_pubsub

    def test_disabled_redis(self):
        """测试禁用 Redis 功能"""
        disabled_redis = RedisManager(self.config, use_redis=False)
        connect_result = self.run_async(disabled_redis.connect())
        self.assertTrue(connect_result)
        self.assertIsNone(disabled_redis.redis)
        save_result = self.run_async(disabled_redis.save_tick_data(self.tick_data))
        self.assertTrue(save_result)
        get_result = self.run_async(disabled_redis.get_latest_tick("BTC/USDT"))
        self.assertIsNone(get_result)

    def test_save_tick_data(self):
        """测试保存 Tick 数据方法"""
        # 使用我们在setUp中创建的mock_pipeline
        save_result = self.run_async(self.redis_manager.save_tick_data(self.tick_data))
        self.assertTrue(save_result)
        # 验证调用了预期的方法
        self.mock_pipeline.hset.assert_called()
        self.mock_pipeline.expire.assert_called()
        self.mock_pipeline.publish.assert_called()
        self.mock_pipeline.execute.assert_called_once()

    def test_save_kline_data(self):
        """测试保存 K 线数据方法"""
        # 使用我们在setUp中创建的mock_pipeline
        save_result = self.run_async(
            self.redis_manager.save_kline_data(self.kline_data)
        )
        self.assertTrue(save_result)
        # 验证调用了预期的方法
        self.mock_pipeline.set.assert_called()
        self.mock_pipeline.publish.assert_called()
        self.mock_pipeline.execute.assert_called_once()

    def test_get_latest_tick(self):
        """测试获取最新 Tick 数据方法"""
        self.mock_redis.get.return_value = json.dumps(self.tick_data.to_dict())
        result = self.run_async(self.redis_manager.get_latest_tick("BTC/USDT"))
        self.assertIsNotNone(result)
        self.assertEqual(result.symbol, "BTC/USDT")
        self.mock_redis.get.assert_called_once()

    def test_get_latest_kline(self):
        """测试获取最新 K 线数据方法"""
        self.mock_redis.get.return_value = json.dumps(self.kline_data.to_dict())
        result = self.run_async(self.redis_manager.get_latest_kline("BTC/USDT", "1h"))
        self.assertIsNotNone(result)
        self.assertEqual(result.symbol, "BTC/USDT")
        self.mock_redis.get.assert_called_once()

    def test_subscribe_tick(self):
        """测试订阅 Tick 数据方法"""

        async def mock_callback(data):
            pass

        # 模拟_listen方法
        with patch.object(
            self.redis_manager, "_listen", new_callable=AsyncMock
        ) as mock_listen:
            # 模拟asyncio.create_task
            with patch("asyncio.create_task") as mock_create_task:
                # 执行测试
                subscribe_result = self.run_async(
                    self.redis_manager.subscribe_tick(["BTC/USDT"], mock_callback)
                )

                # 验证结果
                self.assertTrue(subscribe_result)
                # 验证pubsub.psubscribe被调用(不关心具体参数)
                self.mock_pubsub.psubscribe.assert_called_once()
                # 验证create_task被调用(不关心具体的协程对象)
                mock_create_task.assert_called_once()
                # 验证_listen方法被调用
                mock_listen.assert_called_once()

    def test_subscribe_kline(self):
        """测试订阅 K 线数据方法"""

        async def mock_callback(data):
            pass

        # 模拟_listen方法
        with patch.object(
            self.redis_manager, "_listen", new_callable=AsyncMock
        ) as mock_listen:
            # 模拟asyncio.create_task
            with patch("asyncio.create_task") as mock_create_task:
                # 执行测试
                subscribe_result = self.run_async(
                    self.redis_manager.subscribe_kline(
                        ["BTC/USDT"], "1h", mock_callback
                    )
                )

                # 验证结果
                self.assertTrue(subscribe_result)
                # 验证pubsub.psubscribe被调用(不关心具体参数)
                self.mock_pubsub.psubscribe.assert_called_once()
                # 验证create_task被调用(不关心具体的协程对象)
                mock_create_task.assert_called_once()
                # 验证_listen方法被调用
                mock_listen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
