# timescale_manager.py - TimescaleDB数据库管理器

"""
文件说明：
    这个文件实现了TimescaleDB数据库管理器，继承自db_base.py中的DBManager抽象接口。
    TimescaleDB是基于PostgreSQL的时间序列数据库扩展，专为高效处理时间序列数据而设计。
    它通过"超表(hypertable)"功能自动分区时间序列数据，提供高效的写入和查询性能。

学习目标：
    1. 了解TimescaleDB的特性和优势，特别是在处理时间序列数据方面
    2. 学习asyncpg库进行异步PostgreSQL/TimescaleDB操作
    3. 掌握时间序列数据的高效存储和查询技术
    4. 理解数据库连接池和批量操作的性能优化
"""

import asyncio
import asyncpg
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json
import logging

from .db_base import DBManager
from ..data_collector.models import TickData, KlineData

# 创建日志记录器
logger = logging.getLogger("TimescaleManager")


class TimescaleManager(DBManager):
    """
    TimescaleDB管理器，用于高效存储和查询时间序列数据

    学习点：
    - TimescaleDB是PostgreSQL的扩展，专为时间序列数据优化
    - 超表(hypertable)自动分区数据，提高查询和写入性能
    - 连接池管理多个数据库连接，提高并发性能
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化TimescaleDB管理器

        Args:
            config: 配置参数，至少包含：
                - host: 数据库主机
                - port: 数据库端口
                - user: 数据库用户名
                - password: 数据库密码
                - database: 数据库名称
        """
        super().__init__(config)
        self.pool = None

        # 开发阶段，暴露配置信息以便调试
        # 注意：生产环境应隐藏敏感信息如密码
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 5432)
        self.user = config.get("user", "postgres")
        self.password = config.get("password", "postgres")
        self.database = config.get("database", "gcg_quant")

        logger.info(
            f"TimescaleDB配置: host={self.host}, port={self.port}, database={self.database}, user={self.user}"
        )

    async def connect(self) -> bool:
        """
        连接到TimescaleDB

        Returns:
            bool: 连接是否成功

        学习点：
        - asyncpg提供异步PostgreSQL接口，支持非阻塞操作
        - 连接池(create_pool)管理多个数据库连接，提高并发性能
        - 开发阶段可简化错误处理，直接暴露异常便于调试
        """
        logger.info(f"连接到TimescaleDB: {self.host}:{self.port}/{self.database}")

        # 开发阶段简化错误处理，直接暴露异常
        # 创建连接池
        self.pool = await asyncpg.create_pool(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
        )

        # 初始化数据表
        await self.init_tables()

        logger.info("成功连接到TimescaleDB")
        return True

    async def disconnect(self) -> bool:
        """
        断开与TimescaleDB的连接

        Returns:
            bool: 断开连接是否成功
        """
        logger.info("断开与TimescaleDB的连接")

        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("成功断开与TimescaleDB的连接")

        return True

    async def init_tables(self) -> bool:
        """
        初始化数据表和超表

        Returns:
            bool: 初始化是否成功

        学习点：
        - TimescaleDB扩展为PostgreSQL添加时间序列功能
        - 超表(hypertable)自动将数据按时间分区，提高性能
        - 注意错误处理，某些操作（如创建已存在的表）可能会失败
        """
        logger.info("初始化TimescaleDB表和超表")

        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            # 检查表是否已创建，避免重复执行
            await conn.execute("CREATE TABLE IF NOT EXISTS tick_data (...)")
            # 检查是否已是超表
            is_hypertable = await conn.fetchval(
                "SELECT hypertable_name FROM timescaledb_information.hypertables WHERE table_name = 'tick_data'"
            )
            if not is_hypertable:
                await conn.execute(
                    "SELECT create_hypertable('tick_data', 'time', if_not_exists => TRUE);"
                )
            # 创建TimescaleDB扩展（如果不存在）
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

            # 创建Tick数据表
            await conn.execute(
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

            # 创建K线数据表
            await conn.execute(
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

            # 将表转换为超表
            # 注意：如果表已经是超表，这会抛出异常，所以需要捕获并忽略
            try:
                await conn.execute(
                    "SELECT create_hypertable('tick_data', 'time', if_not_exists => TRUE);"
                )
                logger.info("成功创建tick_data超表")
            except Exception as e:
                # 如果表已经是超表，会抛出异常，这是预期的行为
                logger.info(f"tick_data可能已经是超表: {str(e)}")

            try:
                await conn.execute(
                    "SELECT create_hypertable('kline_data', 'time', if_not_exists => TRUE);"
                )
                logger.info("成功创建kline_data超表")
            except Exception as e:
                logger.info(f"kline_data可能已经是超表: {str(e)}")

            # 创建索引以加速查询
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tick_symbol ON tick_data (symbol);
            """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_kline_symbol_timeframe ON kline_data (symbol, timeframe);
            """
            )

            logger.info("成功初始化TimescaleDB表和索引")
            return True

    async def save_tick_data(
        self, data: Union[TickData, List[TickData]], batch_size: int = 1000
    ) -> bool:
        """
        保存Tick数据

        Args:
            data: 单个Tick数据对象或Tick数据对象列表
            batch_size: 批处理大小，默认1000条

        Returns:
            bool: 保存是否成功

        学习点：
        - 使用executemany进行批量插入，提高性能
        - 批量处理控制内存使用并提高性能
        - TimescaleDB针对时间排序的数据进行了优化
        """
        logger.info("保存Tick数据到TimescaleDB")

        if not self.pool:
            raise RuntimeError("保存Tick数据失败: 未连接到TimescaleDB")

        # 转换为列表以统一处理
        data_list = data if isinstance(data, list) else [data]
        if not data_list:
            return True

        async with self.pool.acquire() as conn:
            # 开始事务
            async with conn.transaction():
                # 按批次处理数据
                for i in range(0, len(data_list), batch_size):
                    batch = data_list[i : i + batch_size]

                    # 批量插入数据
                    values = [
                        (
                            item.datetime,
                            item.symbol,
                            item.price,
                            item.amount,
                            item.side,
                            item.source,
                            item.trade_id,
                        )
                        for item in batch
                    ]

                    # 使用ON CONFLICT更新现有记录
                    await conn.executemany(
                        """
                        INSERT INTO tick_data 
                        (time, symbol, price, amount, side, source, trade_id)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (time, symbol, trade_id) 
                        DO UPDATE SET 
                            price = EXCLUDED.price,
                            amount = EXCLUDED.amount,
                            side = EXCLUDED.side,
                            source = EXCLUDED.source
                    """,
                        values,
                    )

            logger.info(f"成功保存 {len(data_list)} 条Tick数据到TimescaleDB")
            return True

    async def save_kline_data(
        self, data: Union[KlineData, List[KlineData]], batch_size: int = 1000
    ) -> bool:
        """
        保存K线数据

        Args:
            data: 单个K线数据对象或K线数据对象列表
            batch_size: 批处理大小，默认1000条

        Returns:
            bool: 保存是否成功
        """
        logger.info("保存K线数据到TimescaleDB")

        if not self.pool:
            raise RuntimeError("保存K线数据失败: 未连接到TimescaleDB")

        # 转换为列表以统一处理
        data_list = data if isinstance(data, list) else [data]
        if not data_list:
            return True

        async with self.pool.acquire() as conn:
            # 开始事务
            async with conn.transaction():
                # 按批次处理数据
                for i in range(0, len(data_list), batch_size):
                    batch = data_list[i : i + batch_size]

                    # 批量插入数据
                    values = [
                        (
                            item.datetime,
                            item.symbol,
                            item.timeframe,
                            item.open,
                            item.high,
                            item.low,
                            item.close,
                            item.volume,
                            item.source,
                        )
                        for item in batch
                    ]

                    # 使用ON CONFLICT更新现有记录
                    await conn.executemany(
                        """
                        INSERT INTO kline_data 
                        (time, symbol, timeframe, open, high, low, close, volume, source)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (time, symbol, timeframe) 
                        DO UPDATE SET 
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            source = EXCLUDED.source
                    """,
                        values,
                    )

            logger.info(f"成功保存 {len(data_list)} 条K线数据到TimescaleDB")
            return True

    async def query_tick_data(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[TickData]:
        """
        查询Tick数据

        Args:
            symbol: 交易品种符号
            start_time: 开始时间，默认为None（不限制）
            end_time: 结束时间，默认为None（不限制）
            limit: 返回结果数量限制，默认1000条

        Returns:
            List[TickData]: Tick数据列表

        学习点：
        - TimescaleDB针对时间范围查询进行了优化
        - 参数化查询安全且高效
        - 动态构建查询语句，按需添加条件
        """
        logger.info(f"查询 {symbol} 的Tick数据")

        if not self.pool:
            raise RuntimeError("查询Tick数据失败: 未连接到TimescaleDB")

        async with self.pool.acquire() as conn:
            query = """
                SELECT time, symbol, price, amount, side, source, trade_id
                FROM tick_data
                WHERE symbol = $1
            """

            params = [symbol]
            param_idx = 2

            if start_time:
                query += f" AND time >= ${param_idx}"
                params.append(start_time)
                param_idx += 1

            if end_time:
                query += f" AND time <= ${param_idx}"
                params.append(end_time)
                param_idx += 1

            query += " ORDER BY time DESC LIMIT $" + str(param_idx)
            params.append(limit)

            rows = await conn.fetch(query, *params)

            # 转换为TickData对象
            result = []
            for row in rows:
                tick = TickData(
                    symbol=row["symbol"],
                    timestamp=int(row["time"].timestamp() * 1000),
                    datetime=row["time"],
                    price=float(row["price"]),
                    amount=float(row["amount"]),
                    side=row["side"],
                    source=row["source"],
                    trade_id=row["trade_id"],
                )
                result.append(tick)

            logger.info(f"成功查询 {symbol} 的Tick数据，共 {len(result)} 条")
            return result

    async def query_kline_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[KlineData]:
        """
        查询K线数据

        Args:
            symbol: 交易品种符号
            timeframe: 时间周期
            start_time: 开始时间，默认为None（不限制）
            end_time: 结束时间，默认为None（不限制）
            limit: 返回结果数量限制，默认1000条

        Returns:
            List[KlineData]: K线数据列表
        """
        logger.info(f"查询 {symbol} 的 {timeframe} K线数据")

        if not self.pool:
            raise RuntimeError("查询K线数据失败: 未连接到TimescaleDB")

        async with self.pool.acquire() as conn:
            query = """
                SELECT time, symbol, timeframe, open, high, low, close, volume, source
                FROM kline_data
                WHERE symbol = $1 AND timeframe = $2
            """

            params = [symbol, timeframe]
            param_idx = 3

            if start_time:
                query += f" AND time >= ${param_idx}"
                params.append(start_time)
                param_idx += 1

            if end_time:
                query += f" AND time <= ${param_idx}"
                params.append(end_time)
                param_idx += 1

            query += " ORDER BY time DESC LIMIT $" + str(param_idx)
            params.append(limit)

            rows = await conn.fetch(query, *params)

            # 转换为KlineData对象
            result = []
            for row in rows:
                kline = KlineData(
                    symbol=row["symbol"],
                    timestamp=int(row["time"].timestamp() * 1000),
                    datetime=row["time"],
                    timeframe=row["timeframe"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    source=row["source"],
                )
                result.append(kline)

            logger.info(
                f"成功查询 {symbol} 的 {timeframe} K线数据，共 {len(result)} 条"
            )
            return result
