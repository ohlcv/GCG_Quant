# sqlite_manager.py - SQLite数据库管理器

"""
文件说明：
    这个文件实现了SQLite数据库管理器，继承自db_base.py中的DBManager抽象接口。
    SQLite是一个轻量级的嵌入式数据库，不需要单独的服务器进程，适合初期开发和测试。
    该管理器提供了所有必要的数据库操作，包括表创建、数据存储和查询等功能。

学习目标：
    1. 了解SQLite数据库的基本特性和使用方法
    2. 学习如何使用aiosqlite库进行异步SQLite操作
    3. 掌握数据库索引优化和查询构建技术
    4. 理解如何处理批量数据处理和事务管理
"""

import os
import asyncio
import aiosqlite
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import logging

from .db_base import DBManager
from ..data_storage.models import TickData, KlineData

# 创建日志记录器
logger = logging.getLogger("SQLiteManager")


class SQLiteManager(DBManager):
    """
    SQLite数据库管理器，实现对Tick和K线数据的存储和查询

    学习点：
    - 继承抽象基类，实现其所有抽象方法
    - SQLite是文件型数据库，通过文件路径指定数据库位置
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化SQLite管理器

        Args:
            config: 配置参数，至少包含：
                - db_file: SQLite数据库文件路径
        """
        super().__init__(config)
        self.db_file = config.get("db_file", "./data/gcg_quant.db")
        self.connection = None

        # 确保数据库目录存在
        os.makedirs(os.path.dirname(os.path.abspath(self.db_file)), exist_ok=True)

    async def connect(self) -> bool:
        """
        连接到SQLite数据库

        Returns:
            bool: 连接是否成功

        学习点：
        - aiosqlite提供异步SQLite接口，支持非阻塞操作
        - 使用异步上下文管理器（async with）进行资源管理
        """
        try:
            logger.info(f"连接到SQLite数据库: {self.db_file}")

            # 打开数据库连接
            self.connection = await aiosqlite.connect(self.db_file)

            # 启用外键约束
            await self.connection.execute("PRAGMA foreign_keys = ON")

            # 启用日志控制日志
            await self.connection.execute("PRAGMA journal_mode = WAL")

            # 初始化数据表
            await self.init_tables()

            logger.info("成功连接到SQLite数据库")
            return True
        except Exception as e:
            logger.error(f"连接到SQLite数据库失败: {str(e)}")
            return False

    async def disconnect(self) -> bool:
        """
        断开与SQLite数据库的连接

        Returns:
            bool: 断开连接是否成功
        """
        try:
            if self.connection:
                logger.info("断开与SQLite数据库的连接")
                await self.connection.close()
                self.connection = None
                logger.info("成功断开与SQLite数据库的连接")
            return True
        except Exception as e:
            logger.error(f"断开与SQLite数据库的连接失败: {str(e)}")
            return False

    async def init_tables(self) -> bool:
        """
        初始化数据表

        Returns:
            bool: 初始化是否成功

        学习点：
        - 使用SQL DDL语句创建表和索引
        - 索引可以加速查询，但会增加写入开销
        - SQLite支持事务，通过commit确保操作的原子性
        """
        try:
            if not self.connection:
                logger.error("初始化表失败: 未连接到SQLite数据库")
                return False

            # 创建Tick数据表
            await self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tick_data (
                    time TIMESTAMP NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    amount REAL NOT NULL,
                    side TEXT NOT NULL,
                    source TEXT NOT NULL,
                    trade_id TEXT,
                    PRIMARY KEY (time, symbol, trade_id)
                )
            """
            )

            # 创建K线数据表
            await self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS kline_data (
                    time TIMESTAMP NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    source TEXT NOT NULL,
                    PRIMARY KEY (time, symbol, timeframe)
                )
            """
            )

            # 创建索引以加速查询
            await self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tick_symbol_time 
                ON tick_data (symbol, time)
            """
            )

            await self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_kline_symbol_timeframe_time 
                ON kline_data (symbol, timeframe, time)
            """
            )

            # 提交事务
            await self.connection.commit()

            logger.info("成功初始化SQLite数据表和索引")
            return True
        except Exception as e:
            logger.error(f"初始化SQLite数据表失败: {str(e)}")
            return False

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
        - 批量处理提高性能，避免频繁的数据库操作
        - 使用事务确保数据一致性
        - 异常处理和日志记录确保健壮性
        """
        if not self.connection:
            logger.error("保存Tick数据失败: 未连接到SQLite数据库")
            return False

        # 转换为列表以统一处理
        data_list = data if isinstance(data, list) else [data]
        if not data_list:
            return True

        try:
            # 开始事务
            await self.connection.execute("BEGIN TRANSACTION")

            # 按批次处理数据
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i : i + batch_size]

                # 批量插入数据
                await self.connection.executemany(
                    """
                    INSERT OR REPLACE INTO tick_data 
                    (time, symbol, price, amount, side, source, trade_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    [
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
                    ],
                )

            # 提交事务
            await self.connection.commit()

            logger.info(f"成功保存 {len(data_list)} 条Tick数据到SQLite")
            return True
        except Exception as e:
            # 回滚事务
            await self.connection.rollback()
            logger.error(f"保存Tick数据到SQLite失败: {str(e)}")
            return False

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
        if not self.connection:
            logger.error("保存K线数据失败: 未连接到SQLite数据库")
            return False

        # 转换为列表以统一处理
        data_list = data if isinstance(data, list) else [data]
        if not data_list:
            return True

        try:
            # 开始事务
            await self.connection.execute("BEGIN TRANSACTION")

            # 按批次处理数据
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i : i + batch_size]

                # 批量插入数据
                await self.connection.executemany(
                    """
                    INSERT OR REPLACE INTO kline_data 
                    (time, symbol, timeframe, open, high, low, close, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
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
                    ],
                )

            # 提交事务
            await self.connection.commit()

            logger.info(f"成功保存 {len(data_list)} 条K线数据到SQLite")
            return True
        except Exception as e:
            # 回滚事务
            await self.connection.rollback()
            logger.error(f"保存K线数据到SQLite失败: {str(e)}")
            return False

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
        - 动态SQL构建，根据条件添加查询参数
        - 结果集映射，将数据库行转换为对象
        - 参数化查询防止SQL注入
        """
        if not self.connection:
            logger.error("查询Tick数据失败: 未连接到SQLite数据库")
            return []

        try:
            # 构建查询SQL和参数
            query = """
                SELECT time, symbol, price, amount, side, source, trade_id
                FROM tick_data
                WHERE symbol = ?
            """

            params = [symbol]

            # 添加时间条件
            if start_time:
                query += " AND time >= ?"
                params.append(start_time)

            if end_time:
                query += " AND time <= ?"
                params.append(end_time)

            # 添加排序和限制
            query += " ORDER BY time DESC LIMIT ?"
            params.append(limit)

            # 执行查询
            async with self.connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

                # 转换为TickData对象
                result = []
                for row in rows:
                    time, symbol, price, amount, side, source, trade_id = row

                    # 将time (字符串) 转换为datetime对象
                    if isinstance(time, str):
                        time = datetime.fromisoformat(time)

                    tick = TickData(
                        symbol=symbol,
                        timestamp=int(time.timestamp() * 1000),
                        datetime=time,
                        price=float(price),
                        amount=float(amount),
                        side=side,
                        source=source,
                        trade_id=trade_id,
                    )
                    result.append(tick)

                logger.info(f"成功查询 {symbol} 的Tick数据，共 {len(result)} 条")
                return result
        except Exception as e:
            logger.error(f"查询Tick数据失败: {str(e)}")
            return []

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
        if not self.connection:
            logger.error("查询K线数据失败: 未连接到SQLite数据库")
            return []

        try:
            # 构建查询SQL和参数
            query = """
                SELECT time, symbol, timeframe, open, high, low, close, volume, source
                FROM kline_data
                WHERE symbol = ? AND timeframe = ?
            """

            params = [symbol, timeframe]

            # 添加时间条件
            if start_time:
                query += " AND time >= ?"
                params.append(start_time)

            if end_time:
                query += " AND time <= ?"
                params.append(end_time)

            # 添加排序和限制
            query += " ORDER BY time DESC LIMIT ?"
            params.append(limit)

            # 执行查询
            async with self.connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

                # 转换为KlineData对象
                result = []
                for row in rows:
                    (
                        time,
                        symbol,
                        timeframe,
                        open_price,
                        high,
                        low,
                        close,
                        volume,
                        source,
                    ) = row

                    # 将time (字符串) 转换为datetime对象
                    if isinstance(time, str):
                        time = datetime.fromisoformat(time)

                    kline = KlineData(
                        symbol=symbol,
                        timestamp=int(time.timestamp() * 1000),
                        datetime=time,
                        timeframe=timeframe,
                        open=float(open_price),
                        high=float(high),
                        low=float(low),
                        close=float(close),
                        volume=float(volume),
                        source=source,
                    )
                    result.append(kline)

                logger.info(
                    f"成功查询 {symbol} 的 {timeframe} K线数据，共 {len(result)} 条"
                )
                return result
        except Exception as e:
            logger.error(f"查询K线数据失败: {str(e)}")
            return []
