# redis_manager.py - Redis缓存管理器

"""
文件说明：
    这个文件实现了Redis缓存管理器，用于实时数据的缓存和发布/订阅功能。
    Redis是一个内存数据库，提供高速数据读写和发布/订阅机制，适合实时数据处理。
    该管理器支持配置中的"use_redis"选项，可以无缝启用或禁用Redis功能。

学习目标：
    1. 了解Redis的基本特性和优势，特别是在缓存和发布/订阅方面
    2. 学习aioredis库进行异步Redis操作
    3. 掌握合理的缓存策略和过期时间设置
    4. 理解如何设计可选组件，使系统在不同配置下都能正常工作
"""

import asyncio
import aioredis
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import json
import logging

from ..data_collector.models import TickData, KlineData

# 创建日志记录器
logger = logging.getLogger("RedisManager")


class RedisManager:
    """
    Redis管理器，用于实时数据缓存和发布/订阅功能

    学习点：
    - Redis是内存数据库，提供高速读写和发布/订阅机制
    - 可选组件设计，支持无缝启用/禁用
    - 异步Redis操作，充分利用非阻塞I/O
    """

    def __init__(self, config: Dict[str, Any], use_redis: bool = True):
        """
        初始化Redis管理器

        Args:
            config: Redis配置参数
            use_redis: 是否启用Redis，默认为True

        学习点：
        - 通过参数控制组件启用/禁用
        - 配置驱动的初始化
        - 日志记录配置状态，便于调试
        """
        self.config = config
        self.use_redis = use_redis
        self.redis = None
        self.pubsub = None

        # 开发阶段，暴露配置信息以便调试
        # 注意：生产环境应隐藏敏感信息如密码
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6379)
        self.db = config.get("db", 0)
        self.password = config.get("password", "")

        if not self.use_redis:
            logger.info("Redis功能已禁用")
        else:
            logger.info(f"Redis配置: host={self.host}, port={self.port}, db={self.db}")

    async def connect(self) -> bool:
        """
        连接到Redis

        Returns:
            bool: 连接是否成功

        学习点：
        - 条件连接，根据配置决定是否连接
        - 优雅处理禁用状态，不影响系统功能
        - aioredis提供异步Redis接口，支持非阻塞操作
        """
        # 如果Redis禁用，直接返回成功
        if not self.use_redis:
            logger.info("Redis功能已禁用，跳过连接")
            return True

        logger.info(f"连接到Redis: {self.host}:{self.port}/{self.db}")

        # 构建Redis连接URL
        password_part = f":{self.password}@" if self.password else ""
        redis_url = f"redis://{password_part}{self.host}:{self.port}/{self.db}"

        # 开发阶段简化错误处理，直接暴露异常
        self.redis = await aioredis.from_url(redis_url)

        # 创建发布/订阅对象
        self.pubsub = self.redis.pubsub()

        logger.info("成功连接到Redis")
        return True

    async def disconnect(self) -> bool:
        """
        断开与Redis的连接

        Returns:
            bool: 断开连接是否成功
        """
        # 如果Redis禁用或未连接，直接返回成功
        if not self.use_redis or not self.redis:
            return True

        logger.info("断开与Redis的连接")

        await self.redis.close()
        self.redis = None
        self.pubsub = None

        logger.info("成功断开与Redis的连接")
        return True

    async def save_tick_data(self, data: Union[TickData, List[TickData]]) -> bool:
        # 学习点：如果 use_redis=False，此方法直接返回 True，不执行任何操作
        if not self.use_redis or not self.redis:
            return True
        logger.info("保存Tick数据到Redis")
        data_list = data if isinstance(data, list) else [data]
        if not data_list:
            return True
        pipe = self.redis.pipeline()
        for item in data_list:
            # 使用 Hash 结构存储，减少键数量
            hash_key = f"tick:{item.symbol}"
            value = json.dumps(item.to_dict())
            pipe.hset(hash_key, item.timestamp, value)
            pipe.expire(hash_key, 3600)  # 设置过期时间 1 小时
            pipe.hset(f"tick:{item.symbol}:latest", "data", value)
            pipe.publish(f"tick:{item.symbol}", value)
        await pipe.execute()
        logger.info(f"成功保存 {len(data_list)} 条Tick数据到Redis")
        return True

    async def save_kline_data(self, data: Union[KlineData, List[KlineData]]) -> bool:
        """
        保存K线数据到Redis

        Args:
            data: 单个K线数据对象或K线数据对象列表

        Returns:
            bool: 保存是否成功

        学习点：
        - 根据不同时间周期设置不同的过期时间
        - JSON序列化对象，便于存储和传输
        - 频道命名约定，确保订阅者能接收到正确的数据
        """
        # 如果Redis禁用或未连接，直接返回成功
        if not self.use_redis or not self.redis:
            return True

        logger.info("保存K线数据到Redis")

        # 转换为列表以统一处理
        data_list = data if isinstance(data, list) else [data]
        if not data_list:
            return True

        # 使用管道批量操作，提高效率
        pipe = self.redis.pipeline()

        for item in data_list:
            # 构建Redis键，格式: kline:{symbol}:{timeframe}:{timestamp}
            key = f"kline:{item.symbol}:{item.timeframe}:{item.timestamp}"

            # 将数据转换为JSON字符串
            value = json.dumps(item.to_dict())

            # 设置数据，并设置过期时间（根据时间周期设置不同的过期时间）
            expire_seconds = self._get_expire_seconds(item.timeframe)
            pipe.set(key, value, ex=expire_seconds)

            # 将最新的K线数据存储在单独的键中
            latest_key = f"kline:{item.symbol}:{item.timeframe}:latest"
            pipe.set(latest_key, value)

            # 发布数据到对应的频道
            channel = f"kline:{item.symbol}:{item.timeframe}"
            pipe.publish(channel, value)

        # 执行管道中的所有命令
        await pipe.execute()

        logger.info(f"成功保存 {len(data_list)} 条K线数据到Redis")
        return True

    async def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """
        获取最新的Tick数据

        Args:
            symbol: 交易品种符号

        Returns:
            Optional[TickData]: Tick数据对象，如果不存在则返回None

        学习点：
        - 条件返回，在禁用时返回None
        - 错误处理确保接口稳定
        - 对象反序列化，从JSON字符串还原对象
        """
        # 如果Redis禁用或未连接，直接返回None
        if not self.use_redis or not self.redis:
            return None

        logger.info(f"获取 {symbol} 的最新Tick数据")

        try:
            # 获取最新Tick数据的键
            key = f"tick:{symbol}:latest"

            # 获取数据
            value = await self.redis.get(key)
            if not value:
                return None

            # 解析JSON数据
            data = json.loads(value)

            # 创建TickData对象
            tick = TickData.from_dict(data)

            return tick
        except Exception as e:
            logger.error(f"获取最新Tick数据失败: {str(e)}")
            return None

    async def get_latest_kline(
        self, symbol: str, timeframe: str
    ) -> Optional[KlineData]:
        """
        获取最新的K线数据

        Args:
            symbol: 交易品种符号
            timeframe: 时间周期

        Returns:
            Optional[KlineData]: K线数据对象，如果不存在则返回None
        """
        # 如果Redis禁用或未连接，直接返回None
        if not self.use_redis or not self.redis:
            return None

        logger.info(f"获取 {symbol} 的最新 {timeframe} K线数据")

        try:
            # 获取最新K线数据的键
            key = f"kline:{symbol}:{timeframe}:latest"

            # 获取数据
            value = await self.redis.get(key)
            if not value:
                return None

            # 解析JSON数据
            data = json.loads(value)

            # 创建KlineData对象
            kline = KlineData.from_dict(data)

            return kline
        except Exception as e:
            logger.error(f"获取最新K线数据失败: {str(e)}")
            return None

    async def subscribe_tick(self, symbols: List[str], callback: Callable) -> bool:
        """
        订阅Tick数据

        Args:
            symbols: 交易品种符号列表
            callback: 数据回调函数

        Returns:
            bool: 订阅是否成功

        学习点：
        - 发布/订阅模式，实现实时数据推送
        - 回调函数处理，异步接收和处理数据
        - 优雅处理禁用状态，返回False表示不支持
        """
        # 如果Redis禁用或未连接，直接返回失败
        if not self.use_redis or not self.redis or not self.pubsub:
            logger.info("Redis功能已禁用，不支持订阅")
            return False

        logger.info(f"订阅 {symbols} 的Tick数据")

        try:
            # 订阅多个频道
            channels = [f"tick:{symbol}" for symbol in symbols]

            # 注册回调函数
            async def message_handler(message):
                # 解析消息
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    tick = TickData.from_dict(data)
                    await callback([tick])

            # 设置消息处理函数
            self.pubsub.psubscribe(**{channel: message_handler for channel in channels})

            # 启动监听任务
            asyncio.create_task(self._listen())

            logger.info(f"成功订阅 {symbols} 的Tick数据")
            return True
        except Exception as e:
            logger.error(f"订阅Tick数据失败: {str(e)}")
            return False

    async def subscribe_kline(
        self, symbols: List[str], timeframe: str, callback: Callable
    ) -> bool:
        """
        订阅K线数据

        Args:
            symbols: 交易品种符号列表
            timeframe: 时间周期
            callback: 数据回调函数

        Returns:
            bool: 订阅是否成功
        """
        # 如果Redis禁用或未连接，直接返回失败
        if not self.use_redis or not self.redis or not self.pubsub:
            logger.info("Redis功能已禁用，不支持订阅")
            return False

        logger.info(f"订阅 {symbols} 的 {timeframe} K线数据")

        try:
            # 订阅多个频道
            channels = [f"kline:{symbol}:{timeframe}" for symbol in symbols]

            # 注册回调函数
            async def message_handler(message):
                # 解析消息
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    kline = KlineData.from_dict(data)
                    await callback([kline])

            # 设置消息处理函数
            self.pubsub.psubscribe(**{channel: message_handler for channel in channels})

            # 启动监听任务
            asyncio.create_task(self._listen())

            logger.info(f"成功订阅 {symbols} 的 {timeframe} K线数据")
            return True
        except Exception as e:
            logger.error(f"订阅K线数据失败: {str(e)}")
            return False

    async def _listen(self):
        """
        监听Redis发布/订阅消息

        学习点：
        - 异步事件循环，持续监听消息
        - 错误处理确保任务稳定运行
        - 内部方法，以下划线开头表示不应被外部直接调用
        """
        try:
            await self.pubsub.run()
        except asyncio.CancelledError:
            logger.info("Redis发布/订阅监听任务被取消")
        except Exception as e:
            logger.error(f"Redis发布/订阅监听发生错误: {str(e)}")

    def _get_expire_seconds(self, timeframe: str) -> int:
        """
        根据时间周期获取过期时间（秒）

        Args:
            timeframe: 时间周期

        Returns:
            int: 过期时间（秒）

        学习点：
        - 根据业务需求设置合理的过期时间
        - 缓存策略：短周期数据保留较短时间，长周期数据保留较长时间
        - 内部辅助方法，提高代码可读性和可维护性
        """
        # 根据时间周期设置不同的过期时间
        if timeframe == "1m":
            return 3600 * 24  # 1天
        elif timeframe == "5m":
            return 3600 * 24 * 3  # 3天
        elif timeframe == "15m":
            return 3600 * 24 * 7  # 7天
        elif timeframe == "1h":
            return 3600 * 24 * 30  # 30天
        elif timeframe == "4h":
            return 3600 * 24 * 90  # 90天
        elif timeframe == "1d":
            return 3600 * 24 * 365  # 365天
        else:
            return 3600 * 24 * 7  # 默认7天
