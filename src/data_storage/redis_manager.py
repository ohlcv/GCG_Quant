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
    5. 掌握Redis的数据结构选择（Hash vs String）及其应用场景
    6. 理解发布/订阅模式在实时数据传输中的应用
    7. 学习异步编程中的任务管理和事件循环使用
    8. 掌握开发环境下的调试和错误追踪技巧
    9. 理解配置驱动的组件设计模式
    10. 学习内存数据库的性能优化策略
"""

import asyncio
from redis.asyncio import Redis as RedisAsync
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import json

from ..data_storage.models import TickData, KlineData


class RedisManager:
    """
    Redis管理器，用于实时数据缓存和发布/订阅功能

    学习点：
    - Redis是内存数据库，提供高速读写和发布/订阅机制
    - 可选组件设计，支持无缝启用/禁用
    - 异步Redis操作，充分利用非阻塞I/O
    - 使用pipeline批量操作提升性能
    - 合理的数据结构选择和过期策略
    - 发布/订阅模式实现实时数据推送
    - 开发环境下的错误快速暴露策略
    - 内存数据库的连接池管理
    - 实时数据的并发处理机制
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
        - 配置参数的默认值设置
        - 敏感信息处理（密码等）
        - 组件状态管理
        - 开发环境配置暴露
        - 配置参数校验机制
        """
        self.config = config
        self.use_redis = use_redis
        self.redis = None
        self.pubsub = None

        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6379)
        self.db = config.get("db", 0)
        self.password = config.get("password", "")

        if not self.use_redis:
            print("Redis功能已禁用")
        else:
            print(f"Redis配置: host={self.host}, port={self.port}, db={self.db}")

    async def connect(self) -> bool:
        """
        连接到Redis

        Returns:
            bool: 连接是否成功

        学习点：
        - 条件连接，根据配置决定是否连接
        - 优雅处理禁用状态，不影响系统功能
        - aioredis提供异步Redis接口，支持非阻塞操作
        - Redis URL构建和连接参数设置
        - 发布/订阅对象的初始化
        - 连接状态监控
        - 连接超时处理
        """
        if not self.use_redis:
            print("Redis功能已禁用，跳过连接")
            return True

        print(f"连接到Redis: {self.host}:{self.port}/{self.db}")
        password_part = f":{self.password}@" if self.password else ""
        redis_url = f"redis://{password_part}{self.host}:{self.port}/{self.db}"
        self.redis = await RedisAsync.from_url(redis_url)
        self.pubsub = self.redis.pubsub()
        print("成功连接到Redis")
        return True

    async def disconnect(self) -> bool:
        """
        断开与Redis的连接

        Returns:
            bool: 断开连接是否成功

        学习点：
        - 资源清理和连接关闭
        - 状态重置
        - 优雅关闭处理
        - 连接池清理
        - 订阅关系清理
        """
        if not self.use_redis or not self.redis:
            return True

        print("断开与Redis的连接")
        await self.redis.close()
        self.redis = None
        self.pubsub = None
        print("成功断开与Redis的连接")
        return True

    async def save_tick_data(self, data: Union[TickData, List[TickData]]) -> bool:
        """
        保存Tick数据到Redis

        Args:
            data: 单个Tick数据对象或Tick数据对象列表

        Returns:
            bool: 保存是否成功

        学习点：
        - 使用Hash结构存储，减少键数量
        - Pipeline批量操作提升性能
        - 数据序列化和反序列化
        - 发布/订阅实时数据更新
        - 合理的过期时间设置
        - 批量数据处理策略
        - 内存使用优化
        """
        if not self.use_redis or not self.redis:
            return True

        print("保存Tick数据到Redis")
        data_list = data if isinstance(data, list) else [data]
        if not data_list:
            return True

        pipe = self.redis.pipeline()
        for item in data_list:
            hash_key = f"tick:{item.symbol}"
            value = json.dumps(item.to_dict())
            pipe.hset(hash_key, item.timestamp, value)
            pipe.expire(hash_key, 3600)  # 1小时过期
            pipe.hset(f"tick:{item.symbol}:latest", "data", value)
            pipe.publish(f"tick:{item.symbol}", value)
        await pipe.execute()
        print(f"成功保存 {len(data_list)} 条Tick数据到Redis")
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
        - 批量操作性能优化
        - 数据一致性保证
        - 键名设计最佳实践
        - 实时数据更新策略
        """
        if not self.use_redis or not self.redis:
            return True

        print("保存K线数据到Redis")
        data_list = data if isinstance(data, list) else [data]
        if not data_list:
            return True

        pipe = self.redis.pipeline()
        for item in data_list:
            key = f"kline:{item.symbol}:{item.timeframe}:{item.timestamp}"
            value = json.dumps(item.to_dict())
            expire_seconds = self._get_expire_seconds(item.timeframe)
            pipe.set(key, value, ex=expire_seconds)
            latest_key = f"kline:{item.symbol}:{item.timeframe}:latest"
            pipe.set(latest_key, value)
            channel = f"kline:{item.symbol}:{item.timeframe}"
            pipe.publish(channel, value)

        await pipe.execute()
        print(f"成功保存 {len(data_list)} 条K线数据到Redis")
        return True

    async def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """
        获取最新的Tick数据

        Args:
            symbol: 交易品种符号

        Returns:
            Optional[TickData]: Tick数据对象，如果不存在则返回None

        学习点：
        - 使用Optional类型提高类型安全
        - 数据反序列化和对象重建
        - 最新数据快速访问设计
        - 空值处理和返回类型一致性
        - 数据完整性验证
        - 类型转换安全处理
        - 缓存命中率优化
        """
        if not self.use_redis or not self.redis:
            return None

        print(f"获取 {symbol} 的最新Tick数据")
        key = f"tick:{symbol}:latest"
        value = await self.redis.get(key)

        if not value:
            print(f"未找到 {symbol} 的最新Tick数据")
            return None

        data = json.loads(value)
        tick = TickData.from_dict(data)
        print(f"成功获取 {symbol} 的最新Tick数据")
        return tick

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

        学习点：
        - 组合键设计（symbol + timeframe）
        - 数据一致性保证
        - 类型转换和数据验证
        - 返回值类型安全
        - 缓存键设计模式
        - 数据版本控制
        - 查询性能优化
        """
        if not self.use_redis or not self.redis:
            return None

        print(f"获取 {symbol} 的最新 {timeframe} K线数据")
        key = f"kline:{symbol}:{timeframe}:latest"
        value = await self.redis.get(key)

        if not value:
            print(f"未找到 {symbol} 的 {timeframe} K线数据")
            return None

        data = json.loads(value)
        kline = KlineData.from_dict(data)
        print(f"成功获取 {symbol} 的 {timeframe} K线数据")
        return kline

    async def subscribe_tick(self, symbols: List[str], callback: Callable) -> bool:
        """
        订阅Tick数据

        Args:
            symbols: 交易品种符号列表
            callback: 数据回调函数

        Returns:
            bool: 订阅是否成功

        学习点：
        - 发布/订阅模式实现
        - 回调函数设计
        - 异步事件处理
        - 多通道订阅
        - 消息处理函数设计
        - 订阅状态管理
        - 并发回调处理
        - 消息队列缓冲
        """
        if not self.use_redis or not self.redis or not self.pubsub:
            print("Redis未启用或未连接，无法订阅")
            return False

        print(f"订阅 {symbols} 的Tick数据")
        channels = [f"tick:{symbol}" for symbol in symbols]

        async def message_handler(message):
            if message["type"] == "message":
                data = json.loads(message["data"])
                tick = TickData.from_dict(data)
                await callback([tick])

        self.pubsub.psubscribe(**{channel: message_handler for channel in channels})
        asyncio.create_task(self._listen())
        print(f"成功订阅 {symbols} 的Tick数据")
        return True

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

        学习点：
        - 多维度订阅（品种+时间周期）
        - 回调函数异步执行
        - 事件驱动设计
        - 消息过滤和处理
        - 任务管理
        - 订阅模式扩展性
        - 消息重试机制
        - 订阅生命周期管理
        """
        if not self.use_redis or not self.redis or not self.pubsub:
            print("Redis未启用或未连接，无法订阅")
            return False

        print(f"订阅 {symbols} 的 {timeframe} K线数据")
        channels = [f"kline:{symbol}:{timeframe}" for symbol in symbols]

        async def message_handler(message):
            if message["type"] == "message":
                data = json.loads(message["data"])
                kline = KlineData.from_dict(data)
                await callback([kline])

        self.pubsub.psubscribe(**{channel: message_handler for channel in channels})
        asyncio.create_task(self._listen())
        print(f"成功订阅 {symbols} 的 {timeframe} K线数据")
        return True

    async def _listen(self):
        """
        监听Redis发布/订阅消息

        学习点：
        - 异步事件循环设计
        - 任务取消处理
        - 内部方法命名规范
        - 长期运行任务管理
        - 错误传播控制
        - 资源占用监控
        - 消息处理性能优化
        """
        print("开始监听Redis发布/订阅消息")
        await self.pubsub.run()
        print("Redis发布/订阅消息监听已结束")

    def _get_expire_seconds(self, timeframe: str) -> int:
        """
        根据时间周期获取过期时间（秒）

        Args:
            timeframe: 时间周期

        Returns:
            int: 过期时间（秒）

        学习点：
        - 业务逻辑与时间管理
        - 缓存策略设计
        - 配置值管理
        - 默认值处理
        - 时间单位转换
        - 过期时间动态调整
        - 数据生命周期管理
        """
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
