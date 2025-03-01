# main.py - 主程序入口

"""
文件说明：
    这个文件是GCG_Quant系统的主入口，负责初始化配置、启动各组件并协调它们的工作。
    它实现了一个优雅的启动和关闭流程，并根据配置灵活选择数据库类型和是否启用Redis缓存。
    主程序采用面向对象和异步编程范式，提供清晰的系统结构和运行流程。

学习目标：
    1. 了解如何设计一个模块化、可配置的应用程序入口
    2. 学习异步程序的启动和优雅关闭流程
    3. 掌握基于配置的组件初始化和协调方法
    4. 理解命令行参数处理和日志系统初始化
"""

import asyncio
import argparse
import logging
import os
import signal
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# 导入配置和日志模块
from src.config.settings import load_config
from src.utils.logger import setup_logger

# 导入数据库和Redis管理器
from src.data_storage.db_base import create_db_manager
from src.data_storage.redis_manager import RedisManager

# 导入数据采集器
from src.data_collector.exchange_collector import ExchangeCollector
from src.data_collector.file_importer import FileImporter

# 导入数据模型 - 已修正路径
from src.data_storage.models import TickData, KlineData

# 创建日志记录器
logger = logging.getLogger("main")


class GCGQuant:
    """
    GCG_Quant系统主类，负责初始化和协调各组件

    学习点：
    - 面向对象设计：将系统封装为一个类，便于管理状态和组件
    - 依赖注入：通过配置初始化各组件，降低组件间耦合
    - 异步编程：使用async/await处理I/O密集型操作，提高性能
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        初始化GCG_Quant系统

        Args:
            config_file: 配置文件路径，默认为None（使用默认配置）

        学习点：
        - 配置驱动：通过配置文件初始化系统，提高灵活性
        - 延迟初始化：先加载配置，后续再初始化具体组件
        """
        # 加载配置
        self.config = load_config(config_file)
        logger.info(f"GCG_Quant初始化，配置文件: {config_file or '默认配置'}")

        # 记录配置选项
        db_type = self.config["data_storage"]["db_type"]
        use_redis = self.config["data_storage"]["use_redis"]
        logger.info(f"数据库类型: {db_type}, Redis启用: {use_redis}")

        # 组件引用，延迟初始化
        self.db_manager = None
        self.redis_manager = None
        self.exchange_collector = None
        self.file_importer = None

        # 运行状态标志
        self.is_running = False
        self.shutdown_event = asyncio.Event()

    async def initialize(self) -> bool:
        """
        初始化系统组件

        Returns:
            bool: 初始化是否成功

        学习点：
        - 按依赖顺序初始化组件：先存储后采集
        - 错误处理：任何组件初始化失败都会导致整个系统初始化失败
        - 配置条件初始化：根据配置决定是否初始化某些组件
        """
        try:
            logger.info("初始化GCG_Quant组件")

            # 初始化数据库管理器
            db_type = self.config["data_storage"]["db_type"]
            logger.info(f"创建{db_type}数据库管理器")
            self.db_manager = create_db_manager(self.config["data_storage"])

            # 连接数据库
            db_connected = await self.db_manager.connect()
            if not db_connected:
                logger.error(f"{db_type}数据库连接失败")
                return False

            # 根据配置决定是否初始化Redis管理器
            use_redis = self.config["data_storage"]["use_redis"]
            if use_redis:
                logger.info("创建Redis管理器")
                self.redis_manager = RedisManager(
                    self.config["data_storage"]["redis"], use_redis=True
                )

                # 连接Redis
                redis_connected = await self.redis_manager.connect()
                if not redis_connected:
                    logger.error("Redis连接失败")
                    return False
            else:
                logger.info("Redis功能已禁用，跳过Redis管理器初始化")

            # 初始化数据采集器
            logger.info("创建交易所数据采集器")
            self.exchange_collector = ExchangeCollector(
                self.config["data_collector"]["exchange"]
            )

            logger.info("创建文件数据导入器")
            self.file_importer = FileImporter(
                self.config["data_collector"]["file_import"]
            )

            # 连接数据采集器
            collector_connected = await self.exchange_collector.connect()
            if not collector_connected:
                logger.error("交易所数据采集器连接失败")
                return False

            importer_connected = await self.file_importer.connect()
            if not importer_connected:
                logger.error("文件数据导入器连接失败")
                return False

            # 注册数据回调
            await self._register_callbacks()

            logger.info("GCG_Quant组件初始化成功")
            return True
        except Exception as e:
            logger.error(f"GCG_Quant初始化失败: {str(e)}")
            return False

    async def _register_callbacks(self):
        """
        注册数据回调函数

        学习点：
        - 回调函数设计：使用闭包捕获需要的上下文
        - 配置条件回调：根据Redis是否启用决定回调行为
        - 内部方法：使用下划线前缀表示私有方法
        """

        # 注册Tick数据回调
        async def on_tick_data(tick_data):
            # 保存到数据库
            await self.db_manager.save_tick_data(tick_data)

            # 如果启用了Redis，也保存到Redis
            if self.redis_manager:
                await self.redis_manager.save_tick_data(tick_data)

        # 注册K线数据回调
        async def on_kline_data(kline_data):
            # 保存到数据库
            await self.db_manager.save_kline_data(kline_data)

            # 如果启用了Redis，也保存到Redis
            if self.redis_manager:
                await self.redis_manager.save_kline_data(kline_data)

        # 获取交易品种和时间周期
        symbols = self.config["data_collector"]["symbols"]
        timeframes = self.config["data_collector"]["timeframes"]

        # 设置订阅
        for symbol in symbols:
            # 获取初始历史数据
            await self._fetch_initial_data(symbol, timeframes)

        # 订阅实时数据
        await self.exchange_collector.subscribe_tick(symbols, on_tick_data)

        for timeframe in timeframes:
            await self.exchange_collector.subscribe_kline(
                symbols, timeframe, on_kline_data
            )

    async def _fetch_initial_data(self, symbol: str, timeframes: List[str]):
        """
        获取初始历史数据

        Args:
            symbol: 交易品种符号
            timeframes: 时间周期列表

        学习点：
        - 初始化数据：系统启动时获取一些历史数据
        - 错误处理：单个失败不影响整体流程
        """
        try:
            # 获取最近的Tick数据
            start_time = datetime.now() - timedelta(hours=24)  # 最近24小时
            logger.info(f"获取 {symbol} 的初始Tick数据")
            tick_data = await self.exchange_collector.fetch_tick_data(
                symbol, start_time=start_time
            )
            if tick_data:
                # 保存到数据库
                await self.db_manager.save_tick_data(tick_data)

                # 如果启用了Redis，也保存到Redis
                if self.redis_manager:
                    await self.redis_manager.save_tick_data(tick_data)

            # 获取各时间周期的K线数据
            for timeframe in timeframes:
                logger.info(f"获取 {symbol} 的初始 {timeframe} K线数据")
                kline_data = await self.exchange_collector.fetch_kline_data(
                    symbol, timeframe
                )
                if kline_data:
                    # 保存到数据库
                    await self.db_manager.save_kline_data(kline_data)

                    # 如果启用了Redis，也保存到Redis
                    if self.redis_manager:
                        await self.redis_manager.save_kline_data(kline_data)
        except Exception as e:
            logger.error(f"获取 {symbol} 的初始数据失败: {str(e)}")

    async def start(self):
        """
        启动GCG_Quant系统

        学习点：
        - 状态管理：使用标志和事件控制系统状态
        - 信号处理：捕获操作系统信号实现优雅关闭
        - 异步任务调度：创建后台任务并等待完成或取消
        """
        if self.is_running:
            logger.warning("GCG_Quant已经在运行中")
            return

        # 初始化组件
        initialized = await self.initialize()
        if not initialized:
            logger.error("GCG_Quant初始化失败，无法启动")
            return

        # 设置运行状态
        self.is_running = True
        self.shutdown_event.clear()

        # 注册信号处理器，用于优雅关闭
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        logger.info("GCG_Quant启动成功")

        # 等待关闭信号
        await self.shutdown_event.wait()

    async def stop(self):
        """
        停止GCG_Quant系统

        学习点：
        - 优雅关闭：按照依赖的逆序关闭各组件
        - 事件通知：通过事件通知主循环系统已关闭
        - 错误隔离：单个组件关闭失败不影响其他组件
        """
        if not self.is_running:
            return

        logger.info("开始关闭GCG_Quant")

        # 标记系统为非运行状态
        self.is_running = False

        # 关闭数据采集器
        if self.exchange_collector:
            logger.info("关闭交易所数据采集器")
            try:
                await self.exchange_collector.disconnect()
            except Exception as e:
                logger.error(f"关闭交易所数据采集器失败: {str(e)}")

        if self.file_importer:
            logger.info("关闭文件数据导入器")
            try:
                await self.file_importer.disconnect()
            except Exception as e:
                logger.error(f"关闭文件数据导入器失败: {str(e)}")

        # 关闭Redis管理器（如果启用）
        if self.redis_manager:
            logger.info("关闭Redis管理器")
            try:
                await self.redis_manager.disconnect()
            except Exception as e:
                logger.error(f"关闭Redis管理器失败: {str(e)}")

        # 关闭数据库管理器
        if self.db_manager:
            logger.info("关闭数据库管理器")
            try:
                await self.db_manager.disconnect()
            except Exception as e:
                logger.error(f"关闭数据库管理器失败: {str(e)}")

        logger.info("GCG_Quant已关闭")

        # 通知主循环系统已关闭
        self.shutdown_event.set()


async def amain():
    """
    异步主函数

    学习点：
    - 命令行参数处理：使用argparse解析命令行参数
    - 日志系统初始化：在主函数开始时设置日志
    - 异常处理：捕获顶层异常确保程序正常退出
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="GCG_Quant量化交易系统")
    parser.add_argument("-c", "--config", help="配置文件路径")
    parser.add_argument(
        "-l",
        "--log-level",
        help="日志级别",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    # 设置日志级别
    log_level = getattr(logging, args.log_level)
    setup_logger(level=log_level)

    try:
        # 创建并启动GCG_Quant
        app = GCGQuant(args.config)
        await app.start()
    except Exception as e:
        logger.error(f"GCG_Quant运行时发生错误: {str(e)}", exc_info=True)
        return 1

    return 0


def main():
    """
    同步主函数入口

    学习点：
    - 入口函数设计：使用辅助函数封装异步逻辑
    - 退出码处理：返回进程退出码，便于脚本集成
    """
    # 在Windows上要加这一行
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 运行异步主函数
    exit_code = asyncio.run(amain())

    # 返回退出码
    return exit_code


if __name__ == "__main__":
    # 调用主函数并传递退出码
    exit(main())
