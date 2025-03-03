"""
策略管理器模块

本模块实现策略管理器(StrategyManager)，负责管理多个策略的注册、初始化、运行和协调。
策略管理器是策略引擎的核心组件，处理数据分发、信号收集和策略同步等功能。

优化目标:
1. 支持多品种数据分发，确保与 chan.py 和 Backtrader 兼容
2. 增强异步性能，支持实时交易和 WebSocket 推送
3. 提供可视化数据聚合功能，适配 Lightweight Charts
4. 提升错误处理和性能监控能力

设计原则:
1. 模块化设计，职责清晰
2. 支持同步和异步操作
3. 提供灵活的配置选项
4. 实现高效的数据分发机制
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable, Type
from threading import RLock
import time
import uuid

from .interface import IStrategy, IContext
from .signal import Signal

# 创建日志记录器
logger = logging.getLogger("StrategyManager")
logging.basicConfig(level=logging.INFO)


class StrategyManager:
    """
    策略管理器，负责管理多个策略的注册、初始化、运行和协调

    功能:
    1. 策略的注册、初始化和生命周期管理
    2. 多品种数据的分发与处理（K 线、Tick）
    3. 交易信号的收集与筛选
    4. 可视化数据的聚合与推送

    学习点:
    - **线程安全**: 使用 RLock 确保多线程环境下数据一致性。
    - **异步编程**: 通过 asyncio 支持高并发，提升实时性。
    - **日志记录**: 使用 logging 模块跟踪运行状态，便于调试。
    """

    def __init__(self, data_engine=None, global_config: Dict[str, Any] = None):
        """
        初始化策略管理器

        Args:
            data_engine: 数据引擎实例，提供 K 线和 Tick 数据，可选
            global_config: 全局配置，包含通用设置，可选

        学习点:
        - **依赖注入**: 通过参数传入 data_engine，实现松耦合设计。
        - **字典初始化**: 使用 or 提供默认空字典，避免 None 检查。
        """
        self.data_engine = data_engine
        self.global_config = global_config or {}

        # 策略存储
        self.strategies: Dict[str, IStrategy] = {}  # 策略名称 -> 策略实例
        self.strategy_configs: Dict[str, Dict[str, Any]] = {}  # 策略名称 -> 配置
        self.strategy_contexts: Dict[str, IContext] = {}  # 策略名称 -> 上下文

        # 数据追踪
        self.symbols: set = set()  # 所有策略关注的交易品种
        self.symbol_strategies: Dict[str, List[str]] = {}  # 品种 -> 策略名称列表
        self.last_bar_time: Dict[str, datetime] = {}  # 品种 -> 最新K线时间

        # 状态变量
        self.is_running = False
        self.is_async_mode = False
        self.run_id = str(uuid.uuid4())  # 唯一运行标识

        # 信号处理
        self.signal_filters: List[Callable[[Signal], bool]] = []  # 信号过滤器
        self.signal_handlers: List[Callable[[Signal], None]] = []  # 信号处理器

        # 线程安全锁
        self._lock = RLock()

        # 性能统计
        self.perf_stats = {
            "process_count": 0,
            "total_process_time": 0.0,
            "last_process_time": 0.0,
            "signal_count": 0,
        }

        logger.info(f"Strategy Manager initialized with run_id: {self.run_id}")

    def register_strategy(
        self, name: str, strategy_class: Type[IStrategy], config: Dict[str, Any] = None
    ) -> bool:
        """
        注册策略

        Args:
            name: 策略名称，必须唯一
            strategy_class: 策略类，必须是 IStrategy 的子类
            config: 策略配置，可选

        Returns:
            bool: 注册是否成功

        Raises:
            ValueError: 如果策略名称已存在或类无效

        学习点:
        - **类型检查**: 使用 issubclass 确保输入有效，提升代码健壮性。
        - **线程安全**: 使用 with self._lock 保护共享资源。
        """
        with self._lock:
            if name in self.strategies:
                logger.error(f"Strategy '{name}' already registered")
                raise ValueError(f"Strategy '{name}' already registered")

            if not issubclass(strategy_class, IStrategy):
                logger.error(f"Strategy class must be a subclass of IStrategy")
                raise ValueError(f"Strategy class must be a subclass of IStrategy")

            config = config or {}
            symbols = config.get("symbols", [])
            if not symbols:
                logger.warning(f"Strategy '{name}' has no symbols configured")

            try:
                strategy = strategy_class(name, symbols)
                self.strategies[name] = strategy
                self.strategy_configs[name] = config

                for symbol in symbols:
                    self.symbols.add(symbol)
                    self.symbol_strategies.setdefault(symbol, []).append(name)

                logger.info(f"Strategy '{name}' registered with symbols: {symbols}")
                return True
            except Exception as e:
                logger.error(f"Failed to register strategy '{name}': {str(e)}")
                return False

    def initialize_strategies(self) -> bool:
        """
        初始化所有已注册的策略

        Returns:
            bool: 是否全部初始化成功

        学习点:
        - **服务注册**: 在上下文中注入 data_engine 和自身，便于策略访问。
        - **异常处理**: 捕获并记录错误，确保部分失败不影响整体。
        """
        with self._lock:
            success = True
            for name, strategy in self.strategies.items():
                try:
                    context = IContext(self.strategy_configs.get(name, {}))
                    if self.data_engine:
                        context.register_service("data_engine", self.data_engine)
                    context.register_service("strategy_manager", self)
                    strategy.initialize(context)
                    self.strategy_contexts[name] = context
                    logger.info(f"Strategy '{name}' initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize strategy '{name}': {str(e)}")
                    success = False
            return success

    def start(self) -> bool:
        """
        启动所有策略

        Returns:
            bool: 是否启动成功

        学习点:
        - **状态管理**: 检查 is_running 避免重复启动，提升健壮性。
        """
        with self._lock:
            if self.is_running:
                logger.warning("Strategy Manager is already running")
                return True

            try:
                for name, strategy in self.strategies.items():
                    strategy.on_start()
                self.is_running = True
                logger.info(f"Strategy Manager started with run_id: {self.run_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to start Strategy Manager: {str(e)}")
                return False

    def stop(self) -> bool:
        """
        停止所有策略

        Returns:
            bool: 是否停止成功
        """
        with self._lock:
            if not self.is_running:
                logger.warning("Strategy Manager is not running")
                return True

            try:
                for name, strategy in self.strategies.items():
                    strategy.on_stop()
                self.is_running = False
                logger.info("Strategy Manager stopped")
                return True
            except Exception as e:
                logger.error(f"Failed to stop Strategy Manager: {str(e)}")
                return False

    def process_bar(self, bar_data: Dict[str, Dict[str, Any]]) -> List[Signal]:
        """
        处理K线数据（同步版本）

        Args:
            bar_data: 多品种K线数据，{symbol: {timestamp, open, high, low, close, volume}}

        Returns:
            List[Signal]: 生成的信号列表

        学习点:
        - **数据分发**: 根据策略关注的 symbols 筛选数据，提升效率。
        - **性能监控**: 记录处理时间和信号数量，便于优化。
        """
        if not self.is_running:
            logger.warning("Strategy Manager is not running, can't process data")
            return []

        start_time = time.time()
        signal_list = []

        with self._lock:
            # 更新最新K线时间
            for symbol, data in bar_data.items():
                if "timestamp" in data:
                    timestamp = data["timestamp"]
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp)
                    self.last_bar_time[symbol] = timestamp

            # 处理每个策略
            for name, strategy in self.strategies.items():
                try:
                    strategy_symbols = self.strategy_configs.get(name, {}).get(
                        "symbols", []
                    )
                    relevant_data = {
                        s: bar_data[s] for s in strategy_symbols if s in bar_data
                    }
                    if not relevant_data:
                        continue

                    strategy.on_bar(relevant_data)
                    signals = strategy.generate_signals()
                    signal_list.extend(self._filter_and_handle_signals(signals))
                except Exception as e:
                    logger.error(
                        f"Error processing bar data in strategy '{name}': {str(e)}"
                    )

        # 更新性能统计
        self._update_performance_stats(start_time, len(signal_list))
        logger.debug(
            f"Processed bar data in {self.perf_stats['last_process_time']:.6f}s, generated {len(signal_list)} signals"
        )
        return signal_list

    async def process_bar_async(
        self, bar_data: Dict[str, Dict[str, Any]]
    ) -> List[Signal]:
        """
        处理K线数据（异步版本）

        Args:
            bar_data: 多品种K线数据

        Returns:
            List[Signal]: 生成的信号列表

        学习点:
        - **异步并发**: 使用 asyncio.gather 并行处理策略，提升性能。
        - **异常隔离**: return_exceptions=True 确保部分失败不中断整体。
        """
        if not self.is_running:
            logger.warning("Strategy Manager is not running, can't process data")
            return []

        self.is_async_mode = True
        start_time = time.time()
        signal_list = []

        # 更新最新K线时间
        for symbol, data in bar_data.items():
            if "timestamp" in data:
                timestamp = data["timestamp"]
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                self.last_bar_time[symbol] = timestamp

        # 异步处理策略
        tasks = []
        for name, strategy in self.strategies.items():
            strategy_symbols = self.strategy_configs.get(name, {}).get("symbols", [])
            relevant_data = {s: bar_data[s] for s in strategy_symbols if s in bar_data}
            if relevant_data:
                tasks.append(
                    self._process_strategy_bar_async(name, strategy, relevant_data)
                )

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Async bar processing error: {str(result)}")
                elif isinstance(result, list):
                    signal_list.extend(result)

        # 更新性能统计
        self._update_performance_stats(start_time, len(signal_list))
        logger.debug(
            f"Processed bar data asynchronously in {self.perf_stats['last_process_time']:.6f}s, generated {len(signal_list)} signals"
        )
        return signal_list

    async def _process_strategy_bar_async(
        self, name: str, strategy: IStrategy, bar_data: Dict[str, Dict[str, Any]]
    ) -> List[Signal]:
        """
        异步处理单个策略的K线数据

        Args:
            name: 策略名称
            strategy: 策略实例
            bar_data: K线数据

        Returns:
            List[Signal]: 生成的信号列表

        学习点:
        - **异步调用**: 使用 await 调用策略的异步方法，提升并发效率。
        """
        try:
            await strategy.on_bar_async(bar_data)
            signals = await strategy.generate_signals_async()
            return self._filter_and_handle_signals(signals)
        except Exception as e:
            logger.error(
                f"Error processing bar data asynchronously in strategy '{name}': {str(e)}"
            )
            return []

    def process_tick(self, tick_data: Dict[str, Dict[str, Any]]) -> List[Signal]:
        """
        处理Tick数据（同步版本）

        Args:
            tick_data: 多品种Tick数据，{symbol: {timestamp, price, volume}}

        Returns:
            List[Signal]: 生成的信号列表
        """
        if not self.is_running:
            logger.warning("Strategy Manager is not running, can't process data")
            return []

        signal_list = []
        with self._lock:
            for name, strategy in self.strategies.items():
                try:
                    strategy_symbols = self.strategy_configs.get(name, {}).get(
                        "symbols", []
                    )
                    relevant_data = {
                        s: tick_data[s] for s in strategy_symbols if s in tick_data
                    }
                    if not relevant_data:
                        continue

                    strategy.on_tick(relevant_data)
                    signals = strategy.generate_signals()
                    signal_list.extend(self._filter_and_handle_signals(signals))
                except Exception as e:
                    logger.error(
                        f"Error processing tick data in strategy '{name}': {str(e)}"
                    )
        return signal_list

    def collect_signals(self) -> List[Signal]:
        """
        收集所有策略生成的信号（不处理新数据）

        Returns:
            List[Signal]: 当前信号列表

        学习点:
        - **信号聚合**: 从所有策略收集信号，便于统一处理或推送。
        """
        if not self.is_running:
            logger.warning("Strategy Manager is not running, can't collect signals")
            return []

        signal_list = []
        with self._lock:
            for name, strategy in self.strategies.items():
                try:
                    signals = strategy.generate_signals()
                    signal_list.extend(self._filter_and_handle_signals(signals))
                except Exception as e:
                    logger.error(
                        f"Error collecting signals from strategy '{name}': {str(e)}"
                    )
        return signal_list

    def get_visualization_data(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有策略的可视化数据

        Returns:
            Dict[str, Dict[str, Any]]: 策略名称到可视化数据的映射

        学习点:
        - **数据聚合**: 收集并格式化可视化数据，支持实时图表（如 Lightweight Charts）。
        """
        viz_data = {}
        with self._lock:
            for name, strategy in self.strategies.items():
                try:
                    viz_data[name] = strategy.get_visualization_data()
                except Exception as e:
                    logger.error(
                        f"Error getting visualization data from strategy '{name}': {str(e)}"
                    )
                    viz_data[name] = {}
        return viz_data

    def push_visualization_data(self, visualizer) -> None:
        """
        推送可视化数据到图表

        Args:
            visualizer: 可视化模块实例（如 charts/visualizer.py 中的实例）

        学习点:
        - **推送机制**: 将数据主动推送给可视化组件，支持实时更新。
        """
        viz_data = self.get_visualization_data()
        try:
            visualizer.update_chart(viz_data)
            logger.debug("Visualization data pushed successfully")
        except Exception as e:
            logger.error(f"Error pushing visualization data: {str(e)}")

    def add_signal_filter(self, filter_func: Callable[[Signal], bool]) -> None:
        """
        添加信号过滤器

        Args:
            filter_func: 过滤函数，接收 Signal 对象，返回布尔值

        学习点:
        - **函数式编程**: 使用 Callable 实现灵活的信号过滤逻辑。
        """
        with self._lock:
            self.signal_filters.append(filter_func)
            logger.debug(f"Added signal filter: {filter_func.__name__}")

    def add_signal_handler(self, handler_func: Callable[[Signal], None]) -> None:
        """
        添加信号处理器

        Args:
            handler_func: 处理函数，接收 Signal 对象

        学习点:
        - **回调机制**: 支持信号的自定义处理（如推送至交易引擎）。
        """
        with self._lock:
            self.signal_handlers.append(handler_func)
            logger.debug(f"Added signal handler: {handler_func.__name__}")

    def _filter_and_handle_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        过滤并处理信号（内部方法）

        Args:
            signals: 原始信号列表

        Returns:
            List[Signal]: 过滤后的信号列表

        学习点:
        - **列表推导式**: 高效过滤信号，提升性能。
        - **异常隔离**: 在处理信号时捕获异常，避免中断。
        """
        filtered_signals = []
        for signal in signals or []:
            if all(filter_func(signal) for filter_func in self.signal_filters):
                filtered_signals.append(signal)
                for handler in self.signal_handlers:
                    try:
                        handler(signal)
                    except Exception as e:
                        logger.error(f"Error in signal handler: {str(e)}")
        return filtered_signals

    def _update_performance_stats(self, start_time: float, signal_count: int) -> None:
        """
        更新性能统计信息（内部方法）

        Args:
            start_time: 处理开始时间
            signal_count: 处理生成的信号数量

        学习点:
        - **性能监控**: 记录处理时间和信号数量，便于分析瓶颈。
        """
        with self._lock:
            process_time = time.time() - start_time
            self.perf_stats["process_count"] += 1
            self.perf_stats["total_process_time"] += process_time
            self.perf_stats["last_process_time"] = process_time
            self.perf_stats["signal_count"] += signal_count

    def get_strategy(self, name: str) -> Optional[IStrategy]:
        """
        获取策略实例

        Args:
            name: 策略名称

        Returns:
            Optional[IStrategy]: 策略实例或 None
        """
        with self._lock:
            return self.strategies.get(name)

    def get_all_symbols(self) -> set:
        """
        获取所有策略关注的交易品种

        Returns:
            set: 交易品种集合
        """
        with self._lock:
            return self.symbols.copy()

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息

        Returns:
            Dict[str, Any]: 性能统计数据

        学习点:
        - **统计分析**: 提供运行时性能数据，便于优化和调试。
        """
        with self._lock:
            stats = self.perf_stats.copy()
            stats["avg_process_time"] = (
                stats["total_process_time"] / stats["process_count"]
                if stats["process_count"] > 0
                else 0
            )
            stats["strategy_count"] = len(self.strategies)
            stats["symbol_count"] = len(self.symbols)
            return stats

    def is_data_ready(self, symbol: str, current_time: datetime) -> bool:
        """
        检查数据是否准备就绪

        Args:
            symbol: 交易品种
            current_time: 当前时间

        Returns:
            bool: 数据是否及时

        学习点:
        - **时间差计算**: 使用 datetime 比较数据新鲜度，保证实时性。
        """
        with self._lock:
            if symbol not in self.last_bar_time:
                return False
            last_time = self.last_bar_time[symbol]
            time_diff = (current_time - last_time).total_seconds()
            return time_diff <= 300  # 5分钟阈值，可配置化
