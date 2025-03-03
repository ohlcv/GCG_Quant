"""
策略引擎接口定义模块

本模块定义了策略引擎中的核心接口，包括策略接口(IStrategy)和上下文接口(IContext)。
所有具体策略必须实现IStrategy接口，以确保策略引擎能够统一管理和调用。

优化目标:
1. 支持多品种数据处理（与 chan.py 和 Backtrader 兼容）
2. 提供标准化的可视化数据接口（适配 Lightweight Charts）
3. 增强上下文服务访问（支持数据引擎、日志等）
4. 保持同步/异步灵活性（支持实时交易和回测）

设计原则:
1. 接口设计遵循单一职责原则
2. 使用抽象基类(ABC)强制实现必要方法
3. 提供类型提示增强代码可读性和IDE支持
4. 支持异步操作，为高性能处理做准备
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
import asyncio


class IContext:
    """
    策略上下文接口，提供策略运行时所需的环境和服务

    功能:
    1. 访问配置信息（如时间周期、交易对）
    2. 获取依赖服务（如数据引擎、日志）
    3. 存储和共享策略状态（如持仓、指标）

    学习点:
    - **依赖注入**: 通过服务注册和获取，实现松耦合设计，便于单元测试和模块替换。
    - **状态管理**: 使用字典存储运行时状态，支持动态扩展，避免硬编码。
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化策略上下文

        Args:
            config: 配置字典，包含策略所需的各项配置（如交易品种、时间周期）
        """
        self.config = config or {}
        self.state = {}  # 存储策略运行状态
        self.services = {}  # 存储依赖服务（如数据引擎、日志）

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键名
            default: 默认值，如果键不存在则返回此值

        Returns:
            配置值

        学习点:
        - **字典操作**: 使用 get() 方法避免 KeyError，提供默认值支持健壮性。
        """
        return self.config.get(key, default)

    def get_state(self, key: str, default: Any = None) -> Any:
        """
        获取状态

        Args:
            key: 状态键名
            default: 默认值，如果键不存在则返回此值

        Returns:
            状态值
        """
        return self.state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        """
        设置状态

        Args:
            key: 状态键名
            value: 状态值
        """
        self.state[key] = value

    def register_service(self, name: str, service: Any) -> None:
        """
        注册服务

        Args:
            name: 服务名称（如 "data_engine", "logger"）
            service: 服务实例

        学习点:
        - **服务注册**: 动态添加服务，便于扩展（如添加 TA-Lib 计算服务）。
        """
        self.services[name] = service

    def get_service(self, name: str) -> Any:
        """
        获取服务

        Args:
            name: 服务名称

        Returns:
            服务实例

        Raises:
            KeyError: 如果服务不存在

        学习点:
        - **异常处理**: 抛出明确异常，帮助调试和确保服务可用性。
        """
        if name not in self.services:
            raise KeyError(f"Service '{name}' not found in context")
        return self.services[name]

    def get_data_engine(self) -> Any:
        """
        获取数据引擎服务（优化新增）

        Returns:
            数据引擎实例，用于访问 K 线或 Tick 数据

        学习点:
        - **专用接口**: 为高频使用的数据引擎提供便捷访问，避免重复调用 get_service。
        """
        return self.get_service("data_engine")


class IStrategy(ABC):
    """
    策略接口，所有具体策略必须实现此接口

    功能:
    1. 定义策略生命周期（初始化、启动、停止）
    2. 处理多品种市场数据（K 线、Tick、成交）
    3. 生成交易信号和可视化数据

    学习点:
    - **抽象基类 (ABC)**: 使用 @abstractmethod 强制子类实现核心方法，确保一致性。
    - **异步编程**: 通过 async 方法支持高性能实时操作，与 WebSocket 等技术兼容。
    """

    def __init__(self, name: str, symbols: List[str] = None):
        """
        初始化策略

        Args:
            name: 策略名称（唯一标识）
            symbols: 交易品种列表（支持多品种，如 ["BTC/USDT", "ETH/USDT"]）

        学习点:
        - **初始化灵活性**: symbols 参数支持多品种，适配 chan.py 和 Backtrader 的多数据流。
        """
        self.name = name
        self.symbols = symbols or []
        self.context: Optional[IContext] = None
        self.is_running = False

    @abstractmethod
    def initialize(self, context: IContext) -> None:
        """
        策略初始化方法，在策略开始运行前调用

        Args:
            context: 策略上下文，包含配置信息和依赖服务

        学习点:
        - **生命周期管理**: 初始化时设置上下文，为策略运行提供环境支持。
        """
        self.context = context

    @abstractmethod
    def on_bar(self, bar_data: Dict[str, Dict[str, Any]]) -> None:
        """
        处理K线数据（优化为多品种支持）

        当新的K线数据到达时调用此方法，用于更新策略状态和生成信号。

        Args:
            bar_data: 多品种K线数据，格式为 {symbol: {timestamp, open, high, low, close, volume}}

        学习点:
        - **多键字典**: 使用嵌套字典支持多品种数据处理，增强灵活性。
        - **数据驱动**: 通过事件驱动更新策略状态，适配实时和回测场景。
        """
        pass

    def on_tick(self, tick_data: Dict[str, Dict[str, Any]]) -> None:
        """
        处理Tick数据（优化为多品种支持）

        当新的Tick数据到达时调用此方法，用于实时更新策略状态。
        默认实现为空，子类可根据需要重写。

        Args:
            tick_data: 多品种Tick数据，格式为 {symbol: {timestamp, price, volume}}

        学习点:
        - **可选实现**: 提供默认空实现，支持按需扩展（如高频交易）。
        """
        pass

    def on_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        处理成交数据

        当有新的成交发生时调用此方法，用于更新持仓状态。
        默认实现为空，子类可根据需要重写。

        Args:
            trade_data: 成交数据，包含 order_id, symbol, direction, price, volume 等字段

        学习点:
        - **事件处理**: 通过事件更新持仓，保持状态一致性。
        """
        pass

    def on_order(self, order_data: Dict[str, Any]) -> None:
        """
        处理订单数据

        当订单状态发生变化时调用此方法，用于跟踪订单执行情况。
        默认实现为空，子类可根据需要重写。

        Args:
            order_data: 订单数据，包含 order_id, symbol, status 等字段
        """
        pass

    @abstractmethod
    def generate_signals(self) -> List[Dict[str, Any]]:
        """
        生成交易信号

        根据当前策略状态，生成交易信号列表。

        Returns:
            信号列表，每个信号为字典，包含 symbol, direction, price 等字段

        学习点:
        - **信号格式**: 返回标准化的信号列表，便于 StrategyManager 处理和 Backtrader 使用。
        """
        pass

    def on_start(self) -> None:
        """
        策略启动时调用

        可用于初始化运行时资源、启动辅助线程等。
        默认实现仅设置运行标志，子类可根据需要重写。

        学习点:
        - **资源管理**: 在启动时分配资源，确保运行时可用。
        """
        self.is_running = True

    def on_stop(self) -> None:
        """
        策略停止时调用

        可用于释放资源、保存状态等清理工作。
        默认实现仅清除运行标志，子类可根据需要重写。

        学习点:
        - **清理工作**: 确保资源释放，避免内存泄漏。
        """
        self.is_running = False

    async def on_bar_async(self, bar_data: Dict[str, Dict[str, Any]]) -> None:
        """
        异步处理K线数据

        异步版本的 on_bar，用于支持异步操作（如 WebSocket 实时推送）。
        默认实现调用同步版本，子类可重写。

        Args:
            bar_data: 多品种K线数据

        学习点:
        - **异步支持**: 使用 asyncio 支持高并发，确保实时性。
        """
        self.on_bar(bar_data)

    async def generate_signals_async(self) -> List[Dict[str, Any]]:
        """
        异步生成交易信号

        异步版本的 generate_signals，用于支持异步操作。
        默认实现调用同步版本，子类可重写。

        Returns:
            信号列表

        学习点:
        - **异步编程**: 通过 async/await 提高性能，适配高频场景。
        """
        return self.generate_signals()

    def get_visualization_data(self) -> Dict[str, Any]:
        """
        获取用于可视化的数据（优化为 Lightweight Charts 兼容格式）

        返回策略分析结果的可视化数据，如 K 线、笔、线段等。
        默认实现返回空字典，子类根据需要重写。

        Returns:
            可视化数据字典，格式为：
            {
                "klines": [{"time": int, "open": float, "high": float, "low": float, "close": float}],
                "bi": [{"start": {"time": int, "price": float}, "end": {"time": int, "price": float}}],
                "seg": [{"start": {"time": int, "price": float}, "end": {"time": int, "price": float}}],
                "zs": [{"from": {"time": int, "price": float}, "to": {"time": int, "price": float}}],
                "bs": [{"time": int, "position": str, "color": str, "shape": str}]
            }

        学习点:
        - **数据标准化**: 定义与 Lightweight Charts 兼容的格式，确保前端渲染无缝对接。
        - **扩展性**: 返回字典支持动态添加指标，便于后续扩展（如 TA-Lib RSI）。
        """
        return {"klines": [], "bi": [], "seg": [], "zs": [], "bs": []}

    def get_backtest_data(self) -> Dict[str, Any]:
        """
        获取 Backtrader 可用的回测数据（新增）

        返回适合 Backtrader 的数据格式，供回测使用。
        默认实现返回空字典，子类可重写。

        Returns:
            回测数据字典，格式为：
            {
                "symbol": str,
                "data": [{"datetime": datetime, "open": float, "high": float, "low": float, "close": float, "volume": float}]
            }

        学习点:
        - **Backtrader 兼容性**: 提供标准数据格式，便于与 Backtrader 的 PandasData 对接。
        """
        return {"symbol": "", "data": []}
