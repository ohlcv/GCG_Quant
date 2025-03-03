"""
交易信号模块

本模块定义了策略引擎中的交易信号类(Signal)，用于表示策略的交易决策。
Signal类设计为通用格式，支持多种交易品种和信号类型，同时提供可扩展的属性支持。

优化目标:
1. 支持多品种交易信号，确保与 chan.py 和 Backtrader 无缝对接
2. 兼容缠论买卖点特性（如类型、分级）
3. 提供序列化和反序列化功能，支持 WebSocket 传输
4. 增强信号状态管理，确保实时交易可靠性
5. 添加便捷方法，简化信号创建

设计原则:
1. 标准化信号格式，确保下游组件能统一处理
2. 使用枚举规范化信号属性，减少错误
3. 支持灵活的扩展属性，便于未来功能添加
4. 提供输入验证和默认值，提升健壮性
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, Union, List
import json


class SignalDirection(Enum):
    """信号方向枚举

    学习点:
    - **枚举 (Enum)**: 使用枚举定义固定选项，避免字符串硬编码，提高代码可读性和类型安全。
    """

    BUY = "buy"  # 买入
    SELL = "sell"  # 卖出
    CLOSE_LONG = "close_long"  # 平多仓
    CLOSE_SHORT = "close_short"  # 平空仓


class SignalType(Enum):
    """信号类型枚举

    学习点:
    - **枚举扩展性**: 定义常见订单类型，支持未来添加新类型（如条件单）。
    """

    MARKET = "market"  # 市价单
    LIMIT = "limit"  # 限价单
    STOP = "stop"  # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单


class SignalStatus(Enum):
    """信号状态枚举（新增）

    学习点:
    - **状态机**: 通过枚举管理信号生命周期，便于跟踪和调试。
    """

    PENDING = "pending"  # 待处理
    PROCESSED = "processed"  # 已处理
    EXPIRED = "expired"  # 已过期
    FAILED = "failed"  # 处理失败


class Signal:
    """
    交易信号类，表示策略的交易决策

    Signal类用于在策略引擎的各个组件之间传递交易决策信息，
    包含交易方向、价格、数量等基本信息，以及缠论买卖点和附加元数据。

    学习点:
    - **面向对象设计**: 通过类封装信号属性和行为，支持复杂逻辑。
    - **类型提示**: 使用 typing 模块提升代码可读性和 IDE 支持。
    """

    def __init__(
        self,
        symbol: str,
        direction: Union[SignalDirection, str],
        price: Optional[float] = None,
        volume: Optional[float] = None,
        signal_type: Union[SignalType, str] = SignalType.MARKET,
        source_strategy: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        signal_id: Optional[str] = None,
        validity: Optional[int] = None,  # 单位：秒
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        bs_point_type: Optional[List[str]] = None,  # 优化：支持多类型缠论买卖点
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化交易信号

        Args:
            symbol: 交易品种符号（如 "BTC/USDT"）
            direction: 交易方向，可以是 SignalDirection 枚举或字符串
            price: 交易价格，None 表示市价
            volume: 交易数量，None 表示使用系统默认
            signal_type: 信号类型，默认市价单
            source_strategy: 信号来源的策略名称
            timestamp: 信号生成时间戳，默认当前时间
            signal_id: 信号唯一标识，默认自动生成 UUID
            validity: 信号有效期（秒），None 表示永久有效
            stop_loss: 止损价格
            take_profit: 止盈价格
            bs_point_type: 缠论买卖点类型（如 ["1", "2"]），支持多类型
            metadata: 附加信息字典（如缠论特征）

        学习点:
        - **参数默认值**: 使用 None 提供灵活性，避免强制输入。
        - **输入验证**: 在初始化时检查关键参数，确保信号有效性。
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")

        # 基本属性
        self.symbol = symbol
        self.direction = self._parse_direction(direction)
        self.price = price if price is not None else None  # 明确 None 表示市价
        self.volume = volume if volume is not None else None
        self.signal_type = self._parse_signal_type(signal_type)
        self.source_strategy = source_strategy
        self.timestamp = timestamp or datetime.utcnow()  # 使用 UTC 时间统一标准
        self.signal_id = signal_id or str(uuid.uuid4())
        self.validity = validity

        # 交易控制属性
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        # 缠论特定属性（优化：支持多类型）
        self.bs_point_type = (
            bs_point_type if bs_point_type is not None else []
        )  # 空列表表示非缠论信号

        # 扩展属性
        self.metadata = metadata or {}

        # 状态管理（优化：使用枚举）
        self.status = SignalStatus.PENDING
        self.process_time: Optional[datetime] = None
        self.process_result: Optional[Any] = None

    def _parse_direction(
        self, direction: Union[SignalDirection, str]
    ) -> SignalDirection:
        """
        解析交易方向

        Args:
            direction: 交易方向，可以是 SignalDirection 枚举或字符串

        Returns:
            SignalDirection 枚举值

        Raises:
            ValueError: 如果 direction 无效

        学习点:
        - **枚举解析**: 提供灵活的输入方式，支持字符串和枚举，增强兼容性。
        - **异常处理**: 明确的错误信息便于调试。
        """
        if isinstance(direction, SignalDirection):
            return direction
        if isinstance(direction, str):
            try:
                return SignalDirection(direction.lower())
            except ValueError:
                pass
        raise ValueError(f"Invalid signal direction: {direction}")

    def _parse_signal_type(self, signal_type: Union[SignalType, str]) -> SignalType:
        """
        解析信号类型

        Args:
            signal_type: 信号类型，可以是 SignalType 枚举或字符串

        Returns:
            SignalType 枚举值

        Raises:
            ValueError: 如果 signal_type 无效

        学习点:
        - **健壮性**: 统一转为小写处理，减少大小写敏感问题。
        """
        if isinstance(signal_type, SignalType):
            return signal_type
        if isinstance(signal_type, str):
            try:
                return SignalType(signal_type.lower())
            except ValueError:
                pass
        raise ValueError(f"Invalid signal type: {signal_type}")

    def is_valid(self) -> bool:
        """
        检查信号是否有效

        Returns:
            bool: 信号是否有效

        学习点:
        - **时间计算**: 使用 datetime.timedelta 计算时间差，支持有效期管理。
        """
        if self.validity is None:
            return self.status != SignalStatus.EXPIRED
        time_diff = (datetime.utcnow() - self.timestamp).total_seconds()
        return time_diff <= self.validity and self.status != SignalStatus.EXPIRED

    def mark_processed(self, result: Any = None, success: bool = True) -> None:
        """
        标记信号为已处理（优化：添加成功/失败状态）

        Args:
            result: 处理结果
            success: 处理是否成功，决定状态为 PROCESSED 或 FAILED

        学习点:
        - **状态转换**: 通过参数动态设置状态，支持失败场景。
        """
        self.process_time = datetime.utcnow()
        self.status = SignalStatus.PROCESSED if success else SignalStatus.FAILED
        self.process_result = result

    def mark_expired(self) -> None:
        """
        标记信号为已过期（新增）

        学习点:
        - **状态封装**: 提供专用方法管理状态，保持一致性。
        """
        self.status = SignalStatus.EXPIRED
        self.process_time = datetime.utcnow()

    def add_metadata(self, key: str, value: Any) -> None:
        """
        添加元数据

        Args:
            key: 元数据键
            value: 元数据值

        学习点:
        - **灵活扩展**: 字典支持动态添加属性，便于存储缠论特征等。
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        获取元数据

        Args:
            key: 元数据键
            default: 默认值

        Returns:
            元数据值
        """
        return self.metadata.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典（优化：支持 JSON 序列化）

        Returns:
            Dict[str, Any]: 信号的字典表示

        学习点:
        - **序列化**: 将对象转为字典，便于通过 WebSocket 或存储传输。
        - **时间格式**: 使用 ISO 格式确保跨系统兼容性。
        """
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "price": self.price,
            "volume": self.volume,
            "signal_type": self.signal_type.value,
            "source_strategy": self.source_strategy,
            "timestamp": self.timestamp.isoformat(),
            "validity": self.validity,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "bs_point_type": self.bs_point_type,
            "status": self.status.value,
            "process_time": (
                self.process_time.isoformat() if self.process_time else None
            ),
            "process_result": self.process_result,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        """
        从字典创建信号

        Args:
            data: 信号的字典表示

        Returns:
            Signal: 信号实例

        学习点:
        - **反序列化**: 从字典重建对象，支持信号恢复。
        - **错误容忍**: 处理时间字段的多种格式，提升健壮性。
        """
        # 处理时间字段
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        process_time = data.get("process_time")
        if isinstance(process_time, str) and process_time:
            process_time = datetime.fromisoformat(process_time)

        # 创建信号实例
        signal = cls(
            symbol=data["symbol"],
            direction=data["direction"],
            price=data.get("price"),
            volume=data.get("volume"),
            signal_type=data.get("signal_type", SignalType.MARKET),
            source_strategy=data.get("source_strategy"),
            timestamp=timestamp,
            signal_id=data.get("signal_id"),
            validity=data.get("validity"),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            bs_point_type=data.get("bs_point_type"),
            metadata=data.get("metadata", {}),
        )

        # 设置状态
        signal.status = SignalStatus(data.get("status", "pending"))
        signal.process_time = process_time
        signal.process_result = data.get("process_result")

        return signal

    def to_json(self) -> str:
        """
        转换为 JSON 字符串（新增）

        Returns:
            str: JSON 格式的信号表示

        学习点:
        - **JSON 序列化**: 使用 json.dumps 转换为字符串，便于网络传输。
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "Signal":
        """
        从 JSON 字符串创建信号（新增）

        Args:
            json_str: JSON 格式的信号字符串

        Returns:
            Signal: 信号实例

        学习点:
        - **JSON 反序列化**: 从字符串重建对象，支持分布式系统。
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """
        字符串表示

        学习点:
        - **字符串格式化**: 使用 f-string 提供简洁的可读输出。
        """
        return (
            f"Signal(id={self.signal_id}, symbol={self.symbol}, "
            f"direction={self.direction.value}, price={self.price}, "
            f"type={self.signal_type.value}, source={self.source_strategy}, "
            f"status={self.status.value})"
        )

    def __repr__(self) -> str:
        """开发者字符串表示"""
        return self.__str__()

    # 便捷创建方法
    @staticmethod
    def create_buy_signal(
        symbol: str,
        price: Optional[float] = None,
        volume: Optional[float] = None,
        bs_point_type: Optional[List[str]] = None,
        source_strategy: Optional[str] = None,
        **kwargs,
    ) -> "Signal":
        """
        创建买入信号的便捷方法

        Args:
            symbol: 交易品种符号
            price: 买入价格
            volume: 买入数量
            bs_point_type: 缠论买点类型
            source_strategy: 信号来源的策略名称
            **kwargs: 其他参数（如 stop_loss, take_profit）

        Returns:
            Signal: 买入信号

        学习点:
        - **静态方法**: 不依赖实例状态，直接创建对象，便于使用。
        - **kwargs**: 支持灵活的参数扩展。
        """
        return Signal(
            symbol=symbol,
            direction=SignalDirection.BUY,
            price=price,
            volume=volume,
            bs_point_type=bs_point_type,
            source_strategy=source_strategy,
            **kwargs,
        )

    @staticmethod
    def create_sell_signal(
        symbol: str,
        price: Optional[float] = None,
        volume: Optional[float] = None,
        bs_point_type: Optional[List[str]] = None,
        source_strategy: Optional[str] = None,
        **kwargs,
    ) -> "Signal":
        """
        创建卖出信号的便捷方法

        Args:
            symbol: 交易品种符号
            price: 卖出价格
            volume: 卖出数量
            bs_point_type: 缠论卖点类型
            source_strategy: 信号来源的策略名称
            **kwargs: 其他参数

        Returns:
            Signal: 卖出信号
        """
        return Signal(
            symbol=symbol,
            direction=SignalDirection.SELL,
            price=price,
            volume=volume,
            bs_point_type=bs_point_type,
            source_strategy=source_strategy,
            **kwargs,
        )

    @staticmethod
    def create_close_signal(
        symbol: str,
        is_long: bool = True,
        price: Optional[float] = None,
        volume: Optional[float] = None,
        source_strategy: Optional[str] = None,
        **kwargs,
    ) -> "Signal":
        """
        创建平仓信号的便捷方法

        Args:
            symbol: 交易品种符号
            is_long: 是否平多仓，False 表示平空仓
            price: 平仓价格
            volume: 平仓数量
            source_strategy: 信号来源的策略名称
            **kwargs: 其他参数

        Returns:
            Signal: 平仓信号

        学习点:
        - **条件逻辑**: 根据 is_long 参数动态选择方向，增强灵活性。
        """
        direction = (
            SignalDirection.CLOSE_LONG if is_long else SignalDirection.CLOSE_SHORT
        )
        return Signal(
            symbol=symbol,
            direction=direction,
            price=price,
            volume=volume,
            source_strategy=source_strategy,
            **kwargs,
        )
