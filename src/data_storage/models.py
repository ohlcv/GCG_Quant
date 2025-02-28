# models.py - 数据模型定义

"""
文件说明：
    这个文件定义了GCG_Quant系统中的核心数据模型，包括Tick数据和K线数据的结构。
    这些模型用于在不同组件之间传递数据，并提供数据转换功能。

学习目标：
    1. 掌握Python数据类(dataclass)的使用，简化数据模型定义
    2. 理解对象和字典之间的转换方法
    3. 学习类方法(classmethod)和工厂模式的应用
    4. 熟悉时间数据处理和格式转换
"""

from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class TickData:
    """
    Tick数据模型，表示单个交易的信息
    
    学习点：
    - @dataclass装饰器自动生成__init__、__repr__等方法，简化类定义
    - 使用类型注解定义字段，提高代码可读性和IDE支持
    """
    symbol: str  # 交易品种符号
    timestamp: int  # 时间戳（毫秒）
    datetime: datetime  # 日期时间
    price: float  # 价格
    amount: float  # 数量/成交量
    side: str  # 方向，'buy'或'sell'
    source: str  # 数据源
    trade_id: Optional[str] = None  # 交易ID，可能为空
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TickData':
        """
        从字典创建Tick数据对象
        
        Args:
            data: 包含Tick数据的字典
            
        Returns:
            TickData: 新创建的Tick数据对象
            
        学习点：
        - @classmethod装饰器定义类方法，第一个参数是类本身而不是实例
        - 工厂模式：提供创建对象的接口，由子类决定实例化哪个类
        - 'TickData'是返回类型的字符串表示，用于处理前向引用
        """
        if 'symbol' not in data or not data['symbol']:
            raise ValueError("symbol 是必填字段 / symbol is required")
        if 'timestamp' not in data:
            raise ValueError("timestamp 是必填字段 / timestamp is required")
        # 确保 datetime 字段是 datetime 对象
        if isinstance(data.get('datetime'), str):
            data['datetime'] = datetime.fromisoformat(data['datetime'])
        
        return cls(
            symbol=data.get('symbol', ''),
            timestamp=data.get('timestamp', 0),
            datetime=data.get('datetime', datetime.now()),
            price=float(data.get('price', 0.0)),
            amount=float(data.get('amount', 0.0)),
            side=data.get('side', ''),
            source=data.get('source', ''),
            trade_id=data.get('id', None)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            Dict[str, Any]: 包含Tick数据的字典
            
        学习点：
        - 提供对象到字典的转换，便于JSON序列化和数据传输
        - 使用isoformat()将datetime转换为ISO 8601格式的字符串
        """
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'datetime': self.datetime.isoformat(),
            'price': self.price,
            'amount': self.amount,
            'side': self.side,
            'source': self.source,
            'trade_id': self.trade_id
        }

@dataclass
class KlineData:
    """K线数据模型，表示一个时间周期的市场数据汇总"""
    symbol: str  # 交易品种符号
    timestamp: int  # 时间戳（毫秒）
    datetime: datetime  # 日期时间
    timeframe: str  # 时间周期，如'1m', '5m', '1h', '1d'等
    open: float  # 开盘价
    high: float  # 最高价
    low: float  # 最低价
    close: float  # 收盘价
    volume: float  # 成交量
    source: str  # 数据源
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KlineData':
        """从字典创建K线数据对象"""
        # 确保datetime字段是datetime对象
        if isinstance(data.get('datetime'), str):
            data['datetime'] = datetime.fromisoformat(data['datetime'])
        
        return cls(
            symbol=data.get('symbol', ''),
            timestamp=data.get('timestamp', 0),
            datetime=data.get('datetime', datetime.now()),
            timeframe=data.get('timeframe', '1m'),
            open=float(data.get('open', 0.0)),
            high=float(data.get('high', 0.0)),
            low=float(data.get('low', 0.0)),
            close=float(data.get('close', 0.0)),
            volume=float(data.get('volume', 0.0)),
            source=data.get('source', '')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'datetime': self.datetime.isoformat(),
            'timeframe': self.timeframe,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'source': self.source
        }