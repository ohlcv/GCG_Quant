# base_collector.py - 数据采集器基础接口

"""
文件说明：
    这个文件定义了数据采集器的基础抽象接口(BaseCollector)，是所有具体采集器实现的基类。
    它规定了采集器必须实现的基本方法，包括数据获取和连接管理等功能。

学习目标：
    1. 理解抽象基类(ABC)和抽象方法的概念和用法
    2. 掌握Python类型提示(Type Hints)的应用
    3. 学习异步编程(async/await)的基础知识
    4. 了解设计模式中的模板方法模式
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

class BaseCollector(ABC):
    """
    数据采集器基类，定义所有采集器必须实现的接口
    
    学习点：
    - ABC(Abstract Base Class)是Python的抽象基类机制，用于定义接口或抽象类
    - @abstractmethod装饰器标记的方法必须被子类实现，否则实例化子类时会报错
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化采集器
        
        Args:
            config: 配置参数字典
            
        学习点：
        - 构造函数用于初始化对象的属性
        - Dict[str, Any]是类型提示，表示config是一个字典，键为字符串，值为任意类型
        """
        self.config = config
        self.is_running = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        连接到数据源
        
        Returns:
            bool: 连接是否成功
            
        学习点：
        - async关键字定义异步方法，调用时需要使用await
        - 异步方法允许在I/O操作期间（如网络请求）切换执行其他任务，提高效率
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        断开与数据源的连接
        
        Returns:
            bool: 断开连接是否成功
        """
        pass
    
    @abstractmethod
    async def fetch_tick_data(self, symbol: str, start_time: Optional[datetime] = None, 
                             end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        获取Tick数据
        
        Args:
            symbol: 交易品种符号
            start_time: 开始时间，默认为None（获取最新数据）
            end_time: 结束时间，默认为None
            
        Returns:
            List[Dict[str, Any]]: Tick数据列表
            
        学习点：
        - Optional[datetime]表示参数可以是datetime类型或None
        - 默认参数值(=None)使方法调用更灵活
        """
        pass
    
    @abstractmethod
    async def fetch_kline_data(self, symbol: str, timeframe: str, 
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        获取K线数据
        
        Args:
            symbol: 交易品种符号
            timeframe: 时间周期，如'1m', '5m', '1h', '1d'等
            start_time: 开始时间，默认为None（获取最新数据）
            end_time: 结束时间，默认为None
            
        Returns:
            List[Dict[str, Any]]: K线数据列表
        """
        pass
    
    # 学习点：订阅方法在初期版本中简化，符合Grok的建议
    # 这些方法将在第二阶段实现，先预留接口保持设计完整性
    # 订阅方法标记为可选实现，子类可以根据需要覆盖
    
    async def subscribe_tick(self, symbols: List[str], callback) -> bool:
        """
        订阅Tick数据（可选实现）
        
        Args:
            symbols: 交易品种符号列表
            callback: 数据回调函数
            
        Returns:
            bool: 订阅是否成功
            
        学习点：
        - 未使用@abstractmethod，子类可以选择性实现此方法
        - 默认返回False表示不支持订阅功能
        """
        return False
    
    async def subscribe_kline(self, symbols: List[str], timeframe: str, callback) -> bool:
        """
        订阅K线数据（可选实现）
        
        Args:
            symbols: 交易品种符号列表
            timeframe: 时间周期
            callback: 数据回调函数
            
        Returns:
            bool: 订阅是否成功
        """
        return False
    
    async def start(self) -> bool:
        """
        启动采集器
        
        Returns:
            bool: 启动是否成功
            
        学习点：
        - 模板方法模式：定义算法骨架，将一些步骤延迟到子类中实现
        - 此方法依赖子类实现的connect方法，但控制整体流程
        """
        if self.is_running:
            return True
        
        connected = await self.connect()
        if connected:
            self.is_running = True
            return True
        return False
    
    async def stop(self) -> bool:
        """
        停止采集器
        
        Returns:
            bool: 停止是否成功
        """
        if not self.is_running:
            return True
        
        disconnected = await self.disconnect()
        if disconnected:
            self.is_running = False
            return True
        return False