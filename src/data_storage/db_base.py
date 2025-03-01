# db_base.py - 数据库抽象接口

"""
文件说明：
    这个文件定义了数据库操作的抽象接口(DBManager)，用于支持不同类型数据库的切换。
    采用适配器模式和工厂模式设计，实现了配置驱动的数据库选择机制。

学习目标：
    1. 理解设计模式中的适配器模式和工厂模式的应用
    2. 学习如何设计与具体数据库实现解耦的抽象接口
    3. 掌握异步数据库操作的基本模式
    4. 了解配置驱动的组件初始化方法
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Type
from datetime import datetime
import importlib
import logging

# 导入数据模型
from ..data_storage.models import TickData, KlineData

# 创建日志记录器
logger = logging.getLogger("DBManager")


class DBManager(ABC):
    """
    数据库管理器抽象基类，定义了所有数据库管理器必须实现的接口

    学习点：
    - 适配器模式：为不同数据库提供统一接口
    - 抽象基类定义接口规范，子类必须实现这些方法
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据库管理器

        Args:
            config: 数据库配置参数
        """
        self.config = config

    @abstractmethod
    async def connect(self) -> bool:
        """
        连接到数据库

        Returns:
            bool: 连接是否成功
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        断开与数据库的连接

        Returns:
            bool: 断开连接是否成功
        """
        pass

    @abstractmethod
    async def init_tables(self) -> bool:
        """
        初始化数据表

        Returns:
            bool: 初始化是否成功
        """
        pass

    @abstractmethod
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
        - Union类型提示表示参数可以是TickData或List[TickData]
        - 批处理参数用于控制每批次处理的数据量，避免内存压力
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        """
        pass

    @abstractmethod
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
        pass


def create_db_manager(config: Dict[str, Any]) -> DBManager:
    """
    创建数据库管理器实例

    Args:
        config: 数据存储配置

    Returns:
        DBManager: 数据库管理器实例

    Raises:
        ValueError: 如果不支持指定的数据库类型

    学习点：
    - 工厂模式：根据配置创建不同的数据库管理器实例
    - 动态导入：使用importlib根据配置动态加载相应的模块
    - 配置驱动：通过配置项控制组件的具体实现
    """
    # 学习点：工厂模式根据配置创建实例，使用示例：
    # config = {"db_type": "sqlite", "sqlite": {"db_file": "data.db"}}
    # db = create_db_manager(config)
    db_type = config.get("db_type", "sqlite").lower()
    logger.info(f"创建数据库管理器，类型: {db_type}")
    if db_type not in ["sqlite", "timescaledb"]:
        raise ValueError(f"不支持的数据库类型: {db_type}")
    # 学习点：验证配置完整性，避免下游模块因参数缺失失败
    if db_type == "sqlite" and "db_file" not in config:
        raise ValueError("SQLite 配置缺少 db_file 参数")
    if db_type == "timescaledb" and "host" not in config:
        raise ValueError("TimescaleDB 配置缺少 host 参数")
    try:
        if db_type == "sqlite":
            module = importlib.import_module(
                "..data_storage.sqlite_manager", __package__
            )
            return getattr(module, "SQLiteManager")(config["sqlite"])
        elif db_type == "timescaledb":
            module = importlib.import_module(
                "..data_storage.timescale_manager", __package__
            )
            return getattr(module, "TimescaleManager")(config["timescale"])
    except (ImportError, AttributeError) as e:
        logger.error(f"导入数据库管理器失败: {str(e)}")
        raise ValueError(f"创建数据库管理器失败: {str(e)}")
