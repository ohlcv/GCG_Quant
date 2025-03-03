"""
chan_strategy.py - 基于缠论的交易策略

本模块实现了基于chan.py的交易策略，将缠论分析结果转换为交易信号，
支持多品种、多级别联立分析，提供实时可视化数据和回测支持。

学习目标:
1. 掌握如何将chan.py集成到策略系统中，支持多品种数据处理
2. 理解多级别联立分析的原理和实现方法
3. 学习如何将缠论买卖点转换为交易信号
4. 掌握内存优化和缓存设计在量化系统中的应用
5. 理解信号聚合和冲突解决的策略
"""

import os
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from enum import Enum
import logging
from pathlib import Path
import time

# 导入策略接口和信号类
from .interface import IStrategy, IContext
from .signal import Signal, SignalDirection

# 导入chan.py相关类
try:
    from Chan import CChan
    from ChanConfig import CChanConfig
    from Common.CEnum import AUTYPE, KL_TYPE, DATA_SRC
    from KLine.KLine_Unit import CKLine_Unit
except ImportError:
    # 如果无法直接导入，可能需要调整路径
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from Chan import CChan
    from ChanConfig import CChanConfig
    from Common.CEnum import AUTYPE, KL_TYPE, DATA_SRC
    from KLine.KLine_Unit import CKLine_Unit

# 导入日志工具
from logger import get_logger

logger = get_logger("ChanStrategy")


class BuyPointType(Enum):
    """买卖点类型枚举

    学习点:
    - 使用枚举增强类型安全性和代码可读性
    - 便于在信号转换中标准化买卖点类型
    """

    CLASS_1 = "1"  # 一类买卖点
    CLASS_1P = "1p"  # 盘整背驰一类买卖点
    CLASS_2 = "2"  # 二类买卖点
    CLASS_2S = "2s"  # 类二类买卖点
    CLASS_3A = "3a"  # 中枢在一类买卖点后的三类买卖点
    CLASS_3B = "3b"  # 中枢在一类买卖点前的三类买卖点

    @classmethod
    def from_str(cls, type_str: str) -> Optional["BuyPointType"]:
        """从字符串获取枚举值

        学习点:
        - 提供类方法简化字符串转枚举操作
        - 使用Optional表示可能返回None
        """
        for member in cls:
            if member.value == type_str:
                return member
        return None

    @classmethod
    def get_priority(cls, type_str: str) -> int:
        """获取买卖点类型的优先级

        学习点:
        - 不同买卖点类型有不同的信号优先级
        - 用于信号聚合时的冲突解决
        """
        priority_map = {
            cls.CLASS_1.value: 100,
            cls.CLASS_1P.value: 90,
            cls.CLASS_2.value: 80,
            cls.CLASS_3A.value: 70,
            cls.CLASS_3B.value: 60,
            cls.CLASS_2S.value: 50,
        }
        return priority_map.get(type_str, 0)


class TimeframeConverter:
    """时间周期转换工具类

    学习点:
    - 工具类实现KL_TYPE和字符串时间周期的互相转换
    - 支持chan.py和Lightweight Charts的数据格式转换
    """

    # KL_TYPE到字符串的映射
    KL_TYPE_TO_STR = {
        KL_TYPE.K_1S: "1s",
        KL_TYPE.K_3S: "3s",
        KL_TYPE.K_5S: "5s",
        KL_TYPE.K_10S: "10s",
        KL_TYPE.K_15S: "15s",
        KL_TYPE.K_20S: "20s",
        KL_TYPE.K_30S: "30s",
        KL_TYPE.K_1M: "1m",
        KL_TYPE.K_3M: "3m",
        KL_TYPE.K_5M: "5m",
        KL_TYPE.K_10M: "10m",
        KL_TYPE.K_15M: "15m",
        KL_TYPE.K_30M: "30m",
        KL_TYPE.K_60M: "1h",
        KL_TYPE.K_DAY: "1d",
        KL_TYPE.K_WEEK: "1w",
        KL_TYPE.K_MON: "1M",
        KL_TYPE.K_QUARTER: "1Q",
        KL_TYPE.K_YEAR: "1y",
    }

    # 字符串到KL_TYPE的映射
    STR_TO_KL_TYPE = {v: k for k, v in KL_TYPE_TO_STR.items()}

    @classmethod
    def to_string(cls, kl_type: KL_TYPE) -> str:
        """将KL_TYPE转换为字符串表示

        Args:
            kl_type: KL_TYPE枚举值

        Returns:
            对应的字符串表示
        """
        return cls.KL_TYPE_TO_STR.get(kl_type, "unknown")

    @classmethod
    def to_kl_type(cls, timeframe: str) -> Optional[KL_TYPE]:
        """将字符串表示转换为KL_TYPE

        Args:
            timeframe: 时间周期字符串

        Returns:
            对应的KL_TYPE枚举值，如果不存在则返回None
        """
        return cls.STR_TO_KL_TYPE.get(timeframe)

    @classmethod
    def get_seconds(cls, kl_type: KL_TYPE) -> int:
        """获取时间周期对应的秒数

        Args:
            kl_type: KL_TYPE枚举值

        Returns:
            对应的秒数

        学习点:
        - 不同时间周期的秒数计算
        - 用于数据时间对齐和合并
        """
        # 基础周期的秒数映射
        seconds_map = {
            KL_TYPE.K_1S: 1,
            KL_TYPE.K_3S: 3,
            KL_TYPE.K_5S: 5,
            KL_TYPE.K_10S: 10,
            KL_TYPE.K_15S: 15,
            KL_TYPE.K_20S: 20,
            KL_TYPE.K_30S: 30,
            KL_TYPE.K_1M: 60,
            KL_TYPE.K_3M: 3 * 60,
            KL_TYPE.K_5M: 5 * 60,
            KL_TYPE.K_10M: 10 * 60,
            KL_TYPE.K_15M: 15 * 60,
            KL_TYPE.K_30M: 30 * 60,
            KL_TYPE.K_60M: 60 * 60,
            KL_TYPE.K_DAY: 24 * 60 * 60,
            KL_TYPE.K_WEEK: 7 * 24 * 60 * 60,
            KL_TYPE.K_MON: 30 * 24 * 60 * 60,  # 近似值
            KL_TYPE.K_QUARTER: 3 * 30 * 24 * 60 * 60,  # 近似值
            KL_TYPE.K_YEAR: 365 * 24 * 60 * 60,  # 近似值
        }
        return seconds_map.get(kl_type, 0)


class CacheManager:
    """缓存管理器，管理CChan实例和计算结果的缓存

    学习点:
    - 缓存设计：降低重复计算开销
    - 内存管理：定期清理不活跃的缓存项
    - 二级缓存机制：平衡性能和内存占用
    """

    def __init__(self, max_instances: int = 50, max_age_seconds: int = 3600):
        """初始化缓存管理器

        Args:
            max_instances: 最大缓存实例数量
            max_age_seconds: 缓存项最大存活时间(秒)
        """
        self.chan_instances: Dict[str, Dict[str, CChan]] = (
            {}
        )  # symbol -> {config_hash -> CChan}
        self.last_access_time: Dict[str, Dict[str, float]] = (
            {}
        )  # symbol -> {config_hash -> timestamp}
        self.config_hashes: Dict[str, str] = {}  # config_str -> hash
        self.max_instances = max_instances
        self.max_age_seconds = max_age_seconds

    def get_instance(self, symbol: str, config_hash: str) -> Optional[CChan]:
        """获取CChan实例

        Args:
            symbol: 交易品种
            config_hash: 配置哈希值

        Returns:
            CChan实例或None

        学习点:
        - 通过配置哈希值查找实例
        - 记录访问时间用于LRU策略
        """
        if symbol in self.chan_instances and config_hash in self.chan_instances[symbol]:
            # 更新访问时间
            if symbol not in self.last_access_time:
                self.last_access_time[symbol] = {}
            self.last_access_time[symbol][config_hash] = time.time()
            return self.chan_instances[symbol][config_hash]
        return None

    def add_instance(self, symbol: str, config_hash: str, instance: CChan) -> None:
        """添加CChan实例到缓存

        Args:
            symbol: 交易品种
            config_hash: 配置哈希值
            instance: CChan实例

        学习点:
        - 缓存满时使用LRU策略移除最久未访问的项
        - 维护访问时间戳用于清理策略
        """
        # 初始化字典
        if symbol not in self.chan_instances:
            self.chan_instances[symbol] = {}
        if symbol not in self.last_access_time:
            self.last_access_time[symbol] = {}

        # 检查是否需要清理
        self._cleanup_if_needed()

        # 添加实例并记录访问时间
        self.chan_instances[symbol][config_hash] = instance
        self.last_access_time[symbol][config_hash] = time.time()

    def remove_instance(self, symbol: str, config_hash: str) -> None:
        """从缓存中移除实例

        Args:
            symbol: 交易品种
            config_hash: 配置哈希值
        """
        if symbol in self.chan_instances and config_hash in self.chan_instances[symbol]:
            del self.chan_instances[symbol][config_hash]
            if (
                symbol in self.last_access_time
                and config_hash in self.last_access_time[symbol]
            ):
                del self.last_access_time[symbol][config_hash]

    def _cleanup_if_needed(self) -> None:
        """根据需要清理缓存

        学习点:
        - 实现LRU(最近最少使用)清理策略
        - 同时清理过期项和超出容量限制的项
        """
        # 计算当前实例总数
        total_instances = sum(
            len(instances) for instances in self.chan_instances.values()
        )

        # 如果没有超过限制，只清理过期项
        if total_instances <= self.max_instances:
            self._cleanup_expired()
            return

        # 如果超过限制，进行LRU清理
        # 收集所有项及其访问时间
        all_items = []
        for symbol, hashes in self.last_access_time.items():
            for config_hash, access_time in hashes.items():
                all_items.append((symbol, config_hash, access_time))

        # 按访问时间排序
        all_items.sort(key=lambda x: x[2])

        # 计算需要移除的数量
        num_to_remove = total_instances - self.max_instances

        # 移除最旧的项
        for symbol, config_hash, _ in all_items[:num_to_remove]:
            self.remove_instance(symbol, config_hash)

    def _cleanup_expired(self) -> None:
        """清理过期的缓存项

        学习点:
        - 基于时间的清理策略
        - 避免内存泄漏和无用数据占用空间
        """
        now = time.time()
        to_remove = []

        for symbol, hashes in self.last_access_time.items():
            for config_hash, access_time in hashes.items():
                if now - access_time > self.max_age_seconds:
                    to_remove.append((symbol, config_hash))

        for symbol, config_hash in to_remove:
            self.remove_instance(symbol, config_hash)

    def get_config_hash(self, config_str: str) -> str:
        """获取配置字符串的哈希值

        Args:
            config_str: 配置字符串

        Returns:
            哈希字符串

        学习点:
        - 缓存哈希值避免重复计算
        - 降低内存和计算开销
        """
        if config_str not in self.config_hashes:
            # 简单哈希，实际可用更复杂算法
            import hashlib

            self.config_hashes[config_str] = hashlib.md5(
                config_str.encode()
            ).hexdigest()
        return self.config_hashes[config_str]

    def clear(self) -> None:
        """清空所有缓存

        学习点:
        - 完全重置缓存
        - 用于内存紧张时的应急措施
        """
        self.chan_instances.clear()
        self.last_access_time.clear()
        # 不清理config_hashes以保持配置哈希的复用


class ChanStrategy(IStrategy):
    """基于缠论的交易策略

    通过集成chan.py实现多品种、多级别联立分析，将缠论买卖点转换为交易信号。
    支持实时数据处理和回测，提供与Lightweight Charts兼容的可视化数据。

    学习点:
    - 适配器模式：将chan.py功能适配到IStrategy接口
    - 多品种支持：优化设计以处理大量交易品种
    - 内存管理：通过缓存和懒加载平衡性能和资源占用
    """

    def __init__(self, name: str, symbols: List[str] = None):
        """初始化缠论策略

        Args:
            name: 策略名称
            symbols: 交易品种列表

        学习点:
        - 延迟初始化：不在构造函数中创建重量级对象
        - 集合数据结构：使用set存储品种以提高查找效率
        """
        super().__init__(name, symbols)
        self.active_symbols: Set[str] = set(symbols) if symbols else set()
        self.cache_manager = CacheManager()
        self.default_levels = [KL_TYPE.K_DAY, KL_TYPE.K_60M]  # 默认分析级别
        self.signal_cache: Dict[str, List[Signal]] = {}  # 缓存生成的信号
        self.last_signal_time: Dict[str, datetime] = {}  # 上次生成信号的时间
        self.signal_refresh_interval = 300  # 信号刷新间隔(秒)
        self.chan_configs: Dict[str, CChanConfig] = {}  # 按品种缓存配置
        self.new_data_symbols: Set[str] = set()  # 有新数据的品种集合
        self.last_kline_data: Dict[str, Dict[str, Any]] = {}  # 最后处理的K线数据
        self.data_cache: Dict[str, Dict[KL_TYPE, List[Dict[str, Any]]]] = (
            {}
        )  # 原始数据缓存

        # 买卖点类型权重，用于信号聚合
        self.bs_type_weights = {
            "1": 1.0,  # 一类买卖点
            "1p": 0.9,  # 盘整背驰
            "2": 0.8,  # 二类买卖点
            "3a": 0.7,  # 三类买卖点(3-after)
            "3b": 0.6,  # 三类买卖点(3-before)
            "2s": 0.5,  # 类二买卖点
        }

    def initialize(self, context: IContext) -> None:
        """初始化策略

        Args:
            context: 策略上下文

        学习点:
        - 从上下文获取配置
        - 根据配置初始化CChanConfig
        """
        super().initialize(context)

        # 从上下文获取配置
        config = context.config

        # 设置缓存参数
        max_instances = config.get("max_chan_instances", 50)
        max_age_seconds = config.get("chan_cache_age", 3600)
        self.cache_manager = CacheManager(max_instances, max_age_seconds)

        # 设置默认级别
        levels_str = config.get("levels", "day,60m")
        if levels_str:
            self.default_levels = self._parse_levels(levels_str)

        # 设置信号刷新间隔
        self.signal_refresh_interval = config.get("signal_refresh_interval", 300)

        # 设置买卖点类型权重
        weights = config.get("bs_type_weights", None)
        if weights:
            self.bs_type_weights.update(weights)

        logger.info(
            f"ChanStrategy '{self.name}' initialized with {len(self.symbols)} symbols"
        )

    def _parse_levels(self, levels_str: str) -> List[KL_TYPE]:
        """解析级别字符串为KL_TYPE列表

        Args:
            levels_str: 逗号分隔的级别字符串

        Returns:
            KL_TYPE列表

        学习点:
        - 字符串解析为枚举列表
        - 确保级别从大到小排序
        """
        levels = []
        for level in levels_str.split(","):
            level = level.strip().lower()
            kl_type = TimeframeConverter.to_kl_type(level)
            if kl_type:
                levels.append(kl_type)

        # 确保级别从大到小排序
        levels.sort(key=lambda x: -TimeframeConverter.get_seconds(x))
        return levels

    def _get_chan_config(self, symbol: str) -> CChanConfig:
        """获取缠论配置

        Args:
            symbol: 交易品种

        Returns:
            CChanConfig实例

        学习点:
        - 缓存配置对象减少创建开销
        - 根据品种自定义配置
        """
        # 检查缓存
        if symbol in self.chan_configs:
            return self.chan_configs[symbol]

        # 从上下文获取基础配置
        base_config = {}
        if self.context:
            base_config = self.context.get_config("chan_config", {})

        # 从上下文获取品种特定配置
        symbol_config = {}
        if self.context:
            symbol_configs = self.context.get_config("symbol_chan_configs", {})
            symbol_config = symbol_configs.get(symbol, {})

        # 合并配置
        merged_config = {**base_config, **symbol_config}

        # 设置核心参数
        config_dict = {
            "zs_combine": merged_config.get("zs_combine", True),
            "bi_strict": merged_config.get("bi_strict", True),
            "trigger_step": False,  # 非动画模式
            "cal_feature": True,  # 计算特征用于生成信号
            "cal_bsp": True,  # 计算买卖点
            # 多级别联立时不检查K线对齐
            "kl_data_check": False,
            # 允许多级别缺失
            "max_kl_misalgin_cnt": merged_config.get("max_kl_misalgin_cnt", 10),
            "max_kl_inconsistent_cnt": merged_config.get("max_kl_inconsistent_cnt", 10),
        }

        # 买卖点相关配置
        bs_config = {
            "divergence_rate": merged_config.get("divergence_rate", 0.9),
            "min_zs_cnt": merged_config.get("min_zs_cnt", 1),
            "max_bs2_rate": merged_config.get("max_bs2_rate", 0.9999),
            "bs1_peak": merged_config.get("bs1_peak", True),
            "macd_algo": merged_config.get("macd_algo", "peak"),
            "bs_type": merged_config.get("bs_type", "1,1p,2,2s,3a,3b"),
        }
        config_dict.update(bs_config)

        # 创建并缓存配置
        config = CChanConfig(config_dict)
        self.chan_configs[symbol] = config

        return config

    def _get_or_create_chan(self, symbol: str, begin_time: str = None) -> CChan:
        """获取或创建CChan实例

        Args:
            symbol: 交易品种
            begin_time: 开始时间

        Returns:
            CChan实例

        学习点:
        - 延迟初始化策略
        - 缓存重量级对象减少内存占用
        """
        # 获取配置
        config = self._get_chan_config(symbol)

        # 生成配置哈希
        config_str = f"{symbol}:{str(config.__dict__)}:{begin_time}"
        config_hash = self.cache_manager.get_config_hash(config_str)

        # 尝试从缓存获取
        chan = self.cache_manager.get_instance(symbol, config_hash)
        if chan:
            return chan

        # 如果没有缓存，创建新实例
        try:
            # 默认数据源
            data_src = DATA_SRC.BAO_STOCK
            if symbol.startswith("HK."):
                data_src = DATA_SRC.FUTU
            elif symbol.startswith("US."):
                data_src = DATA_SRC.FUTU

            # 确定开始时间
            if begin_time is None:
                # 如果有上下文，尝试从上下文获取
                if self.context:
                    begin_time = self.context.get_config("begin_time")

                # 默认值
                if begin_time is None:
                    # 默认取今天往前一年
                    begin_time = (datetime.now() - timedelta(days=365)).strftime(
                        "%Y-%m-%d"
                    )

            # 从数据缓存转换为KLine_Unit
            extra_kl = self._prepare_extra_kl_data(symbol)

            # 创建CChan实例
            chan = CChan(
                code=symbol,
                begin_time=begin_time,
                end_time=None,  # 取到最新
                data_src=data_src,
                lv_list=self.default_levels,
                config=config,
                autype=AUTYPE.QFQ,  # 前复权
                extra_kl=extra_kl,
            )

            # 添加到缓存
            self.cache_manager.add_instance(symbol, config_hash, chan)

            logger.info(f"Created new CChan instance for {symbol}")
            return chan

        except Exception as e:
            logger.error(f"Error creating CChan for {symbol}: {str(e)}")
            return None

    def _prepare_extra_kl_data(
        self, symbol: str
    ) -> Optional[Dict[KL_TYPE, List[CKLine_Unit]]]:
        """准备额外的K线数据（用于trigger_load）

        Args:
            symbol: 交易品种

        Returns:
            级别到K线单元列表的映射，如果没有缓存数据则返回None

        学习点:
        - 转换原始数据为chan.py需要的格式
        - 不同级别数据的组织和处理
        """
        if symbol not in self.data_cache or not self.data_cache[symbol]:
            return None

        extra_kl = {}
        for kl_type, klines in self.data_cache[symbol].items():
            kline_units = []
            for idx, kl in enumerate(klines):
                try:
                    # 转换为CKLine_Unit
                    kline_unit = self._dict_to_kline_unit(kl, idx, kl_type)
                    kline_units.append(kline_unit)
                except Exception as e:
                    logger.error(f"Error converting K-line data for {symbol}: {str(e)}")
            if kline_units:
                extra_kl[kl_type] = kline_units

        return extra_kl if extra_kl else None

    def _dict_to_kline_unit(
        self, kl_dict: Dict[str, Any], idx: int, kl_type: KL_TYPE
    ) -> CKLine_Unit:
        """将字典转换为K线单元

        Args:
            kl_dict: K线数据字典
            idx: 索引
            kl_type: K线类型

        Returns:
            CKLine_Unit实例

        学习点:
        - 数据格式转换
        - 字段映射和类型转换
        """
        from datetime import datetime
        from Common.CTime import CTime

        # 提取时间
        time_val = (
            kl_dict.get("timestamp") or kl_dict.get("time") or kl_dict.get("datetime")
        )
        if isinstance(time_val, str):
            try:
                time_val = datetime.fromisoformat(time_val.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                time_val = datetime.now()
        elif isinstance(time_val, (int, float)):
            # 假设是毫秒时间戳
            time_val = datetime.fromtimestamp(time_val / 1000)

        # 创建CTime对象
        ctime = CTime(
            time_val.year, time_val.month, time_val.day, time_val.hour, time_val.minute
        )

        # 提取价格数据
        from Common.CEnum import DATA_FIELD

        item_dict = {
            DATA_FIELD.FIELD_TIME: ctime,
            DATA_FIELD.FIELD_OPEN: float(kl_dict.get("open")),
            DATA_FIELD.FIELD_CLOSE: float(kl_dict.get("close")),
            DATA_FIELD.FIELD_HIGH: float(kl_dict.get("high")),
            DATA_FIELD.FIELD_LOW: float(kl_dict.get("low")),
        }

        # 可选字段
        if "volume" in kl_dict:
            item_dict[DATA_FIELD.FIELD_VOLUME] = float(kl_dict.get("volume"))
        if "turnover" in kl_dict:
            item_dict[DATA_FIELD.FIELD_TURNOVER] = float(kl_dict.get("turnover"))
        if "turnrate" in kl_dict or "turnoverRate" in kl_dict:
            item_dict[DATA_FIELD.FIELD_TURNRATE] = float(
                kl_dict.get("turnrate") or kl_dict.get("turnoverRate", 0)
            )

        # 创建K线单元
        return CKLine_Unit(idx, kl_type, item_dict)

    def on_bar(self, bar_data: Dict[str, Dict[str, Any]]) -> None:
        """处理K线数据

        Args:
            bar_data: 多品种K线数据

        学习点:
        - 数据缓存：仅更新有新数据的品种缓存
        - 标记修改：记录需要重新计算的品种
        """
        # 清空新数据品种集合
        self.new_data_symbols.clear()

        # 处理每个品种的数据
        for symbol, data in bar_data.items():
            if symbol not in self.active_symbols:
                continue

            # 记录最新数据
            self.last_kline_data[symbol] = data

            # 标记为有新数据
            self.new_data_symbols.add(symbol)

            # 更新数据缓存
            timeframe = data.get("timeframe", "1d")
            kl_type = TimeframeConverter.to_kl_type(timeframe) or KL_TYPE.K_DAY

            if symbol not in self.data_cache:
                self.data_cache[symbol] = {}
            if kl_type not in self.data_cache[symbol]:
                self.data_cache[symbol][kl_type] = []

            # 将数据添加到缓存
            self.data_cache[symbol][kl_type].append(data)

            # 限制缓存大小
            max_cache_size = 1000  # 每个级别最多缓存的K线数量
            if len(self.data_cache[symbol][kl_type]) > max_cache_size:
                self.data_cache[symbol][kl_type] = self.data_cache[symbol][kl_type][
                    -max_cache_size:
                ]

    async def on_bar_async(self, bar_data: Dict[str, Dict[str, Any]]) -> None:
        """异步处理K线数据

        Args:
            bar_data: 多品种K线数据

        学习点:
        - 异步处理：避免阻塞主线程
        - 与同步版本保持一致的行为
        """
        # 调用同步版本处理数据
        self.on_bar(bar_data)

    def generate_signals(self) -> List[Signal]:
        """生成交易信号

        Returns:
            信号列表

        学习点:
        - 信号生成：根据缠论买卖点生成交易信号
        - 信号聚合：处理多个品种的多个信号
        - 缓存机制：减少重复计算
        """
        signals = []

        # 如果没有新数据，直接返回空列表
        if not self.new_data_symbols:
            return signals

        # 处理有新数据的品种
        for symbol in self.new_data_symbols:
            # 检查是否需要刷新信号
            now = datetime.now()
            last_time = self.last_signal_time.get(symbol)
            if (
                last_time
                and (now - last_time).total_seconds() < self.signal_refresh_interval
            ):
                # 如果上次生成信号时间距离现在太近，使用缓存
                if symbol in self.signal_cache:
                    signals.extend(self.signal_cache[symbol])
                continue

            # 获取缠论实例
            chan = self._get_or_create_chan(symbol)
            if not chan:
                logger.warning(f"Failed to get CChan instance for {symbol}")
                continue

            # 更新实例
            try:
                self._update_chan_with_new_data(chan, symbol)
            except Exception as e:
                logger.error(f"Error updating CChan for {symbol}: {str(e)}")
                continue

            # 生成信号
            symbol_signals = self._extract_signals_from_chan(chan, symbol)

            # 缓存信号
            self.signal_cache[symbol] = symbol_signals
            self.last_signal_time[symbol] = now

            # 添加到结果
            signals.extend(symbol_signals)

        # 信号聚合处理
        signals = self._aggregate_signals(signals)

        return signals

    async def generate_signals_async(self) -> List[Signal]:
        """异步生成交易信号

        Returns:
            信号列表

        学习点:
        - 并行处理：同时处理多个品种
        - 异步优化：提高实时性能
        """
        # 如果没有新数据，直接返回空列表
        if not self.new_data_symbols:
            return []

        # 使用异步任务处理每个品种
        tasks = []
        for symbol in self.new_data_symbols:
            tasks.append(self._generate_symbol_signals_async(symbol))

        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 收集结果
        signals = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error generating signals: {str(result)}")
            elif isinstance(result, list):
                signals.extend(result)

        # 信号聚合处理
        signals = self._aggregate_signals(signals)

        return signals

    async def _generate_symbol_signals_async(self, symbol: str) -> List[Signal]:
        """异步生成单个品种的信号

        Args:
            symbol: 交易品种

        Returns:
            信号列表

        学习点:
        - 隔离错误：单个品种错误不影响其他品种
        - 共享代码：复用同步方法的逻辑
        """
        # 检查是否需要刷新信号
        now = datetime.now()
        last_time = self.last_signal_time.get(symbol)
        if (
            last_time
            and (now - last_time).total_seconds() < self.signal_refresh_interval
        ):
            # 如果上次生成信号时间距离现在太近，使用缓存
            return self.signal_cache.get(symbol, [])

        # 获取缠论实例
        chan = self._get_or_create_chan(symbol)
        if not chan:
            logger.warning(f"Failed to get CChan instance for {symbol}")
            return []

        # 更新实例
        try:
            self._update_chan_with_new_data(chan, symbol)
        except Exception as e:
            logger.error(f"Error updating CChan for {symbol}: {str(e)}")
            return []

        # 生成信号
        symbol_signals = self._extract_signals_from_chan(chan, symbol)

        # 缓存信号
        self.signal_cache[symbol] = symbol_signals
        self.last_signal_time[symbol] = now

        return symbol_signals

    def _update_chan_with_new_data(self, chan: CChan, symbol: str) -> None:
        """使用新数据更新CChan实例

        Args:
            chan: CChan实例
            symbol: 交易品种

        学习点:
        - 增量更新：使用trigger_load而非重新计算
        - 数据格式转换：处理不同级别的数据
        """
        # 准备K线数据
        extra_kl = self._prepare_extra_kl_data(symbol)

        # 如果没有数据，直接返回
        if not extra_kl:
            return

        # 更新CChan实例
        chan.trigger_load(extra_kl)

    def _extract_signals_from_chan(self, chan: CChan, symbol: str) -> List[Signal]:
        """从CChan实例提取交易信号

        Args:
            chan: CChan实例
            symbol: 交易品种

        Returns:
            信号列表

        学习点:
        - 信号转换：将缠论买卖点转换为交易信号
        - 多级别处理：处理不同级别的买卖点
        - 信号过滤：过滤低质量信号
        """
        signals = []

        # 处理每个级别
        for i, lv in enumerate(chan.lv_list):
            try:
                kl_data = chan[i]

                # 获取买卖点列表
                bsp_list = (
                    kl_data.bs_point_lst.lst if hasattr(kl_data, "bs_point_lst") else []
                )

                # 获取自定义买卖点列表
                cbsp_list = (
                    kl_data.cbsp_strategy if hasattr(kl_data, "cbsp_strategy") else []
                )

                # 提取买卖点信号
                for bsp in bsp_list:
                    # 只考虑最近的买卖点
                    time_threshold = datetime.now() - timedelta(days=7)
                    if (
                        hasattr(bsp, "klu")
                        and hasattr(bsp.klu, "time")
                        and bsp.klu.time.to_datetime() < time_threshold
                    ):
                        continue

                    # 过滤不确定的买卖点
                    if hasattr(bsp, "is_sure") and not bsp.is_sure:
                        continue

                    # 转换为信号
                    signal = self._convert_bsp_to_signal(bsp, symbol, lv)
                    if signal:
                        signals.append(signal)

                # 提取自定义买卖点信号
                for cbsp in cbsp_list:
                    # 只考虑还未开仓或平仓的买卖点
                    if hasattr(cbsp, "is_open") and cbsp.is_open:
                        continue
                    if hasattr(cbsp, "is_cover") and cbsp.is_cover:
                        continue

                    # 转换为信号
                    signal = self._convert_cbsp_to_signal(cbsp, symbol, lv)
                    if signal:
                        signals.append(signal)

            except Exception as e:
                logger.error(
                    f"Error extracting signals for {symbol} at level {lv}: {str(e)}"
                )

        return signals

    def _convert_bsp_to_signal(
        self, bsp: Any, symbol: str, lv: KL_TYPE
    ) -> Optional[Signal]:
        """将买卖点转换为信号

        Args:
            bsp: 买卖点对象
            symbol: 交易品种
            lv: K线级别

        Returns:
            信号对象，如果无法转换则返回None

        学习点:
        - 对象转换：将特定域对象转换为通用信号
        - 属性提取：处理动态属性访问
        - 错误处理：优雅处理属性缺失
        """
        try:
            # 提取买卖点类型
            bs_type = None
            if hasattr(bsp, "type"):
                bs_type = [str(t) for t in bsp.type]
            elif hasattr(bsp, "type2str"):
                type_str = bsp.type2str()
                bs_type = type_str.split(",")

            if not bs_type:
                return None

            # 提取价格
            price = None
            if hasattr(bsp, "price"):
                price = bsp.price
            elif hasattr(bsp, "klu"):
                if hasattr(bsp.klu, "close"):
                    price = bsp.klu.close
                elif hasattr(bsp.klu, "price"):
                    price = bsp.klu.price

            if price is None:
                logger.warning(f"Cannot determine price for BSP in {symbol}")
                return None

            # 提取方向
            is_buy = None
            if hasattr(bsp, "is_buy"):
                is_buy = bsp.is_buy

            if is_buy is None:
                logger.warning(f"Cannot determine direction for BSP in {symbol}")
                return None

            # 提取时间戳
            timestamp = datetime.now()
            if hasattr(bsp, "klu") and hasattr(bsp.klu, "time"):
                timestamp = bsp.klu.time.to_datetime()

            # 创建元数据
            metadata = {
                "level": TimeframeConverter.to_string(lv),
                "is_bsp": True,
                "confidence": self._calculate_signal_confidence(bsp, bs_type),
            }

            # 添加买卖点特征
            if hasattr(bsp, "features"):
                metadata["features"] = bsp.features

            # 创建信号
            direction = SignalDirection.BUY if is_buy else SignalDirection.SELL

            signal = Signal(
                symbol=symbol,
                direction=direction,
                price=price,
                signal_type="market",
                source_strategy=self.name,
                timestamp=timestamp,
                bs_point_type=bs_type,
                metadata=metadata,
            )

            return signal

        except Exception as e:
            logger.error(f"Error converting BSP to signal for {symbol}: {str(e)}")
            return None

    def _convert_cbsp_to_signal(
        self, cbsp: Any, symbol: str, lv: KL_TYPE
    ) -> Optional[Signal]:
        """将自定义买卖点转换为信号

        Args:
            cbsp: 自定义买卖点对象
            symbol: 交易品种
            lv: K线级别

        Returns:
            信号对象，如果无法转换则返回None

        学习点:
        - 特殊处理：处理自定义买卖点的特殊属性
        - 数据富化：添加丰富的元数据
        """
        try:
            # 提取买卖点类型
            bs_type = []
            if hasattr(cbsp, "bs_type"):
                bs_type = [cbsp.bs_type]
            elif hasattr(cbsp, "type2str"):
                type_str = cbsp.type2str()
                bs_type = type_str.split(",")

            # 提取价格
            price = None
            if hasattr(cbsp, "open_price"):
                price = cbsp.open_price
            elif hasattr(cbsp, "price"):
                price = cbsp.price
            elif hasattr(cbsp, "klu"):
                if hasattr(cbsp.klu, "close"):
                    price = cbsp.klu.close
                elif hasattr(cbsp.klu, "price"):
                    price = cbsp.klu.price

            if price is None:
                logger.warning(f"Cannot determine price for CBSP in {symbol}")
                return None

            # 提取方向
            is_buy = None
            if hasattr(cbsp, "is_buy"):
                is_buy = cbsp.is_buy

            if is_buy is None:
                logger.warning(f"Cannot determine direction for CBSP in {symbol}")
                return None

            # 提取时间戳
            timestamp = datetime.now()
            if hasattr(cbsp, "klu") and hasattr(cbsp.klu, "time"):
                timestamp = cbsp.klu.time.to_datetime()

            # 提取止损价
            stop_loss = None
            if hasattr(cbsp, "sl_price"):
                stop_loss = cbsp.sl_price

            # 提取目标价
            take_profit = None
            if hasattr(cbsp, "target_price"):
                take_profit = cbsp.target_price

            # 创建元数据
            metadata = {
                "level": TimeframeConverter.to_string(lv),
                "is_cbsp": True,
                "confidence": self._calculate_signal_confidence(cbsp, bs_type),
            }

            # 添加自定义买卖点特征
            if hasattr(cbsp, "features"):
                metadata["features"] = cbsp.features

            # 添加关联的BSP
            if hasattr(cbsp, "bsp") and cbsp.bsp:
                if hasattr(cbsp.bsp, "type2str"):
                    metadata["related_bsp_type"] = cbsp.bsp.type2str()

            # 添加信号强度
            if hasattr(cbsp, "strength"):
                metadata["strength"] = cbsp.strength

            # 创建信号
            direction = SignalDirection.BUY if is_buy else SignalDirection.SELL

            signal = Signal(
                symbol=symbol,
                direction=direction,
                price=price,
                signal_type="market",
                source_strategy=self.name,
                timestamp=timestamp,
                bs_point_type=bs_type,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )

            return signal

        except Exception as e:
            logger.error(f"Error converting CBSP to signal for {symbol}: {str(e)}")
            return None

    def _calculate_signal_confidence(self, bsp: Any, bs_type: List[str]) -> float:
        """计算信号的置信度

        Args:
            bsp: 买卖点对象
            bs_type: 买卖点类型列表

        Returns:
            置信度得分(0-1)

        学习点:
        - 买卖点评分：基于类型和特征计算置信度
        - 特征融合：整合多个特征的评分
        """
        # 基础置信度(根据买卖点类型)
        base_confidence = 0.0
        for bs in bs_type:
            base_confidence = max(base_confidence, self.bs_type_weights.get(bs, 0.0))

        # 调整系数
        adjustment = 0.0

        # 根据特征调整
        if hasattr(bsp, "features"):
            features = bsp.features

            # 示例: 根据背驰强度调整(如果有)
            divergence = features.get("divergence_rate", 0.0)
            if isinstance(divergence, (int, float)) and divergence > 0:
                # 背驰强度越大越好
                adjustment += min(0.2, divergence * 0.2)

            # 示例：根据zs_cnt调整(如果有)
            zs_cnt = features.get("zs_cnt", 0)
            if isinstance(zs_cnt, int) and zs_cnt > 1:
                # 中枢数量越多越好
                adjustment += min(0.1, (zs_cnt - 1) * 0.05)

            # 示例：根据模型分数调整(如果有)
            model_score = features.get("model_score", 0.0)
            if isinstance(model_score, (int, float)) and model_score > 0:
                adjustment += min(0.3, model_score * 0.3)

        # 计算最终置信度
        confidence = min(1.0, base_confidence + adjustment)

        return confidence

    def _aggregate_signals(self, signals: List[Signal]) -> List[Signal]:
        """聚合信号，处理冲突

        Args:
            signals: 原始信号列表

        Returns:
            聚合后的信号列表

        学习点:
        - 信号聚合：合并同一品种的多个信号
        - 冲突解决：处理同方向或相反方向的冲突信号
        - 优先级机制：基于买卖点类型和置信度的优先级
        """
        if not signals:
            return []

        # 按品种分组
        symbol_signals = {}
        for signal in signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)

        aggregated_signals = []

        # 处理每个品种的信号
        for symbol, sym_signals in symbol_signals.items():
            if len(sym_signals) == 1:
                # 只有一个信号，直接添加
                aggregated_signals.append(sym_signals[0])
                continue

            # 按方向分组
            buy_signals = [s for s in sym_signals if s.direction == SignalDirection.BUY]
            sell_signals = [
                s for s in sym_signals if s.direction == SignalDirection.SELL
            ]

            # 有买有卖，冲突场景
            if buy_signals and sell_signals:
                # 获取最高置信度的买卖信号
                best_buy = max(
                    buy_signals,
                    key=lambda s: (
                        s.metadata.get("confidence", 0.0) if s.metadata else 0.0
                    ),
                )
                best_sell = max(
                    sell_signals,
                    key=lambda s: (
                        s.metadata.get("confidence", 0.0) if s.metadata else 0.0
                    ),
                )

                # 比较置信度，保留最高的
                if best_buy.metadata.get("confidence", 0.0) > best_sell.metadata.get(
                    "confidence", 0.0
                ):
                    aggregated_signals.append(best_buy)
                else:
                    aggregated_signals.append(best_sell)

            # 只有买信号
            elif buy_signals:
                # 获取最高置信度的买信号
                best_buy = max(
                    buy_signals,
                    key=lambda s: (
                        s.metadata.get("confidence", 0.0) if s.metadata else 0.0
                    ),
                )
                aggregated_signals.append(best_buy)

            # 只有卖信号
            elif sell_signals:
                # 获取最高置信度的卖信号
                best_sell = max(
                    sell_signals,
                    key=lambda s: (
                        s.metadata.get("confidence", 0.0) if s.metadata else 0.0
                    ),
                )
                aggregated_signals.append(best_sell)

        return aggregated_signals

    def get_visualization_data(self) -> Dict[str, Any]:
        """获取可视化数据

        Returns:
            可视化数据字典，格式兼容Lightweight Charts

        学习点:
        - 数据转换：将chan.py数据转换为前端图表格式
        - 多品种支持：按品种组织数据
        - 格式兼容：生成符合Lightweight Charts要求的数据
        """
        result = {}

        # 处理有新数据的品种
        for symbol in self.active_symbols:
            if symbol not in self.last_kline_data:
                continue

            # 获取缠论实例
            chan = self._get_or_create_chan(symbol)
            if not chan:
                continue

            symbol_result = {}

            # 获取每个级别的数据
            for i, lv in enumerate(chan.lv_list):
                try:
                    level_str = TimeframeConverter.to_string(lv)
                    kl_data = chan[i]

                    # 转换K线数据
                    klines = self._convert_klines_for_chart(kl_data)

                    # 转换笔
                    bi_lines = self._convert_bi_for_chart(kl_data)

                    # 转换线段
                    seg_lines = self._convert_seg_for_chart(kl_data)

                    # 转换中枢
                    zs_areas = self._convert_zs_for_chart(kl_data)

                    # 转换买卖点
                    bs_markers = self._convert_bs_for_chart(kl_data)

                    # 组装级别数据
                    level_data = {
                        "klines": klines,
                        "bi": bi_lines,
                        "seg": seg_lines,
                        "zs": zs_areas,
                        "bs": bs_markers,
                        "info": {"code": symbol, "level": level_str},
                    }

                    symbol_result[level_str] = level_data

                except Exception as e:
                    logger.error(
                        f"Error converting visualization data for {symbol} at level {lv}: {str(e)}"
                    )

            # 添加品种数据
            if symbol_result:
                result[symbol] = symbol_result

        return result

    def _convert_klines_for_chart(self, kl_data: Any) -> List[Dict[str, Any]]:
        """转换K线数据为图表格式

        Args:
            kl_data: K线数据

        Returns:
            图表格式的K线数据列表

        学习点:
        - 数据转换：将内部格式转换为图表兼容格式
        - 数据格式化：时间和价格的格式化
        """
        klines = []

        # 获取K线列表
        kl_list = []
        if hasattr(kl_data, "lst"):
            for klc in kl_data.lst:
                if hasattr(klc, "lst"):
                    for klu in klc.lst:
                        kl_list.append(klu)

        # 转换为图表格式
        for klu in kl_list:
            if not hasattr(klu, "time") or not hasattr(klu, "open"):
                continue

            # 转换时间
            time_val = (
                klu.time.to_datetime().isoformat()
                if hasattr(klu.time, "to_datetime")
                else str(klu.time)
            )

            # 创建K线数据
            kline = {
                "time": time_val,
                "open": klu.open,
                "high": klu.high,
                "low": klu.low,
                "close": klu.close,
            }

            # 添加可选字段
            if hasattr(klu, "volume"):
                kline["volume"] = klu.volume

            klines.append(kline)

        return klines

    def _convert_bi_for_chart(self, kl_data: Any) -> List[Dict[str, Any]]:
        """转换笔数据为图表格式

        Args:
            kl_data: K线数据

        Returns:
            图表格式的笔数据列表

        学习点:
        - 线段表示：表示具有方向的连接线
        - 虚实区分：通过线型区分确定和未确定的笔
        """
        bi_lines = []

        # 获取笔列表
        if not hasattr(kl_data, "bi_list") or not hasattr(kl_data.bi_list, "bi_list"):
            return bi_lines

        # 转换为图表格式
        for bi in kl_data.bi_list.bi_list:
            try:
                if not hasattr(bi, "get_begin_klu") or not hasattr(bi, "get_end_klu"):
                    continue

                # 获取起止K线
                begin_klu = bi.get_begin_klu()
                end_klu = bi.get_end_klu()

                if not begin_klu or not end_klu:
                    continue

                # 获取起止时间和价格
                begin_time = (
                    begin_klu.time.to_datetime().isoformat()
                    if hasattr(begin_klu.time, "to_datetime")
                    else str(begin_klu.time)
                )
                end_time = (
                    end_klu.time.to_datetime().isoformat()
                    if hasattr(end_klu.time, "to_datetime")
                    else str(end_klu.time)
                )

                begin_price = bi.get_begin_val()
                end_price = bi.get_end_val()

                # 创建笔数据
                bi_line = {
                    "start": {"time": begin_time, "price": begin_price},
                    "end": {"time": end_time, "price": end_price},
                    "style": {
                        "color": "blue" if bi.dir.value > 0 else "green",
                        "lineStyle": 0 if bi.is_sure else 1,  # 0-实线，1-虚线
                        "lineWidth": 1,
                    },
                }

                bi_lines.append(bi_line)
            except Exception as e:
                # 跳过有问题的笔
                logger.debug(f"Error converting bi: {str(e)}")
                continue

        return bi_lines

    def _convert_seg_for_chart(self, kl_data: Any) -> List[Dict[str, Any]]:
        """转换线段数据为图表格式

        Args:
            kl_data: K线数据

        Returns:
            图表格式的线段数据列表

        学习点:
        - 线段处理：与笔类似但样式不同
        - 异常处理：安全处理可能缺失的属性
        """
        seg_lines = []

        # 获取线段列表
        if not hasattr(kl_data, "seg_list") or not hasattr(kl_data.seg_list, "lst"):
            return seg_lines

        # 转换为图表格式
        for seg in kl_data.seg_list.lst:
            try:
                if not hasattr(seg, "get_begin_klu") or not hasattr(seg, "get_end_klu"):
                    continue

                # 获取起止K线
                begin_klu = seg.get_begin_klu()
                end_klu = seg.get_end_klu()

                if not begin_klu or not end_klu:
                    continue

                # 获取起止时间和价格
                begin_time = (
                    begin_klu.time.to_datetime().isoformat()
                    if hasattr(begin_klu.time, "to_datetime")
                    else str(begin_klu.time)
                )
                end_time = (
                    end_klu.time.to_datetime().isoformat()
                    if hasattr(end_klu.time, "to_datetime")
                    else str(end_klu.time)
                )

                begin_price = seg.get_begin_val()
                end_price = seg.get_end_val()

                # 创建线段数据
                seg_line = {
                    "start": {"time": begin_time, "price": begin_price},
                    "end": {"time": end_time, "price": end_price},
                    "style": {
                        "color": "red" if seg.dir.value > 0 else "purple",
                        "lineStyle": 0 if seg.is_sure else 1,  # 0-实线，1-虚线
                        "lineWidth": 2,
                    },
                }

                seg_lines.append(seg_line)
            except Exception as e:
                # 跳过有问题的线段
                logger.debug(f"Error converting seg: {str(e)}")
                continue

        return seg_lines

    def _convert_zs_for_chart(self, kl_data: Any) -> List[Dict[str, Any]]:
        """转换中枢数据为图表格式

        Args:
            kl_data: K线数据

        Returns:
            图表格式的中枢数据列表

        学习点:
        - 区域表示：使用矩形表示区域
        - 中枢特性：处理中枢的时间范围和价格范围
        """
        zs_areas = []

        # 获取中枢列表
        if not hasattr(kl_data, "zs_list") or not hasattr(kl_data.zs_list, "zs_lst"):
            return zs_areas

        # 转换为图表格式
        for zs in kl_data.zs_list.zs_lst:
            try:
                if not hasattr(zs, "begin") or not hasattr(zs, "end"):
                    continue

                # 获取中枢的时间范围
                begin_time = (
                    zs.begin.time.to_datetime().isoformat()
                    if hasattr(zs.begin.time, "to_datetime")
                    else str(zs.begin.time)
                )
                end_time = (
                    zs.end.time.to_datetime().isoformat()
                    if hasattr(zs.end.time, "to_datetime")
                    else str(zs.end.time)
                )

                # 获取中枢的价格范围
                high_price = zs.high
                low_price = zs.low

                # 创建中枢数据
                zs_area = {
                    "from": {"time": begin_time, "price": low_price},
                    "to": {"time": end_time, "price": high_price},
                    "style": {
                        "backgroundColor": "rgba(255, 165, 0, 0.2)",
                        "borderColor": "rgba(255, 165, 0, 1)",
                        "borderWidth": 1,
                    },
                }

                zs_areas.append(zs_area)
            except Exception as e:
                # 跳过有问题的中枢
                logger.debug(f"Error converting zs: {str(e)}")
                continue

        return zs_areas

    def _convert_bs_for_chart(self, kl_data: Any) -> List[Dict[str, Any]]:
        """转换买卖点数据为图表格式

        Args:
            kl_data: K线数据

        Returns:
            图表格式的买卖点数据列表

        学习点:
        - 标记表示：使用箭头标记买卖点
        - 类型处理：区分不同类型的买卖点
        """
        bs_markers = []

        # 获取买卖点列表
        bsp_list = []
        if hasattr(kl_data, "bs_point_lst") and hasattr(kl_data.bs_point_lst, "lst"):
            bsp_list.extend(kl_data.bs_point_lst.lst)

        # 获取自定义买卖点列表
        if hasattr(kl_data, "cbsp_strategy"):
            bsp_list.extend(kl_data.cbsp_strategy)

        # 转换为图表格式
        for bsp in bsp_list:
            try:
                if not hasattr(bsp, "klu") or not hasattr(bsp.klu, "time"):
                    continue

                # 获取K线时间
                time_val = (
                    bsp.klu.time.to_datetime().isoformat()
                    if hasattr(bsp.klu.time, "to_datetime")
                    else str(bsp.klu.time)
                )

                # 获取买卖点类型
                bs_type = "unknown"
                if hasattr(bsp, "type2str"):
                    bs_type = bsp.type2str()
                elif hasattr(bsp, "bs_type"):
                    bs_type = str(bsp.bs_type)

                # 获取方向
                is_buy = False
                if hasattr(bsp, "is_buy"):
                    is_buy = bsp.is_buy

                # 确定标记文本
                text = bs_type
                if hasattr(bsp, "is_cbsp"):
                    text = f"C{bs_type}"  # 自定义买卖点前加C

                # 确定标记位置
                position = "belowBar" if is_buy else "aboveBar"

                # 确定标记颜色
                color = "#2196F3" if is_buy else "#FF5252"

                # 确定标记形状
                shape = "arrowUp" if is_buy else "arrowDown"

                # 创建买卖点数据
                bs_marker = {
                    "time": time_val,
                    "position": position,
                    "color": color,
                    "shape": shape,
                    "text": text,
                }

                bs_markers.append(bs_marker)
            except Exception as e:
                # 跳过有问题的买卖点
                logger.debug(f"Error converting bs: {str(e)}")
                continue

        return bs_markers

    def get_backtest_data(self) -> Dict[str, Any]:
        """获取回测数据

        Returns:
            回测数据字典，包含每个品种的K线和信号数据

        学习点:
        - 回测数据：提供给Backtrader使用的数据格式
        - 品种聚合：按品种整理数据
        """
        result = {}

        # 处理每个品种
        for symbol in self.active_symbols:
            if symbol not in self.last_kline_data:
                continue

            # 获取缠论实例
            chan = self._get_or_create_chan(symbol)
            if not chan:
                continue

            symbol_data = {"symbol": symbol, "data": []}

            # 获取每个级别的数据
            for i, lv in enumerate(chan.lv_list):
                try:
                    level_str = TimeframeConverter.to_string(lv)
                    kl_data = chan[i]

                    # 转换K线数据为回测格式
                    kl_list = []
                    if hasattr(kl_data, "lst"):
                        for klc in kl_data.lst:
                            if hasattr(klc, "lst"):
                                for klu in klc.lst:
                                    kl_list.append(klu)

                    bt_data = []
                    for klu in kl_list:
                        if not hasattr(klu, "time") or not hasattr(klu, "open"):
                            continue

                        # 转换时间
                        dt = (
                            klu.time.to_datetime()
                            if hasattr(klu.time, "to_datetime")
                            else datetime.now()
                        )

                        # 创建K线数据
                        bar_data = {
                            "datetime": dt,
                            "open": klu.open,
                            "high": klu.high,
                            "low": klu.low,
                            "close": klu.close,
                            "volume": getattr(klu, "volume", 0),
                        }

                        bt_data.append(bar_data)

                    # 添加级别数据
                    if bt_data:
                        symbol_data[f"data_{level_str}"] = bt_data

                except Exception as e:
                    logger.error(
                        f"Error converting backtest data for {symbol} at level {lv}: {str(e)}"
                    )

            # 添加信号数据
            signals = self._extract_signals_from_chan(chan, symbol)
            if signals:
                symbol_data["signals"] = [signal.to_dict() for signal in signals]

            # 添加品种数据
            if symbol_data.get("data") or any(
                k.startswith("data_") and v for k, v in symbol_data.items()
            ):
                result[symbol] = symbol_data

        return result

    def on_start(self) -> None:
        """策略启动时调用

        学习点:
        - 生命周期管理：在策略启动时准备资源
        - 预加载：预热常用数据
        """
        super().on_start()

        # 清空缓存
        self.signal_cache.clear()
        self.last_signal_time.clear()

        logger.info(f"ChanStrategy '{self.name}' started")

    def on_stop(self) -> None:
        """策略停止时调用

        学习点:
        - 资源清理：释放不再需要的资源
        - 状态持久化：可以在这里保存状态（如果需要）
        """
        super().on_stop()

        # 清空缓存
        self.cache_manager.clear()
        self.signal_cache.clear()
        self.last_signal_time.clear()

        logger.info(f"ChanStrategy '{self.name}' stopped")
