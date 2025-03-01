# file_importer.py - 文件数据导入器

"""
文件说明：
    这个文件实现了文件数据导入器(FileImporter)，继承自base_collector.py中的BaseCollector抽象接口。
    它负责从本地CSV或JSON文件导入历史交易数据，包括Tick数据和K线数据，用于回测或初始化数据库。
    实现了文件扫描、数据预处理和异步读取，支持按交易品种和时间段灵活查询。

学习目标：
    1. 了解如何处理CSV和JSON格式的金融数据
    2. 学习异步文件I/O操作，提高读取效率
    3. 掌握数据预处理和转换的方法
    4. 理解数据导入与系统其他组件的集成方式
"""

import os
import asyncio
import csv
import json
import glob
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging
import aiofiles

from ..config.constants import (
    SUPPORTED_TIMEFRAMES,
    FILE_EXT_CSV,
    FILE_EXT_JSON,
    TRADE_SIDE_BUY,
    TRADE_SIDE_SELL,
)
from ..utils.time_utils import now, parse_time, datetime_to_ms
from .base_collector import BaseCollector
from ..data_storage.models import TickData, KlineData

# 创建日志记录器
logger = logging.getLogger("FileImporter")


class FileImporter(BaseCollector):
    """
    文件数据导入器，用于从本地文件导入历史数据

    学习点：
    - 继承抽象基类，实现所有抽象方法
    - 异步文件I/O提高读取效率
    - 支持多种文件格式的数据导入
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化文件数据导入器

        Args:
            config: 配置参数，至少包含：
                - data_dir: 数据目录

        学习点：
        - 配置驱动的组件初始化
        - 延迟扫描和加载，避免启动时间过长
        """
        super().__init__(config)

        # 获取配置参数
        self.data_dir = config.get("data_dir", "./data/raw")

        # 文件缓存，存储扫描到的文件信息
        self.file_cache = {
            "tick": {},  # 按交易品种存储Tick数据文件
            "kline": {},  # 按交易品种和时间周期存储K线数据文件
        }

        # 文件元数据缓存，存储文件的时间范围等信息
        self.file_metadata = {}

        # 已加载的数据缓存
        self.data_cache = {"tick": {}, "kline": {}}

        # 确保数据目录存在
        os.makedirs(os.path.dirname(os.path.abspath(self.data_dir)), exist_ok=True)

    async def connect(self) -> bool:
        """
        连接到数据源（扫描文件）

        Returns:
            bool: 连接是否成功

        学习点：
        - "连接"在这里是扫描和索引文件
        - 使用异步方法可以避免长时间阻塞
        """
        try:
            logger.info(f"扫描数据目录: {self.data_dir}")

            # 扫描数据目录
            await self._scan_files()

            logger.info(f"成功扫描数据目录，找到: {len(self.file_metadata)} 个文件")
            return True
        except Exception as e:
            logger.error(f"扫描数据目录失败: {str(e)}")
            return False

    async def disconnect(self) -> bool:
        """
        断开与数据源的连接（清理缓存）

        Returns:
            bool: 断开连接是否成功

        学习点：
        - 清理资源，释放内存
        - 简单实现，适合文件导入场景
        """
        try:
            logger.info("清理文件缓存")

            # 清理缓存
            self.file_cache = {"tick": {}, "kline": {}}
            self.file_metadata = {}
            self.data_cache = {"tick": {}, "kline": {}}

            logger.info("成功清理文件缓存")
            return True
        except Exception as e:
            logger.error(f"清理文件缓存失败: {str(e)}")
            return False

    async def fetch_tick_data(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取Tick数据

        Args:
            symbol: 交易品种符号
            start_time: 开始时间，默认为None
            end_time: 结束时间，默认为None

        Returns:
            List[Dict[str, Any]]: Tick数据列表

        学习点：
        - 根据时间范围从文件中提取数据
        - 处理不同时间格式和时区
        - 数据过滤和排序
        """
        try:
            logger.info(f"从文件获取 {symbol} 的Tick数据")

            # 检查是否有该交易品种的文件
            if symbol not in self.file_cache["tick"]:
                logger.warning(f"未找到 {symbol} 的Tick数据文件")
                return []

            # 获取该交易品种的所有文件
            files = self.file_cache["tick"][symbol]
            if not files:
                return []

            # 根据时间范围筛选文件
            selected_files = []
            for file_path in files:
                metadata = self.file_metadata.get(file_path)
                if not metadata:
                    continue

                # 检查时间范围是否有交集
                if start_time and metadata["end_time"] < start_time:
                    continue
                if end_time and metadata["start_time"] > end_time:
                    continue

                selected_files.append(file_path)

            # 从选定的文件中读取数据
            result = []
            for file_path in selected_files:
                # 根据文件扩展名选择读取方法
                ext = os.path.splitext(file_path)[1].lower()
                if ext == FILE_EXT_CSV:
                    data = await self._read_tick_csv(
                        file_path, symbol, start_time, end_time
                    )
                elif ext == FILE_EXT_JSON:
                    data = await self._read_tick_json(
                        file_path, symbol, start_time, end_time
                    )
                else:
                    logger.warning(f"不支持的文件格式: {ext}")
                    continue

                result.extend(data)

            # 按时间排序
            result.sort(key=lambda x: x["timestamp"])

            logger.info(f"成功获取 {symbol} 的Tick数据，共 {len(result)} 条")
            return result
        except Exception as e:
            logger.error(f"获取 {symbol} 的Tick数据失败: {str(e)}")
            return []

    async def fetch_kline_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取K线数据

        Args:
            symbol: 交易品种符号
            timeframe: 时间周期，如'1m', '5m', '1h', '1d'等
            start_time: 开始时间，默认为None
            end_time: 结束时间，默认为None

        Returns:
            List[Dict[str, Any]]: K线数据列表

        学习点：
        - 处理不同时间周期的K线数据
        - 文件筛选策略，减少I/O操作
        - 数据验证和转换
        """
        try:
            logger.info(f"从文件获取 {symbol} 的 {timeframe} K线数据")

            # 检查时间周期是否支持
            if timeframe not in SUPPORTED_TIMEFRAMES:
                logger.warning(f"不支持的时间周期: {timeframe}")
                return []

            # 检查是否有该交易品种和时间周期的文件
            key = f"{timeframe}"
            if (
                key not in self.file_cache["kline"]
                or symbol not in self.file_cache["kline"][key]
            ):
                logger.warning(f"未找到 {symbol} 的 {timeframe} K线数据文件")
                return []

            # 获取该交易品种和时间周期的所有文件
            files = self.file_cache["kline"][key].get(symbol, [])
            if not files:
                return []

            # 根据时间范围筛选文件
            selected_files = []
            for file_path in files:
                metadata = self.file_metadata.get(file_path)
                if not metadata:
                    continue

                # 检查时间范围是否有交集
                if start_time and metadata["end_time"] < start_time:
                    continue
                if end_time and metadata["start_time"] > end_time:
                    continue

                selected_files.append(file_path)

            # 从选定的文件中读取数据
            result = []
            for file_path in selected_files:
                # 根据文件扩展名选择读取方法
                ext = os.path.splitext(file_path)[1].lower()
                if ext == FILE_EXT_CSV:
                    data = await self._read_kline_csv(
                        file_path, symbol, timeframe, start_time, end_time
                    )
                elif ext == FILE_EXT_JSON:
                    data = await self._read_kline_json(
                        file_path, symbol, timeframe, start_time, end_time
                    )
                else:
                    logger.warning(f"不支持的文件格式: {ext}")
                    continue

                result.extend(data)

            # 按时间排序
            result.sort(key=lambda x: x["timestamp"])

            logger.info(
                f"成功获取 {symbol} 的 {timeframe} K线数据，共 {len(result)} 条"
            )
            return result
        except Exception as e:
            logger.error(f"获取 {symbol} 的 {timeframe} K线数据失败: {str(e)}")
            return []

    async def subscribe_tick(self, symbols: List[str], callback: Callable) -> bool:
        """
        订阅Tick数据（文件数据通常不需要订阅，但实现接口以保持一致性）

        Args:
            symbols: 交易品种符号列表
            callback: 数据回调函数

        Returns:
            bool: 订阅是否成功

        学习点：
        - 基类方法的可选实现
        - 对于文件数据，"订阅"意味着监控文件变化
        """
        logger.info(f"文件数据导入器不支持订阅功能")
        return False

    async def subscribe_kline(
        self, symbols: List[str], timeframe: str, callback: Callable
    ) -> bool:
        """
        订阅K线数据（文件数据通常不需要订阅，但实现接口以保持一致性）

        Args:
            symbols: 交易品种符号列表
            timeframe: 时间周期
            callback: 数据回调函数

        Returns:
            bool: 订阅是否成功
        """
        logger.info(f"文件数据导入器不支持订阅功能")
        return False

    async def _scan_files(self):
        """
        扫描数据目录，索引所有数据文件

        学习点：
        - 文件模式匹配和递归扫描
        - 文件分类和组织
        - 并发处理提高效率
        """
        logger.info(f"扫描数据目录: {self.data_dir}")

        # 清理旧缓存
        self.file_cache = {"tick": {}, "kline": {}}
        self.file_metadata = {}

        # 递归搜索所有CSV和JSON文件
        tick_pattern = os.path.join(self.data_dir, "**", "*tick*")
        kline_pattern = os.path.join(self.data_dir, "**", "*kline*")

        # 查找所有可能的文件
        tick_files = []
        tick_files.extend(glob.glob(tick_pattern + FILE_EXT_CSV, recursive=True))
        tick_files.extend(glob.glob(tick_pattern + FILE_EXT_JSON, recursive=True))

        kline_files = []
        kline_files.extend(glob.glob(kline_pattern + FILE_EXT_CSV, recursive=True))
        kline_files.extend(glob.glob(kline_pattern + FILE_EXT_JSON, recursive=True))

        # 处理Tick数据文件
        for file_path in tick_files:
            try:
                # 分析文件名，提取交易品种
                filename = os.path.basename(file_path)
                symbol = self._extract_symbol_from_filename(filename)

                if not symbol:
                    logger.warning(f"无法从文件名提取交易品种: {filename}")
                    continue

                # 获取文件元数据
                metadata = await self._get_file_metadata(file_path)
                if not metadata:
                    continue

                # 添加到缓存
                if symbol not in self.file_cache["tick"]:
                    self.file_cache["tick"][symbol] = []

                self.file_cache["tick"][symbol].append(file_path)
                self.file_metadata[file_path] = metadata
            except Exception as e:
                logger.error(f"处理Tick数据文件失败: {file_path}, 错误: {str(e)}")

        # 处理K线数据文件
        for file_path in kline_files:
            try:
                # 分析文件名，提取交易品种和时间周期
                filename = os.path.basename(file_path)
                symbol, timeframe = self._extract_symbol_and_timeframe(filename)

                if not symbol or not timeframe:
                    logger.warning(f"无法从文件名提取交易品种或时间周期: {filename}")
                    continue

                # 检查时间周期是否支持
                if timeframe not in SUPPORTED_TIMEFRAMES:
                    logger.warning(f"不支持的时间周期: {timeframe}, 文件: {filename}")
                    continue

                # 获取文件元数据
                metadata = await self._get_file_metadata(file_path)
                if not metadata:
                    continue

                # 添加到缓存
                key = f"{timeframe}"
                if key not in self.file_cache["kline"]:
                    self.file_cache["kline"][key] = {}

                if symbol not in self.file_cache["kline"][key]:
                    self.file_cache["kline"][key][symbol] = []

                self.file_cache["kline"][key][symbol].append(file_path)
                self.file_metadata[file_path] = metadata
            except Exception as e:
                logger.error(f"处理K线数据文件失败: {file_path}, 错误: {str(e)}")

        logger.info(
            f"扫描完成，找到 {len(tick_files)} 个Tick数据文件和 {len(kline_files)} 个K线数据文件"
        )

    async def _get_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        获取文件元数据，包括时间范围等信息

        Args:
            file_path: 文件路径

        Returns:
            Optional[Dict[str, Any]]: 文件元数据

        学习点：
        - 文件预处理，提取关键信息
        - 异步文件读取
        - 错误处理和恢复
        """
        stat = os.stat(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        metadata = {
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "start_time": None,
            "end_time": None,
            "count": 0,
        }
        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()
            if ext == FILE_EXT_CSV:
                reader = csv.DictReader(content.splitlines())
                rows = list(reader)
                if rows:
                    metadata["start_time"] = parse_time(
                        rows[0].get("time", rows[0].get("datetime"))
                    )
                    metadata["end_time"] = parse_time(
                        rows[-1].get("time", rows[-1].get("datetime"))
                    )
                    metadata["count"] = len(rows)
            return metadata

    def _extract_symbol_from_filename(self, filename: str) -> Optional[str]:
        """
        从文件名提取交易品种

        Args:
            filename: 文件名

        Returns:
            Optional[str]: 交易品种，如果无法提取则返回None

        学习点：
        - 文件命名约定
        - 正则表达式或字符串处理
        """
        # 假设文件名格式为: symbol_tick_date.csv 或 tick_symbol_date.csv
        # 例如: BTCUSDT_tick_20250101.csv 或 tick_BTCUSDT_20250101.csv

        # 简单实现，实际应用中可能需要更复杂的逻辑或正则表达式
        parts = filename.split("_")

        if "tick" in parts:
            # 寻找不是'tick'和不以数字开头的部分
            for part in parts:
                if part != "tick" and (not part[0].isdigit()):
                    # 移除可能的文件扩展名
                    symbol = part.split(".")[0]
                    return symbol

        return None

    def _extract_symbol_and_timeframe(self, filename: str) -> tuple:
        """
        从文件名提取交易品种和时间周期

        Args:
            filename: 文件名

        Returns:
            tuple: (交易品种, 时间周期)，如果无法提取则返回(None, None)

        学习点：
        - 更复杂的字符串处理
        - 多重信息提取
        """
        # 假设文件名格式为: symbol_kline_timeframe_date.csv 或 kline_symbol_timeframe_date.csv
        # 例如: BTCUSDT_kline_1m_20250101.csv 或 kline_BTCUSDT_1m_20250101.csv

        # 简单实现，实际应用中可能需要更复杂的逻辑或正则表达式
        parts = filename.split("_")

        symbol = None
        timeframe = None

        if "kline" in parts:
            # 查找时间周期
            for part in parts:
                if part in SUPPORTED_TIMEFRAMES:
                    timeframe = part
                    break

            # 寻找不是'kline'、不是时间周期和不以数字开头的部分
            for part in parts:
                if part != "kline" and part != timeframe and (not part[0].isdigit()):
                    # 移除可能的文件扩展名
                    symbol = part.split(".")[0]
                    break

        return symbol, timeframe

    async def _read_tick_csv(
        self,
        file_path: str,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        从CSV文件读取Tick数据

        Args:
            file_path: 文件路径
            symbol: 交易品种符号
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            List[Dict[str, Any]]: Tick数据列表

        学习点：
        - 异步CSV文件读取
        - 数据过滤和转换
        - 内存效率处理大文件
        """
        try:
            result = []

            async with aiofiles.open(file_path, "r", newline="") as f:
                content = await f.read()
                reader = csv.DictReader(content.splitlines())

                for row in reader:
                    # 尝试提取时间
                    if "time" in row:
                        time_str = row["time"]
                    elif "datetime" in row:
                        time_str = row["datetime"]
                    elif "timestamp" in row:
                        # 假设是毫秒时间戳
                        timestamp = int(row["timestamp"])
                        dt = datetime.fromtimestamp(timestamp / 1000)
                    else:
                        # 无法提取时间，跳过
                        continue

                    # 如果有时间字符串，转换为datetime
                    if "time_str" in locals():
                        try:
                            dt = parse_time(time_str)
                        except:
                            # 尝试其他格式
                            try:
                                dt = datetime.fromisoformat(time_str)
                            except:
                                # 无法解析时间，跳过
                                continue

                    # 检查时间范围
                    if start_time and dt < start_time:
                        continue
                    if end_time and dt > end_time:
                        continue

                    # 提取价格和数量
                    try:
                        price = float(row.get("price", 0))
                        amount = float(row.get("amount", 0))
                    except:
                        # 无法提取价格或数量，跳过
                        continue

                    # 提取交易方向
                    side = row.get("side", "buy")
                    if side.lower() in ["buy", "bid", "long"]:
                        side = TRADE_SIDE_BUY
                    else:
                        side = TRADE_SIDE_SELL

                    # 创建Tick数据
                    tick = {
                        "symbol": symbol,
                        "timestamp": datetime_to_ms(dt),
                        "datetime": dt,
                        "price": price,
                        "amount": amount,
                        "side": side,
                        "source": "file",
                        "trade_id": row.get("id") or row.get("trade_id"),
                    }

                    result.append(tick)

            return result
        except Exception as e:
            logger.error(f"读取Tick CSV文件失败: {file_path}, 错误: {str(e)}")
            return []

    async def _read_tick_json(
        self,
        file_path: str,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        从JSON文件读取Tick数据

        Args:
            file_path: 文件路径
            symbol: 交易品种符号
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            List[Dict[str, Any]]: Tick数据列表

        学习点：
        - 异步JSON文件读取
        - JSON数据结构处理
        - 数据验证和转换
        """
        try:
            result = []

            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                data = json.loads(content)

                # 确保数据是列表
                if not isinstance(data, list):
                    if isinstance(data, dict) and "data" in data:
                        # 某些格式可能将数据放在'data'字段中
                        data = data["data"]
                    else:
                        # 无法处理的格式
                        return []

                for item in data:
                    # 尝试提取时间
                    if "time" in item:
                        time_str = item["time"]
                    elif "datetime" in item:
                        time_str = item["datetime"]
                    elif "timestamp" in item:
                        # 假设是毫秒时间戳
                        timestamp = int(item["timestamp"])
                        dt = datetime.fromtimestamp(timestamp / 1000)
                    else:
                        # 无法提取时间，跳过
                        continue

                    # 如果有时间字符串，转换为datetime
                    if "time_str" in locals():
                        try:
                            dt = parse_time(time_str)
                        except:
                            # 尝试其他格式
                            try:
                                dt = datetime.fromisoformat(time_str)
                            except:
                                # 无法解析时间，跳过
                                continue

                    # 检查时间范围
                    if start_time and dt < start_time:
                        continue
                    if end_time and dt > end_time:
                        continue

                    # 提取价格和数量
                    try:
                        price = float(item.get("price", 0))
                        amount = float(item.get("amount", 0))
                    except:
                        # 无法提取价格或数量，跳过
                        continue

                    # 提取交易方向
                    side = item.get("side", "buy")
                    if side.lower() in ["buy", "bid", "long"]:
                        side = TRADE_SIDE_BUY
                    else:
                        side = TRADE_SIDE_SELL

                    # 创建Tick数据
                    tick = {
                        "symbol": symbol,
                        "timestamp": datetime_to_ms(dt),
                        "datetime": dt,
                        "price": price,
                        "amount": amount,
                        "side": side,
                        "source": "file",
                        "trade_id": item.get("id") or item.get("trade_id"),
                    }

                    result.append(tick)

            return result
        except Exception as e:
            logger.error(f"读取Tick JSON文件失败: {file_path}, 错误: {str(e)}")
            return []

    async def _read_kline_csv(
        self,
        file_path: str,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        从CSV文件读取K线数据

        Args:
            file_path: 文件路径
            symbol: 交易品种符号
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            List[Dict[str, Any]]: K线数据列表

        学习点：
        - CSV文件中K线数据的结构
        - 时间周期验证
        - OHLCV数据提取
        """
        try:
            result = []

            async with aiofiles.open(file_path, "r", newline="") as f:
                content = await f.read()
                reader = csv.DictReader(content.splitlines())

                for row in reader:
                    # 尝试提取时间
                    if "time" in row:
                        time_str = row["time"]
                    elif "datetime" in row:
                        time_str = row["datetime"]
                    elif "timestamp" in row:
                        # 假设是毫秒时间戳
                        timestamp = int(row["timestamp"])
                        dt = datetime.fromtimestamp(timestamp / 1000)
                    else:
                        # 无法提取时间，跳过
                        continue

                    # 如果有时间字符串，转换为datetime
                    if "time_str" in locals():
                        try:
                            dt = parse_time(time_str)
                        except:
                            # 尝试其他格式
                            try:
                                dt = datetime.fromisoformat(time_str)
                            except:
                                # 无法解析时间，跳过
                                continue

                    # 检查时间范围
                    if start_time and dt < start_time:
                        continue
                    if end_time and dt > end_time:
                        continue

                    # 提取OHLCV数据
                    try:
                        open_price = float(row.get("open", 0))
                        high = float(row.get("high", 0))
                        low = float(row.get("low", 0))
                        close = float(row.get("close", 0))
                        volume = float(row.get("volume", 0))
                    except:
                        # 无法提取OHLCV数据，跳过
                        continue

                    # 创建K线数据
                    kline = {
                        "symbol": symbol,
                        "timestamp": datetime_to_ms(dt),
                        "datetime": dt,
                        "timeframe": timeframe,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "source": "file",
                    }

                    result.append(kline)

            return result
        except Exception as e:
            logger.error(f"读取K线CSV文件失败: {file_path}, 错误: {str(e)}")
            return []

    async def _read_kline_json(
        self,
        file_path: str,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        从JSON文件读取K线数据

        Args:
            file_path: 文件路径
            symbol: 交易品种符号
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            List[Dict[str, Any]]: K线数据列表

        学习点：
        - JSON文件中K线数据的结构
        - 灵活处理不同的JSON格式
        - 数据标准化和验证
        """
        try:
            result = []

            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                data = json.loads(content)

                # 确保数据是列表
                if not isinstance(data, list):
                    if isinstance(data, dict) and "data" in data:
                        # 某些格式可能将数据放在'data'字段中
                        data = data["data"]
                    else:
                        # 无法处理的格式
                        return []

                for item in data:
                    # 尝试提取时间
                    if "time" in item:
                        time_str = item["time"]
                    elif "datetime" in item:
                        time_str = item["datetime"]
                    elif "timestamp" in item:
                        # 假设是毫秒时间戳
                        timestamp = int(item["timestamp"])
                        dt = datetime.fromtimestamp(timestamp / 1000)
                    else:
                        # 无法提取时间，跳过
                        continue

                    # 如果有时间字符串，转换为datetime
                    if "time_str" in locals():
                        try:
                            dt = parse_time(time_str)
                        except:
                            # 尝试其他格式
                            try:
                                dt = datetime.fromisoformat(time_str)
                            except:
                                # 无法解析时间，跳过
                                continue

                    # 检查时间范围
                    if start_time and dt < start_time:
                        continue
                    if end_time and dt > end_time:
                        continue

                    # 提取OHLCV数据
                    try:
                        open_price = float(item.get("open", 0))
                        high = float(item.get("high", 0))
                        low = float(item.get("low", 0))
                        close = float(item.get("close", 0))
                        volume = float(item.get("volume", 0))
                    except:
                        # 无法提取OHLCV数据，跳过
                        continue

                    # 创建K线数据
                    kline = {
                        "symbol": symbol,
                        "timestamp": datetime_to_ms(dt),
                        "datetime": dt,
                        "timeframe": timeframe,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "source": "file",
                    }

                    result.append(kline)

            return result
        except Exception as e:
            logger.error(f"读取K线JSON文件失败: {file_path}, 错误: {str(e)}")
            return []
