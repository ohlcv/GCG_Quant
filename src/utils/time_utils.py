# time_utils.py - 时间工具函数

"""
文件说明：
    这个文件提供了GCG_Quant系统中处理时间和日期的工具函数。
    时间处理在量化交易系统中非常重要，需要处理不同时区、时间格式和时间周期。
    这些工具函数简化了系统中的时间处理操作，确保一致性和准确性。

学习目标：
    1. 了解Python中的时间和日期处理
    2. 学习处理不同时间格式和时区的技巧
    3. 掌握量化交易中常用的时间处理模式
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Union, Tuple

# 导入常量
from ..config.constants import SUPPORTED_TIMEFRAMES


def now_ms() -> int:
    """
    获取当前时间的毫秒时间戳

    Returns:
        int: 毫秒时间戳

    学习点：
    - 使用time.time()获取秒级时间戳
    - 转换为毫秒时间戳以匹配交易所API
    """
    return int(time.time() * 1000)


def now() -> datetime:
    """
    获取当前UTC时间

    Returns:
        datetime: 当前UTC时间

    学习点：
    - 使用UTC时间避免时区问题
    - datetime对象便于时间计算和格式化
    """
    return datetime.now(timezone.utc)


def ms_to_datetime(ms: int) -> datetime:
    """
    将毫秒时间戳转换为datetime对象

    Args:
        ms: 毫秒时间戳

    Returns:
        datetime: 对应的datetime对象（UTC时区）

    学习点：
    - 毫秒时间戳是交易所API常用的时间格式
    - 转换为datetime对象便于后续处理
    """
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def datetime_to_ms(dt: datetime) -> int:
    """
    将datetime对象转换为毫秒时间戳

    Args:
        dt: datetime对象

    Returns:
        int: 毫秒时间戳

    学习点：
    - 确保datetime有时区信息，避免时区问题
    - 转换为毫秒时间戳以匹配交易所API
    """
    # 如果没有时区信息，假设为UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return int(dt.timestamp() * 1000)


def parse_timeframe(timeframe: str) -> Tuple[int, str]:
    """
    解析时间周期字符串，如"1m", "1h", "1d"

    Args:
        timeframe: 时间周期字符串

    Returns:
        Tuple[int, str]: (数值, 单位)，如(1, "m")

    Raises:
        ValueError: 如果时间周期格式无效

    学习点：
    - 解析时间周期字符串，提取数值和单位
    - 验证时间周期格式，确保有效性
    """
    # 验证时间周期格式
    if not timeframe or not isinstance(timeframe, str):
        raise ValueError(f"无效的时间周期格式: {timeframe}")

    # 检查是否在支持的时间周期列表中
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"不支持的时间周期: {timeframe}，支持的时间周期: {SUPPORTED_TIMEFRAMES}"
        )

    # 解析数值和单位
    for i, c in enumerate(timeframe):
        if not c.isdigit():
            value = int(timeframe[:i])
            unit = timeframe[i:]
            return value, unit

    # 如果没有找到单位，抛出异常
    raise ValueError(f"无效的时间周期格式: {timeframe}")


def timeframe_to_seconds(timeframe: str) -> int:
    """
    将时间周期转换为秒数

    Args:
        timeframe: 时间周期字符串，如"1m", "1h", "1d"

    Returns:
        int: 对应的秒数

    Raises:
        ValueError: 如果时间周期格式无效或不支持

    学习点：
    - 不同时间单位转换为秒数
    - 便于时间计算和比较
    """
    # 解析时间周期
    value, unit = parse_timeframe(timeframe)

    # 转换为秒数
    if unit == "m":
        return value * 60
    elif unit == "h":
        return value * 60 * 60
    elif unit == "d":
        return value * 60 * 60 * 24
    else:
        raise ValueError(f"不支持的时间周期单位: {unit}")


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    """
    将时间周期转换为timedelta对象

    Args:
        timeframe: 时间周期字符串，如"1m", "1h", "1d"

    Returns:
        timedelta: 对应的timedelta对象

    学习点：
    - timedelta对象便于时间计算
    - 与datetime对象配合使用很方便
    """
    # 解析时间周期
    value, unit = parse_timeframe(timeframe)

    # 转换为timedelta
    if unit == "m":
        return timedelta(minutes=value)
    elif unit == "h":
        return timedelta(hours=value)
    elif unit == "d":
        return timedelta(days=value)
    else:
        raise ValueError(f"不支持的时间周期单位: {unit}")


def align_time_to_timeframe(dt: datetime, timeframe: str) -> datetime:
    """
    将时间对齐到时间周期的整数倍

    Args:
        dt: 需要对齐的时间
        timeframe: 时间周期字符串，如"1m", "1h", "1d"

    Returns:
        datetime: 对齐后的时间

    学习点：
    - 时间对齐在K线处理中非常重要
    - 确保时间戳与交易所的K线时间一致
    """
    # 解析时间周期
    value, unit = parse_timeframe(timeframe)

    # 复制时间对象，避免修改原对象
    aligned = dt.replace(microsecond=0)

    # 按照不同时间单位进行对齐
    if unit == "m":
        minute = (aligned.minute // value) * value
        aligned = aligned.replace(minute=minute, second=0)
    elif unit == "h":
        aligned = aligned.replace(minute=0, second=0)
        hour = (aligned.hour // value) * value
        aligned = aligned.replace(hour=hour)
    elif unit == "d":
        aligned = aligned.replace(hour=0, minute=0, second=0)
        # 如果不是1d，可能需要更复杂的逻辑来对齐日期

    return aligned


def format_time(dt: Optional[datetime] = None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    格式化时间

    Args:
        dt: 要格式化的时间，默认为当前时间
        fmt: 格式化字符串，默认为"%Y-%m-%d %H:%M:%S"

    Returns:
        str: 格式化后的时间字符串

    学习点：
    - 使用strftime格式化时间为字符串
    - 不同的格式适用于不同场景
    """
    # 如果没有提供时间，使用当前时间
    if dt is None:
        dt = now()

    return dt.strftime(fmt)


def parse_time(time_str: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """
    解析时间字符串

    Args:
        time_str: 时间字符串
        fmt: 格式化字符串，默认为"%Y-%m-%d %H:%M:%S"

    Returns:
        datetime: 解析后的datetime对象（UTC时区）

    Raises:
        ValueError: 如果时间字符串格式不匹配

    学习点：
    - 使用strptime解析时间字符串
    - 添加时区信息，避免时区问题
    """
    dt = datetime.strptime(time_str, fmt)
    # 添加UTC时区信息
    return dt.replace(tzinfo=timezone.utc)


def get_start_time_from_timeframe(timeframe: str, bars: int = 100) -> datetime:
    """
    根据时间周期和K线数量，计算起始时间

    Args:
        timeframe: 时间周期字符串，如"1m", "1h", "1d"
        bars: K线数量，默认为100

    Returns:
        datetime: 起始时间

    学习点：
    - 根据需要的历史K线数量计算起始时间
    - 便于获取历史数据
    """
    # 获取时间周期对应的timedelta
    delta = timeframe_to_timedelta(timeframe)

    # 计算总时间跨度
    total_delta = delta * bars

    # 从当前时间减去总时间跨度
    return now() - total_delta


def get_time_range(timeframe: str, bars: int = 100) -> Tuple[datetime, datetime]:
    """
    根据时间周期和K线数量，计算时间范围

    Args:
        timeframe: 时间周期字符串，如"1m", "1h", "1d"
        bars: K线数量，默认为100

    Returns:
        Tuple[datetime, datetime]: (起始时间, 结束时间)

    学习点：
    - 计算获取历史数据的时间范围
    - 返回元组包含起始和结束时间
    """
    # 获取起始时间
    start_time = get_start_time_from_timeframe(timeframe, bars)

    # 结束时间为当前时间
    end_time = now()

    return start_time, end_time
