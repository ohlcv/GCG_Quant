# time_utils.py - 基于Arrow的时间工具函数

"""
文件说明：
    这个文件提供了GCG_Quant系统中处理时间和日期的工具函数。
    时间处理在量化交易系统中非常重要，需要处理不同时区、时间格式和时间周期。
    本模块使用Arrow库简化时间处理，同时保持原有API兼容性。

学习目标：
    1. 了解Arrow库的使用及其优势
    2. 学习处理不同时间格式和时区的技巧
    3. 掌握量化交易中常用的时间处理模式
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Union, Tuple, List
import arrow
from arrow import Arrow

# 导入常量
# 从原constants.py文件导入，或直接定义
SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]


def now_ms() -> int:
    """
    获取当前时间的毫秒时间戳

    Returns:
        int: 毫秒时间戳

    学习点：
    - 使用arrow.utcnow()获取当前UTC时间
    - 转换为毫秒时间戳以匹配交易所API
    """
    return int(arrow.utcnow().float_timestamp * 1000)


def now() -> datetime:
    """
    获取当前UTC时间

    Returns:
        datetime: 当前UTC时间

    学习点：
    - 使用Arrow获取UTC时间
    - 保持返回datetime对象以兼容原API
    """
    return arrow.utcnow().datetime


def ms_to_datetime(ms: int) -> datetime:
    """
    将毫秒时间戳转换为datetime对象

    Args:
        ms: 毫秒时间戳

    Returns:
        datetime: 对应的datetime对象（UTC时区）

    学习点：
    - 使用arrow.get()处理时间戳
    - 保持返回datetime对象以兼容原API
    """
    return arrow.get(ms / 1000).to("utc").datetime


def datetime_to_ms(dt: Union[datetime, Arrow]) -> int:
    """
    将datetime对象转换为毫秒时间戳

    Args:
        dt: datetime对象或Arrow对象

    Returns:
        int: 毫秒时间戳

    学习点：
    - 处理不同类型的时间对象
    - 确保正确的时区处理
    """
    # 如果是Arrow对象，直接使用
    if isinstance(dt, Arrow):
        return int(dt.float_timestamp * 1000)

    # 如果是datetime对象
    # 确保有时区信息，没有则假设为UTC
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
    seconds = timeframe_to_seconds(timeframe)
    return timedelta(seconds=seconds)


def align_time_to_timeframe(dt: Union[datetime, Arrow], timeframe: str) -> datetime:
    """
    将时间对齐到时间周期的整数倍

    Args:
        dt: 需要对齐的时间
        timeframe: 时间周期字符串，如"1m", "1h", "1d"

    Returns:
        datetime: 对齐后的时间

    学习点：
    - 使用Arrow简化时间对齐逻辑
    - 时间对齐在K线处理中非常重要
    """
    # 将datetime转换为Arrow对象
    if isinstance(dt, datetime):
        dt_arrow = arrow.get(dt)
    else:
        dt_arrow = dt

    # 确保时间戳精确到秒
    dt_arrow = dt_arrow.floor("second")

    # 获取时间周期的秒数
    period_seconds = timeframe_to_seconds(timeframe)

    # 计算时间戳除以周期的整数倍
    timestamp = int(dt_arrow.timestamp())
    aligned_timestamp = (timestamp // period_seconds) * period_seconds

    # 转换回datetime对象
    return arrow.get(aligned_timestamp).datetime


def format_time(
    dt: Optional[Union[datetime, Arrow]] = None, fmt: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """
    格式化时间

    Args:
        dt: 要格式化的时间，默认为当前时间
        fmt: 格式化字符串，默认为"%Y-%m-%d %H:%M:%S"

    Returns:
        str: 格式化后的时间字符串

    学习点：
    - 使用Arrow的format功能简化格式化
    - 支持多种时间对象类型
    """
    # 如果没有提供时间，使用当前时间
    if dt is None:
        dt = arrow.utcnow()

    # 如果是datetime对象，转换为Arrow对象
    if isinstance(dt, datetime):
        dt = arrow.get(dt)

    # 使用Arrow的format方法格式化时间
    return dt.format(fmt)


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
    - 使用Arrow简化时间解析
    - 智能解析，支持多种常见格式
    """
    try:
        # 尝试使用指定格式解析
        if fmt:
            dt = arrow.get(time_str, fmt)
        else:
            # 使用Arrow的智能解析
            dt = arrow.get(time_str)

        # 返回带有UTC时区的datetime对象
        return dt.to("utc").datetime
    except Exception:
        # 尝试其他常见格式
        common_formats = [
            "YYYY-MM-DD HH:mm:ss",
            "YYYY-MM-DD",
            "YYYY/MM/DD HH:mm:ss",
            "YYYY/MM/DD",
            "DD-MM-YYYY HH:mm:ss",
            "DD-MM-YYYY",
            "MM/DD/YYYY HH:mm:ss",
            "MM/DD/YYYY",
            "HH:mm:ss",
        ]

        for fmt in common_formats:
            try:
                dt = arrow.get(time_str, fmt)
                return dt.to("utc").datetime
            except Exception:
                continue

        # 如果所有尝试都失败，抛出异常
        raise ValueError(f"无法解析时间字符串: {time_str}")


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
    # 获取当前时间
    now_dt = arrow.utcnow()

    # 获取时间周期对应的秒数
    seconds = timeframe_to_seconds(timeframe)

    # 计算总时间跨度（秒）
    total_seconds = seconds * bars

    # 从当前时间减去总时间跨度
    start_time = now_dt.shift(seconds=-total_seconds)

    # 返回datetime对象
    return start_time.datetime


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
    end_time = arrow.utcnow().datetime

    return start_time, end_time


def utc_to_local(dt: Union[datetime, Arrow]) -> datetime:
    """
    将UTC时间转换为本地时间

    Args:
        dt: UTC时间

    Returns:
        datetime: 本地时间

    学习点：
    - Arrow简化时区转换
    - 自动处理夏令时
    """
    # 如果是datetime对象，先转换为Arrow对象
    if isinstance(dt, datetime):
        dt = arrow.get(dt)

    # 确保是UTC时间
    dt = dt.to("utc")

    # 转换为本地时间
    local_dt = dt.to("local")

    # 返回datetime对象
    return local_dt.datetime


def local_to_utc(dt: Union[datetime, Arrow]) -> datetime:
    """
    将本地时间转换为UTC时间

    Args:
        dt: 本地时间

    Returns:
        datetime: UTC时间

    学习点：
    - 本地时区到UTC的转换
    - 保持API一致性
    """
    # 如果是datetime对象，先转换为Arrow对象
    if isinstance(dt, datetime):
        # 如果没有时区信息，假设为本地时间
        if dt.tzinfo is None:
            dt = arrow.get(dt).replace(tzinfo=arrow.now().tzinfo)
        else:
            dt = arrow.get(dt)

    # 转换为UTC时间
    utc_dt = dt.to("utc")

    # 返回datetime对象
    return utc_dt.datetime


def is_same_day(dt1: Union[datetime, Arrow], dt2: Union[datetime, Arrow]) -> bool:
    """
    判断两个时间是否是同一天

    Args:
        dt1: 第一个时间
        dt2: 第二个时间

    Returns:
        bool: 是否是同一天

    学习点：
    - 使用Arrow的floor功能进行日期比较
    - 支持不同类型的时间对象
    """
    # 将时间对象转换为Arrow对象
    if isinstance(dt1, datetime):
        dt1 = arrow.get(dt1)
    if isinstance(dt2, datetime):
        dt2 = arrow.get(dt2)

    # 对齐到天
    day1 = dt1.floor("day")
    day2 = dt2.floor("day")

    # 比较是否是同一天
    return day1 == day2


def round_time(dt: Union[datetime, Arrow], round_to: str = "minute") -> datetime:
    """
    将时间舍入到指定单位

    Args:
        dt: 要舍入的时间
        round_to: 舍入单位，可选值: 'second', 'minute', 'hour', 'day'

    Returns:
        datetime: 舍入后的时间

    学习点：
    - Arrow的floor和ceil功能简化时间舍入
    - 支持多种舍入单位
    """
    # 将时间对象转换为Arrow对象
    if isinstance(dt, datetime):
        dt = arrow.get(dt)

    # 根据舍入单位进行舍入
    if round_to == "second":
        rounded = dt.floor("second")
    elif round_to == "minute":
        rounded = dt.floor("minute")
    elif round_to == "hour":
        rounded = dt.floor("hour")
    elif round_to == "day":
        rounded = dt.floor("day")
    else:
        raise ValueError(f"不支持的舍入单位: {round_to}")

    # 返回datetime对象
    return rounded.datetime


def generate_time_series(
    start: Union[datetime, Arrow], end: Union[datetime, Arrow], interval: str
) -> List[datetime]:
    """
    生成时间序列

    Args:
        start: 开始时间
        end: 结束时间
        interval: 时间间隔，如"1m", "1h", "1d"

    Returns:
        List[datetime]: 时间序列列表

    学习点：
    - 使用Arrow生成均匀的时间序列
    - 支持不同的时间间隔
    """
    # 将时间对象转换为Arrow对象
    if isinstance(start, datetime):
        start = arrow.get(start)
    if isinstance(end, datetime):
        end = arrow.get(end)

    # 获取间隔秒数
    seconds = timeframe_to_seconds(interval)

    # 生成时间序列
    result = []
    current = start
    while current <= end:
        result.append(current.datetime)
        current = current.shift(seconds=seconds)

    return result


# 使用示例
if __name__ == "__main__":
    # 获取当前时间
    current_time = now()
    print(f"当前时间: {current_time}")

    # 格式化时间
    formatted = format_time(current_time)
    print(f"格式化时间: {formatted}")

    # 解析时间字符串
    parsed = parse_time("2025-01-01 12:00:00")
    print(f"解析后的时间: {parsed}")

    # 时间对齐
    aligned = align_time_to_timeframe(current_time, "1h")
    print(f"对齐到小时: {aligned}")

    # 获取时间范围
    start, end = get_time_range("1h", 24)
    print(f"过去24小时的时间范围: {start} 到 {end}")

    # 时区转换
    local_time = utc_to_local(current_time)
    print(f"本地时间: {local_time}")

    # 生成时间序列
    time_series = generate_time_series(start, end, "1h")
    print(f"时间序列长度: {len(time_series)}")
    print(f"时间序列前5个: {time_series[:5]}")
