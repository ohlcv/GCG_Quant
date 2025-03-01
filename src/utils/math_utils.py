# math_utils.py - 数学工具模块

"""
文件说明：
    这个文件提供了GCG_Quant系统中常用的数学和统计函数。
    包含金融计算、统计分析、数值处理等工具函数。
    这些函数用于策略开发、数据分析和性能评估等场景。

学习目标：
    1. 了解量化交易中常用的数学和统计方法
    2. 学习如何实现移动平均、标准差等指标计算
    3. 掌握数值处理和精度控制的技术
"""

import math
import numpy as np
from typing import List, Union, Tuple, Optional, Callable, Dict, Any
from decimal import Decimal, ROUND_HALF_UP, getcontext


def round_to_precision(value: float, precision: float) -> float:
    """
    将数值四舍五入到指定精度

    Args:
        value: 要四舍五入的数值
        precision: 精度，如0.01表示精确到0.01

    Returns:
        float: 四舍五入后的数值

    Examples:
        >>> round_to_precision(1.234, 0.01)
        1.23
        >>> round_to_precision(1.235, 0.01)
        1.24
        >>> round_to_precision(1.234, 0.1)
        1.2
    """
    if precision <= 0:
        raise ValueError("精度必须大于0")

    # 使用Decimal确保精确的四舍五入
    getcontext().rounding = ROUND_HALF_UP
    factor = 1 / precision
    return float(
        Decimal(str(value))
        * Decimal(str(factor)).quantize(Decimal("1"))
        / Decimal(str(factor))
    )


def round_to_tick_size(value: float, tick_size: float) -> float:
    """
    将数值四舍五入到最接近的刻度大小

    Args:
        value: 要四舍五入的数值
        tick_size: 刻度大小，如交易所的最小价格变动单位

    Returns:
        float: 四舍五入后的数值

    Examples:
        >>> round_to_tick_size(1.234, 0.01)
        1.23
        >>> round_to_tick_size(1.237, 0.01)
        1.24
        >>> round_to_tick_size(1.234, 0.1)
        1.2
    """
    return round_to_precision(value, tick_size)


def round_decimal_places(value: float, decimal_places: int) -> float:
    """
    将数值四舍五入到指定小数位数

    Args:
        value: 要四舍五入的数值
        decimal_places: 小数位数

    Returns:
        float: 四舍五入后的数值

    Examples:
        >>> round_decimal_places(1.234, 2)
        1.23
        >>> round_decimal_places(1.235, 2)
        1.24
        >>> round_decimal_places(1.234, 1)
        1.2
    """
    if decimal_places < 0:
        raise ValueError("小数位数必须大于等于0")

    # 使用Decimal确保精确的四舍五入
    getcontext().rounding = ROUND_HALF_UP
    return float(Decimal(str(value)).quantize(Decimal(f'0.{"0" * decimal_places}')))


def truncate_decimal_places(value: float, decimal_places: int) -> float:
    """
    截断数值到指定小数位数（不进行四舍五入）

    Args:
        value: 要截断的数值
        decimal_places: 小数位数

    Returns:
        float: 截断后的数值

    Examples:
        >>> truncate_decimal_places(1.239, 2)
        1.23
        >>> truncate_decimal_places(1.234, 1)
        1.2
    """
    if decimal_places < 0:
        raise ValueError("小数位数必须大于等于0")

    factor = 10**decimal_places
    return math.floor(value * factor) / factor


def calculate_percentage_change(initial: float, final: float) -> float:
    """
    计算百分比变化

    Args:
        initial: 初始值
        final: 最终值

    Returns:
        float: 百分比变化，例如0.05表示5%的增长

    Examples:
        >>> calculate_percentage_change(100, 105)
        0.05
        >>> calculate_percentage_change(100, 95)
        -0.05
    """
    if initial == 0:
        return float("inf") if final > 0 else float("-inf") if final < 0 else 0

    return (final - initial) / initial


def calculate_absolute_change(initial: float, final: float) -> float:
    """
    计算绝对变化

    Args:
        initial: 初始值
        final: 最终值

    Returns:
        float: 绝对变化
    """
    return final - initial


def calculate_sma(data: List[float], window: int) -> List[float]:
    """
    计算简单移动平均 (Simple Moving Average)

    Args:
        data: 数据列表
        window: 窗口大小

    Returns:
        List[float]: 简单移动平均列表，长度为len(data) - window + 1
    """
    if window <= 0:
        raise ValueError("窗口大小必须大于0")

    if window > len(data):
        return []

    return [sum(data[i : i + window]) / window for i in range(len(data) - window + 1)]


def calculate_ema(
    data: List[float], window: int, smoothing: float = 2.0
) -> List[float]:
    """
    计算指数移动平均 (Exponential Moving Average)

    Args:
        data: 数据列表
        window: 窗口大小
        smoothing: 平滑因子，默认为2.0

    Returns:
        List[float]: 指数移动平均列表，长度与data相同
    """
    if window <= 0:
        raise ValueError("窗口大小必须大于0")

    if not data:
        return []

    # 计算权重
    alpha = smoothing / (1 + window)

    # 初始化EMA
    ema = [data[0]]

    # 计算EMA
    for i in range(1, len(data)):
        ema.append(data[i] * alpha + ema[i - 1] * (1 - alpha))

    return ema


def calculate_wma(data: List[float], window: int) -> List[float]:
    """
    计算加权移动平均 (Weighted Moving Average)

    Args:
        data: 数据列表
        window: 窗口大小

    Returns:
        List[float]: 加权移动平均列表，长度为len(data) - window + 1
    """
    if window <= 0:
        raise ValueError("窗口大小必须大于0")

    if window > len(data):
        return []

    weights = list(range(1, window + 1))
    sum_weights = sum(weights)

    result = []
    for i in range(len(data) - window + 1):
        weighted_sum = sum(data[i + j] * weights[j] for j in range(window))
        result.append(weighted_sum / sum_weights)

    return result


def calculate_standard_deviation(data: List[float], ddof: int = 0) -> float:
    """
    计算标准差

    Args:
        data: 数据列表
        ddof: 自由度缩减值，默认为0（总体标准差）

    Returns:
        float: 标准差
    """
    if not data:
        return float("nan")

    if len(data) <= ddof:
        return float("nan")

    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - ddof)
    return math.sqrt(variance)


def calculate_rolling_standard_deviation(
    data: List[float], window: int, ddof: int = 0
) -> List[float]:
    """
    计算滚动标准差

    Args:
        data: 数据列表
        window: 窗口大小
        ddof: 自由度缩减值，默认为0（总体标准差）

    Returns:
        List[float]: 滚动标准差列表，长度为len(data) - window + 1
    """
    if window <= 0:
        raise ValueError("窗口大小必须大于0")

    if window > len(data):
        return []

    result = []
    for i in range(len(data) - window + 1):
        window_data = data[i : i + window]
        std = calculate_standard_deviation(window_data, ddof)
        result.append(std)

    return result


def calculate_correlation(data1: List[float], data2: List[float]) -> float:
    """
    计算两个数据列表的相关系数

    Args:
        data1: 第一个数据列表
        data2: 第二个数据列表

    Returns:
        float: 相关系数，范围为[-1, 1]
    """
    if len(data1) != len(data2):
        raise ValueError("两个数据列表长度必须相同")

    if len(data1) <= 1:
        return float("nan")

    # 计算均值
    mean1 = sum(data1) / len(data1)
    mean2 = sum(data2) / len(data2)

    # 计算协方差和标准差
    covariance = sum((data1[i] - mean1) * (data2[i] - mean2) for i in range(len(data1)))
    std1 = math.sqrt(sum((x - mean1) ** 2 for x in data1))
    std2 = math.sqrt(sum((x - mean2) ** 2 for x in data2))

    # 避免除以零
    if std1 == 0 or std2 == 0:
        return 0

    return covariance / (std1 * std2)


def calculate_rolling_correlation(
    data1: List[float], data2: List[float], window: int
) -> List[float]:
    """
    计算滚动相关系数

    Args:
        data1: 第一个数据列表
        data2: 第二个数据列表
        window: 窗口大小

    Returns:
        List[float]: 滚动相关系数列表，长度为len(data1) - window + 1
    """
    if len(data1) != len(data2):
        raise ValueError("两个数据列表长度必须相同")

    if window <= 1:
        raise ValueError("窗口大小必须大于1")

    if window > len(data1):
        return []

    result = []
    for i in range(len(data1) - window + 1):
        correlation = calculate_correlation(
            data1[i : i + window], data2[i : i + window]
        )
        result.append(correlation)

    return result


def calculate_rsi(data: List[float], window: int = 14) -> List[float]:
    """
    计算相对强弱指数 (Relative Strength Index)

    Args:
        data: 价格数据列表
        window: 窗口大小，默认为14

    Returns:
        List[float]: RSI值列表，长度为len(data) - window
    """
    if len(data) <= window:
        return []

    # 计算价格变化
    changes = [data[i] - data[i - 1] for i in range(1, len(data))]

    # 分离正负变化
    gains = [max(0, change) for change in changes]
    losses = [max(0, -change) for change in changes]

    # 计算初始平均值
    avg_gain = sum(gains[:window]) / window
    avg_loss = sum(losses[:window]) / window

    rsi_values = []

    # 计算第一个RSI值
    if avg_loss == 0:
        rsi_values.append(100)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100 - (100 / (1 + rs)))

    # 计算其余RSI值
    for i in range(window, len(changes)):
        # 更新平均值
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window

        # 计算RSI
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))

    return rsi_values


def calculate_macd(
    data: List[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[List[float], List[float], List[float]]:
    """
    计算MACD (Moving Average Convergence Divergence)

    Args:
        data: 价格数据列表
        fast_period: 快速EMA周期，默认为12
        slow_period: 慢速EMA周期，默认为26
        signal_period: 信号线EMA周期，默认为9

    Returns:
        Tuple[List[float], List[float], List[float]]: (MACD线, 信号线, 柱状图)
    """
    if len(data) <= slow_period:
        return [], [], []

    # 计算快速和慢速EMA
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)

    # 计算MACD线
    macd_line = [fast_ema[i] - slow_ema[i] for i in range(len(fast_ema))]

    # 计算信号线（MACD的EMA）
    signal_line = calculate_ema(macd_line, signal_period)

    # 计算柱状图
    histogram = [macd_line[i] - signal_line[i] for i in range(len(signal_line))]

    # 截短，使三个列表长度相同
    min_len = min(len(macd_line), len(signal_line), len(histogram))
    macd_line = macd_line[-min_len:]
    signal_line = signal_line[-min_len:]
    histogram = histogram[-min_len:]

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    data: List[float], window: int = 20, num_std: float = 2.0
) -> Tuple[List[float], List[float], List[float]]:
    """
    计算布林带 (Bollinger Bands)

    Args:
        data: 价格数据列表
        window: 窗口大小，默认为20
        num_std: 标准差倍数，默认为2.0

    Returns:
        Tuple[List[float], List[float], List[float]]: (中轨, 上轨, 下轨)
    """
    if len(data) < window:
        return [], [], []

    # 计算简单移动平均 (中轨)
    middle_band = calculate_sma(data, window)

    # 计算滚动标准差
    std_values = calculate_rolling_standard_deviation(data, window)

    # 计算上轨和下轨
    upper_band = [
        middle_band[i] + num_std * std_values[i] for i in range(len(middle_band))
    ]
    lower_band = [
        middle_band[i] - num_std * std_values[i] for i in range(len(middle_band))
    ]

    return middle_band, upper_band, lower_band


def calculate_atr(
    high: List[float], low: List[float], close: List[float], window: int = 14
) -> List[float]:
    """
    计算平均真实范围 (Average True Range)

    Args:
        high: 最高价列表
        low: 最低价列表
        close: 收盘价列表
        window: 窗口大小，默认为14

    Returns:
        List[float]: ATR值列表
    """
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("输入列表长度必须相同")

    if len(high) <= 1:
        return []

    # 计算真实范围 (True Range)
    tr_values = []

    # 第一个TR只使用当日的高低价差
    tr_values.append(high[0] - low[0])

    # 计算其余TR值
    for i in range(1, len(high)):
        # TR = max(high - low, |high - prev_close|, |low - prev_close|)
        tr = max(
            high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])
        )
        tr_values.append(tr)

    # 计算ATR（TR的移动平均）
    if len(tr_values) < window:
        return []

    # 计算第一个ATR值
    atr_values = [sum(tr_values[:window]) / window]

    # 计算其余ATR值（使用Wilder平滑法）
    for i in range(window, len(tr_values)):
        atr = (atr_values[-1] * (window - 1) + tr_values[i]) / window
        atr_values.append(atr)

    return atr_values


def calculate_sharpe_ratio(
    returns: List[float], risk_free_rate: float = 0.0, annualization_factor: float = 252
) -> float:
    """
    计算夏普比率 (Sharpe Ratio)

    Args:
        returns: 日收益率列表
        risk_free_rate: 无风险利率，默认为0
        annualization_factor: 年化因子，默认为252（交易日）

    Returns:
        float: 夏普比率
    """
    if not returns:
        return float("nan")

    # 计算平均收益率
    mean_return = sum(returns) / len(returns)

    # 计算收益率标准差
    std_dev = calculate_standard_deviation(returns)

    if std_dev == 0:
        return float("inf") if mean_return > risk_free_rate else float("-inf")

    # 计算夏普比率
    sharpe = (mean_return - risk_free_rate) / std_dev

    # 年化夏普比率
    sharpe_annualized = sharpe * math.sqrt(annualization_factor)

    return sharpe_annualized


def calculate_sortino_ratio(
    returns: List[float], risk_free_rate: float = 0.0, annualization_factor: float = 252
) -> float:
    """
    计算索提诺比率 (Sortino Ratio)

    Args:
        returns: 日收益率列表
        risk_free_rate: 无风险利率，默认为0
        annualization_factor: 年化因子，默认为252（交易日）

    Returns:
        float: 索提诺比率
    """
    if not returns:
        return float("nan")

    # 计算平均收益率
    mean_return = sum(returns) / len(returns)

    # 计算下行波动率（仅考虑负收益率）
    negative_returns = [r for r in returns if r < 0]

    # 如果没有负收益率
    if not negative_returns:
        return float("inf") if mean_return > risk_free_rate else float("-inf")

    # 计算下行标准差
    downside_std_dev = math.sqrt(
        sum(r**2 for r in negative_returns) / len(negative_returns)
    )

    # 计算索提诺比率
    sortino = (mean_return - risk_free_rate) / downside_std_dev

    # 年化索提诺比率
    sortino_annualized = sortino * math.sqrt(annualization_factor)

    return sortino_annualized


def calculate_max_drawdown(prices: List[float]) -> Tuple[float, int, int]:
    """
    计算最大回撤

    Args:
        prices: 价格列表

    Returns:
        Tuple[float, int, int]: (最大回撤比例, 峰值索引, 谷值索引)
    """
    if not prices:
        return 0.0, -1, -1

    # 初始化
    max_dd = 0.0
    max_price = prices[0]
    peak_idx = 0
    trough_idx = 0
    temp_peak_idx = 0

    # 计算最大回撤
    for i in range(1, len(prices)):
        if prices[i] > max_price:
            max_price = prices[i]
            temp_peak_idx = i
        else:
            dd = (max_price - prices[i]) / max_price
            if dd > max_dd:
                max_dd = dd
                peak_idx = temp_peak_idx
                trough_idx = i

    return max_dd, peak_idx, trough_idx


def calculate_compound_interest(
    principal: float, rate: float, time: int, frequency: int = 1
) -> float:
    """
    计算复利

    Args:
        principal: 本金
        rate: 年利率（小数形式，如0.05表示5%）
        time: 时间（年）
        frequency: 每年计息次数，默认为1（年复利）

    Returns:
        float: 最终金额
    """
    return principal * (1 + rate / frequency) ** (frequency * time)


def calculate_simple_interest(principal: float, rate: float, time: int) -> float:
    """
    计算单利

    Args:
        principal: 本金
        rate: 年利率（小数形式，如0.05表示5%）
        time: 时间（年）

    Returns:
        float: 最终金额
    """
    return principal * (1 + rate * time)


def calculate_cagr(
    initial_value: float, final_value: float, time_years: float
) -> float:
    """
    计算复合年增长率 (Compound Annual Growth Rate)

    Args:
        initial_value: 初始值
        final_value: 最终值
        time_years: 时间（年）

    Returns:
        float: CAGR值，以小数表示
    """
    if initial_value <= 0 or time_years <= 0:
        return float("nan")

    return (final_value / initial_value) ** (1 / time_years) - 1


def linear_interpolation(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """
    线性插值

    Args:
        x: 要插值的点
        x0: 已知点1的x坐标
        y0: 已知点1的y坐标
        x1: 已知点2的x坐标
        y1: 已知点2的y坐标

    Returns:
        float: 插值结果
    """
    if x1 == x0:
        return (y0 + y1) / 2

    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def calculate_z_score(value: float, mean: float, std_dev: float) -> float:
    """
    计算Z分数

    Args:
        value: 值
        mean: 均值
        std_dev: 标准差

    Returns:
        float: Z分数
    """
    if std_dev == 0:
        return 0.0

    return (value - mean) / std_dev


def normalize_data(data: List[float], method: str = "minmax") -> List[float]:
    """
    数据归一化

    Args:
        data: 数据列表
        method: 归一化方法，可选 'minmax' 或 'zscore'

    Returns:
        List[float]: 归一化后的数据
    """
    if not data:
        return []

    if method == "minmax":
        # Min-Max归一化: (x - min) / (max - min)
        min_val = min(data)
        max_val = max(data)

        if max_val == min_val:
            return [0.5] * len(data)

        return [(x - min_val) / (max_val - min_val) for x in data]

    elif method == "zscore":
        # Z-Score归一化: (x - mean) / std_dev
        mean = sum(data) / len(data)
        std_dev = calculate_standard_deviation(data)

        if std_dev == 0:
            return [0.0] * len(data)

        return [(x - mean) / std_dev for x in data]

    else:
        raise ValueError(f"不支持的归一化方法: {method}")


def exponential_smoothing(data: List[float], alpha: float) -> List[float]:
    """
    指数平滑

    Args:
        data: 数据列表
        alpha: 平滑因子，范围(0, 1)

    Returns:
        List[float]: 平滑后的数据
    """
    if not data:
        return []

    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha必须在(0, 1)范围内")

    result = [data[0]]

    for i in range(1, len(data)):
        smoothed = alpha * data[i] + (1 - alpha) * result[i - 1]
        result.append(smoothed)

    return result


def sigmoid(x: float) -> float:
    """
    Sigmoid函数

    Args:
        x: 输入值

    Returns:
        float: Sigmoid函数值
    """
    return 1 / (1 + math.exp(-x))


def tanh(x: float) -> float:
    """
    双曲正切函数

    Args:
        x: 输入值

    Returns:
        float: 双曲正切函数值
    """
    return math.tanh(x)


def relu(x: float) -> float:
    """
    ReLU (Rectified Linear Unit) 函数

    Args:
        x: 输入值

    Returns:
        float: ReLU函数值
    """
    return max(0, x)


def calculate_distance(
    point1: Union[List[float], Tuple[float, ...]],
    point2: Union[List[float], Tuple[float, ...]],
    method: str = "euclidean",
) -> float:
    """
    计算两点之间的距离

    Args:
        point1: 第一个点的坐标
        point2: 第二个点的坐标
        method: 距离计算方法，可选 'euclidean', 'manhattan', 'chebyshev'

    Returns:
        float: 距离
    """
    if len(point1) != len(point2):
        raise ValueError("两点的维度必须相同")

    if method == "euclidean":
        # 欧几里得距离
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

    elif method == "manhattan":
        # 曼哈顿距离
        return sum(abs(a - b) for a, b in zip(point1, point2))

    elif method == "chebyshev":
        # 切比雪夫距离
        return max(abs(a - b) for a, b in zip(point1, point2))

    else:
        raise ValueError(f"不支持的距离计算方法: {method}")


def logistic_regression_probability(
    features: List[float], weights: List[float], bias: float
) -> float:
    """
    计算逻辑回归概率

    Args:
        features: 特征列表
        weights: 权重列表
        bias: 偏置

    Returns:
        float: 概率，范围[0, 1]
    """
    if len(features) != len(weights):
        raise ValueError("特征和权重的长度必须相同")

    # 计算线性组合
    z = bias + sum(f * w for f, w in zip(features, weights))

    # 应用sigmoid函数
    return sigmoid(z)


def calculate_moving_median(data: List[float], window: int) -> List[float]:
    """
    计算移动中位数

    Args:
        data: 数据列表
        window: 窗口大小

    Returns:
        List[float]: 移动中位数列表
    """
    if window <= 0:
        raise ValueError("窗口大小必须大于0")

    if window > len(data):
        return []

    result = []
    for i in range(len(data) - window + 1):
        window_data = sorted(data[i : i + window])
        median = (
            window_data[window // 2]
            if window % 2 == 1
            else (window_data[window // 2 - 1] + window_data[window // 2]) / 2
        )
        result.append(median)

    return result


def calculate_percentile(data: List[float], percentile: float) -> float:
    """
    计算百分位数

    Args:
        data: 数据列表
        percentile: 百分位数，范围[0, 100]

    Returns:
        float: 百分位数值
    """
    if not data:
        return float("nan")

    if percentile < 0 or percentile > 100:
        raise ValueError("百分位数必须在[0, 100]范围内")

    # 排序数据
    sorted_data = sorted(data)

    # 计算索引
    index = percentile / 100 * (len(sorted_data) - 1)

    # 如果索引是整数，直接返回对应值
    if index.is_integer():
        return sorted_data[int(index)]

    # 否则进行线性插值
    lower_idx = int(index)
    upper_idx = lower_idx + 1

    return sorted_data[lower_idx] + (
        sorted_data[upper_idx] - sorted_data[lower_idx]
    ) * (index - lower_idx)


def calculate_beta(asset_returns: List[float], market_returns: List[float]) -> float:
    """
    计算资产相对于市场的Beta系数

    Args:
        asset_returns: 资产收益率列表
        market_returns: 市场收益率列表

    Returns:
        float: Beta系数
    """
    if len(asset_returns) != len(market_returns):
        raise ValueError("资产收益率和市场收益率的长度必须相同")

    if len(asset_returns) <= 1:
        return float("nan")

    # 计算协方差
    mean_asset = sum(asset_returns) / len(asset_returns)
    mean_market = sum(market_returns) / len(market_returns)

    covariance = sum(
        (a - mean_asset) * (m - mean_market)
        for a, m in zip(asset_returns, market_returns)
    ) / len(asset_returns)

    # 计算市场方差
    market_variance = sum((m - mean_market) ** 2 for m in market_returns) / len(
        market_returns
    )

    # 避免除以零
    if market_variance == 0:
        return 0

    # 计算Beta
    return covariance / market_variance


def calculate_alpha(
    asset_returns: List[float], market_returns: List[float], risk_free_rate: float
) -> float:
    """
    计算资产相对于市场的Alpha系数

    Args:
        asset_returns: 资产收益率列表
        market_returns: 市场收益率列表
        risk_free_rate: 无风险利率

    Returns:
        float: Alpha系数
    """
    if len(asset_returns) != len(market_returns):
        raise ValueError("资产收益率和市场收益率的长度必须相同")

    if len(asset_returns) == 0:
        return float("nan")

    # 计算平均收益率
    mean_asset = sum(asset_returns) / len(asset_returns)
    mean_market = sum(market_returns) / len(market_returns)

    # 计算Beta
    beta = calculate_beta(asset_returns, market_returns)

    # 计算Alpha
    alpha = mean_asset - (risk_free_rate + beta * (mean_market - risk_free_rate))

    return alpha


def calculate_kelly_criterion(win_rate: float, win_loss_ratio: float) -> float:
    """
    计算凯利公式

    Args:
        win_rate: 胜率，范围[0, 1]
        win_loss_ratio: 赢亏比（赢的平均额/亏的平均额）

    Returns:
        float: 凯利比例，建议的仓位比例
    """
    if win_rate <= 0 or win_rate >= 1:
        raise ValueError("胜率必须在(0, 1)范围内")

    if win_loss_ratio <= 0:
        raise ValueError("赢亏比必须大于0")

    # 计算凯利比例
    kelly = win_rate - (1 - win_rate) / win_loss_ratio

    # 如果凯利比例为负数，返回0
    return max(0, kelly)


def calculate_var(returns: List[float], confidence_level: float = 0.95) -> float:
    """
    计算历史VaR (Value at Risk)

    Args:
        returns: 收益率列表
        confidence_level: 置信水平，默认为0.95

    Returns:
        float: VaR值，表示在给定置信水平下的最大亏损
    """
    if not returns:
        return float("nan")

    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("置信水平必须在(0, 1)范围内")

    # 排序收益率
    sorted_returns = sorted(returns)

    # 计算索引
    index = len(sorted_returns) * (1 - confidence_level)

    # 如果索引是整数，直接返回对应值
    if index.is_integer():
        return -sorted_returns[int(index)]

    # 否则进行线性插值
    lower_idx = int(index)
    upper_idx = min(lower_idx + 1, len(sorted_returns) - 1)

    var = -(
        sorted_returns[lower_idx]
        + (sorted_returns[upper_idx] - sorted_returns[lower_idx]) * (index - lower_idx)
    )

    return var


# 使用示例
if __name__ == "__main__":
    # 测试四舍五入函数
    value = 1.234
    tick_size = 0.01
    print(
        f"原始值: {value}, 四舍五入到{tick_size}: {round_to_tick_size(value, tick_size)}"
    )

    # 测试简单移动平均
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    window = 3
    sma = calculate_sma(data, window)
    print(f"数据: {data}")
    print(f"简单移动平均 (窗口={window}): {sma}")

    # 测试指数移动平均
    ema = calculate_ema(data, window)
    print(f"指数移动平均 (窗口={window}): {ema}")

    # 测试加权移动平均
    wma = calculate_wma(data, window)
    print(f"加权移动平均 (窗口={window}): {wma}")

    # 测试标准差
    std_dev = calculate_standard_deviation(data)
    print(f"标准差: {std_dev}")

    # 测试相关系数
    data2 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    correlation = calculate_correlation(data, data2)
    print(f"相关系数: {correlation}")

    # 测试RSI
    rsi = calculate_rsi(data)
    print(f"RSI: {rsi}")

    # 测试布林带
    middle, upper, lower = calculate_bollinger_bands(data)
    print(f"布林带中轨: {middle}")
    print(f"布林带上轨: {upper}")
    print(f"布林带下轨: {lower}")

    # 测试MACD
    macd_line, signal_line, histogram = calculate_macd(data)
    print(f"MACD线: {macd_line}")
    print(f"信号线: {signal_line}")
    print(f"柱状图: {histogram}")

    # 测试最大回撤
    prices = [100, 110, 105, 95, 85, 90, 95, 100, 90, 80]
    max_dd, peak_idx, trough_idx = calculate_max_drawdown(prices)
    print(f"价格: {prices}")
    print(f"最大回撤: {max_dd:.2%}, 峰值索引: {peak_idx}, 谷值索引: {trough_idx}")

    # 测试复利
    principal = 1000
    rate = 0.05
    time = 5
    compound = calculate_compound_interest(principal, rate, time)
    print(f"本金: {principal}, 年利率: {rate:.0%}, 时间: {time}年")
    print(f"复利: {compound:.2f}")

    # 测试凯利公式
    win_rate = 0.6
    win_loss_ratio = 2.0
    kelly = calculate_kelly_criterion(win_rate, win_loss_ratio)
    print(f"胜率: {win_rate:.0%}, 赢亏比: {win_loss_ratio}")
    print(f"凯利比例: {kelly:.2%}")
