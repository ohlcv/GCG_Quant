# string_utils.py - 字符串处理工具模块

"""
文件说明：
    这个文件提供了GCG_Quant系统中处理字符串的工具函数。
    包含交易对符号标准化、数值格式化等常用功能。
    这些工具函数使系统中的字符串处理更加一致和可靠。

学习目标：
    1. 了解字符串处理的常用技术
    2. 学习如何规范化交易品种符号
    3. 掌握数值的格式化展示方法
"""

import re
from typing import Any, Optional, Dict, List, Union, Tuple
import difflib


def denormalize_symbol(symbol: str, target_format: str = "none") -> str:
    """
    将标准化的交易对符号转换为无分隔符或指定格式

    Args:
        symbol: 交易对符号，如 "BTC/USDT"
        target_format: 目标格式，可选值:
                      "none" - 无分隔符 (BTCUSDT)
                      "dash" - 使用'-'分隔 (BTC-USDT)
                      "underscore" - 使用'_'分隔 (BTC_USDT)

    Returns:
        str: 转换后的交易对符号，如 "BTCUSDT"
    """
    # 先标准化，确保格式一致
    normalized = normalize_symbol(symbol)

    # 分割为基础货币和报价货币
    if "/" in normalized:
        base, quote = normalized.split("/")
    else:
        return normalized  # 无法处理的格式，返回原符号

    # 根据目标格式转换
    if target_format == "none":
        return f"{base}{quote}"
    elif target_format == "dash":
        return f"{base}-{quote}"
    elif target_format == "underscore":
        return f"{base}_{quote}"
    else:
        return f"{base}{quote}"  # 默认无分隔符


def normalize_symbol(symbol: str, separator: str = "/") -> str:
    """
    标准化交易对符号

    Args:
        symbol: 交易对符号，如 "BTCUSDT", "BTC/USDT", "BTC-USDT"
        separator: 标准分隔符，默认为"/"

    Returns:
        str: 标准化后的交易对符号，如 "BTC/USDT"
    """
    # 移除所有空白字符
    symbol = symbol.strip().replace(" ", "").upper()

    # 如果已经包含分隔符，检查格式
    if separator in symbol:
        parts = symbol.split(separator)
        if len(parts) == 2:
            return symbol

    # 尝试识别常见的分隔符
    for sep in ["-", "_", ".", ":"]:
        if sep in symbol:
            parts = symbol.split(sep)
            if len(parts) == 2:
                return f"{parts[0]}{separator}{parts[1]}"

    # 尝试识别常见的模式
    common_quote_currencies = [
        "USDT",
        "USD",
        "BTC",
        "ETH",
        "BNB",
        "BUSD",
        "DAI",
        "USDC",
    ]

    for quote in common_quote_currencies:
        if symbol.endswith(quote):
            base = symbol[: -len(quote)]
            return f"{base}{separator}{quote}"

    # 无法识别的格式，返回原样
    return symbol


def split_symbol(symbol: str, separator: str = "/") -> Tuple[str, str]:
    """
    分割交易对符号为基础货币和报价货币

    Args:
        symbol: 交易对符号，如 "BTC/USDT"
        separator: 分隔符，默认为"/"

    Returns:
        Tuple[str, str]: (基础货币, 报价货币)，如 ("BTC", "USDT")
    """
    # 先标准化符号
    normalized = normalize_symbol(symbol, separator)

    # 分割为基础货币和报价货币
    if separator in normalized:
        parts = normalized.split(separator)
        if len(parts) == 2:
            return parts[0], parts[1]

    # 无法分割，返回原符号和空字符串
    return normalized, ""


def format_price(price: Union[float, int], decimals: Optional[int] = None) -> str:
    """
    格式化价格，自动处理小数位数或使用指定小数位数

    Args:
        price: 价格数值
        decimals: 小数位数，None表示自动确定

    Returns:
        str: 格式化后的价格字符串
    """
    if price is None:
        return "N/A"

    # 如果指定了小数位数
    if decimals is not None:
        return f"{price:.{decimals}f}"

    # 自动确定小数位数
    price_str = str(price)
    if "." in price_str:
        integer_part, decimal_part = price_str.split(".")

        # 如果整数部分为0，保留更多小数位数
        if int(integer_part) == 0:
            # 找到第一个非零小数位
            for i, digit in enumerate(decimal_part):
                if digit != "0":
                    # 保留到第一个非零位之后的4位
                    return f"{price:.{i+4}f}"

            # 如果全是0，保留8位小数
            return f"{price:.8f}"
        else:
            # 整数部分不为0，保留2位小数
            return f"{price:.2f}"
    else:
        # 没有小数部分，保留2位小数
        return f"{price:.2f}"


def format_quantity(quantity: Union[float, int], decimals: Optional[int] = None) -> str:
    """
    格式化数量，处理整数和小数

    Args:
        quantity: 数量数值
        decimals: 小数位数，None表示自动确定

    Returns:
        str: 格式化后的数量字符串
    """
    if quantity is None:
        return "N/A"

    # 如果指定了小数位数
    if decimals is not None:
        return f"{quantity:.{decimals}f}"

    # 如果是整数，不显示小数部分
    if quantity == int(quantity):
        return f"{int(quantity)}"

    # 自动确定小数位数
    quantity_str = str(quantity)
    if "." in quantity_str:
        _, decimal_part = quantity_str.split(".")
        # 计算有效小数位数（末尾的0不计算）
        effective_decimals = len(decimal_part.rstrip("0"))
        return f"{quantity:.{effective_decimals}f}"
    else:
        return f"{int(quantity)}"


def format_percentage(
    value: Union[float, int], decimals: int = 2, include_sign: bool = True
) -> str:
    """
    格式化百分比

    Args:
        value: 百分比值，如0.156表示15.6%
        decimals: 小数位数，默认为2
        include_sign: 是否包含正负号，默认为True

    Returns:
        str: 格式化后的百分比字符串
    """
    if value is None:
        return "N/A"

    # 转换为百分比
    percentage = value * 100

    # 格式化字符串
    if include_sign and percentage > 0:
        return f"+{percentage:.{decimals}f}%"
    else:
        return f"{percentage:.{decimals}f}%"


def abbreviate_number(number: Union[float, int]) -> str:
    """
    缩写大数字，如1200变为1.2K

    Args:
        number: 要缩写的数字

    Returns:
        str: 缩写后的字符串
    """
    if number is None:
        return "N/A"

    abs_number = abs(number)
    sign = "-" if number < 0 else ""

    if abs_number < 1000:
        return (
            f"{sign}{abs_number:.2f}".rstrip("0").rstrip(".")
            if "." in f"{abs_number:.2f}"
            else f"{sign}{abs_number}"
        )
    elif abs_number < 1000000:
        return f"{sign}{abs_number/1000:.2f}K".rstrip("0").rstrip(".") + "K"
    elif abs_number < 1000000000:
        return f"{sign}{abs_number/1000000:.2f}M".rstrip("0").rstrip(".") + "M"
    elif abs_number < 1000000000000:
        return f"{sign}{abs_number/1000000000:.2f}B".rstrip("0").rstrip(".") + "B"
    else:
        return f"{sign}{abs_number/1000000000000:.2f}T".rstrip("0").rstrip(".") + "T"


def sanitize_filename(filename: str) -> str:
    """
    净化文件名，移除不允许的字符

    Args:
        filename: 原始文件名

    Returns:
        str: 净化后的文件名
    """
    # 替换Windows和Unix系统下不允许的文件名字符
    invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
    sanitized = re.sub(invalid_chars, "_", filename)

    # 移除起始和结束的空格和点
    sanitized = sanitized.strip(". ")

    # 确保文件名不为空
    if not sanitized:
        sanitized = "unnamed_file"

    return sanitized


def truncate_string(text: str, max_length: int, ellipsis: str = "...") -> str:
    """
    截断字符串到指定长度，添加省略号

    Args:
        text: 原始字符串
        max_length: 最大长度
        ellipsis: 省略号，默认为"..."

    Returns:
        str: 截断后的字符串
    """
    if len(text) <= max_length:
        return text

    # 确保max_length > len(ellipsis)
    if max_length <= len(ellipsis):
        return ellipsis[:max_length]

    return text[: max_length - len(ellipsis)] + ellipsis


def string_similarity(str1: str, str2: str) -> float:
    """
    计算两个字符串的相似度

    Args:
        str1: 第一个字符串
        str2: 第二个字符串

    Returns:
        float: 相似度，范围0-1
    """
    return difflib.SequenceMatcher(None, str1, str2).ratio()


def find_closest_match(query: str, choices: List[str]) -> Optional[str]:
    """
    从选择列表中找到最接近的匹配

    Args:
        query: 查询字符串
        choices: 选择列表

    Returns:
        Optional[str]: 最接近的匹配，如果choices为空则返回None
    """
    if not choices:
        return None

    # 使用difflib找到最佳匹配
    matches = difflib.get_close_matches(query, choices, n=1, cutoff=0.0)

    if matches:
        return matches[0]
    else:
        return choices[0]  # 如果没有匹配，返回第一个选择


def extract_digits(text: str) -> str:
    """
    从字符串中提取数字

    Args:
        text: 包含数字的字符串

    Returns:
        str: 提取的数字字符串
    """
    return "".join(c for c in text if c.isdigit() or c == ".")


def camel_to_snake(name: str) -> str:
    """
    将驼峰命名法转换为蛇形命名法

    Args:
        name: 驼峰命名的字符串，如 "camelCase"

    Returns:
        str: 蛇形命名的字符串，如 "camel_case"
    """
    # 在大写字母前添加下划线，然后转换为小写
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def snake_to_camel(name: str) -> str:
    """
    将蛇形命名法转换为驼峰命名法

    Args:
        name: 蛇形命名的字符串，如 "snake_case"

    Returns:
        str: 驼峰命名的字符串，如 "snakeCase"
    """
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def format_table(
    data: List[Dict[str, Any]], columns: Optional[List[str]] = None
) -> str:
    """
    格式化数据为ASCII表格

    Args:
        data: 数据列表，每个元素是一个字典
        columns: 要显示的列，None表示使用所有列
    Returns:
        str: 格式化的ASCII表格
    """
    if not data:
        return "空数据集"

    # 确定要显示的列
    if columns is None:
        columns = list(data[0].keys())

    # 计算每列的最大宽度
    widths = {col: len(col) for col in columns}
    for row in data:
        for col in columns:
            if col in row:
                widths[col] = max(widths[col], len(str(row[col])))

    # 创建表格
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)
    rows = []

    for row in data:
        formatted_row = " | ".join(
            str(row.get(col, "")).ljust(widths[col]) for col in columns
        )
        rows.append(formatted_row)

    return "\n".join([header, separator] + rows)


def mask_sensitive_data(text: str, mask_char: str = "*") -> str:
    """
    遮蔽敏感数据，如API密钥、密码等

    Args:
        text: 包含敏感数据的文本
        mask_char: 用于遮蔽的字符，默认为"*"

    Returns:
        str: 遮蔽后的文本
    """
    if not text:
        return ""

    # 如果文本长度小于6，全部遮蔽
    if len(text) < 6:
        return mask_char * len(text)

    # 保留前两个和后两个字符，中间全部遮蔽
    visible_chars = 2
    return (
        text[:visible_chars]
        + mask_char * (len(text) - visible_chars * 2)
        + text[-visible_chars:]
    )


def parse_key_value_string(
    text: str, delimiter: str = ";", separator: str = "="
) -> Dict[str, str]:
    """
    解析键值对字符串

    Args:
        text: 键值对字符串，如 "key1=value1;key2=value2"
        delimiter: 键值对之间的分隔符，默认为";"
        separator: 键和值之间的分隔符，默认为"="

    Returns:
        Dict[str, str]: 解析后的键值对字典
    """
    result = {}

    # 分割为键值对
    pairs = text.split(delimiter)

    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue

        # 分割键和值
        if separator in pair:
            key, value = pair.split(separator, 1)
            result[key.strip()] = value.strip()
        else:
            # 如果没有分隔符，将整个部分作为键，值为空字符串
            result[pair.strip()] = ""

    return result


def to_fixed_width(text: str, width: int, align: str = "left") -> str:
    """
    将文本转换为固定宽度

    Args:
        text: 要转换的文本
        width: 固定宽度
        align: 对齐方式，可选 "left", "right", "center"

    Returns:
        str: 固定宽度的文本
    """
    if len(text) > width:
        return text[:width]

    if align == "left":
        return text.ljust(width)
    elif align == "right":
        return text.rjust(width)
    elif align == "center":
        return text.center(width)
    else:
        raise ValueError(f"不支持的对齐方式: {align}")


# 使用示例
if __name__ == "__main__":
    # 测试交易对符号标准化
    symbols = ["BTCUSDT", "BTC-USDT", "BTC_USDT", "BTC/USDT"]
    for symbol in symbols:
        print(f"原始符号: {symbol}, 标准化后: {normalize_symbol(symbol)}")

    # 测试分割交易对符号
    symbol = "BTC/USDT"
    base, quote = split_symbol(symbol)
    print(f"交易对: {symbol}, 基础货币: {base}, 报价货币: {quote}")

    # 测试价格格式化
    prices = [0.00001234, 0.1234, 1.234, 12.34, 123.4, 1234]
    for price in prices:
        print(f"原始价格: {price}, 格式化后: {format_price(price)}")

    # 测试数值格式化
    numbers = [1, 1.0, 1.2, 1.23, 1.234, 1.2345]
    for number in numbers:
        print(f"原始数值: {number}, 格式化后: {format_quantity(number)}")

    # 测试百分比格式化
    percentages = [-0.1234, -0.01, 0, 0.01, 0.1234]
    for percentage in percentages:
        print(f"原始百分比: {percentage}, 格式化后: {format_percentage(percentage)}")

    # 测试数字缩写
    numbers = [123, 1234, 12345, 123456, 1234567, 12345678, 123456789, 1234567890]
    for number in numbers:
        print(f"原始数字: {number}, 缩写后: {abbreviate_number(number)}")

    # 测试相似度计算
    str1 = "BTC/USDT"
    str2 = "BTCUSDT"
    print(f"字符串1: {str1}, 字符串2: {str2}, 相似度: {string_similarity(str1, str2)}")

    # 测试表格格式化
    data = [
        {"name": "BTC", "price": 50000, "change": 0.05},
        {"name": "ETH", "price": 4000, "change": -0.02},
        {"name": "BNB", "price": 500, "change": 0.01},
    ]
    print("\n表格格式化示例:")
    print(format_table(data))
