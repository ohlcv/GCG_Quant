# logger.py - 基于Loguru的日志工具

"""
文件说明：
    这个文件提供了GCG_Quant系统的日志功能，基于Loguru库实现。
    Loguru提供了简洁的API和强大的功能，包括结构化日志、彩色输出、文件轮转等。
    系统中所有组件都使用这里定义的日志工具来记录日志。

学习目标：
    1. 了解Loguru库的使用方法和优势
    2. 学习如何配置日志格式、级别和处理器
    3. 掌握在大型项目中统一管理日志的最佳实践
"""

import os
import sys
from typing import Optional, Any, Dict, Union
from pathlib import Path
from datetime import datetime

from loguru import logger


def setup_logger(
    name: str = "gcg_quant",
    level: str = "INFO",
    log_file: Optional[str] = None,
    retention: str = "30 days",
    rotation: str = "00:00",  # 默认改为每天午夜轮转
    format_string: Optional[str] = None,
) -> Any:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称，默认为"gcg_quant"
        level: 日志级别，默认为INFO
        log_file: 日志文件路径，默认为None（不记录到文件）
        retention: 日志保留时间，默认为30天
        rotation: 日志轮转策略，默认为"00:00"（每天午夜轮转）
        format_string: 自定义日志格式，默认为None（使用预定义格式）

    Returns:
        配置好的日志记录器

    学习点：
    - Loguru使用简单的API配置日志
    - 支持彩色输出和结构化日志
    - 内置文件轮转和保留功能
    """
    # 移除默认处理器
    logger.remove()

    # 设置默认格式
    if format_string is None:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        )
    else:
        console_format = format_string
        file_format = format_string

    # 添加控制台处理器
    logger.add(sys.stdout, format=console_format, level=level, colorize=True)

    # 如果提供了日志文件路径，添加文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)

        # 添加文件处理器
        logger.add(
            log_file,
            format=file_format,
            level=level,
            rotation=rotation,  # 默认每天午夜轮转
            retention=retention,  # 保留30天的日志
            compression="zip",  # 压缩旧日志文件
            encoding="utf-8",  # 使用utf-8编码
        )

    # 返回绑定了名称的日志记录器
    return logger.bind(name=name)


def get_logger(name: str = "gcg_quant") -> Any:
    """
    获取已配置的日志记录器，如果不存在则新建一个

    Args:
        name: 日志记录器名称，默认为"gcg_quant"

    Returns:
        日志记录器

    学习点：
    - 使用Loguru的bind功能创建命名的日志记录器
    - 简化日志记录器获取流程
    - 默认使用按日期轮转的日志文件
    """
    # 检查是否已经有配置过的logger
    if not logger._core.handlers:
        # 创建日志目录
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # 构建日志文件路径 - 使用日期格式化而不是时间戳
        log_file = os.path.join(log_dir, f"{name}_{{time:YYYY-MM-DD}}.log")

        # 设置日志记录器 - 使用每日轮转
        setup_logger(
            name=name, level="INFO", log_file=log_file, rotation="00:00"  # 每天午夜轮转
        )

    # 返回绑定了名称的日志记录器
    return logger.bind(name=name)


def setup_daily_logger(
    name: str = "gcg_quant", level: str = "INFO", log_dir: Optional[str] = None
) -> Any:
    """
    设置每日轮转的日志记录器

    Args:
        name: 日志记录器名称，默认为"gcg_quant"
        level: 日志级别，默认为INFO
        log_dir: 日志目录，默认为当前目录下的logs目录

    Returns:
        配置好的日志记录器

    学习点：
    - 日志按日期轮转，便于管理长期运行的系统
    - 统一的日志存储位置，便于查找和分析
    """
    # 设置默认日志目录
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")

    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 构建日志文件路径，使用{time}占位符实现按日期轮转
    log_file = os.path.join(log_dir, f"{name}_{{time:YYYY-MM-DD}}.log")

    # 设置日志记录器
    return setup_logger(
        name=name,
        level=level,
        log_file=log_file,
        rotation="00:00",  # 每天午夜轮转
        retention="30 days",  # 保留30天的日志
    )


def log_exception(e: Exception, message: str = "发生异常") -> None:
    """
    记录异常信息

    Args:
        e: 异常对象
        message: 额外信息，默认为"发生异常"

    学习点：
    - Loguru支持异常堆栈的详细记录
    - 简化异常日志记录流程
    """
    logger.exception(f"{message}: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 获取默认日志记录器
    log = get_logger()

    # 记录不同级别的日志
    log.debug("这是一条调试信息")
    log.info("这是一条信息")
    log.warning("这是一条警告")
    log.error("这是一条错误")

    # 记录异常
    try:
        1 / 0
    except Exception as e:
        log_exception(e, "除零错误")

    # 记录结构化数据
    log.info("用户登录成功: {user_id}", user_id=12345)

    # 不同模块使用不同名称的日志记录器
    db_log = get_logger("database")
    db_log.info("数据库连接成功")
