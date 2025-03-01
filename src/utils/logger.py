# logger.py - 日志工具

"""
文件说明：
    这个文件提供了GCG_Quant系统的日志功能，基于Python标准库的logging模块。
    它设置了统一的日志格式，并支持输出到控制台和文件，方便调试和问题排查。
    系统中所有组件都使用这里定义的日志工具来记录日志。

学习目标：
    1. 了解Python日志系统的设计和使用
    2. 学习如何配置日志格式、级别和处理器
    3. 掌握在大型项目中统一管理日志的最佳实践
"""

import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# 导入常量
from ..config.constants import (
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_CRITICAL,
)


def setup_logger(name="gcg_quant", level=None, log_file=None):
    """
    设置日志记录器

    Args:
        name: 日志记录器名称，默认为"gcg_quant"
        level: 日志级别，默认为INFO
        log_file: 日志文件路径，默认为None（不记录到文件）

    Returns:
        Logger: 配置好的日志记录器

    学习点：
    - 日志级别控制输出详细程度
    - 多个处理器可以同时输出到不同目标
    - 统一的日志格式简化日志分析
    """
    # 设置默认日志级别
    if level is None:
        level = logging.INFO

    # 创建或获取日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 如果日志记录器已经有处理器，说明已经配置过，直接返回
    if logger.handlers:
        return logger

    # 创建日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 如果提供了日志文件路径，添加文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 创建按大小轮转的文件处理器（最大10MB，保存5个备份）
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name="gcg_quant"):
    """
    获取已配置的日志记录器，如果不存在则新建一个

    Args:
        name: 日志记录器名称，默认为"gcg_quant"

    Returns:
        Logger: 日志记录器

    学习点：
    - 复用已配置的日志记录器，避免重复配置
    - 子模块可以创建独立的日志记录器，方便分析问题
    """
    # 获取已存在的日志记录器，如果不存在则新建一个
    logger = logging.getLogger(name)

    # 如果日志记录器没有配置过，使用默认配置
    if not logger.handlers:
        # 获取当前时间作为日志文件名的一部分
        today = datetime.now().strftime("%Y%m%d")

        # 创建日志目录
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # 构建日志文件路径
        log_file = os.path.join(log_dir, f"{today}_{name}.log")

        # 设置日志记录器
        logger = setup_logger(name, level=logging.INFO, log_file=log_file)

    return logger


def setup_daily_logger(name="gcg_quant", level=None, log_dir=None):
    """
    设置每日轮转的日志记录器

    Args:
        name: 日志记录器名称，默认为"gcg_quant"
        level: 日志级别，默认为INFO
        log_dir: 日志目录，默认为当前目录下的logs目录

    Returns:
        Logger: 配置好的日志记录器

    学习点：
    - 日志按日期轮转，便于管理长期运行的系统
    - 统一的日志存储位置，便于查找和分析
    """
    # 设置默认日志级别
    if level is None:
        level = logging.INFO

    # 设置默认日志目录
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")

    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 创建或获取日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 如果日志记录器已经有处理器，说明已经配置过，直接返回
    if logger.handlers:
        return logger

    # 创建日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 构建日志文件路径
    log_file = os.path.join(log_dir, f"{name}.log")

    # 创建按日期轮转的文件处理器（每天轮转，保存30天的日志）
    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=30
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
