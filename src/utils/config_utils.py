# config_utils.py - 配置工具模块

"""
文件说明：
    这个文件提供了GCG_Quant系统的配置管理增强功能。
    扩展了基础配置管理，支持环境变量覆盖、配置验证和敏感信息保护等功能。
    系统中所有组件可以使用这里的工具函数来读取、验证和管理配置。

学习目标：
    1. 了解配置管理的最佳实践
    2. 学习如何使用环境变量增强配置安全性
    3. 掌握配置验证和合并的技术
"""

import os
import yaml
import json
import re
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path


def load_yaml_config(config_file: str) -> Dict[str, Any]:
    """
    加载YAML格式的配置文件

    Args:
        config_file: 配置文件路径

    Returns:
        Dict[str, Any]: 配置字典

    Raises:
        FileNotFoundError: 如果配置文件不存在
        yaml.YAMLError: 如果YAML解析失败
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def load_json_config(config_file: str) -> Dict[str, Any]:
    """
    加载JSON格式的配置文件

    Args:
        config_file: 配置文件路径

    Returns:
        Dict[str, Any]: 配置字典

    Raises:
        FileNotFoundError: 如果配置文件不存在
        json.JSONDecodeError: 如果JSON解析失败
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def save_yaml_config(config: Dict[str, Any], config_file: str) -> bool:
    """
    保存配置到YAML文件

    Args:
        config: 配置字典
        config_file: 配置文件路径

    Returns:
        bool: 保存是否成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)

        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)

        return True
    except Exception as e:
        print(f"保存配置文件失败: {str(e)}")
        return False


def save_json_config(config: Dict[str, Any], config_file: str, indent: int = 4) -> bool:
    """
    保存配置到JSON文件

    Args:
        config: 配置字典
        config_file: 配置文件路径
        indent: JSON缩进，默认为4

    Returns:
        bool: 保存是否成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=indent)

        return True
    except Exception as e:
        print(f"保存配置文件失败: {str(e)}")
        return False


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并两个配置字典，override覆盖base的值

    Args:
        base: 基础配置
        override: 覆盖配置

    Returns:
        Dict[str, Any]: 合并后的配置
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 如果两边都是字典，递归合并
            result[key] = deep_merge(result[key], value)
        else:
            # 否则直接覆盖
            result[key] = value

    return result


def apply_env_override(config: Dict[str, Any], prefix: str = "GCG_") -> Dict[str, Any]:
    """
    应用环境变量覆盖配置

    环境变量格式: PREFIX_SECTION_KEY=value
    例如: GCG_DATABASE_HOST=localhost 将覆盖 config['database']['host']

    Args:
        config: 配置字典
        prefix: 环境变量前缀

    Returns:
        Dict[str, Any]: 应用环境变量后的配置字典
    """
    result = config.copy()

    # 获取所有匹配前缀的环境变量
    env_vars = {k: v for k, v in os.environ.items() if k.startswith(prefix)}

    for env_name, env_value in env_vars.items():
        # 移除前缀
        key_path = env_name[len(prefix) :].lower()

        # 分割路径，例如 DATABASE_HOST -> ['database', 'host']
        parts = key_path.split("_")

        # 从根开始导航
        current = result

        # 导航到最后一个部分前
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # 最后一个部分，设置值
                current[part] = _convert_env_value(env_value)
            else:
                # 如果路径不存在，创建一个新的字典
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}

                current = current[part]

    return result


def _convert_env_value(value: str) -> Any:
    """
    转换环境变量值为适当的类型

    Args:
        value: 环境变量值

    Returns:
        Any: 转换后的值
    """
    # 布尔值
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False

    # 数字
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # 列表 (逗号分隔)
    if "," in value:
        return [_convert_env_value(item.strip()) for item in value.split(",")]

    # 默认为字符串
    return value


def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """
    验证配置是否符合模式定义

    Args:
        config: 要验证的配置字典
        schema: 模式定义，格式为:
            {
                "key": {
                    "type": "string|int|float|bool|list|dict",  # 必需
                    "required": True|False,  # 可选，默认False
                    "default": value,  # 可选
                    "choices": [value1, value2, ...],  # 可选
                    "min": min_value,  # 可选，用于数字
                    "max": max_value,  # 可选，用于数字
                    "regex": "pattern",  # 可选，用于字符串
                    "schema": {...}  # 可选，用于嵌套dict
                    "item_schema": {...}  # 可选，用于list的每个元素
                }
            }

    Returns:
        List[str]: 错误消息列表，如果为空则验证通过
    """
    errors = []
    _validate_dict_against_schema(config, schema, errors, path=[])
    return errors


def _validate_dict_against_schema(
    config: Dict[str, Any], schema: Dict[str, Any], errors: List[str], path: List[str]
) -> None:
    """
    递归验证字典配置

    Args:
        config: 要验证的配置字典
        schema: 模式定义
        errors: 错误消息列表，将添加新错误
        path: 当前路径，用于生成错误消息
    """
    # 检查必需字段
    for key, field_schema in schema.items():
        field_path = path + [key]
        field_path_str = ".".join(field_path)

        # 检查字段是否存在
        if key not in config:
            if field_schema.get("required", False):
                errors.append(f"缺少必需字段: {field_path_str}")

            # 如果有默认值，添加到配置中
            if "default" in field_schema:
                config[key] = field_schema["default"]

            # 字段不存在，不需要进一步验证
            continue

        # 获取字段值和期望类型
        value = config[key]
        expected_type = field_schema.get("type")

        # 验证类型
        if expected_type:
            type_valid = False

            if expected_type == "string" and isinstance(value, str):
                type_valid = True
            elif (
                expected_type == "int"
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                type_valid = True
            elif (
                expected_type == "float"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                type_valid = True
            elif expected_type == "bool" and isinstance(value, bool):
                type_valid = True
            elif expected_type == "list" and isinstance(value, list):
                type_valid = True
            elif expected_type == "dict" and isinstance(value, dict):
                type_valid = True

            if not type_valid:
                errors.append(
                    f"字段 {field_path_str} 应为 {expected_type} 类型，但得到 {type(value).__name__}"
                )
                continue

        # 验证选项值
        if "choices" in field_schema and value not in field_schema["choices"]:
            choices_str = ", ".join(str(c) for c in field_schema["choices"])
            errors.append(
                f"字段 {field_path_str} 值 {value} 不在允许的选项中: [{choices_str}]"
            )

        # 验证数字范围
        if expected_type in ("int", "float"):
            if "min" in field_schema and value < field_schema["min"]:
                errors.append(
                    f"字段 {field_path_str} 值 {value} 小于最小值 {field_schema['min']}"
                )

            if "max" in field_schema and value > field_schema["max"]:
                errors.append(
                    f"字段 {field_path_str} 值 {value} 大于最大值 {field_schema['max']}"
                )

        # 验证字符串正则匹配
        if expected_type == "string" and "regex" in field_schema:
            if not re.match(field_schema["regex"], value):
                errors.append(
                    f"字段 {field_path_str} 值 {value} 不匹配模式 {field_schema['regex']}"
                )

        # 递归验证嵌套字典
        if expected_type == "dict" and "schema" in field_schema:
            _validate_dict_against_schema(
                value, field_schema["schema"], errors, field_path
            )

        # 验证列表中的每个元素
        if expected_type == "list" and "item_schema" in field_schema:
            for i, item in enumerate(value):
                if not isinstance(item, dict):
                    errors.append(f"字段 {field_path_str}[{i}] 应为字典类型")
                    continue

                _validate_dict_against_schema(
                    item, field_schema["item_schema"], errors, field_path + [f"[{i}]"]
                )


def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    根据路径获取配置值

    Args:
        config: 配置字典
        path: 配置路径，使用.分隔，例如 "database.host"
        default: 默认值，如果路径不存在则返回该值

    Returns:
        Any: 配置值或默认值
    """
    if not path:
        return default

    parts = path.split(".")
    current = config

    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default

        current = current[part]

    return current


def set_config_value(config: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    """
    根据路径设置配置值

    Args:
        config: 配置字典
        path: 配置路径，使用.分隔，例如 "database.host"
        value: 要设置的值

    Returns:
        Dict[str, Any]: 更新后的配置字典
    """
    if not path:
        return config

    result = config.copy()
    parts = path.split(".")
    current = result

    # 导航到最后一个部分前
    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            # 最后一个部分，设置值
            current[part] = value
        else:
            # 如果路径不存在，创建一个新的字典
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}

            current = current[part]

    return result


# 示例使用
if __name__ == "__main__":
    # 示例配置
    config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "password",
        },
        "logging": {"level": "INFO", "file": "app.log"},
    }

    # 示例模式
    schema = {
        "database": {
            "type": "dict",
            "required": True,
            "schema": {
                "host": {"type": "string", "required": True},
                "port": {"type": "int", "required": True, "min": 1, "max": 65535},
                "user": {"type": "string", "required": True},
                "password": {"type": "string", "required": True},
            },
        },
        "logging": {
            "type": "dict",
            "required": False,
            "schema": {
                "level": {
                    "type": "string",
                    "required": False,
                    "default": "INFO",
                    "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                },
                "file": {"type": "string", "required": False},
            },
        },
        "api": {
            "type": "dict",
            "required": False,
            "default": {"enabled": False},
            "schema": {"enabled": {"type": "bool", "required": True}},
        },
    }

    # 验证配置
    errors = validate_config(config, schema)
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("配置验证通过")

    # 使用环境变量覆盖配置
    os.environ["GCG_DATABASE_HOST"] = "db.example.com"
    os.environ["GCG_LOGGING_LEVEL"] = "DEBUG"

    config = apply_env_override(config)
    print(f"应用环境变量后的配置: {config}")

    # 获取和设置配置值
    db_host = get_config_value(config, "database.host")
    print(f"数据库主机: {db_host}")

    config = set_config_value(config, "database.port", 5433)
    print(f"更新后的配置: {config}")
