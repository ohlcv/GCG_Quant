# settings.py - 配置设置模块

import os
import yaml
from typing import Dict, Any, Optional

# 默认配置
DEFAULT_CONFIG = {
    # 数据采集配置
    "data_collector": {
        "exchange": {
            "exchange_id": "binance",
            "api_key": "",
            "secret": "",
            "timeout": 30000,
            "use_websocket": False,  # 学习点：新增WebSocket配置选项，第一阶段使用轮询，第二阶段可升级为WebSocket
        },
        "file_import": {
            "data_dir": "./data/raw"
        },
        "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
    },
    
    # 数据存储配置
    "data_storage": {
        # 学习点：增加数据库类型配置，支持SQLite和TimescaleDB的切换
        "db_type": "sqlite",  # 可选值: "sqlite", "timescaledb"
        "sqlite": {
            "db_file": "./data/gcg_quant.db"
        },
        "timescale": {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "postgres",
            "database": "gcg_quant"
        },
        # 学习点：增加Redis启用选项，可以在不需要实时缓存时禁用
        "use_redis": True,
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": ""
        },
        # 学习点：批量处理配置，控制每批次处理的数据量，避免内存压力
        "batch_size": 1000
    },
    
    # 日志配置
    "logging": {
        "level": "INFO",  # 学习点：可在运行时动态调整的日志级别
        "json_format": False,
        "retention_days": 30
    }
}

def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置，如果未指定配置文件，则使用默认配置
    
    Args:
        config_file: 配置文件路径，默认为None
        
    Returns:
        Dict[str, Any]: 配置字典
    """
    # 学习点：深拷贝配置，避免修改默认配置
    config = DEFAULT_CONFIG.copy()
    
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                
            # 递归合并配置
            merge_configs(config, user_config)
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
    
    return config

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并配置字典
    
    Args:
        base: 基础配置
        override: 覆盖配置
        
    Returns:
        Dict[str, Any]: 合并后的配置
    """
    # 学习点：递归合并字典，保留所有层级的配置
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            merge_configs(base[key], value)
        else:
            base[key] = value
    
    return base

def save_config(config: Dict[str, Any], config_file: str) -> bool:
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        config_file: 配置文件路径
        
    Returns:
        bool: 保存是否成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return True
    except Exception as e:
        print(f"保存配置文件失败: {str(e)}")
        return False