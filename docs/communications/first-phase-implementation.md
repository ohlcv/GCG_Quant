# GCG_Quant项目第一阶段实施细则

## 一、总体规划

根据之前达成的融合方案，GCG_Quant项目将采用Grok提出的六大模块清晰划分（数据采集、存储、分析、交易执行、可视化、日志），同时在内部结构上参考Claude的细分设计。第一阶段将优先实现**数据采集**和**数据存储**模块，为后续功能奠定基础。

## 二、第一阶段目标

1. 构建基础项目结构
2. 实现数据采集模块
3. 实现数据存储模块
4. 实现基础日志系统
5. 构建简单的测试与验证程序

## 三、具体实施细则

### 1. 目录结构设计

第一阶段采用简化的目录结构，同时为后续扩展预留空间：

```
GCG_Quant/
├── docs/                         # 项目文档
│   ├── architecture/             # 架构文档
│   ├── communications/           # 交流记录
│   └── README.md                 # 文档索引
├── src/                          # 源代码
│   ├── config/                   # 配置管理
│   │   ├── settings.py           # 配置设置
│   │   └── constants.py          # 常量定义
│   ├── data_collector/           # 数据采集模块
│   │   ├── __init__.py
│   │   ├── base_collector.py     # 基础采集器接口
│   │   ├── exchange_collector.py # 交易所数据采集器
│   │   └── file_importer.py      # 文件数据导入器
│   ├── data_storage/             # 数据存储模块
│   │   ├── __init__.py
│   │   ├── timescale_manager.py  # TimescaleDB管理
│   │   ├── redis_manager.py      # Redis管理
│   │   └── models.py             # 数据模型定义
│   ├── utils/                    # 工具类
│   │   ├── __init__.py
│   │   ├── logger.py             # 日志工具(Loguru)
│   │   └── time_utils.py         # 时间工具函数
│   └── main.py                   # 主程序入口
├── tests/                        # 测试目录
│   ├── __init__.py
│   ├── test_collector.py         # 数据采集测试
│   └── test_storage.py           # 数据存储测试
├── .gitignore                    # Git忽略文件
├── requirements.txt              # 依赖管理
├── README.md                     # 项目介绍
└── setup.py                      # 安装脚本
```

### 2. 数据采集模块设计

#### 2.1 基础采集器接口 (base_collector.py)

简化后的BaseCollector接口，仅包含基本的数据获取功能：

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

class BaseCollector(ABC):
    """数据采集器基类，定义所有采集器必须实现的接口"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化采集器"""
        self.config = config
        
    @abstractmethod
    async def connect(self) -> bool:
        """连接到数据源"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """断开与数据源的连接"""
        pass
    
    @abstractmethod
    async def fetch_tick_data(self, symbol: str, 
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """获取Tick数据"""
        pass
    
    @abstractmethod
    async def fetch_kline_data(self, symbol: str, timeframe: str,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """获取K线数据"""
        pass
```

### 3. 数据存储模块设计

#### 3.1 数据模型定义 (models.py)

```python
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class TickData:
    """Tick数据模型"""
    symbol: str  # 交易品种符号
    timestamp: int  # 时间戳（毫秒）
    datetime: datetime  # 日期时间
    price: float  # 价格
    amount: float  # 数量/成交量
    side: str  # 方向，'buy'或'sell'
    source: str  # 数据源
    trade_id: Optional[str] = None  # 交易ID，可能为空

@dataclass
class KlineData:
    """K线数据模型"""
    symbol: str  # 交易品种符号
    timestamp: int  # 时间戳（毫秒）
    datetime: datetime  # 日期时间
    timeframe: str  # 时间周期，如'1m', '5m', '1h', '1d'等
    open: float  # 开盘价
    high: float  # 最高价
    low: float  # 最低价
    close: float  # 收盘价
    volume: float  # 成交量
    source: str  # 数据源
```

## 四、依赖管理

### requirements.txt

```
# 核心依赖
asyncio>=3.4.3
aiohttp>=3.8.1

# 数据采集
ccxt>=1.90.0
pandas>=1.3.5
numpy>=1.21.5

# 数据存储
asyncpg>=0.25.0
aioredis>=2.0.1

# 配置管理
pyyaml>=6.0

# 日志
loguru>=0.6.0

# 工具
python-dateutil>=2.8.2

# 测试
pytest>=7.0.0
pytest-asyncio>=0.18.3
```

## 五、第一阶段实施计划

### 1. 环境准备 (预计1天)

1. 安装Python环境（Python 3.9+）
2. 安装PostgreSQL和TimescaleDB扩展
3. 安装Redis服务
4. 克隆项目仓库并创建开发分支
5. 安装项目依赖

### 2. 数据采集模块实现 (预计2天)

1. 实现基础采集器接口
2. 实现交易所数据采集器
3. 实现文件数据导入器
4. 编写单元测试
5. 进行集成测试

### 3. 数据存储模块实现 (预计2天)

1. 实现数据模型定义
2. 实现TimescaleDB管理器
3. 实现Redis管理器
4. 编写单元测试
5. 进行集成测试

### 4. 工具和配置模块实现 (预计1天)

1. 实现日志工具
2. 实现配置管理
3. 定义常量
4. 进行各模块集成测试

### 5. 主程序实现 (预计1天)

1. 实现主程序类
2. 实现命令行接口
3. 进行端到端测试
4. 修复问题和优化

### 6. 文档和示例 (预计1天)

1. 编写详细文档
2. 创建示例配置
3. 编写使用教程
4. 准备演示

## 六、下一步计划

完成第一阶段实施后，将进入第二阶段，开始实现数据分析模块，重点整合chan.py进行缠论分析。同时，将扩展数据采集和存储模块，支持更多数据源和更复杂的查询。

## 七、优化说明

1. 简化了BaseCollector接口，将订阅功能推迟到第二阶段实现
2. 优化了数据存储结构，使用Redis Hash结构减少键数量
3. 增加了批量处理支持，提高数据处理效率
4. 添加了详细的学习点注释，便于理解和学习
5. 预留了WebSocket接口，为第二阶段实现做准备