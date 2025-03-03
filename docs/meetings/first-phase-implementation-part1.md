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

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

class BaseCollector(ABC):
    """数据采集器基类，定义所有采集器必须实现的接口"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化采集器
        
        Args:
            config: 配置参数字典
        """
        self.config = config
        self.is_running = False
        
    @abstractmethod
    async def connect(self) -> bool:
        """
        连接到数据源
        
        Returns:
            bool: 连接是否成功
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        断开与数据源的连接
        
        Returns:
            bool: 断开连接是否成功
        """
        pass
    
    @abstractmethod
    async def fetch_tick_data(self, symbol: str, start_time: Optional[datetime] = None, 
                             end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        获取Tick数据
        
        Args:
            symbol: 交易品种符号
            start_time: 开始时间，默认为None（获取最新数据）
            end_time: 结束时间，默认为None
            
        Returns:
            List[Dict[str, Any]]: Tick数据列表
        """
        pass
    
    @abstractmethod
    async def fetch_kline_data(self, symbol: str, timeframe: str, 
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        获取K线数据
        
        Args:
            symbol: 交易品种符号
            timeframe: 时间周期，如'1m', '5m', '1h', '1d'等
            start_time: 开始时间，默认为None（获取最新数据）
            end_time: 结束时间，默认为None
            
        Returns:
            List[Dict[str, Any]]: K线数据列表
        """
        pass
    
    async def start(self) -> bool:
        """
        启动采集器
        
        Returns:
            bool: 启动是否成功
        """
        if self.is_running:
            return True
        
        connected = await self.connect()
        if connected:
            self.is_running = True
            return True
        return False
    
    async def stop(self) -> bool:
        """
        停止采集器
        
        Returns:
            bool: 停止是否成功
        """
        if not self.is_running:
            return True
        
        disconnected = await self.disconnect()
        if disconnected:
            self.is_running = False
            return True
        return False
```

#### 2.2 交易所数据采集器 (exchange_collector.py)

```python
import asyncio
import ccxt.async_support as ccxt
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import time
import pandas as pd

from .base_collector import BaseCollector
from ..utils.logger import get_logger

logger = get_logger("ExchangeCollector")

class ExchangeCollector(BaseCollector):
    """交易所数据采集器，用于从各交易所API获取数据"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化交易所采集器
        
        Args:
            config: 配置参数，至少包含：
                - exchange_id: 交易所ID，如'binance', 'okex'等
                - api_key: API密钥（可选）
                - secret: API密钥（可选）
                - timeout: 请求超时时间（可选）
        """
        super().__init__(config)
        
        # 根据配置创建ccxt交易所实例
        exchange_id = config.get('exchange_id', '').lower()
        
        # 检查是否支持该交易所
        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"不支持的交易所: {exchange_id}")
        
        # 创建交易所实例
        exchange_class = getattr(ccxt, exchange_id)
        
        # 提取交易所参数
        exchange_params = {
            'apiKey': config.get('api_key', ''),
            'secret': config.get('secret', ''),
            'timeout': config.get('timeout', 30000),  # 默认30秒
            'enableRateLimit': True,  # 启用请求频率限制
        }
        
        self.exchange = exchange_class(exchange_params)
        self.exchange_id = exchange_id
        
        # 订阅回调函数和任务
        self.tick_callbacks = {}
        self.kline_callbacks = {}
        self.subscription_tasks = []
        
    async def connect(self) -> bool:
        """
        连接到交易所，加载市场信息
        
        Returns:
            bool: 连接是否成功
        """
        try:
            logger.info(f"连接到交易所 {self.exchange_id}")
            await self.exchange.load_markets()
            logger.info(f"成功连接到交易所 {self.exchange_id}")
            return True
        except Exception as e:
            logger.error(f"连接到交易所 {self.exchange_id} 失败: {str(e)}")
            return False
    
    async def disconnect(self) -> bool:
        """
        断开与交易所的连接，关闭所有订阅
        
        Returns:
            bool: 断开连接是否成功
        """
        try:
            # 取消所有订阅任务
            for task in self.subscription_tasks:
                task.cancel()
            
            logger.info(f"正在断开与交易所 {self.exchange_id} 的连接")
            await self.exchange.close()
            logger.info(f"已断开与交易所 {self.exchange_id} 的连接")
            return True
        except Exception as e:
            logger.error(f"断开与交易所 {self.exchange_id} 的连接失败: {str(e)}")
            return False
    
    async def fetch_tick_data(self, symbol: str, start_time: Optional[datetime] = None, 
                             end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        获取Tick数据，由于多数交易所不支持历史Tick数据的获取，
        因此这里返回最新的Tick数据（通常是最新的交易）
        
        Args:
            symbol: 交易品种符号
            start_time: 开始时间（大多数交易所API不支持）
            end_time: 结束时间（大多数交易所API不支持）
            
        Returns:
            List[Dict[str, Any]]: Tick数据列表
        """
        try:
            normalized_symbol = self._normalize_symbol(symbol)
            logger.info(f"获取 {normalized_symbol} 的Tick数据")
            
            # 获取最新交易
            trades = await self.exchange.fetch_trades(normalized_symbol)
            
            # 转换为统一格式
            result = []
            for trade in trades:
                result.append({
                    'symbol': symbol,
                    'timestamp': trade['timestamp'],
                    'datetime': trade['datetime'],
                    'price': trade['price'],
                    'amount': trade['amount'],
                    'side': trade['side'],  # 'buy' or 'sell'
                    'source': self.exchange_id
                })
            
            logger.info(f"成功获取 {normalized_symbol} 的Tick数据，共 {len(result)} 条")
            return result
        except Exception as e:
            logger.error(f"获取 {symbol} 的Tick数据失败: {str(e)}")
            return []
    
    async def fetch_kline_data(self, symbol: str, timeframe: str, 
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        获取K线数据
        
        Args:
            symbol: 交易品种符号
            timeframe: 时间周期，如'1m', '5m', '1h', '1d'等
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[Dict[str, Any]]: K线数据列表
        """
        try:
            normalized_symbol = self._normalize_symbol(symbol)
            logger.info(f"获取 {normalized_symbol} 的 {timeframe} K线数据")
            
            # 检查交易所是否支持该时间周期
            if timeframe not in self.exchange.timeframes:
                logger.error(f"交易所 {self.exchange_id} 不支持 {timeframe} 时间周期")
                return []
            
            # 如果提供了开始时间，则转换为毫秒时间戳
            since = int(start_time.timestamp() * 1000) if start_time else None
            
            # 获取K线数据
            ohlcv = await self.exchange.fetch_ohlcv(normalized_symbol, timeframe, since)
            
            # 转换为统一格式
            result = []
            for candle in ohlcv:
                timestamp, open_price, high, low, close, volume = candle
                
                # 如果设置了结束时间，且当前K线时间超过结束时间，则跳过
                if end_time and timestamp > end_time.timestamp() * 1000:
                    continue
                    
                result.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'datetime': datetime.fromtimestamp(timestamp / 1000).isoformat(),
                    'timeframe': timeframe,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                    'source': self.exchange_id
                })
            
            logger.info(f"成功获取 {normalized_symbol} 的 {timeframe} K线数据，共 {len(result)} 条")
            return result
        except Exception as e:
            logger.error(f"获取 {symbol} 的 {timeframe} K线数据失败: {str(e)}")
            return []
    
    async def _poll_tick_data(self, symbols: List[str]):
        """
        轮询获取Tick数据并回调
        
        Args:
            symbols: 交易品种符号列表
        """
        logger.info(f"开始轮询 {symbols} 的Tick数据")
        
        # 记录上次获取的最新交易ID，避免重复
        last_trade_ids = {symbol: None for symbol in symbols}
        
        try:
            while True:
                for symbol in symbols:
                    if symbol in self.tick_callbacks:
                        try:
                            normalized_symbol = self._normalize_symbol(symbol)
                            trades = await self.exchange.fetch_trades(normalized_symbol)
                            
                            # 过滤出新的交易
                            new_trades = []
                            for trade in trades:
                                # 如果这是新的交易，则添加
                                if last_trade_ids[symbol] is None or trade['id'] > last_trade_ids[symbol]:
                                    new_trades.append({
                                        'symbol': symbol,
                                        'timestamp': trade['timestamp'],
                                        'datetime': trade['datetime'],
                                        'price': trade['price'],
                                        'amount': trade['amount'],
                                        'side': trade['side'],
                                        'source': self.exchange_id
                                    })
                            
                            # 更新最新交易ID
                            if trades and 'id' in trades[-1]:
                                last_trade_ids[symbol] = trades[-1]['id']
                            
                            # 如果有新交易，则回调
                            if new_trades:
                                logger.debug(f"获取到 {symbol} 的 {len(new_trades)} 条新Tick数据")
                                # 调用回调函数
                                callback = self.tick_callbacks[symbol]
                                await callback(new_trades)
                        except Exception as e:
                            logger.error(f"轮询 {symbol} 的Tick数据失败: {str(e)}")
                
                # 轮询间隔，根据需要调整
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info(f"停止轮询 {symbols} 的Tick数据")
        except Exception as e:
            logger.error(f"轮询Tick数据时发生错误: {str(e)}")
    
    async def _poll_kline_data(self, symbols: List[str], timeframe: str):
        """
        轮询获取K线数据并回调
        
        Args:
            symbols: 交易品种符号列表
            timeframe: 时间周期
        """
        logger.info(f"开始轮询 {symbols} 的 {timeframe} K线数据")
        
        # 记录上次获取的最新K线时间戳，避免重复
        last_candle_timestamps = {symbol: None for symbol in symbols}
        
        try:
            while True:
                for symbol in symbols:
                    key = f"{symbol}_{timeframe}"
                    if key in self.kline_callbacks:
                        try:
                            normalized_symbol = self._normalize_symbol(symbol)
                            ohlcv = await self.exchange.fetch_ohlcv(normalized_symbol, timeframe)
                            
                            # 过滤出新的K线
                            new_candles = []
                            for candle in ohlcv:
                                timestamp, open_price, high, low, close, volume = candle
                                
                                # 如果这是新的K线，则添加
                                if last_candle_timestamps[symbol] is None or timestamp > last_candle_timestamps[symbol]:
                                    new_candles.append({
                                        'symbol': symbol,
                                        'timestamp': timestamp,
                                        'datetime': datetime.fromtimestamp(timestamp / 1000).isoformat(),
                                        'timeframe': timeframe,
                                        'open': open_price,
                                        'high': high,
                                        'low': low,
                                        'close': close,
                                        'volume': volume,
                                        'source': self.exchange_id
                                    })
                            
                            # 更新最新K线时间戳
                            if ohlcv:
                                last_candle_timestamps[symbol] = ohlcv[-1][0]
                            
                            # 如果有新K线，则回调
                            if new_candles:
                                logger.debug(f"获取到 {symbol} 的 {len(new_candles)} 条新 {timeframe} K线数据")
                                # 调用回调函数
                                callback = self.kline_callbacks[key]
                                await callback(new_candles)
                        except Exception as e:
                            logger.error(f"轮询 {symbol} 的 {timeframe} K线数据失败: {str(e)}")
                
                # 轮询间隔，根据时间周期调整
                # 对于较短的时间周期如1m，可以缩短间隔
                poll_interval = 10  # 默认10秒
                if timeframe == '1m':
                    poll_interval = 5
                elif timeframe == '1h':
                    poll_interval = 30
                elif timeframe == '1d':
                    poll_interval = 60
                
                await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            logger.info(f"停止轮询 {symbols} 的 {timeframe} K线数据")
        except Exception as e:
            logger.error(f"轮询K线数据时发生错误: {str(e)}")
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        将通用的交易对符号格式转换为特定交易所的格式
        
        Args:
            symbol: 通用交易对符号，如'BTC/USDT'
            
        Returns:
            str: 交易所特定格式的交易对符号
        """
        # 如果交易所需要特定格式，可以在这里处理
        # 例如，有些交易所使用'BTCUSDT'而不是'BTC/USDT'
        # 默认情况下，保持原样
        return symbol
```

#### 2.3 文件数据导入器 (file_importer.py)

```python
import pandas as pd
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from .base_collector import BaseCollector
from ..utils.logger import get_logger

logger = get_logger("FileImporter")

class FileImporter(BaseCollector):
    """文件数据导入器，用于从CSV等文件导入历史数据"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化文件导入器
        
        Args:
            config: 配置参数，至少包含：
                - data_dir: 数据文件目录
        """
        super().__init__(config)
        self.data_dir = config.get('data_dir', './data')
        
    async def connect(self) -> bool:
        """
        检查数据目录是否存在
        
        Returns:
            bool: 检查是否成功
        """
        try:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            return True
        except Exception as e:
            logger.error(f"检查数据目录失败: {str(e)}")
            return False
    
    async def disconnect(self) -> bool:
        """
        断开连接（对于文件导入器而言，没有实际操作）
        
        Returns:
            bool: 操作是否成功
        """
        return True
    
    async def fetch_tick_data(self, symbol: str, start_time: Optional[datetime] = None, 
                             end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        从文件导入Tick数据
        
        Args:
            symbol: 交易品种符号，用于定位文件
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[Dict[str, Any]]: Tick数据列表
        """
        try:
            # 构建文件路径
            file_path = self._get_tick_file_path(symbol)
            
            if not os.path.exists(file_path):
                logger.error(f"Tick数据文件不存在: {file_path}")
                return []
            
            logger.info(f"从文件导入 {symbol} 的Tick数据: {file_path}")
            
            # 读取CSV文件
            df = pd.read_csv(file_path, parse_dates=['datetime'])
            
            # 根据时间范围过滤
            if start_time:
                df = df[df['datetime'] >= start_time]
            if end_time:
                df = df[df['datetime'] <= end_time]
            
            # 转换为字典列表
            result = df.to_dict('records')
            logger.info(f"成功导入 {symbol} 的Tick数据，共 {len(result)} 条")
            return result
        except Exception as e:
            logger.error(f"导入 {symbol} 的Tick数据失败: {str(e)}")
            return []
    
    async def fetch_kline_data(self, symbol: str, timeframe: str, 
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        从文件导入K线数据
        
        Args:
            symbol: 交易品种符号，用于定位文件
            timeframe: 时间周期，如'1m', '5m', '1h', '1d'等
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[Dict[str, Any]]: K线数据列表
        """
        try:
            # 构建文件路径
            file_path = self._get_kline_file_path(symbol, timeframe)
            
            if not os.path.exists(file_path):
                logger.error(f"K线数据文件不存在: {file_path}")
                return []
            
            logger.info(f"从文件导入 {symbol} 的 {timeframe} K线数据: {file_path}")
            
            # 读取CSV文件
            df = pd.read_csv(file_path, parse_dates=['datetime'])
            
            # 根据时间范围过滤
            if start_time:
                df = df[df['datetime'] >= start_time]
            if end_time:
                df = df[df['datetime'] <= end_time]
            
            # 转换为字典列表
            result = df.to_dict('records')
            logger.info(f"成功导入 {symbol} 的 {timeframe} K线数据，共 {len(result)} 条")
            return result
        except Exception as e:
            logger.error(f"导入 {symbol} 的 {timeframe} K线数据失败: {str(e)}")
            return []
    
    def _get_tick_file_path(self, symbol: str) -> str:
        """
        获取Tick数据文件路径
        
        Args:
            symbol: 交易品种符号
            
        Returns:
            str: 文件路径
        """
        # 将符号中的/替换为_
        symbol_safe = symbol.replace('/', '_')
        return os.path.join(self.data_dir, f"tick_{symbol_safe}.csv")
    
    def _get_kline_file_path(self, symbol: str, timeframe: str) -> str:
        """
        获取K线数据文件路径
        
        Args:
            symbol: 交易品种符号
            timeframe: 时间周期
            
        Returns:
            str: 文件路径
        """
        # 将符号中的/替换为_
        