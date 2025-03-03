### 2.3 文件数据导入器 (file_importer.py) (续)

```python
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
        symbol_safe = symbol.replace('/', '_')
        return os.path.join(self.data_dir, f"kline_{symbol_safe}_{timeframe}.csv")
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TickData':
        """从字典创建Tick数据对象"""
        # 确保datetime字段是datetime对象
        if isinstance(data.get('datetime'), str):
            data['datetime'] = datetime.fromisoformat(data['datetime'])
        
        return cls(
            symbol=data.get('symbol', ''),
            timestamp=data.get('timestamp', 0),
            datetime=data.get('datetime', datetime.now()),
            price=float(data.get('price', 0.0)),
            amount=float(data.get('amount', 0.0)),
            side=data.get('side', ''),
            source=data.get('source', ''),
            trade_id=data.get('id', None)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'datetime': self.datetime.isoformat(),
            'price': self.price,
            'amount': self.amount,
            'side': self.side,
            'source': self.source,
            'trade_id': self.trade_id
        }

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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KlineData':
        """从字典创建K线数据对象"""
        # 确保datetime字段是datetime对象
        if isinstance(data.get('datetime'), str):
            data['datetime'] = datetime.fromisoformat(data['datetime'])
        
        return cls(
            symbol=data.get('symbol', ''),
            timestamp=data.get('timestamp', 0),
            datetime=data.get('datetime', datetime.now()),
            timeframe=data.get('timeframe', '1m'),
            open=float(data.get('open', 0.0)),
            high=float(data.get('high', 0.0)),
            low=float(data.get('low', 0.0)),
            close=float(data.get('close', 0.0)),
            volume=float(data.get('volume', 0.0)),
            source=data.get('source', '')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'datetime': self.datetime.isoformat(),
            'timeframe': self.timeframe,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'source': self.source
        }
```

#### 3.2 TimescaleDB管理器 (timescale_manager.py)

```python
import asyncio
import asyncpg
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json

from ..utils.logger import get_logger
from .models import TickData, KlineData

logger = get_logger("TimescaleManager")

class TimescaleManager:
    """TimescaleDB管理器，用于存储和查询时间序列数据"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化TimescaleDB管理器
        
        Args:
            config: 配置参数，至少包含：
                - host: 数据库主机
                - port: 数据库端口
                - user: 数据库用户名
                - password: 数据库密码
                - database: 数据库名称
        """
        self.config = config
        self.pool = None
        
    async def connect(self) -> bool:
        """
        连接到TimescaleDB
        
        Returns:
            bool: 连接是否成功
        """
        try:
            logger.info("连接到TimescaleDB")
            
            self.pool = await asyncpg.create_pool(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 5432),
                user=self.config.get('user', 'postgres'),
                password=self.config.get('password', 'postgres'),
                database=self.config.get('database', 'gcg_quant')
            )
            
            # 初始化数据库表和超表
            await self._init_database()
            
            logger.info("成功连接到TimescaleDB")
            return True
        except Exception as e:
            logger.error(f"连接到TimescaleDB失败: {str(e)}")
            return False
    
    async def disconnect(self) -> bool:
        """
        断开与TimescaleDB的连接
        
        Returns:
            bool: 断开连接是否成功
        """
        try:
            if self.pool:
                logger.info("断开与TimescaleDB的连接")
                await self.pool.close()
                logger.info("成功断开与TimescaleDB的连接")
            return True
        except Exception as e:
            logger.error(f"断开与TimescaleDB的连接失败: {str(e)}")
            return False
    
    async def _init_database(self) -> bool:
        """
        初始化数据库表和超表
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            async with self.pool.acquire() as conn:
                # 创建TimescaleDB扩展（如果不存在）
                await conn.execute('CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;')
                
                # 创建Tick数据表
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS tick_data (
                        time TIMESTAMPTZ NOT NULL,
                        symbol TEXT NOT NULL,
                        price DOUBLE PRECISION NOT NULL,
                        amount DOUBLE PRECISION NOT NULL,
                        side TEXT NOT NULL,
                        source TEXT NOT NULL,
                        trade_id TEXT
                    );
                ''')
                
                # 创建K线数据表
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS kline_data (
                        time TIMESTAMPTZ NOT NULL,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        open DOUBLE PRECISION NOT NULL,
                        high DOUBLE PRECISION NOT NULL,
                        low DOUBLE PRECISION NOT NULL,
                        close DOUBLE PRECISION NOT NULL,
                        volume DOUBLE PRECISION NOT NULL,
                        source TEXT NOT NULL
                    );
                ''')
                
                # 将表转换为超表（如果尚未转换）
                # 超表是TimescaleDB的特殊表类型，用于高效处理时间序列数据
                try:
                    await conn.execute('''
                        SELECT create_hypertable('tick_data', 'time', if_not_exists => TRUE);
                    ''')
                    
                    await conn.execute('''
                        SELECT create_hypertable('kline_data', 'time', if_not_exists => TRUE);
                    ''')
                except Exception as e:
                    logger.warning(f"创建超表时发生警告（可能已存在）: {str(e)}")
                
                # 创建索引以加速查询
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_tick_symbol ON tick_data (symbol);
                ''')
                
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_kline_symbol_timeframe ON kline_data (symbol, timeframe);
                ''')
                
                logger.info("成功初始化TimescaleDB表和索引")
                return True
        except Exception as e:
            logger.error(f"初始化TimescaleDB失败: {str(e)}")
            return False
    
    async def save_tick_data(self, data: Union[TickData, List[TickData]]) -> bool:
        """
        保存Tick数据
        
        Args:
            data: 单个Tick数据对象或Tick数据对象列表
            
        Returns:
            bool: 保存是否成功
        """
        if not self.pool:
            logger.error("保存Tick数据失败: 未连接到TimescaleDB")
            return False
        
        # 转换为列表以统一处理
        data_list = data if isinstance(data, list) else [data]
        if not data_list:
            return True
        
        try:
            async with self.pool.acquire() as conn:
                # 批量插入数据
                values = [
                    (item.datetime, item.symbol, item.price, 
                     item.amount, item.side, item.source, item.trade_id)
                    for item in data_list
                ]
                
                await conn.executemany('''
                    INSERT INTO tick_data 
                    (time, symbol, price, amount, side, source, trade_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                ''', values)
                
                logger.info(f"成功保存 {len(data_list)} 条Tick数据")
                return True
        except Exception as e:
            logger.error(f"保存Tick数据失败: {str(e)}")
            return False
    
    async def save_kline_data(self, data: Union[KlineData, List[KlineData]]) -> bool:
        """
        保存K线数据
        
        Args:
            data: 单个K线数据对象或K线数据对象列表
            
        Returns:
            bool: 保存是否成功
        """
        if not self.pool:
            logger.error("保存K线数据失败: 未连接到TimescaleDB")
            return False
        
        # 转换为列表以统一处理
        data_list = data if isinstance(data, list) else [data]
        if not data_list:
            return True
        
        try:
            async with self.pool.acquire() as conn:
                # 批量插入数据
                values = [
                    (item.datetime, item.symbol, item.timeframe, 
                     item.open, item.high, item.low, item.close, 
                     item.volume, item.source)
                    for item in data_list
                ]
                
                await conn.executemany('''
                    INSERT INTO kline_data 
                    (time, symbol, timeframe, open, high, low, close, volume, source)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ''', values)
                
                logger.info(f"成功保存 {len(data_list)} 条K线数据")
                return True
        except Exception as e:
            logger.error(f"保存K线数据失败: {str(e)}")
            return False
    
    async def query_tick_data(self, symbol: str, 
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None,
                             limit: int = 1000) -> List[TickData]:
        """
        查询Tick数据
        
        Args:
            symbol: 交易品种符号
            start_time: 开始时间，默认为None（不限制）
            end_time: 结束时间，默认为None（不限制）
            limit: 返回结果数量限制，默认1000条
            
        Returns:
            List[TickData]: Tick数据列表
        """
        if not self.pool:
            logger.error("查询Tick数据失败: 未连接到TimescaleDB")
            return []
        
        try:
            async with self.pool.acquire() as conn:
                query = '''
                    SELECT time, symbol, price, amount, side, source, trade_id
                    FROM tick_data
                    WHERE symbol = $1
                '''
                
                params = [symbol]
                param_idx = 2
                
                if start_time:
                    query += f" AND time >= ${param_idx}"
                    params.append(start_time)
                    param_idx += 1
                
                if end_time:
                    query += f" AND time <= ${param_idx}"
                    params.append(end_time)
                    param_idx += 1
                
                query += " ORDER BY time DESC LIMIT $" + str(param_idx)
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                # 转换为TickData对象
                result = []
                for row in rows:
                    tick = TickData(
                        symbol=row['symbol'],
                        timestamp=int(row['time'].timestamp() * 1000),
                        datetime=row['time'],
                        price=float(row['price']),
                        amount=float(row['amount']),
                        side=row['side'],
                        source=row['source'],
                        trade_id=row['trade_id']
                    )
                    result.append(tick)
                
                logger.info(f"成功查询 {symbol} 的Tick数据，共 {len(result)} 条")
                return result
        except Exception as e:
            logger.error(f"查询Tick数据失败: {str(e)}")
            return []
    
    async def query_kline_data(self, symbol: str, timeframe: str,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              limit: int = 1000) -> List[KlineData]:
        """
        查询K线数据
        
        Args:
            symbol: 交易品种符号
            timeframe: 时间周期
            start_time: 开始时间，默认为None（不限制）
            end_time: 结束时间，默认为None（不限制）
            limit: 返回结果数量限制，默认1000条
            
        Returns:
            List[KlineData]: K线数据列表
        """
        if not self.pool:
            logger.error("查询K线数据失败: 未连接到TimescaleDB")
            return []
        
        try:
            async with self.pool.acquire() as conn:
                query = '''
                    SELECT time, symbol, timeframe, open, high, low, close, volume, source
                    FROM kline_data
                    WHERE symbol = $1 AND timeframe = $2
                '''
                
                params = [symbol, timeframe]
                param_idx = 3
                
                if start_time:
                    query += f" AND time >= ${param_idx}"
                    params.append(start_time)
                    param_idx += 1
                
                if end_time:
                    query += f" AND time <= ${param_idx}"
                    params.append(end_time)
                    param_idx += 1
                
                query += " ORDER BY time DESC LIMIT $" + str(param_idx)
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                # 转换为KlineData对象
                result = []
                for row in rows:
                    kline = KlineData(
                        symbol=row['symbol'],
                        timestamp=int(row['time'].timestamp() * 1000),
                        datetime=row['time'],
                        timeframe=row['timeframe'],
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=float(row['volume']),
                        source=row['source']
                    )
                    result.append(kline)
                
                logger.info(f"成功查询 {symbol} 的 {timeframe} K线数据，共 {len(result)} 条")
                return result
        except Exception as e:
            logger.error(f"查询K线数据失败: {str(e)}")
            return []
```

#### 3.3 Redis管理器 (redis_manager.py)

```python
import asyncio
from redis.asyncio import Redis as RedisAsync
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

from ..utils.logger import get_logger
from .models import TickData, KlineData

logger = get_logger("RedisManager")

class RedisManager:
    """Redis管理器，用于缓存实时数据和发布/订阅功能"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Redis管理器
        
        Args:
            config: 配置参数，至少包含：
                - host: Redis主机
                - port: Redis端口
                - db: Redis数据库索引
                - password: Redis密码（可选）
        """
        self.config = config
        self.redis = None
        self.pubsub = None
        
    async def connect(self) -> bool:
        """
        连接到Redis
        
        Returns:
            bool: 连接是否成功
        """
        try:
            logger.info("连接到Redis")
            
            # 构建Redis连接URL
            password = self.config.get('password', '')
            password_part = f":{password}@" if password else ""
            host = self.config.get('host', 'localhost')
            port = self.config.get('port', 6379)
            db = self.config.get('db', 0)
            
            redis_url = f"redis://{password_part}{host}:{port}/{db}"
            self.redis = await aioredis.from_url(redis_url)
            
            # 创建发布/订阅对象
            self.pubsub = self.redis.pubsub()
            
            logger.info("成功连接到Redis")
            return True
        except Exception as e:
            logger.error(f"连接到Redis失败: {str(e)}")
            return False
    
    async def disconnect(self) -> bool:
        """
        断开与Redis的连接
        
        Returns:
            bool: 断开连接是否成功
        """
        try:
            if self.redis:
                logger.info("断开与Redis的连接")
                await self.redis.close()
                logger.info("成功断开与Redis的连接")
            return True
        except Exception as e:
            logger.error(f"断开与Redis的连接失败: {str(e)}")
            return False
    
    async def save_tick_data(self, data: Union[TickData, List[TickData]]) -> bool:
        """
        保存Tick数据到Redis
        
        Args:
            data: 单个Tick数据对象或Tick数据对象列表
            
        Returns:
            bool: 保存是否成功
        """
        if not self.redis:
            logger.error("保存Tick数据失败: 未连接到Redis")
            return False
        
        # 转换为列表以统一处理
        data_list = data if isinstance(data, list) else [data]
        if not data_list:
            return True
        
        try:
            # 使用管道批量操作，提高效率
            pipe = self.redis.pipeline()
            
            for item in data_list:
                # 构建Redis键，格式: tick:{symbol}:{timestamp}
                key = f"tick:{item.symbol}:{item.timestamp}"
                
                # 将数据转换为JSON字符串
                value = json.dumps(item.to_dict())
                
                # 设置数据，并设置过期时间（例如1小时）
                expire_seconds = 3600  # 1小时
                pipe.set(key, value, ex=expire_seconds)
                
                # 将最新的Tick数据存储在单独的键中
                latest_key = f"tick:{item.symbol}:latest"
                pipe.set(latest_key, value)
                
                # 发布数据到对应的频道
                channel = f"tick:{item.symbol}"
                pipe.publish(channel, value)
            
            # 执行管道中的所有命令
            await pipe.execute()
            
            logger.info(f"成功保存 {len(data_list)} 条Tick数据到Redis")
            return True
        except Exception as e:
            logger.error(f"保存Tick数据到Redis失败: {str(e)}")
            return False
    
    async def save_kline_data(self, data: Union[KlineData, List[KlineData]]) -> bool:
        """
        保存K线数据到Redis
        
        Args:
            data: 单个K线数据对象或K线数据对象列表
            
        Returns:
            bool: 保存是否成功
        """
        if not self.redis:
            logger.error("保存K线数据失败: 未连接到Redis")
            return False
        
        # 转换为列表以统一处理
        data_list = data if isinstance(data, list) else [data]
        if not data_list:
            return True
        
        try:
            # 使用管道批量操作，提高效率
            pipe = self.redis.pipeline()
            
            for item in data_list:
                # 构建Redis键，格式: kline:{symbol}:{timeframe}:{timestamp}
                key = f"kline:{item.symbol}:{item.timeframe}:{item.timestamp}"
                
                # 将数据转换为JSON字符串
                value = json.dumps(item.to_dict())
                
                # 设置数据，并设置过期时间（根据时间周期设置不同的过期时间）
                expire_seconds = self._get_expire_seconds(item.timeframe)
                pipe.set(key, value, ex=expire_seconds)
                
                # 将最新的K线数据存储在单独的键中
                latest_key = f"kline:{item.symbol}:{item.timeframe}:latest"
                pipe.set(latest_key, value)
                
                # 发布数据到对应的频道
                channel = f"kline:{item.symbol}:{item.timeframe}"
                pipe.publish(channel, value)
            
            # 执行管道中的所有命令
            await pipe.execute()
            
            logger.info(f"成功保存 {len(data_list)} 条K线数据到Redis")
            return True
        except Exception as e:
            logger.error(f"保存K线数据到Redis失败: {str(e)}")
            return False
    
    async def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """
        获取最新的Tick数据
        
        Args:
            symbol: 交易品种符号
            
        Returns:
            Optional[TickData]: Tick数据对象，如果不存在则返回None
        """
        if not self.redis:
            logger.error("获取最新Tick数据失败: 未连接到Redis")
            return None
        
        try:
            # 获取最新Tick数据的键
            key = f"tick:{symbol}:latest"
            
            # 获取数据
            value = await self.redis.get(key)
            if not value:
                return None
            
            # 解析JSON数据
            data = json.loads(value)
            
            # 创建TickData对象
            tick = TickData.from_dict(data)
            
            return tick
        except Exception as e:
            logger.error(f"获取最新Tick数据失败: {str(e)}")
            return None
    
    async def get_latest_kline(self, symbol: str, timeframe: str) -> Optional[KlineData]:
        """
        获取最新的K线数据
        
        Args:
            symbol: 交易品种符号
            timeframe: 时间周期
            
        Returns:
            Optional[KlineData]: K线数据对象，如果不存在则返回None
        """
        if not self.redis:
            logger.error("获取最新K线数据失败: 未连接到Redis")
            return None
        
        try:
            # 获取最新K线数据的键
            key = f"kline:{symbol}:{timeframe}:latest"
            
            # 获取数据
            value = await self.redis.get(key)
            if not value:
                return None
            
            # 解析JSON数据
            data = json.loads(value)
            
            # 创建KlineData对象
            kline = KlineData.from_dict(data)
            
            return kline
        except Exception as e:
            logger.error(f"获取最新K线数据失败: {str(e)}")
            return None
    
    async def subscribe_tick(self, symbols: List[str], callback) -> bool:
        """
        订阅Tick数据
        
        Args:
            symbols: 交易品种符号列表
            callback: 数据回调函数
            
        Returns:
            bool: 订阅是否成功
        """
        if not self.redis or not self.pubsub:
            logger.error("订阅Tick数据失败: 未连接到Redis")
            return False
        
        try:
            # 订阅多个频道
            channels = [f"tick:{symbol}" for symbol in symbols]
            
            # 注册回调函数
            async def message_handler(message):
                # 解析消息
                if message['type'] == 'message':
                    data = json.loads(message['data'])
                    tick = TickData.from_dict(data)
                    await callback([tick])
            
            # 设置消息处理函数
            self.pubsub.psubscribe(**{channel: message_handler for channel in channels})
            
            # 启动监听任务
            asyncio.create_task(self._listen())
            
            logger.info(f"成功订阅 {symbols} 的Tick数据")
            return True
        except Exception as e:
            logger.error(f"订阅Tick数据失败: {str(e)}")
            return False
    
    async def subscribe_kline(self, symbols: List[str], timeframe: str, callback) -> bool:
        """
        订阅K线数据
        
        Args:
            symbols: 交易品种符号列表
            timeframe: 时间周期
            callback: 数据回调函数
            
        Returns:
            bool: 订阅是否成功
        """
        if not self.redis or not self.pubsub:
            logger.error("订阅K线数据失败: 未连接到Redis")
            return False
        
        try:
            # 订阅多个频道
            channels = [f"kline:{symbol}:{timeframe}" for symbol in symbols]
            
            # 注册回调函数
            async def message_handler(message):
                # 解析消息
                if message['type'] == 'message':
                    data = json.loads(message['data'])
                    kline = KlineData.from_dict(data)
                    await callback([kline])
            
            # 设置消息处理函数
            self.pubsub.psubscribe(**{channel: message_handler for channel in channels})
            
            # 启动监听任务
            asyncio.create_task(self._listen())
            
            logger.info(f"成功订阅 {symbols} 的 {timeframe} K线数据")
            return True
        except Exception as e:
            logger.error(f"订阅K线数据失败: {str(e)}")
            return False
    
    async def _listen(self):
        """监听Redis发布/订阅消息"""
        try:
            await self.pubsub.run()
        except asyncio.CancelledError:
            logger.info("Redis发布/订阅监听任务被取消")
        except Exception as e:
            logger.error(f"Redis发布/订阅监听发生错误: {str(e)}")
    
    def _get_expire_seconds(self, timeframe: str) -> int:
        """
        根据时间周期获取过期时间（秒）
        
        Args:
            timeframe: 时间周期
            
        Returns:
            int: 过期时间（秒）
        """
        # 根据时间周期设置不同的过期时间
        if timeframe == '1m':
            return 3600 * 24  # 1天
        elif timeframe == '5m':
            return 3600 * 24 * 3  # 3天
        