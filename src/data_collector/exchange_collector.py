# exchange_collector.py - 交易所数据采集器

"""
文件说明：
    这个文件实现了交易所数据采集器(ExchangeCollector)，继承自base_collector.py中的BaseCollector抽象接口。
    它负责从各大交易所API获取实时的Tick数据和K线数据，支持多种交易品种和时间周期。
    采用异步编程模式，能高效处理大量并发请求，同时提供数据订阅功能。

学习目标：
    1. 了解如何使用ccxt库与各大交易所API交互
    2. 学习异步HTTP请求和WebSocket连接的使用方法
    3. 掌握实时数据采集和处理的技术
    4. 理解数据转换和规范化的重要性
"""

import asyncio
import ccxt.async_support as ccxt
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging
import json

from ..config.constants import (
    SUPPORTED_EXCHANGES,
    TRADE_SIDE_BUY,
    TRADE_SIDE_SELL,
    DATA_TYPE_TICK,
    DATA_TYPE_KLINE,
)
from ..utils.time_utils import now, now_ms, ms_to_datetime
from .base_collector import BaseCollector
from ..data_storage.models import TickData, KlineData

# 创建日志记录器
logger = logging.getLogger("ExchangeCollector")


class ExchangeCollector(BaseCollector):
    """
    交易所数据采集器，负责从交易所API获取实时数据
    
    学习点：
    - 继承抽象基类，必须实现所有抽象方法
    - 使用ccxt库统一处理不同交易所的API差异
    - 异步编程提高网络I/O密集型操作的效率
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化交易所数据采集器
        
        Args:
            config: 配置参数，至少包含：
                - exchange_id: 交易所ID，如'binance'
                - api_key: API密钥（可选）
                - secret: API密钥（可选）
                - timeout: 请求超时时间，单位毫秒
                - use_websocket: 是否使用WebSocket连接
        
        学习点：
        - 初始化时根据配置创建交易所实例
        - 使用字典存储不同的数据订阅
        """
        super().__init__(config)
        
        # 获取配置参数
        self.exchange_id = config.get('exchange_id', 'binance').lower()
        self.api_key = config.get('api_key', '')
        self.secret = config.get('secret', '')
        self.timeout = config.get('timeout', 30000)  # 默认30秒
        self.use_websocket = config.get('use_websocket', False)
        
        # 验证交易所ID
        if self.exchange_id not in SUPPORTED_EXCHANGES:
            raise ValueError(f"不支持的交易所: {self.exchange_id}，支持的交易所: {SUPPORTED_EXCHANGES}")
        
        # 交易所实例，延迟初始化
        self.exchange = None
        
        # 数据订阅
        self.tick_subscriptions = {}
        self.kline_subscriptions = {}
        
        # WebSocket连接
        self.ws_client = None
    
    async def connect(self) -> bool:
        """
        连接到交易所API
        
        Returns:
            bool: 连接是否成功
        
        学习点：
        - 使用ccxt库创建交易所实例，支持各大交易所
        - 设置API配置和连接参数
        - 异步连接确保应用响应性
        """
        try:
            logger.info(f"连接到交易所: {self.exchange_id}")
            
            # 创建交易所实例
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.secret,
                'timeout': self.timeout,
                'enableRateLimit': True,  # 启用请求频率限制，避免被禁止访问
                'options': {'defaultType': 'spot'}  # 默认使用现货市场
            })
            
            # 加载市场信息
            await self.exchange.load_markets()
            
            # 如果需要WebSocket连接，建立连接
            if self.use_websocket:
                # WebSocket连接的实现方式取决于具体交易所
                # 部分交易所可能需要单独的WebSocket库
                # TODO: 实现WebSocket连接
                logger.info("WebSocket功能尚未实现，使用HTTP轮询代替")
            
            logger.info(f"成功连接到交易所: {self.exchange_id}")
            return True
        except Exception as e:
            logger.error(f"连接到交易所失败: {str(e)}")
            return False
    
    async def disconnect(self) -> bool:
        """
        断开与交易所API的连接
        
        Returns:
            bool: 断开连接是否成功
        
        学习点：
        - 清理资源，关闭连接
        - 优雅关闭确保不留下资源泄漏
        """
        try:
            logger.info(f"断开与交易所的连接: {self.exchange_id}")
            
            if self.exchange:
                await self.exchange.close()
                self.exchange = None
            
            # 关闭WebSocket连接
            if self.ws_client:
                # TODO: 关闭WebSocket连接
                self.ws_client = None
            
            # 清理订阅
            self.tick_subscriptions = {}
            self.kline_subscriptions = {}
            
            logger.info(f"成功断开与交易所的连接: {self.exchange_id}")
            return True
        except Exception as e:
            logger.error(f"断开与交易所的连接失败: {str(e)}")
            return False
    
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
        
        学习点：
        - 从交易所API获取交易历史（Tick数据）
        - 时间参数处理和转换
        - 标准化不同交易所的数据格式
        """
        try:
            if not self.exchange:
                raise RuntimeError("未连接到交易所")
            
            logger.info(f"获取 {symbol} 的Tick数据")
            
            # 准备请求参数
            params = {}
            
            # 添加时间参数（如果提供）
            if start_time:
                params['since'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['limit'] = 1000  # 通常交易所限制每次请求的数量
            
            # 标准化交易品种符号
            exchange_symbol = self._format_symbol(symbol)
            
            # 获取交易历史
            trades = await self.exchange.fetch_trades(exchange_symbol, params=params)
            
            # 转换为标准格式
            result = []
            for trade in trades:
                tick = {
                    'symbol': symbol,
                    'timestamp': trade['timestamp'],
                    'datetime': ms_to_datetime(trade['timestamp']),
                    'price': float(trade['price']),
                    'amount': float(trade['amount']),
                    'side': TRADE_SIDE_BUY if trade['side'] == 'buy' else TRADE_SIDE_SELL,
                    'source': self.exchange_id,
                    'trade_id': trade.get('id')
                }
                result.append(tick)
            
            logger.info(f"成功获取 {symbol} 的Tick数据，共 {len(result)} 条")
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
            start_time: 开始时间，默认为None（获取最新数据）
            end_time: 结束时间，默认为None
            
        Returns:
            List[Dict[str, Any]]: K线数据列表
        
        学习点：
        - 从交易所API获取OHLCV数据（K线）
        - 处理不同时间周期的K线数据
        - 数据转换和时间对齐
        """
        try:
            if not self.exchange:
                raise RuntimeError("未连接到交易所")
            
            logger.info(f"获取 {symbol} 的 {timeframe} K线数据")
            
            # 准备请求参数
            params = {}
            
            # 添加时间参数（如果提供）
            if start_time:
                params['since'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['limit'] = 1000  # 通常交易所限制每次请求的数量
            
            # 标准化交易品种符号
            exchange_symbol = self._format_symbol(symbol)
            
            # 获取K线数据
            ohlcv = await self.exchange.fetch_ohlcv(exchange_symbol, timeframe, params=params)
            
            # 转换为标准格式
            result = []
            for candle in ohlcv:
                timestamp, open_price, high, low, close, volume = candle
                kline = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'datetime': ms_to_datetime(timestamp),
                    'timeframe': timeframe,
                    'open': float(open_price),
                    'high': float(high),
                    'low': float(low),
                    'close': float(close),
                    'volume': float(volume),
                    'source': self.exchange_id
                }
                result.append(kline)
            
            logger.info(f"成功获取 {symbol} 的 {timeframe} K线数据，共 {len(result)} 条")
            return result
        except Exception as e:
            logger.error(f"获取 {symbol} 的 {timeframe} K线数据失败: {str(e)}")
            return []
    
    async def subscribe_tick(self, symbols: List[str], callback: Callable) -> bool:
        """
        订阅Tick数据
        
        Args:
            symbols: 交易品种符号列表
            callback: 数据回调函数
            
        Returns:
            bool: 订阅是否成功
        
        学习点：
        - 订阅模式实现实时数据更新
        - 使用回调函数处理数据
        - WebSocket vs HTTP轮询的选择
        """
        try:
            if not self.exchange:
                raise RuntimeError("未连接到交易所")
            
            logger.info(f"订阅 {symbols} 的Tick数据")
            
            if self.use_websocket:
                # TODO: 实现WebSocket订阅
                logger.info("WebSocket订阅功能尚未实现，使用HTTP轮询代替")
            
            # 保存订阅信息
            for symbol in symbols:
                if symbol not in self.tick_subscriptions:
                    self.tick_subscriptions[symbol] = {
                        'callback': callback,
                        'task': None
                    }
                    
                    # 创建轮询任务
                    task = asyncio.create_task(self._poll_tick_data(symbol, callback))
                    self.tick_subscriptions[symbol]['task'] = task
            
            logger.info(f"成功订阅 {symbols} 的Tick数据")
            return True
        except Exception as e:
            logger.error(f"订阅 {symbols} 的Tick数据失败: {str(e)}")
            return False
    
    async def subscribe_kline(self, symbols: List[str], timeframe: str, callback: Callable) -> bool:
        """
        订阅K线数据
        
        Args:
            symbols: 交易品种符号列表
            timeframe: 时间周期
            callback: 数据回调函数
            
        Returns:
            bool: 订阅是否成功
        
        学习点：
        - 为不同时间周期的K线数据实现订阅
        - 处理多个交易品种的订阅
        - 轮询间隔根据时间周期调整
        """
        try:
            if not self.exchange:
                raise RuntimeError("未连接到交易所")
            
            logger.info(f"订阅 {symbols} 的 {timeframe} K线数据")
            
            if self.use_websocket:
                # TODO: 实现WebSocket订阅
                logger.info("WebSocket订阅功能尚未实现，使用HTTP轮询代替")
            
            # 保存订阅信息
            key = f"{timeframe}"
            if key not in self.kline_subscriptions:
                self.kline_subscriptions[key] = {}
            
            for symbol in symbols:
                if symbol not in self.kline_subscriptions[key]:
                    self.kline_subscriptions[key][symbol] = {
                        'callback': callback,
                        'task': None
                    }
                    
                    # 创建轮询任务
                    task = asyncio.create_task(self._poll_kline_data(symbol, timeframe, callback))
                    self.kline_subscriptions[key][symbol]['task'] = task
            
            logger.info(f"成功订阅 {symbols} 的 {timeframe} K线数据")
            return True
        except Exception as e:
            logger.error(f"订阅 {symbols} 的 {timeframe} K线数据失败: {str(e)}")
            return False
    
    async def _poll_tick_data(self, symbol: str, callback: Callable):
        """
        轮询Tick数据
        
        Args:
            symbol: 交易品种符号
            callback: 数据回调函数
            
        学习点：
        - 无限循环轮询，直到任务取消
        - 错误处理和重试机制
        - 使用异步睡眠避免阻塞
        """
        last_timestamp = None  # 改为时间戳过滤
        while True:
            trades = await self.fetch_tick_data(symbol)
            if trades:
                new_trades = []
                for trade in trades:
                    if last_timestamp is None or trade['timestamp'] > last_timestamp:
                        new_trades.append(trade)
                if new_trades:
                    last_timestamp = new_trades[-1]['timestamp']
                    tick_data = [TickData.from_dict(trade) for trade in new_trades]
                    await callback(tick_data)
            await asyncio.sleep(5)
    
    async def _poll_kline_data(self, symbol: str, timeframe: str, callback: Callable):
        """
        轮询K线数据
        
        Args:
            symbol: 交易品种符号
            timeframe: 时间周期
            callback: 数据回调函数
            
        学习点：
        - 轮询间隔根据时间周期调整
        - 只处理新的K线数据
        - 时间周期与轮询频率的平衡
        """
        try:
            logger.info(f"开始轮询 {symbol} 的 {timeframe} K线数据")
            
            # 记录最后处理的时间戳
            last_timestamp = None
            
            # 根据时间周期设置轮询间隔
            poll_interval = self._get_poll_interval(timeframe)
            
            while True:
                try:
                    # 获取最新K线数据
                    klines = await self.fetch_kline_data(symbol, timeframe)
                    
                    if klines:
                        # 过滤已处理的K线
                        new_klines = []
                        
                        if last_timestamp is not None:
                            for kline in klines:
                                if kline['timestamp'] > last_timestamp:
                                    new_klines.append(kline)
                        else:
                            new_klines = klines
                        
                        if new_klines:
                            # 更新最后处理的时间戳
                            last_timestamp = new_klines[-1]['timestamp']
                            
                            # 转换为KlineData对象
                            kline_data = [KlineData.from_dict(kline) for kline in new_klines]
                            
                            # 调用回调函数
                            await callback(kline_data)
                except Exception as e:
                    logger.error(f"轮询 {symbol} 的 {timeframe} K线数据发生错误: {str(e)}")
                
                # 等待一段时间再次轮询
                await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            logger.info(f"停止轮询 {symbol} 的 {timeframe} K线数据")
        except Exception as e:
            logger.error(f"轮询 {symbol} 的 {timeframe} K线数据失败: {str(e)}")
    
    def _format_symbol(self, symbol: str) -> str:
        """
        格式化交易品种符号，适配交易所API
        
        Args:
            symbol: 交易品种符号
            
        Returns:
            str: 格式化后的交易品种符号
            
        学习点：
        - 不同交易所的符号格式差异
        - 工具方法简化重复代码
        """
        # 如果已经是正确格式，直接返回
        if '/' in symbol:
            return symbol
        
        # 尝试根据常见模式转换
        # 例如：BTCUSDT -> BTC/USDT
        if 'USDT' in symbol:
            return symbol.replace('USDT', '/USDT')
        elif 'USD' in symbol:
            return symbol.replace('USD', '/USD')
        elif 'BTC' in symbol:
            return symbol.replace('BTC', '/BTC')
        elif 'ETH' in symbol:
            return symbol.replace('ETH', '/ETH')
        
        # 无法识别的格式，返回原样
        return symbol
    
    def _get_poll_interval(self, timeframe: str) -> int:
        """
        根据时间周期获取轮询间隔（秒）
        
        Args:
            timeframe: 时间周期
            
        Returns:
            int: 轮询间隔（秒）
            
        学习点：
        - 轮询频率与时间周期的匹配
        - 避免过于频繁的API请求
        """
        # 根据时间周期设置不同的轮询间隔
        if timeframe == '1m':
            return 10  # 10秒
        elif timeframe == '5m':
            return 30  # 30秒
        elif timeframe == '15m':
            return 60  # 1分钟
        elif timeframe == '1h':
            return 120  # 2分钟
        elif timeframe == '4h':
            return 300  # 5分钟
        elif timeframe == '1d':
            return 600  # 10分钟
        else:
            return 60  # 默认1分钟