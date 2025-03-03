# GCG_Quant策略引擎开发规划

## 1. 概述

策略引擎是GCG_Quant系统的核心组件，负责执行交易策略逻辑、处理市场数据、生成交易信号，并与其他系统组件（如数据引擎、回测引擎和实盘引擎）进行交互。本文档详细规划策略引擎的架构设计、接口定义、组件关系以及开发路线图。

## 2. 系统架构

### 2.1 整体架构

GCG_Quant系统分为以下主要组件：

1. **数据引擎**：负责市场数据的获取、处理和存储
2. **策略引擎**：负责策略逻辑执行和信号生成
3. **回测引擎**：基于Backtrader，负责策略的历史模拟测试
4. **实盘引擎**：负责实际交易执行和订单管理
5. **监控分析引擎**：负责系统监控和性能分析

各组件之间通过标准化接口进行交互，确保系统的解耦和灵活性。

### 2.2 策略引擎架构

策略引擎内部架构如下：

```
┌─────────────────────────────────────────────────────────────┐
│                       策略引擎                              │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  策略接口   │    │  策略管理器  │    │  信号生成器  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │            │
│         ▼                  ▼                  ▼            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ 具体策略实现 │    │ 策略参数管理 │    │ 信号过滤/聚合 │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  数据引擎   │    │  回测引擎   │    │  实盘引擎   │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 3. 主要组件

### 3.1 策略接口（Strategy Interface）

定义所有策略必须实现的标准接口，确保策略可以被策略引擎统一管理。

```python
# 策略接口定义
class IStrategy:
    # 核心方法定义
    pass
```

### 3.2 策略管理器（Strategy Manager）

负责管理多个策略的注册、初始化、运行和协调。

```python
# 策略管理器
class StrategyManager:
    # 核心方法定义
    pass
```

### 3.3 信号生成器（Signal Generator）

负责将策略的决策转换为标准化的交易信号。

```python
# 信号生成器
class SignalGenerator:
    # 核心方法定义
    pass
```

### 3.4 策略参数管理（Strategy Parameter Manager）

负责管理和优化策略参数。

```python
# 策略参数管理
class StrategyParameterManager:
    # 核心方法定义
    pass
```

### 3.5 信号过滤/聚合（Signal Filter/Aggregator）

对多个策略生成的信号进行过滤和聚合，解决可能的冲突。

```python
# 信号过滤器
class SignalFilter:
    # 核心方法定义
    pass

# 信号聚合器
class SignalAggregator:
    # 核心方法定义
    pass
```

## 4. 核心类和接口定义

### 4.1 策略接口（IStrategy）

```python
class IStrategy:
    """
    策略接口，所有具体策略必须实现此接口
    """
    
    def initialize(self, context):
        """
        策略初始化方法
        
        Args:
            context: 策略上下文，包含配置信息和依赖服务
        """
        pass
    
    def on_bar(self, bar_data):
        """
        处理K线数据
        
        Args:
            bar_data: K线数据，包含开高低收量等信息
        """
        pass
    
    def on_tick(self, tick_data):
        """
        处理Tick数据
        
        Args:
            tick_data: Tick数据，包含最新价格和交易量等信息
        """
        pass
    
    def on_trade(self, trade_data):
        """
        处理成交数据
        
        Args:
            trade_data: 成交数据，包含成交价格、数量和方向等信息
        """
        pass
    
    def on_order(self, order_data):
        """
        处理订单数据
        
        Args:
            order_data: 订单数据，包含订单状态和成交信息等
        """
        pass
    
    def generate_signals(self):
        """
        生成交易信号
        
        Returns:
            List[Signal]: 交易信号列表
        """
        return []
    
    def on_stop(self):
        """
        策略停止时的清理工作
        """
        pass

    def get_visualization_data(self):
        """
        获取用于可视化的数据（如缠论指标）
        Returns:
            Dict: 可视化数据（K线、指标等）
        """
        pass
```

### 4.2 交易信号（Signal）

```python
class Signal:
    """
    交易信号类，表示策略的交易决策
    """
    
    def __init__(self, symbol, direction, price=None, volume=None, 
                 signal_type="market", source_strategy=None, timestamp=None,
                 validity=None, stop_loss=None, take_profit=None, comments=None):
        """
        初始化交易信号
        
        Args:
            symbol: 交易品种符号
            direction: 交易方向，"buy"或"sell"
            price: 交易价格，None表示市价
            volume: 交易数量
            signal_type: 信号类型，如"market", "limit", "stop"等
            source_strategy: 信号来源的策略名称
            timestamp: 信号生成时间戳
            validity: 信号有效期
            stop_loss: 止损价格
            take_profit: 止盈价格
            comments: 附加信息
        """
        pass
```

### 4.3 策略管理器（StrategyManager）

```python
class StrategyManager:
    """
    策略管理器，负责管理多个策略的注册、初始化和运行
    """
    
    def __init__(self, data_engine=None):
        """
        初始化策略管理器
        
        Args:
            data_engine: 数据引擎实例
        """
        pass
    
    def register_strategy(self, name, strategy_class, config=None):
        """
        注册策略
        
        Args:
            name: 策略名称
            strategy_class: 策略类
            config: 策略配置
        """
        pass
    
    def initialize_strategies(self, global_config=None):
        """
        初始化所有已注册的策略
        
        Args:
            global_config: 全局配置
        """
        pass
    
    def process_bar(self, bar_data):
        """
        将K线数据分发给所有策略
        
        Args:
            bar_data: K线数据
        """
        pass
    
    def process_tick(self, tick_data):
        """
        将Tick数据分发给所有策略
        
        Args:
            tick_data: Tick数据
        """
        pass
    
    def collect_signals(self):
        """
        收集所有策略生成的信号
        
        Returns:
            List[Signal]: 所有策略生成的信号列表
        """
        pass
    
    def stop_all(self):
        """
        停止所有策略
        """
        pass

    def push_visualization_data(self, visualizer):
        """
        推送可视化数据到图表
        Args:
            visualizer: 可视化模块实例
        """
        viz_data = {}
        for name, strategy in self.strategies.items():
            viz_data[name] = strategy.get_visualization_data()
        visualizer.update_chart(viz_data)
```

### 4.4 缠论策略（ChanStrategy）

```python
class ChanStrategy(IStrategy):
    """
    基于缠论的交易策略
    """
    
    def __init__(self, symbols, config=None):
        """
        初始化缠论策略
        
        Args:
            symbols: 交易品种列表
            config: 策略配置
        """
        pass
    
    def initialize(self, context):
        """
        策略初始化
        
        Args:
            context: 策略上下文
        """
        pass
    
    def on_bar(self, bar_data):
        """
        处理K线数据，更新缠论分析
        
        Args:
            bar_data: K线数据
        """
        pass
    
    def generate_signals(self):
        """
        基于缠论分析生成交易信号
        
        Returns:
            List[Signal]: 交易信号列表
        """
        pass

    def get_visualization_data(self):
        data = {}
        for symbol, analyzer in self.chan_analyzers.items():
            data[symbol] = {
                'kline': self.get_latest_kline(symbol),
                'fractals': analyzer.get_fractals(),
                'strokes': analyzer.get_strokes(),
                'segments': analyzer.get_segments(),
                'zones': analyzer.get_zones()
            }
        return data
```

### 4.5 机器学习策略（MLStrategy）

```python
class MLStrategy(IStrategy):
    """
    基于机器学习的交易策略
    """
    
    def __init__(self, symbols, model_path=None, config=None):
        """
        初始化机器学习策略
        
        Args:
            symbols: 交易品种列表
            model_path: 模型文件路径
            config: 策略配置
        """
        pass
    
    def initialize(self, context):
        """
        策略初始化，加载模型
        
        Args:
            context: 策略上下文
        """
        pass
    
    def on_bar(self, bar_data):
        """
        处理K线数据，提取特征
        
        Args:
            bar_data: K线数据
        """
        pass
    
    def generate_signals(self):
        """
        基于模型预测生成交易信号
        
        Returns:
            List[Signal]: 交易信号列表
        """
        pass
```

## 5. 与其他组件的接口

### 5.1 与数据引擎的接口

策略引擎从数据引擎接收市场数据，通过以下接口进行交互：

```python
class DataEngineInterface:
    """
    数据引擎接口
    """
    
    def subscribe_bars(self, symbols, callback):
        """
        订阅K线数据
        
        Args:
            symbols: 交易品种列表
            callback: 数据回调函数
        """
        pass
    
    def subscribe_ticks(self, symbols, callback):
        """
        订阅Tick数据
        
        Args:
            symbols: 交易品种列表
            callback: 数据回调函数
        """
        pass
    
    def get_historical_bars(self, symbol, timeframe, start_time, end_time):
        """
        获取历史K线数据
        
        Args:
            symbol: 交易品种
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List: 历史K线数据列表
        """
        pass
```

### 5.2 与回测引擎的接口

策略引擎需要将生成的信号传递给回测引擎，通过以下接口进行交互：

```python
class BacktestEngineInterface:
    """
    回测引擎接口
    """
    
    def initialize(self, start_date, end_date, symbols, initial_capital):
        """
        初始化回测引擎
        
        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期
            symbols: 交易品种列表
            initial_capital: 初始资金
        """
        pass
    
    def process_signals(self, signals):
        """
        处理交易信号
        
        Args:
            signals: 交易信号列表
        """
        pass
    
    def run_backtest(self):
        """
        运行回测
        
        Returns:
            Dict: 回测结果
        """
        pass
```

### 5.3 与实盘引擎的接口

策略引擎需要将生成的信号传递给实盘引擎，通过以下接口进行交互：

```python
class LiveTradingEngineInterface:
    """
    实盘引擎接口
    """
    
    def initialize(self, config):
        """
        初始化实盘引擎
        
        Args:
            config: 配置信息
        """
        pass
    
    def process_signals(self, signals):
        """
        处理交易信号
        
        Args:
            signals: 交易信号列表
        """
        pass
    
    def get_account_info(self):
        """
        获取账户信息
        
        Returns:
            Dict: 账户信息
        """
        pass
```

## 6. 与Backtrader的集成

### 6.1 Backtrader适配器

为了将策略引擎与Backtrader集成，需要创建一个适配器：

```python
class BacktraderAdapter:
    """
    Backtrader适配器，将策略引擎与Backtrader集成
    """
    
    def __init__(self, strategy_manager):
        """
        初始化适配器
        
        Args:
            strategy_manager: 策略管理器实例
        """
        pass
    
    def create_cerebro(self, config):
        """
        创建Backtrader的Cerebro实例
        
        Args:
            config: 配置信息
            
        Returns:
            Cerebro: Backtrader的Cerebro实例
        """
        pass
    
    def add_strategy(self, cerebro):
        """
        将策略添加到Cerebro
        
        Args:
            cerebro: Cerebro实例
        """
        pass
    
    def run_backtest(self, cerebro):
        """
        运行回测
        
        Args:
            cerebro: Cerebro实例
            
        Returns:
            Dict: 回测结果
        """
        pass
```

### 6.2 Backtrader策略类

在Backtrader中，需要创建一个特殊的策略类，该类将接收来自策略引擎的信号：

```python
class BacktraderStrategy:
    """
    Backtrader策略类，接收来自策略引擎的信号
    """
    
    def __init__(self):
        """
        初始化Backtrader策略
        """
        pass
    
    def next(self):
        """
        Backtrader的next方法，每个bar调用一次
        """
        pass
    
    def process_signals(self, signals):
        """
        处理来自策略引擎的信号
        
        Args:
            signals: 信号列表
        """
        pass
```

## 7. 与Chan.py的集成

### 7.1 Chan.py适配器

为了将chan.py与策略引擎集成，需要创建一个适配器：

```python
class ChanAdapter:
    """
    Chan.py适配器，将chan.py与策略引擎集成
    """
    
    def __init__(self):
        """
        初始化适配器
        """
        pass
    
    def create_analyzer(self, config):
        """
        创建chan.py分析器
        
        Args:
            config: 配置信息
            
        Returns:
            Object: chan.py分析器实例
        """
        pass
    
    def process_bar(self, analyzer, bar_data):
        """
        使用chan.py分析器处理K线数据
        
        Args:
            analyzer: chan.py分析器实例
            bar_data: K线数据
        """
        pass
    
    def get_features(self, analyzer):
        """
        获取chan.py分析结果作为特征
        
        Args:
            analyzer: chan.py分析器实例
            
        Returns:
            Dict: 特征字典
        """
        pass
    
    def get_signals(self, analyzer):
        """
        基于chan.py分析结果生成信号
        
        Args:
            analyzer: chan.py分析器实例
            
        Returns:
            List: 信号列表
        """
        pass
```

## 8. 数据结构

### 8.1 K线数据（Bar）

```python
class Bar:
    """
    K线数据结构
    """
    
    def __init__(self, symbol, timestamp, open_price, high_price, low_price, close_price, 
                 volume, timeframe="1m", source=None):
        """
        初始化K线数据
        
        Args:
            symbol: 交易品种
            timestamp: 时间戳
            open_price: 开盘价
            high_price: 最高价
            low_price: 最低价
            close_price: 收盘价
            volume: 成交量
            timeframe: 时间周期
            source: 数据来源
        """
        pass
```

### 8.2 Tick数据（Tick）

```python
class Tick:
    """
    Tick数据结构
    """
    
    def __init__(self, symbol, timestamp, price, volume, direction=None, source=None):
        """
        初始化Tick数据
        
        Args:
            symbol: 交易品种
            timestamp: 时间戳
            price: 价格
            volume: 成交量
            direction: 方向，"buy"或"sell"
            source: 数据来源
        """
        pass
```

### 8.3 订单数据（Order）

```python
class Order:
    """
    订单数据结构
    """
    
    def __init__(self, symbol, order_id, order_type, direction, price, volume, 
                 status, timestamp, source_strategy=None):
        """
        初始化订单数据
        
        Args:
            symbol: 交易品种
            order_id: 订单ID
            order_type: 订单类型，如"market", "limit", "stop"等
            direction: 交易方向，"buy"或"sell"
            price: 价格
            volume: 数量
            status: 订单状态
            timestamp: 时间戳
            source_strategy: 来源策略
        """
        pass
```

## 9. 开发路线图

策略引擎的开发将分为以下几个步骤：

### 9.1 第一步：基础架构（2周）

1. 设计和实现策略接口（IStrategy）
2. 设计和实现信号类（Signal）
3. 设计和实现策略管理器（StrategyManager）
4. 设计数据结构（Bar, Tick, Order）
5. 实现简单的数据引擎接口

### 9.2 第二步：缠论策略集成（2周）

1. 实现Chan.py适配器
2. 实现基于缠论的策略类（ChanStrategy）
3. 测试缠论策略的数据处理和信号生成

### 补充步：可视化集成（2周）
1. 实现 Lightweight Charts 图表基础功能。 
2. 整合策略引擎的可视化数据输出。 
3. 测试实时更新和缠论指标绘制。

### 9.3 第三步：Backtrader集成（2周）

1. 实现Backtrader适配器
2. 设计Backtrader策略类
3. 测试策略引擎与Backtrader的集成
4. 实现回测结果的收集和分析

### 9.4 第四步：机器学习集成（2周）

1. 设计特征提取接口
2. 实现机器学习策略类（MLStrategy）
3. 实现模型训练和预测功能
4. 测试机器学习策略的信号生成

### 9.5 第五步：策略优化与完善（2周）

1. 实现策略参数管理
2. 实现信号过滤和聚合
3. 添加性能监控和日志记录
4. 全面测试和性能优化

## 10. 实现注意事项

1. **性能优化**：
   - 使用异步编程提高I/O密集型操作的效率
   - 优化数据处理流程，减少不必要的计算
   - 考虑使用缓存减少重复计算

2. **错误处理**：
   - 实现全面的错误捕获和处理机制
   - 添加合适的日志记录，便于调试
   - 实现故障恢复机制

3. **扩展性**：
   - 保持接口的一致性和稳定性
   - 使用依赖注入，减少组件之间的耦合
   - 提供清晰的文档和示例

4. **测试**：
   - 编写单元测试，确保各组件功能正确
   - 进行集成测试，验证组件之间的交互
   - 进行性能测试，确保系统能够处理大量数据

## 11. 参考资料

1. Chan.py文档：https://github.com/Vespa314/chan.py
2. Backtrader文档：https://www.backtrader.com/docu/
3. 《量化交易之路》- 使用Python构建自己的量化交易系统

## 12. 附录

### 12.1 配置示例

```yaml
# 策略引擎配置示例
strategy_engine:
  log_level: INFO
  signal_expiry: 300  # 信号有效期（秒）
  
  strategies:
    chan_strategy:
      enabled: true
      symbols: ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
      parameters:
        kl_period: "30m"
        chan_config:
          bi_th: 0.7
          bi_min_k: 5
          use_new_rule: true
          use_zs: true
          zs_mode: "no_cross"
    
    ml_strategy:
      enabled: false
      symbols: ["BTC/USDT"]
      parameters:
        model_path: "models/btc_model.pkl"
        prediction_threshold: 0.7
        feature_window: 20
```

### 12.2 使用示例

```python
# 策略引擎使用示例
from strategies.chan_strategy import ChanStrategy
from strategy_engine.manager import StrategyManager

# 创建策略管理器
strategy_manager = StrategyManager()

# 注册策略
strategy_manager.register_strategy(
    name="chan_strategy",
    strategy_class=ChanStrategy,
    config={
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "parameters": {
            "kl_period": "30m",
            "chan_config": {
                "bi_th": 0.7,
                "bi_min_k": 5
            }
        }
    }
)

# 初始化策略
strategy_manager.initialize_strategies()

# 处理K线数据
bar_data = {...}  # K线数据
strategy_manager.process_bar(bar_data)

# 收集信号
signals = strategy_manager.collect_signals()

# 将信号传递给回测或实盘引擎
backtest_engine.process_signals(signals)
```
