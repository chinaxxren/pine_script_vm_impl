from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum, auto
from ..plotting import Plotter, PlotStyle, Color, LineStyle, Chart, ChartType, ChartData
from ..indicators import (
    SMA, EMA, WMA, RSI, MACD, ATR, BollingerBands,
    ADX, SuperTrend, OBV, MoneyFlowIndex,
    KDJ, CCI, WR
)
from ..backtest import (
    BacktestEngine, Strategy, Order, OrderType, OrderSide,
    PerformanceAnalyzer, RiskManager, PositionSide
)

class OpCode(Enum):
    LOAD_CONST = auto()
    STORE_VAR = auto()
    LOAD_VAR = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    CALL = auto()
    RETURN = auto()
    LOAD_BUILTIN = auto()
    LOAD_INDICATOR = auto()
    JUMP = auto()
    JUMP_IF_FALSE = auto()
    JUMP_IF_TRUE = auto()
    JUMP_IF_FALSE_KEEP = auto()
    JUMP_IF_TRUE_KEEP = auto()
    COMPARISON = auto()
    PLOT = auto()           # 绘制数据
    PLOT_SHAPE = auto()     # 绘制形状
    PLOT_ARROW = auto()     # 绘制箭头
    PLOT_BGCOLOR = auto()   # 绘制背景色
    PLOT_CANDLE = auto()    # 绘制K线
    STRATEGY_ENTRY = auto() # 策略开仓
    STRATEGY_EXIT = auto()  # 策略平仓
    STRATEGY_CLOSE = auto() # 策略清仓

@dataclass
class Instruction:
    opcode: OpCode
    operand: Any = None

class PineStrategy(Strategy):
    """Pine Script策略"""
    
    def __init__(self, vm):
        super().__init__()
        self.vm = vm
        self.data = []
        
    def on_bar(self, timestamp, open_price, high, low, close, volume):
        """处理K线数据"""
        # 更新数据
        self.data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        # 运行策略逻辑
        self.vm.execute_strategy(self.data[-1])

class PineVM:
    def __init__(self):
        self.stack: List[Any] = []
        self.variables: Dict[str, Any] = {}
        self.instructions: List[Instruction] = []
        self.ip: int = 0
        self.call_stack: List[int] = []
        self.builtin_functions: Dict[str, Callable] = {}
        self.indicators: Dict[str, Callable] = {}
        self.signals: Dict[str, Any] = {}
        self.levels: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        
        # 绘图相关
        self.plotter = Plotter()
        self.chart = Chart()
        self.current_plot_options = None
        
        # 初始化技术指标
        self._init_indicators()
        
        # 回测相关
        self.backtest_engine = BacktestEngine()
        self.strategy = PineStrategy(self)
        self.performance_analyzer = PerformanceAnalyzer(100000)  # 初始资金10万
        self.risk_manager = RiskManager(
            max_position_size=1000,
            max_drawdown=0.2,
            stop_loss_pct=0.1,
            take_profit_pct=0.3
        )
        
    def _init_indicators(self):
        """初始化技术指标"""
        self.indicators = {
            # 移动平均线
            'sma': SMA(),
            'ema': EMA(),
            'wma': WMA(),
            
            # 震荡指标
            'rsi': RSI(),
            'macd': MACD(),
            'kdj': KDJ(),
            'cci': CCI(),
            'wr': WR(),
            
            # 波动率指标
            'atr': ATR(),
            'bb': BollingerBands(),
            
            # 趋势指标
            'adx': ADX(),
            'supertrend': SuperTrend(),
            
            # 成交量指标
            'obv': OBV(),
            'mfi': MoneyFlowIndex()
        }
        
    def calculate_indicator(self, name: str, data: List[float], **kwargs) -> Any:
        """计算技术指标
        
        Args:
            name: 指标名称
            data: 输入数据
            **kwargs: 指标参数
            
        Returns:
            计算结果
        """
        if name not in self.indicators:
            raise ValueError(f"Unknown indicator: {name}")
            
        indicator = self.indicators[name]
        result = indicator.calculate(np.array(data), **kwargs)
        
        return result.values

    def setup_chart_data(self, data: ChartData) -> None:
        """设置图表数据"""
        self.chart.set_data(data)
        
    def push(self, value: Any) -> None:
        self.stack.append(value)

    def pop(self) -> Any:
        if not self.stack:
            raise RuntimeError("Stack underflow")
        return self.stack.pop()

    def execute(self, script: str) -> Optional[Any]:
        self.ip = 0
        while self.ip < len(self.instructions):
            instruction = self.instructions[self.ip]
            self.execute_instruction(instruction)
            self.ip += 1
        return self.pop() if self.stack else None

    def execute_strategy(self, bar_data: dict) -> None:
        """执行策略逻辑"""
        # 更新市场数据
        self.variables.update({
            'open': bar_data['open'],
            'high': bar_data['high'],
            'low': bar_data['low'],
            'close': bar_data['close'],
            'volume': bar_data['volume'],
            'timestamp': bar_data['timestamp']
        })
        
        # 执行策略指令
        while self.ip < len(self.instructions):
            self.execute_instruction(self.instructions[self.ip])
            self.ip += 1
            
    def execute_instruction(self, instruction: Instruction) -> None:
        if instruction.opcode == OpCode.LOAD_CONST:
            self.push(instruction.operand)
        elif instruction.opcode == OpCode.STORE_VAR:
            value = self.pop()
            self.variables[instruction.operand] = value
        elif instruction.opcode == OpCode.LOAD_VAR:
            if instruction.operand not in self.variables:
                raise RuntimeError(f"Variable '{instruction.operand}' not found")
            self.push(self.variables[instruction.operand])
        elif instruction.opcode == OpCode.ADD:
            b = self.pop()
            a = self.pop()
            self.push(a + b)
        elif instruction.opcode == OpCode.SUB:
            b = self.pop()
            a = self.pop()
            self.push(a - b)
        elif instruction.opcode == OpCode.MUL:
            b = self.pop()
            a = self.pop()
            self.push(a * b)
        elif instruction.opcode == OpCode.DIV:
            b = self.pop()
            a = self.pop()
            if b == 0:
                raise RuntimeError("Division by zero")
            self.push(a / b)
        elif instruction.opcode == OpCode.CALL:
            func_name = instruction.operand
            if func_name in self.builtin_functions:
                func = self.builtin_functions[func_name]
                args = [self.pop() for _ in range(func.__code__.co_argcount)]
                args.reverse()
                result = func(*args)
                self.push(result)
            else:
                raise RuntimeError(f"Function '{func_name}' not found")
        elif instruction.opcode == OpCode.JUMP:
            self.ip = instruction.operand - 1
        elif instruction.opcode == OpCode.JUMP_IF_FALSE:
            condition = self.pop()
            if not condition:
                self.ip = instruction.operand - 1
        elif instruction.opcode == OpCode.JUMP_IF_TRUE:
            condition = self.pop()
            if condition:
                self.ip = instruction.operand - 1
        elif instruction.opcode == OpCode.JUMP_IF_FALSE_KEEP:
            condition = self.pop()
            self.push(condition)
            if not condition:
                self.ip = instruction.operand - 1
        elif instruction.opcode == OpCode.JUMP_IF_TRUE_KEEP:
            condition = self.pop()
            self.push(condition)
            if condition:
                self.ip = instruction.operand - 1
        elif instruction.opcode == OpCode.COMPARISON:
            b = self.pop()
            a = self.pop()
            result = False
            
            if instruction.operand == '==':
                result = a == b
            elif instruction.operand == '!=':
                result = a != b
            elif instruction.operand == '>':
                result = a > b
            elif instruction.operand == '>=':
                result = a >= b
            elif instruction.operand == '<':
                result = a < b
            elif instruction.operand == '<=':
                result = a <= b
            
            self.push(result)
        elif instruction.opcode == OpCode.LOAD_BUILTIN:
            func_name = instruction.operand
            if func_name in self.builtin_functions:
                self.push(self.builtin_functions[func_name])
            else:
                raise RuntimeError(f"Builtin function '{func_name}' not found")
        elif instruction.opcode == OpCode.LOAD_INDICATOR:
            indicator_name = instruction.operand
            if indicator_name in self.indicators:
                self.push(self.indicators[indicator_name])
            else:
                raise RuntimeError(f"Indicator '{indicator_name}' not found")
        elif instruction.opcode == OpCode.PLOT:
            # 获取绘图参数
            plot_style = self.pop()
            data = self.pop()
            options = self.pop() if instruction.operand else {}
            
            # 设置默认值
            options.setdefault('color', Color.BLUE)
            options.setdefault('line_style', LineStyle.SOLID)
            options.setdefault('width', 1.0)
            options.setdefault('opacity', 1.0)
            
            # 根据绘图样式调用相应的方法
            if plot_style == PlotStyle.LINE:
                self.plotter.line(data, **options)
                
            elif plot_style == PlotStyle.HISTOGRAM:
                self.plotter.histogram(data, **options)
                
            elif plot_style == PlotStyle.SCATTER:
                self.plotter.scatter(data, **options)
                
            elif plot_style == PlotStyle.AREA:
                self.plotter.area(data, **options)
                
            elif plot_style == PlotStyle.BAND:
                upper = data[:, 0]
                lower = data[:, 1]
                self.plotter.band(upper, lower, **options)
                
            elif plot_style == PlotStyle.MARKER:
                self.plotter.marker(data, **options)
                
            elif plot_style == PlotStyle.LABEL:
                self.plotter.label(data, options.get('text', ''), **options)
                
            elif plot_style == PlotStyle.ARROW:
                start = data[:, 0]
                end = data[:, 1]
                self.plotter.arrow(start, end, **options)
                
            elif plot_style == PlotStyle.BGCOLOR:
                self.plotter.bgcolor(data, **options)
                
        elif instruction.opcode == OpCode.PLOT_SHAPE:
            value = self.pop()
            location = self.pop()
            options = instruction.operand or {}
            options['style'] = PlotStyle.CIRCLES
            self.plotter.plot([location, value], options)
            
        elif instruction.opcode == OpCode.PLOT_ARROW:
            direction = self.pop()
            location = self.pop()
            options = instruction.operand or {}
            options['style'] = PlotStyle.ARROWS
            self.plotter.plot([location, direction], options)
            
        elif instruction.opcode == OpCode.PLOT_BGCOLOR:
            color = self.pop()
            options = instruction.operand or {}
            options['style'] = PlotStyle.BACKGROUND
            self.plotter.plot([color], options)
            
        elif instruction.opcode == OpCode.PLOT_CANDLE:
            close = self.pop()
            low = self.pop()
            high = self.pop()
            open_price = self.pop()
            options = instruction.operand or {}
            
            if not self.chart.data:
                # 创建新的图表数据
                self.chart.set_data(ChartData(
                    timestamp=[str(len(self.chart.data.timestamp) if self.chart.data else 0)],
                    open=[open_price],
                    high=[high],
                    low=[low],
                    close=[close]
                ))
            else:
                # 添加到现有数据
                self.chart.data.timestamp.append(str(len(self.chart.data.timestamp)))
                self.chart.data.open.append(open_price)
                self.chart.data.high.append(high)
                self.chart.data.low.append(low)
                self.chart.data.close.append(close)
                
        elif instruction.opcode == OpCode.STRATEGY_ENTRY:
            # 开仓
            direction = self.pop()  # 1为多头，-1为空头
            price = self.pop()
            quantity = self.pop()
            
            order = Order(
                timestamp=self.variables['timestamp'],
                symbol='default',
                order_type=OrderType.MARKET,
                side=OrderSide.BUY if direction > 0 else OrderSide.SELL,
                price=price,
                quantity=quantity
            )
            
            # 检查风险
            if self.risk_manager.check_position_size(order):
                self.backtest_engine.place_order(order)
                
        elif instruction.opcode == OpCode.STRATEGY_EXIT:
            # 平仓
            price = self.pop()
            
            # 获取当前持仓
            position = self.backtest_engine.positions.get('default')
            if position:
                order = Order(
                    timestamp=self.variables['timestamp'],
                    symbol='default',
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
                    price=price,
                    quantity=position.quantity
                )
                self.backtest_engine.place_order(order)
                
        elif instruction.opcode == OpCode.STRATEGY_CLOSE:
            # 清仓
            for symbol, position in self.backtest_engine.positions.items():
                order = Order(
                    timestamp=self.variables['timestamp'],
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
                    price=self.variables['close'],
                    quantity=position.quantity
                )
                self.backtest_engine.place_order(order)
        elif instruction.opcode == OpCode.LOAD_INDICATOR:
            # 获取指标名称和参数
            indicator_name = self.pop()
            params = {}
            
            if indicator_name in self.indicators:
                indicator = self.indicators[indicator_name]
                
                # 根据指标类型获取所需数据
                if indicator.category in ["Trend", "Oscillator"]:
                    params['high'] = self.variables['high']
                    params['low'] = self.variables['low']
                    params['close'] = self.variables['close']
                    
                elif indicator.category == "Volume":
                    params['close'] = self.variables['close']
                    params['volume'] = self.variables['volume']
                    
                # 获取其他参数
                if instruction.operand:
                    params.update(self.pop())
                    
                # 计算指标值
                result = indicator.calculate(**params)
                
                # 将结果压入栈
                self.push(result.values)
                
                # 记录信号和水平
                self.signals[indicator_name] = result.signals
                self.levels[indicator_name] = result.levels
                
                # 记录元数据
                if result.metadata:
                    self.metadata[indicator_name] = result.metadata
                    
            else:
                raise ValueError(f"Unknown indicator: {indicator_name}")
        else:
            raise RuntimeError(f"Unknown opcode: {instruction.opcode}")

    def load_bytecode(self, instructions: List[Instruction]) -> None:
        self.instructions = instructions
        self.ip = 0
        self.stack.clear()
        self.variables.clear()
        
    def show_plots(self) -> None:
        """显示所有图表"""
        if self.chart.data:
            self.chart.show()
        self.plotter.show()
        
    def clear_plots(self) -> None:
        """清除所有图表"""
        self.chart.clear()
        self.plotter.clear()
