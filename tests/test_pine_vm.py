"""
Pine VM 测试
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.vm.pine_vm import PineVM
from src.compiler.compiler import Compiler
from src.types.pine_types import PineValue, PineType

class TestPineVM(unittest.TestCase):
    """Pine VM 测试"""
    
    def setUp(self):
        """准备测试数据"""
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        self.data = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 100,
            'high': np.random.randn(len(dates)).cumsum() + 102,
            'low': np.random.randn(len(dates)).cumsum() + 98,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # 创建编译器和 VM
        self.compiler = Compiler()
        self.vm = PineVM()
        
    def test_basic_operations(self):
        """测试基本运算"""
        code = """
a = 1 + 2 * 3
b = a - 4 / 2
c = (a + b) * 2
"""
        # 编译并运行
        compiled = self.compiler.compile(code)
        self.vm.load(compiled)
        self.vm.run()
        
        # 验证结果
        self.assertEqual(self.vm.get_var('a').value, 7)
        self.assertEqual(self.vm.get_var('b').value, 5)
        self.assertEqual(self.vm.get_var('c').value, 24)
        
    def test_pine_types(self):
        """测试 Pine 类型"""
        code = """
var float myFloat = 1.23
var int myInt = 456
var string myString = "hello"
var bool myBool = true
"""
        # 编译并运行
        compiled = self.compiler.compile(code)
        self.vm.load(compiled)
        self.vm.run()
        
        # 验证类型和值
        self.assertEqual(self.vm.get_var('myFloat').type, PineType.FLOAT)
        self.assertEqual(self.vm.get_var('myInt').type, PineType.INT)
        self.assertEqual(self.vm.get_var('myString').type, PineType.STRING)
        self.assertEqual(self.vm.get_var('myBool').type, PineType.BOOL)
        
        self.assertAlmostEqual(self.vm.get_var('myFloat').value, 1.23)
        self.assertEqual(self.vm.get_var('myInt').value, 456)
        self.assertEqual(self.vm.get_var('myString').value, "hello")
        self.assertEqual(self.vm.get_var('myBool').value, True)
        
    def test_series_operations(self):
        """测试序列操作"""
        code = """
sma = ta.sma(close, 10)
highest = ta.highest(high, 20)
lowest = ta.lowest(low, 20)
"""
        # 编译并运行
        compiled = self.compiler.compile(code)
        self.vm.load(compiled)
        self.vm.set_data(self.data)
        self.vm.run()
        
        # 验证结果
        sma = self.vm.get_var('sma').value
        highest = self.vm.get_var('highest').value
        lowest = self.vm.get_var('lowest').value
        
        self.assertIsInstance(sma, pd.Series)
        self.assertIsInstance(highest, pd.Series)
        self.assertIsInstance(lowest, pd.Series)
        
        self.assertEqual(len(sma), len(self.data))
        self.assertEqual(len(highest), len(self.data))
        self.assertEqual(len(lowest), len(self.data))
        
    def test_strategy_commands(self):
        """测试策略命令"""
        code = """
strategy("Test Strategy", overlay=true)

longCondition = ta.crossover(ta.sma(close, 10), ta.sma(close, 20))
shortCondition = ta.crossunder(ta.sma(close, 10), ta.sma(close, 20))

if longCondition
    strategy.entry("Long", strategy.long)
else if shortCondition
    strategy.close("Long")
"""
        # 编译并运行
        compiled = self.compiler.compile(code)
        self.vm.load(compiled)
        self.vm.set_data(self.data)
        self.vm.run()
        
        # 验证策略执行结果
        trades = self.vm.get_trades()
        self.assertIsInstance(trades, list)
        self.assertTrue(len(trades) > 0)
        
        # 验证交易记录格式
        for trade in trades:
            self.assertIn('entry_time', trade)
            self.assertIn('entry_price', trade)
            self.assertIn('exit_time', trade)
            self.assertIn('exit_price', trade)
            self.assertIn('pnl', trade)
            
    def test_plotting_commands(self):
        """测试绘图命令"""
        code = """
//@version=5
indicator("Test Indicator", overlay=true)

sma1 = ta.sma(close, 10)
sma2 = ta.sma(close, 20)

plot(sma1, "Fast MA", color=color.blue)
plot(sma2, "Slow MA", color=color.red)

plotshape(ta.crossover(sma1, sma2), "Buy", style=shape.triangleup, location=location.belowbar, color=color.green)
plotshape(ta.crossunder(sma1, sma2), "Sell", style=shape.triangledown, location=location.abovebar, color=color.red)
"""
        # 编译并运行
        compiled = self.compiler.compile(code)
        self.vm.load(compiled)
        self.vm.set_data(self.data)
        self.vm.run()
        
        # 验证绘图命令
        plots = self.vm.get_plots()
        self.assertTrue(len(plots) > 0)
        
        # 验证绘图数据格式
        for plot in plots:
            self.assertIn('type', plot)
            self.assertIn('name', plot)
            self.assertIn('data', plot)
            
    def test_error_handling(self):
        """测试错误处理"""
        # 测试除零错误
        code = "result = 1 / 0"
        compiled = self.compiler.compile(code)
        self.vm.load(compiled)
        with self.assertRaises(ZeroDivisionError):
            self.vm.run()
            
        # 测试未定义变量
        code = "result = undefined_var + 1"
        compiled = self.compiler.compile(code)
        self.vm.load(compiled)
        with self.assertRaises(Exception):
            self.vm.run()
            
        # 测试类型错误
        code = 'result = "string" + 1'
        compiled = self.compiler.compile(code)
        self.vm.load(compiled)
        with self.assertRaises(TypeError):
            self.vm.run()
            
    def test_function_calls(self):
        """测试函数调用"""
        code = """
//@function
myFunction(float price, int length) =>
    result = ta.sma(price, length)
    result

output = myFunction(close, 10)
"""
        # 编译并运行
        compiled = self.compiler.compile(code)
        self.vm.load(compiled)
        self.vm.set_data(self.data)
        self.vm.run()
        
        # 验证函数输出
        output = self.vm.get_var('output').value
        self.assertIsInstance(output, pd.Series)
        self.assertEqual(len(output), len(self.data))
        
    def test_variable_scope(self):
        """测试变量作用域"""
        code = """
var float global = 0.0

//@function
updateGlobal(float value) =>
    global := value
    global

if close > open
    updateGlobal(close)
else
    updateGlobal(open)
"""
        # 编译并运行
        compiled = self.compiler.compile(code)
        self.vm.load(compiled)
        self.vm.set_data(self.data)
        self.vm.run()
        
        # 验证全局变量
        global_var = self.vm.get_var('global')
        self.assertIsInstance(global_var.value, float)
        
if __name__ == '__main__':
    unittest.main()
