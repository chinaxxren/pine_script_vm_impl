from typing import List, Optional, Dict
import numpy as np
from .base import Indicator, IndicatorResult

class SMA(Indicator):
    """简单移动平均线"""
    
    def __init__(self):
        super().__init__()
        self.description = "Simple Moving Average"
        self.category = "Moving Average"
        
    def calculate(self, data: np.ndarray, period: int = 14) -> IndicatorResult:
        """计算简单移动平均线
        
        Args:
            data: 输入数据
            period: 周期
        
        Returns:
            IndicatorResult: 计算结果
        """
        self.validate_input(data)
        if period < 1:
            raise ValueError("Period must be >= 1")
            
        values = np.zeros_like(data)
        for i in range(period - 1, len(data)):
            values[i] = np.mean(data[i-period+1:i+1])
            
        signals = self.generate_signals(values)
        levels = self.calculate_levels(values)
        
        return IndicatorResult(values=values, signals=signals, levels=levels)
        
    def generate_signals(self, values: np.ndarray) -> Dict[str, List[bool]]:
        """生成交易信号"""
        signals = {
            'crossover': [],   # 价格上穿均线
            'crossunder': []   # 价格下穿均线
        }
        
        for i in range(1, len(values)):
            signals['crossover'].append(
                values[i-1] < values[i] and values[i-1] > values[i]
            )
            signals['crossunder'].append(
                values[i-1] > values[i] and values[i-1] < values[i]
            )
            
        return signals

class EMA(Indicator):
    """指数移动平均线"""
    
    def __init__(self):
        super().__init__()
        self.description = "Exponential Moving Average"
        self.category = "Moving Average"
        
    def calculate(self, data: np.ndarray, period: int = 14) -> IndicatorResult:
        """计算指数移动平均线
        
        Args:
            data: 输入数据
            period: 周期
        
        Returns:
            IndicatorResult: 计算结果
        """
        self.validate_input(data)
        if period < 1:
            raise ValueError("Period must be >= 1")
            
        alpha = 2.0 / (period + 1)
        values = np.zeros_like(data)
        values[0] = data[0]
        
        for i in range(1, len(data)):
            values[i] = alpha * data[i] + (1 - alpha) * values[i-1]
            
        signals = self.generate_signals(values)
        levels = self.calculate_levels(values)
        
        return IndicatorResult(values=values, signals=signals, levels=levels)
        
    def generate_signals(self, values: np.ndarray) -> Dict[str, List[bool]]:
        """生成交易信号"""
        signals = {
            'crossover': [],   # 价格上穿均线
            'crossunder': []   # 价格下穿均线
        }
        
        for i in range(1, len(values)):
            signals['crossover'].append(
                values[i-1] < values[i] and values[i-1] > values[i]
            )
            signals['crossunder'].append(
                values[i-1] > values[i] and values[i-1] < values[i]
            )
            
        return signals

class WMA(Indicator):
    """加权移动平均线"""
    
    def __init__(self):
        super().__init__()
        self.description = "Weighted Moving Average"
        self.category = "Moving Average"
        
    def calculate(self, data: np.ndarray, period: int = 14) -> IndicatorResult:
        """计算加权移动平均线
        
        Args:
            data: 输入数据
            period: 周期
        
        Returns:
            IndicatorResult: 计算结果
        """
        self.validate_input(data)
        if period < 1:
            raise ValueError("Period must be >= 1")
            
        weights = np.arange(1, period + 1)
        values = np.zeros_like(data)
        
        for i in range(period - 1, len(data)):
            values[i] = np.sum(data[i-period+1:i+1] * weights) / np.sum(weights)
            
        signals = self.generate_signals(values)
        levels = self.calculate_levels(values)
        
        return IndicatorResult(values=values, signals=signals, levels=levels)
        
    def generate_signals(self, values: np.ndarray) -> Dict[str, List[bool]]:
        """生成交易信号"""
        signals = {
            'crossover': [],   # 价格上穿均线
            'crossunder': []   # 价格下穿均线
        }
        
        for i in range(1, len(values)):
            signals['crossover'].append(
                values[i-1] < values[i] and values[i-1] > values[i]
            )
            signals['crossunder'].append(
                values[i-1] > values[i] and values[i-1] < values[i]
            )
            
        return signals
