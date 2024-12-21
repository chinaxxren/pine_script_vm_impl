"""
动量指标
"""

import numpy as np
from .base import Indicator, IndicatorResult

class CCI(Indicator):
    """商品通道指数"""
    
    def __init__(self):
        super().__init__()
        self.description = "Commodity Channel Index"
        self.category = "Momentum"
        
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int = 20) -> IndicatorResult:
        """计算CCI
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            period: 周期
            
        Returns:
            IndicatorResult: 计算结果
        """
        self.validate_input(high)
        self.validate_input(low)
        self.validate_input(close)
        
        # 计算典型价格
        tp = (high + low + close) / 3
        
        # 计算移动平均
        sma = np.zeros_like(tp)
        for i in range(period - 1, len(tp)):
            sma[i] = np.mean(tp[i-period+1:i+1])
        
        # 计算平均绝对偏差
        mad = np.zeros_like(tp)
        for i in range(period - 1, len(tp)):
            mad[i] = np.mean(np.abs(tp[i-period+1:i+1] - sma[i]))
        
        # 计算CCI
        cci = np.zeros_like(tp)
        mask = mad != 0
        cci[mask] = (tp[mask] - sma[mask]) / (0.015 * mad[mask])
        
        # 处理前period-1个值
        cci[:period-1] = 0
        
        signals = self.generate_signals(cci)
        levels = self.calculate_levels()
        
        metadata = {
            'typical_price': tp,
            'sma': sma,
            'mad': mad
        }
        
        return IndicatorResult(values=cci, signals=signals, levels=levels, metadata=metadata)
        
    def generate_signals(self, cci: np.ndarray) -> dict:
        """生成交易信号"""
        signals = {
            'overbought': [],    # CCI > 100
            'oversold': [],      # CCI < -100
            'extreme_high': [],  # CCI > 200
            'extreme_low': []    # CCI < -200
        }
        
        for value in cci:
            signals['overbought'].append(value > 100)
            signals['oversold'].append(value < -100)
            signals['extreme_high'].append(value > 200)
            signals['extreme_low'].append(value < -200)
            
        return signals
        
    def calculate_levels(self) -> dict:
        """计算重要水平"""
        return {
            'overbought': 100.0,
            'oversold': -100.0,
            'extreme_high': 200.0,
            'extreme_low': -200.0
        }

class WilliamsR(Indicator):
    """威廉指标"""
    
    def __init__(self):
        super().__init__()
        self.description = "Williams %R"
        self.category = "Momentum"
        
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int = 14) -> IndicatorResult:
        """计算威廉指标
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            period: 周期
            
        Returns:
            IndicatorResult: 计算结果
        """
        self.validate_input(high)
        self.validate_input(low)
        self.validate_input(close)
        
        # 计算最高价和最低价
        highest_high = np.zeros_like(high)
        lowest_low = np.zeros_like(low)
        
        for i in range(len(close)):
            if i < period:
                highest_high[i] = np.max(high[:i+1])
                lowest_low[i] = np.min(low[:i+1])
            else:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
                
        # 计算威廉指标
        wr = np.zeros_like(close)
        denom = highest_high - lowest_low
        mask = denom != 0
        wr[mask] = -100 * (highest_high[mask] - close[mask]) / denom[mask]
        
        signals = self.generate_signals(wr)
        levels = self.calculate_levels()
        
        metadata = {
            'highest_high': highest_high,
            'lowest_low': lowest_low
        }
        
        return IndicatorResult(values=wr, signals=signals, levels=levels, metadata=metadata)
        
    def generate_signals(self, wr: np.ndarray) -> dict:
        """生成交易信号"""
        signals = {
            'overbought': [],  # %R < -80
            'oversold': []     # %R > -20
        }
        
        for value in wr:
            signals['overbought'].append(value < -80)
            signals['oversold'].append(value > -20)
            
        return signals
        
    def calculate_levels(self) -> dict:
        """计算重要水平"""
        return {
            'overbought': -80.0,
            'oversold': -20.0,
            'middle': -50.0
        }
