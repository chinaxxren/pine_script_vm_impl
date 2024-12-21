from typing import List, Optional, Dict, Tuple
import numpy as np
from .base import Indicator, IndicatorResult

class ATR(Indicator):
    """平均真实波幅"""
    
    def __init__(self):
        super().__init__()
        self.description = "Average True Range"
        self.category = "Volatility"
        
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                 period: int = 14) -> IndicatorResult:
        """计算ATR
        
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
        
        if period < 1:
            raise ValueError("Period must be >= 1")
            
        # 计算真实波幅
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
            
        # 计算ATR
        values = np.zeros_like(tr)
        values[period-1] = np.mean(tr[:period])
        
        for i in range(period, len(tr)):
            values[i] = (values[i-1] * (period-1) + tr[i]) / period
            
        signals = self.generate_signals(values)
        levels = self.calculate_levels(values)
        
        return IndicatorResult(values=values, signals=signals, levels=levels)
        
    def generate_signals(self, values: np.ndarray) -> Dict[str, List[bool]]:
        """生成交易信号"""
        signals = {
            'high_volatility': [],  # 波动率高于平均值的2倍
            'low_volatility': []    # 波动率低于平均值的0.5倍
        }
        
        mean_atr = np.mean(values[values > 0])
        
        for value in values:
            signals['high_volatility'].append(value > mean_atr * 2)
            signals['low_volatility'].append(value < mean_atr * 0.5)
            
        return signals

class BollingerBands(Indicator):
    """布林带"""
    
    def __init__(self):
        super().__init__()
        self.description = "Bollinger Bands"
        self.category = "Volatility"
        
    def calculate(self, data: np.ndarray, 
                 period: int = 20, 
                 deviations: float = 2.0) -> IndicatorResult:
        """计算布林带
        
        Args:
            data: 输入数据
            period: 周期
            deviations: 标准差倍数
            
        Returns:
            IndicatorResult: 计算结果
        """
        self.validate_input(data)
        
        if period < 1:
            raise ValueError("Period must be >= 1")
            
        # 计算中轨（简单移动平均线）
        middle = np.zeros_like(data)
        for i in range(period - 1, len(data)):
            middle[i] = np.mean(data[i-period+1:i+1])
            
        # 计算标准差
        std = np.zeros_like(data)
        for i in range(period - 1, len(data)):
            std[i] = np.std(data[i-period+1:i+1])
            
        # 计算上轨和下轨
        upper = middle + deviations * std
        lower = middle - deviations * std
        
        values = np.column_stack((upper, middle, lower))
        
        signals = self.generate_signals(values)
        levels = self.calculate_levels(values)
        
        metadata = {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
        
        return IndicatorResult(values=values, signals=signals, levels=levels, metadata=metadata)
        
    def generate_signals(self, values: np.ndarray) -> Dict[str, List[bool]]:
        """生成交易信号"""
        upper = values[:, 0]
        middle = values[:, 1]
        lower = values[:, 2]
        
        signals = {
            'price_above_upper': [],    # 价格突破上轨
            'price_below_lower': [],    # 价格突破下轨
            'price_cross_middle': []    # 价格穿过中轨
        }
        
        for i in range(1, len(values)):
            signals['price_above_upper'].append(
                values[i-1, 1] <= values[i-1, 0] and 
                values[i, 1] > values[i, 0]
            )
            signals['price_below_lower'].append(
                values[i-1, 1] >= values[i-1, 2] and 
                values[i, 1] < values[i, 2]
            )
            signals['price_cross_middle'].append(
                (values[i-1, 1] < values[i-1, 1] and 
                 values[i, 1] > values[i, 1]) or
                (values[i-1, 1] > values[i-1, 1] and 
                 values[i, 1] < values[i, 1])
            )
            
        return signals
