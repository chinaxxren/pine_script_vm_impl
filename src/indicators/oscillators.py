"""
振荡器指标
"""

import numpy as np
from .base import Indicator, IndicatorResult

class RSI(Indicator):
    """相对强弱指标"""
    
    def __init__(self):
        super().__init__()
        self.description = "Relative Strength Index"
        self.category = "Oscillator"
        
    def calculate(self, data: np.ndarray, period: int = 14) -> IndicatorResult:
        """计算 RSI
        
        Args:
            data: 输入数据
            period: 周期
            
        Returns:
            IndicatorResult: 计算结果
        """
        self.validate_input(data)
        
        # 计算价格变化
        deltas = np.zeros_like(data)
        deltas[1:] = data[1:] - data[:-1]
        
        # 分离上涨和下跌
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # 计算平均值
        avg_gains = np.zeros_like(gains)
        avg_losses = np.zeros_like(losses)
        
        # 第一个周期
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        # 后续周期
        for i in range(period + 1, len(data)):
            avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i]) / period
            avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i]) / period
            
        # 计算相对强度和 RSI
        rs = np.where(avg_losses != 0, avg_gains / avg_losses, 100)
        rsi = np.where(avg_losses == 0, 100, 100 - (100 / (1 + rs)))
        
        # 处理开始的 period 个值
        rsi[:period] = 50
        
        signals = self.generate_signals(rsi)
        levels = self.calculate_levels(rsi)
        
        return IndicatorResult(values=rsi, signals=signals, levels=levels)
        
    def generate_signals(self, rsi: np.ndarray) -> dict:
        """生成交易信号"""
        signals = {
            'oversold': [],    # 超卖
            'overbought': [],  # 超买
            'bullish': [],     # 看多
            'bearish': []      # 看空
        }
        
        for i in range(len(rsi)):
            signals['oversold'].append(rsi[i] < 30)
            signals['overbought'].append(rsi[i] > 70)
            signals['bullish'].append(rsi[i] > 50)
            signals['bearish'].append(rsi[i] < 50)
            
        return signals
        
    def calculate_levels(self, rsi: np.ndarray) -> dict:
        """计算重要水平"""
        return {
            'oversold': 30.0,
            'overbought': 70.0,
            'neutral': 50.0
        }
        
class Stochastic(Indicator):
    """随机指标"""
    
    def __init__(self):
        super().__init__()
        self.description = "Stochastic Oscillator"
        self.category = "Oscillator"
        
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 k_period: int = 14, d_period: int = 3) -> IndicatorResult:
        """计算随机指标
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            k_period: K值周期
            d_period: D值周期
            
        Returns:
            IndicatorResult: 计算结果
        """
        self.validate_input(high)
        self.validate_input(low)
        self.validate_input(close)
        
        # 计算 %K
        lowest_low = np.zeros_like(low)
        highest_high = np.zeros_like(high)
        
        for i in range(len(close)):
            if i < k_period:
                lowest_low[i] = np.min(low[:i+1])
                highest_high[i] = np.max(high[:i+1])
            else:
                lowest_low[i] = np.min(low[i-k_period+1:i+1])
                highest_high[i] = np.max(high[i-k_period+1:i+1])
                
        # 处理除零情况
        denom = highest_high - lowest_low
        k = np.where(denom != 0,
                    100 * (close - lowest_low) / denom,
                    50)  # 当最高价等于最低价时，返回50
        
        # 计算 %D
        d = np.zeros_like(k)
        for i in range(len(k)):
            if i < d_period:
                d[i] = np.mean(k[:i+1])
            else:
                d[i] = np.mean(k[i-d_period+1:i+1])
            
        values = {
            'k': k,
            'd': d
        }
        
        signals = self.generate_signals(k, d)
        levels = self.calculate_levels(k)
        
        return IndicatorResult(values=values, signals=signals, levels=levels)
        
    def generate_signals(self, k: np.ndarray, d: np.ndarray) -> dict:
        """生成交易信号"""
        signals = {
            'oversold': [],       # 超卖
            'overbought': [],     # 超买
            'bullish_cross': [],  # 金叉
            'bearish_cross': []   # 死叉
        }
        
        for i in range(1, len(k)):
            signals['oversold'].append(k[i] < 20)
            signals['overbought'].append(k[i] > 80)
            signals['bullish_cross'].append(
                k[i-1] < d[i-1] and k[i] > d[i]
            )
            signals['bearish_cross'].append(
                k[i-1] > d[i-1] and k[i] < d[i]
            )
            
        return signals
        
    def calculate_levels(self, k: np.ndarray) -> dict:
        """计算重要水平"""
        return {
            'oversold': 20.0,
            'overbought': 80.0,
            'neutral': 50.0
        }

class KDJ(Indicator):
    """随机指标KDJ"""
    
    def __init__(self):
        super().__init__()
        self.description = "Stochastic KDJ"
        self.category = "Oscillator"
        
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 k_period: int = 9, d_period: int = 3, j_period: int = 3) -> IndicatorResult:
        """计算KDJ指标
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            k_period: K值周期
            d_period: D值周期
            j_period: J值周期
            
        Returns:
            IndicatorResult: 计算结果
        """
        self.validate_input(high)
        self.validate_input(low)
        self.validate_input(close)
        
        # 计算RSV (Raw Stochastic Value)
        lowest_low = np.zeros_like(low)
        highest_high = np.zeros_like(high)
        
        for i in range(len(close)):
            if i < k_period:
                lowest_low[i] = np.min(low[:i+1])
                highest_high[i] = np.max(high[:i+1])
            else:
                lowest_low[i] = np.min(low[i-k_period+1:i+1])
                highest_high[i] = np.max(high[i-k_period+1:i+1])
                
        # 处理除零情况
        denom = highest_high - lowest_low
        rsv = np.where(denom != 0,
                      100 * (close - lowest_low) / denom,
                      50)  # 当最高价等于最低价时，返回50
        
        # 计算K值
        k = np.zeros_like(rsv)
        k[0] = 50  # 初始值
        for i in range(1, len(rsv)):
            k[i] = (2/3) * k[i-1] + (1/3) * rsv[i]
        
        # 计算D值
        d = np.zeros_like(k)
        d[0] = 50  # 初始值
        for i in range(1, len(k)):
            d[i] = (2/3) * d[i-1] + (1/3) * k[i]
        
        # 计算J值
        j = 3 * k - 2 * d
        
        values = {
            'k': k,
            'd': d,
            'j': j
        }
        
        signals = self.generate_signals(k, d, j)
        levels = self.calculate_levels()
        
        return IndicatorResult(values=values, signals=signals, levels=levels)
        
    def generate_signals(self, k: np.ndarray, d: np.ndarray, j: np.ndarray) -> dict:
        """生成交易信号"""
        signals = {
            'oversold': [],       # 超卖
            'overbought': [],     # 超买
            'bullish_cross': [],  # 金叉
            'bearish_cross': []   # 死叉
        }
        
        for i in range(1, len(k)):
            # 超买超卖信号
            signals['oversold'].append(k[i] < 20 and d[i] < 20)
            signals['overbought'].append(k[i] > 80 and d[i] > 80)
            
            # 金叉死叉信号
            signals['bullish_cross'].append(
                k[i-1] < d[i-1] and k[i] > d[i]
            )
            signals['bearish_cross'].append(
                k[i-1] > d[i-1] and k[i] < d[i]
            )
            
        return signals
        
    def calculate_levels(self) -> dict:
        """计算重要水平"""
        return {
            'oversold': 20.0,
            'overbought': 80.0,
            'neutral': 50.0
        }
