"""
成交量指标
"""

import numpy as np
from .base import Indicator, IndicatorResult

class OBV(Indicator):
    """能量潮指标"""
    
    def __init__(self):
        super().__init__()
        self.description = "On Balance Volume"
        self.category = "Volume"
        
    def calculate(self, close: np.ndarray, volume: np.ndarray) -> IndicatorResult:
        """计算OBV
        
        Args:
            close: 收盘价
            volume: 成交量
            
        Returns:
            IndicatorResult: 计算结果
        """
        self.validate_input(close)
        self.validate_input(volume)
        
        obv = np.zeros_like(volume)
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
                
        signals = self.generate_signals(obv)
        levels = self.calculate_levels(obv)
        
        return IndicatorResult(values=obv, signals=signals, levels=levels)
        
    def generate_signals(self, values: np.ndarray) -> dict:
        """生成交易信号"""
        signals = {
            'increasing': [],  # OBV上升
            'decreasing': []   # OBV下降
        }
        
        for i in range(1, len(values)):
            signals['increasing'].append(values[i] > values[i-1])
            signals['decreasing'].append(values[i] < values[i-1])
            
        return signals
        
    def calculate_levels(self, values: np.ndarray) -> dict:
        """计算重要水平"""
        return {
            'zero': 0.0
        }

class AccumulationDistribution(Indicator):
    """累积/派发线指标"""
    
    def __init__(self):
        super().__init__()
        self.description = "Accumulation/Distribution Line"
        self.category = "Volume"
        
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> IndicatorResult:
        """计算累积/派发线
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            volume: 成交量
            
        Returns:
            IndicatorResult: 计算结果
        """
        self.validate_input(high)
        self.validate_input(low)
        self.validate_input(close)
        self.validate_input(volume)
        
        # 计算资金流量乘数
        high_low = high - low
        close_low = close - low
        high_close = high - close
        
        mfm = np.where(
            high_low == 0,
            0,
            ((close_low - high_close) / high_low)
        )
        
        # 计算资金流量量
        mfv = mfm * volume
        
        # 计算累积/派发线
        ad = np.zeros_like(volume)
        ad[0] = mfv[0]
        
        for i in range(1, len(volume)):
            ad[i] = ad[i-1] + mfv[i]
            
        signals = self.generate_signals(ad)
        levels = self.calculate_levels(ad)
        
        metadata = {
            'money_flow_multiplier': mfm,
            'money_flow_volume': mfv
        }
        
        return IndicatorResult(values=ad, signals=signals, levels=levels, metadata=metadata)
        
    def generate_signals(self, values: np.ndarray) -> dict:
        """生成交易信号"""
        signals = {
            'increasing': [],  # A/D线上升
            'decreasing': [],  # A/D线下降
            'divergence': []   # A/D线与价格背离
        }
        
        for i in range(1, len(values)):
            signals['increasing'].append(values[i] > values[i-1])
            signals['decreasing'].append(values[i] < values[i-1])
            signals['divergence'].append(False)  # 背离信号需要价格数据才能计算
            
        return signals
        
    def calculate_levels(self, values: np.ndarray) -> dict:
        """计算重要水平"""
        return {
            'zero': 0.0
        }

class MoneyFlowIndex(Indicator):
    """资金流量指标"""
    
    def __init__(self):
        super().__init__()
        self.description = "Money Flow Index"
        self.category = "Volume"
        
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 volume: np.ndarray, period: int = 14) -> IndicatorResult:
        """计算MFI
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            volume: 成交量
            period: 周期
            
        Returns:
            IndicatorResult: 计算结果
        """
        self.validate_input(high)
        self.validate_input(low)
        self.validate_input(close)
        self.validate_input(volume)
        
        # 计算典型价格
        typical_price = (high + low + close) / 3
        
        # 计算资金流
        money_flow = typical_price * volume
        
        # 计算正向和负向资金流
        positive_flow = np.zeros_like(money_flow)
        negative_flow = np.zeros_like(money_flow)
        
        for i in range(1, len(typical_price)):
            if typical_price[i] > typical_price[i-1]:
                positive_flow[i] = money_flow[i]
            elif typical_price[i] < typical_price[i-1]:
                negative_flow[i] = money_flow[i]
                
        # 计算周期内的资金流
        positive_mf = np.zeros_like(positive_flow)
        negative_mf = np.zeros_like(negative_flow)
        
        for i in range(period - 1, len(positive_flow)):
            positive_mf[i] = np.sum(positive_flow[i-period+1:i+1])
            negative_mf[i] = np.sum(negative_flow[i-period+1:i+1])
            
        # 计算资金流比率和MFI
        mfr = np.where(negative_mf != 0, positive_mf / negative_mf, 100)
        mfi = 100 - (100 / (1 + mfr))
        
        signals = self.generate_signals(mfi)
        levels = self.calculate_levels(mfi)
        
        metadata = {
            'positive_flow': positive_flow,
            'negative_flow': negative_flow,
            'money_flow_ratio': mfr
        }
        
        return IndicatorResult(values=mfi, signals=signals, levels=levels, metadata=metadata)
        
    def generate_signals(self, values: np.ndarray) -> dict:
        """生成交易信号"""
        signals = {
            'overbought': [],    # MFI > 80
            'oversold': [],      # MFI < 20
            'bullish': [],       # MFI上升
            'bearish': []        # MFI下降
        }
        
        for i in range(1, len(values)):
            signals['overbought'].append(values[i] > 80)
            signals['oversold'].append(values[i] < 20)
            signals['bullish'].append(
                values[i-1] < values[i] and values[i-1] < 50 and values[i] > 50
            )
            signals['bearish'].append(
                values[i-1] > values[i] and values[i-1] > 50 and values[i] < 50
            )
            
        return signals
        
    def calculate_levels(self, values: np.ndarray) -> dict:
        """计算重要水平"""
        return {
            'overbought': 80.0,
            'oversold': 20.0,
            'middle': 50.0
        }
