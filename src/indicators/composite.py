from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
from .base import Indicator, IndicatorResult
from .moving_averages import SMA, EMA
from .trend import MACD
from .oscillators import RSI, Stochastic
from .volume import OBV, MoneyFlowIndex
from .supertrend import SuperTrend
from .ichimoku import IchimokuCloud

class CompositeIndicator(Indicator):
    """组合指标"""
    
    def __init__(self, indicators: List[Indicator],
                 combine_func: Callable[[List[np.ndarray]], np.ndarray]):
        super().__init__()
        self.indicators = indicators
        self.combine_func = combine_func
        
    def calculate(self, **kwargs) -> IndicatorResult:
        """计算组合指标值"""
        # 计算每个子指标
        results = []
        signals = {}
        
        for indicator in self.indicators:
            result = indicator.calculate(**kwargs)
            results.append(result.values)
            signals.update(result.signals)
            
        # 组合结果
        values = self.combine_func(results)
        
        return IndicatorResult(values=values, signals=signals)

class TrendConfirmation(CompositeIndicator):
    """趋势确认指标"""
    
    def __init__(self, period: int = 14):
        # 使用MACD和RSI组合
        macd = MACD()
        rsi = RSI()
        
        def combine(results: List[np.ndarray]) -> np.ndarray:
            macd_hist = results[0]
            rsi_values = results[1]
            
            # 归一化
            macd_norm = (macd_hist - np.min(macd_hist)) / (np.max(macd_hist) - np.min(macd_hist))
            rsi_norm = rsi_values / 100
            
            # 组合信号
            return (macd_norm + rsi_norm) / 2
            
        super().__init__([macd, rsi], combine)

class VolumePriceDivergence(CompositeIndicator):
    """量价背离指标"""
    
    def __init__(self, period: int = 14):
        # 使用OBV和MFI组合
        obv = OBV()
        mfi = MoneyFlowIndex(period)
        
        def combine(results: List[np.ndarray]) -> np.ndarray:
            obv_values = results[0]
            mfi_values = results[1]
            
            # 归一化
            obv_norm = (obv_values - np.min(obv_values)) / (np.max(obv_values) - np.min(obv_values))
            mfi_norm = mfi_values / 100
            
            # 计算背离
            return np.abs(obv_norm - mfi_norm)
            
        super().__init__([obv, mfi], combine)

class MarketRegime(CompositeIndicator):
    """市场状态指标"""
    
    def __init__(self, period: int = 20):
        # 使用EMA和RSI组合
        ema = EMA(period)
        rsi = RSI()
        
        def combine(results: List[np.ndarray]) -> np.ndarray:
            ema_values = results[0]
            rsi_values = results[1]
            
            # 归一化
            ema_trend = np.gradient(ema_values)
            ema_norm = (ema_trend - np.min(ema_trend)) / (np.max(ema_trend) - np.min(ema_trend))
            rsi_norm = rsi_values / 100
            
            # 组合信号
            regime = (ema_norm + rsi_norm) / 2
            return regime
            
        super().__init__([ema, rsi], combine)

def create_trend_filter(threshold: float = 0.6):
    """创建趋势过滤器"""
    def trend_filter(signals: Dict[str, Any]) -> bool:
        if 'trend_strength' in signals:
            return signals['trend_strength'] > threshold
        return True
    return trend_filter

def create_volatility_filter(atr_threshold: float = 0.02):
    """创建波动率过滤器"""
    def volatility_filter(signals: Dict[str, Any]) -> bool:
        if 'atr' in signals:
            return signals['atr'] > atr_threshold
        return True
    return volatility_filter

def create_momentum_filter(rsi_threshold: float = 30):
    """创建动量过滤器"""
    def momentum_filter(signals: Dict[str, Any]) -> bool:
        if 'rsi' in signals:
            return signals['rsi'] > rsi_threshold
        return True
    return momentum_filter
