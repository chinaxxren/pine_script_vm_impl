from typing import Dict, Any
import numpy as np
from .base import Indicator, IndicatorResult

class IchimokuCloud(Indicator):
    """
    一目均衡图（Ichimoku Cloud）指标
    包含五条线：
    1. 转换线（Conversion Line）：9日的最高点和最低点的平均值
    2. 基准线（Base Line）：26日的最高点和最低点的平均值
    3. 先行带A（Leading Span A）：转换线和基准线的平均值，向前推26天
    4. 先行带B（Leading Span B）：52日的最高点和最低点的平均值，向前推26天
    5. 延迟线（Lagging Span）：收盘价向后推26天
    """
    
    def __init__(self, 
                 conversion_period: int = 9,
                 base_period: int = 26,
                 lagging_span2_period: int = 52,
                 displacement: int = 26):
        """
        初始化一目均衡图指标
        
        参数:
            conversion_period: 转换线周期（默认9）
            base_period: 基准线周期（默认26）
            lagging_span2_period: 先行带B周期（默认52）
            displacement: 位移周期（默认26）
        """
        super().__init__()
        self.conversion_period = conversion_period
        self.base_period = base_period
        self.lagging_span2_period = lagging_span2_period
        self.displacement = displacement
        
    def _get_period_high_low(self, high: np.ndarray, low: np.ndarray, period: int) -> tuple:
        """计算指定周期的最高点和最低点"""
        highs = np.array([np.nan] * len(high))
        lows = np.array([np.nan] * len(low))
        
        for i in range(period - 1, len(high)):
            highs[i] = np.max(high[i-period+1:i+1])
            lows[i] = np.min(low[i-period+1:i+1])
            
        return highs, lows
        
    def calculate(self, **kwargs) -> IndicatorResult:
        """
        计算一目均衡图指标
        
        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            
        返回:
            包含一目均衡图各条线的 IndicatorResult
        """
        high = kwargs.get('high')
        low = kwargs.get('low')
        close = kwargs.get('close')
        
        if high is None or low is None or close is None:
            raise ValueError("Missing required price data")
            
        # 计算转换线
        conversion_highs, conversion_lows = self._get_period_high_low(high, low, self.conversion_period)
        conversion = (conversion_highs + conversion_lows) / 2
        
        # 计算基准线
        base_highs, base_lows = self._get_period_high_low(high, low, self.base_period)
        base = (base_highs + base_lows) / 2
        
        # 计算先行带A
        leading_span_a = (conversion + base) / 2
        
        # 计算先行带B
        span2_highs, span2_lows = self._get_period_high_low(high, low, self.lagging_span2_period)
        leading_span_b = (span2_highs + span2_lows) / 2
        
        # 计算延迟线（收盘价向后移动）
        lagging_span = np.roll(close, self.displacement)
        
        # 向前移动先行带
        leading_span_a = np.roll(leading_span_a, -self.displacement)
        leading_span_b = np.roll(leading_span_b, -self.displacement)
        
        # 生成信号
        signals = {
            'conversion': conversion,  # 转换线
            'base': base,  # 基准线
            'leading_span_a': leading_span_a,  # 先行带A
            'leading_span_b': leading_span_b,  # 先行带B
            'lagging_span': lagging_span  # 延迟线
        }
        
        # 返回云层作为主要指标值（先行带A和B之间的区域）
        cloud = leading_span_a - leading_span_b
        
        return IndicatorResult(cloud, signals)
