from typing import Dict, Any
import numpy as np
from .base import Indicator, IndicatorResult
from .volatility import ATR

class SuperTrend(Indicator):
    """
    SuperTrend 指标
    SuperTrend = 价格 +/- (multiplier * ATR)
    """
    
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        """
        初始化 SuperTrend 指标
        
        参数:
            period: ATR 周期
            multiplier: ATR 乘数
        """
        super().__init__()
        self.period = period
        self.multiplier = multiplier
        self.atr = ATR(period)
        
    def calculate(self, **kwargs) -> IndicatorResult:
        """
        计算 SuperTrend 值
        
        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            
        返回:
            包含 SuperTrend 值和信号的 IndicatorResult
        """
        high = kwargs.get('high')
        low = kwargs.get('low')
        close = kwargs.get('close')
        
        if high is None or low is None or close is None:
            raise ValueError("Missing required price data")
            
        # 计算 ATR
        atr_result = self.atr.calculate(**kwargs)
        atr = atr_result.values
        
        # 计算基本上轨和下轨
        basic_upperband = (high + low) / 2 + self.multiplier * atr
        basic_lowerband = (high + low) / 2 - self.multiplier * atr
        
        # 初始化最终上下轨
        final_upperband = np.zeros_like(close)
        final_lowerband = np.zeros_like(close)
        supertrend = np.zeros_like(close)
        
        # 初始化趋势方向
        trend = np.zeros_like(close)  # 1 表示上涨趋势，-1 表示下跌趋势
        
        # 第一个值初始化
        final_upperband[0] = basic_upperband[0]
        final_lowerband[0] = basic_lowerband[0]
        trend[0] = 1 if close[0] > basic_upperband[0] else -1
        
        # 计算 SuperTrend
        for i in range(1, len(close)):
            # 计算上轨
            if basic_upperband[i] < final_upperband[i-1] or close[i-1] > final_upperband[i-1]:
                final_upperband[i] = basic_upperband[i]
            else:
                final_upperband[i] = final_upperband[i-1]
                
            # 计算下轨
            if basic_lowerband[i] > final_lowerband[i-1] or close[i-1] < final_lowerband[i-1]:
                final_lowerband[i] = basic_lowerband[i]
            else:
                final_lowerband[i] = final_lowerband[i-1]
                
            # 确定趋势
            if trend[i-1] == 1:  # 之前是上涨趋势
                if close[i] < final_lowerband[i]:
                    trend[i] = -1  # 转为下跌趋势
                else:
                    trend[i] = 1
            else:  # 之前是下跌趋势
                if close[i] > final_upperband[i]:
                    trend[i] = 1  # 转为上涨趋势
                else:
                    trend[i] = -1
                    
            # 根据趋势确定 SuperTrend 值
            supertrend[i] = final_upperband[i] if trend[i] == -1 else final_lowerband[i]
            
        # 生成交易信号
        signals = {
            'trend': trend,
            'upperband': final_upperband,
            'lowerband': final_lowerband
        }
        
        return IndicatorResult(supertrend, signals)
