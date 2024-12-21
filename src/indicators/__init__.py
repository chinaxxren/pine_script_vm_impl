"""
指标模块
"""

from .base import Indicator, IndicatorResult
from .moving_averages import SMA, EMA, WMA
from .trend import MACD, ADX, DI, ATR
from .oscillators import RSI, Stochastic, KDJ
from .volatility import BollingerBands, ATR
from .volume import OBV, AccumulationDistribution, MoneyFlowIndex
from .composite import TrendConfirmation, VolumePriceDivergence, MarketRegime
from .supertrend import SuperTrend
from .ichimoku import IchimokuCloud
from .momentum import CCI, WilliamsR as WR

__all__ = [
    'SMA', 'EMA', 'WMA',
    'MACD', 'ADX', 'DI', 'ATR',
    'RSI', 'Stochastic', 'KDJ',
    'BollingerBands', 'ATR',
    'OBV', 'AccumulationDistribution', 'MoneyFlowIndex',
    'TrendConfirmation', 'VolumePriceDivergence', 'MarketRegime',
    'SuperTrend',
    'IchimokuCloud',
    'CCI', 'WR'
]
