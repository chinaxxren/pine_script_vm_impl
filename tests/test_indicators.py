"""
技术指标测试
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.indicators.base import Indicator
from src.indicators.moving_averages import SMA, EMA, WMA
from src.indicators.trend import MACD, ADX
from src.indicators.oscillators import RSI, Stochastic
from src.indicators.volatility import BollingerBands, ATR
from src.indicators.volume import OBV, AccumulationDistribution
from src.indicators.composite import SuperTrend, IchimokuCloud

class TestIndicatorBase(unittest.TestCase):
    """基础指标测试"""
    
    def setUp(self):
        """准备测试数据"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        self.prices = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 100,
            'high': np.random.randn(len(dates)).cumsum() + 102,
            'low': np.random.randn(len(dates)).cumsum() + 98,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
class TestMovingAverages(TestIndicatorBase):
    """移动平均测试"""
    
    def test_sma(self):
        """测试简单移动平均"""
        length = 10
        sma = SMA(length)
        result = sma.calculate(self.prices['close'])
        
        # 验证结果
        self.assertEqual(len(result), len(self.prices))
        self.assertTrue(pd.isna(result[:length-1]).all())  # 前 length-1 个值应该是 NaN
        self.assertTrue(pd.notna(result[length-1:]).all())  # 之后的值应该都有效
        
        # 验证计算正确性
        manual_sma = self.prices['close'].rolling(window=length).mean()
        pd.testing.assert_series_equal(result, manual_sma)
        
    def test_ema(self):
        """测试指数移动平均"""
        length = 10
        ema = EMA(length)
        result = ema.calculate(self.prices['close'])
        
        # 验证结果
        self.assertEqual(len(result), len(self.prices))
        self.assertTrue(pd.isna(result[:length-1]).all())
        self.assertTrue(pd.notna(result[length-1:]).all())
        
        # 验证计算正确性
        alpha = 2 / (length + 1)
        manual_ema = self.prices['close'].ewm(alpha=alpha, adjust=False).mean()
        pd.testing.assert_series_equal(result, manual_ema)
        
    def test_wma(self):
        """测试加权移动平均"""
        length = 10
        wma = WMA(length)
        result = wma.calculate(self.prices['close'])
        
        # 验证结果
        self.assertEqual(len(result), len(self.prices))
        self.assertTrue(pd.isna(result[:length-1]).all())
        self.assertTrue(pd.notna(result[length-1:]).all())
        
class TestTrendIndicators(TestIndicatorBase):
    """趋势指标测试"""
    
    def test_macd(self):
        """测试 MACD"""
        macd = MACD(12, 26, 9)
        macd_line, signal_line, histogram = macd.calculate(self.prices['close'])
        
        # 验证结果
        self.assertEqual(len(macd_line), len(self.prices))
        self.assertEqual(len(signal_line), len(self.prices))
        self.assertEqual(len(histogram), len(self.prices))
        
        # 验证计算关系
        pd.testing.assert_series_equal(histogram, macd_line - signal_line)
        
    def test_adx(self):
        """测试 ADX"""
        length = 14
        adx = ADX(length)
        adx_line, plus_di, minus_di = adx.calculate(
            self.prices['high'],
            self.prices['low'],
            self.prices['close']
        )
        
        # 验证结果
        self.assertEqual(len(adx_line), len(self.prices))
        self.assertEqual(len(plus_di), len(self.prices))
        self.assertEqual(len(minus_di), len(self.prices))
        
        # ADX 应该在 0-100 之间
        self.assertTrue((adx_line >= 0).all() and (adx_line <= 100).all())
        
class TestOscillators(TestIndicatorBase):
    """振荡器测试"""
    
    def test_rsi(self):
        """测试 RSI"""
        length = 14
        rsi = RSI(length)
        result = rsi.calculate(self.prices['close'])
        
        # 验证结果
        self.assertEqual(len(result), len(self.prices))
        
        # RSI 应该在 0-100 之间
        self.assertTrue((result >= 0).all() and (result <= 100).all())
        
    def test_stochastic(self):
        """测试随机指标"""
        k_length = 14
        d_length = 3
        stoch = Stochastic(k_length, d_length)
        k_line, d_line = stoch.calculate(
            self.prices['high'],
            self.prices['low'],
            self.prices['close']
        )
        
        # 验证结果
        self.assertEqual(len(k_line), len(self.prices))
        self.assertEqual(len(d_line), len(self.prices))
        
        # 值应该在 0-100 之间
        self.assertTrue((k_line >= 0).all() and (k_line <= 100).all())
        self.assertTrue((d_line >= 0).all() and (d_line <= 100).all())
        
class TestVolatilityIndicators(TestIndicatorBase):
    """波动率指标测试"""
    
    def test_bollinger_bands(self):
        """测试布林带"""
        length = 20
        multiplier = 2
        bb = BollingerBands(length, multiplier)
        upper, middle, lower = bb.calculate(self.prices['close'])
        
        # 验证结果
        self.assertEqual(len(upper), len(self.prices))
        self.assertEqual(len(middle), len(self.prices))
        self.assertEqual(len(lower), len(self.prices))
        
        # 验证布林带关系
        self.assertTrue((upper >= middle).all())
        self.assertTrue((middle >= lower).all())
        
    def test_atr(self):
        """测试 ATR"""
        length = 14
        atr = ATR(length)
        result = atr.calculate(
            self.prices['high'],
            self.prices['low'],
            self.prices['close']
        )
        
        # 验证结果
        self.assertEqual(len(result), len(self.prices))
        self.assertTrue((result >= 0).all())  # ATR 应该是非负的
        
class TestVolumeIndicators(TestIndicatorBase):
    """成交量指标测试"""
    
    def test_obv(self):
        """测试 OBV"""
        obv = OBV()
        result = obv.calculate(self.prices['close'], self.prices['volume'])
        
        # 验证结果
        self.assertEqual(len(result), len(self.prices))
        
    def test_accumulation_distribution(self):
        """测试 A/D 线"""
        ad = AccumulationDistribution()
        result = ad.calculate(
            self.prices['high'],
            self.prices['low'],
            self.prices['close'],
            self.prices['volume']
        )
        
        # 验证结果
        self.assertEqual(len(result), len(self.prices))
        
class TestCompositeIndicators(TestIndicatorBase):
    """复合指标测试"""
    
    def test_supertrend(self):
        """测试 SuperTrend"""
        length = 10
        multiplier = 3
        supertrend = SuperTrend(length, multiplier)
        trend, direction = supertrend.calculate(
            self.prices['high'],
            self.prices['low'],
            self.prices['close']
        )
        
        # 验证结果
        self.assertEqual(len(trend), len(self.prices))
        self.assertEqual(len(direction), len(self.prices))
        
        # direction 应该是 1 或 -1
        self.assertTrue(((direction == 1) | (direction == -1) | pd.isna(direction)).all())
        
    def test_ichimoku(self):
        """测试一目均衡表"""
        ichimoku = IchimokuCloud()
        tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku.calculate(
            self.prices['high'],
            self.prices['low']
        )
        
        # 验证结果
        self.assertEqual(len(tenkan), len(self.prices))
        self.assertEqual(len(kijun), len(self.prices))
        self.assertEqual(len(senkou_a), len(self.prices))
        self.assertEqual(len(senkou_b), len(self.prices))
        self.assertEqual(len(chikou), len(self.prices))
        
if __name__ == '__main__':
    unittest.main()
