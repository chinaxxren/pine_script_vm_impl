"""
测试振荡器指标
"""

import unittest
import numpy as np
from src.indicators.oscillators import RSI, Stochastic

class TestRSI(unittest.TestCase):
    """测试RSI指标"""
    
    def setUp(self):
        self.rsi = RSI()
        
    def test_rsi_calculation(self):
        """测试RSI计算"""
        data = np.array([10, 12, 11, 13, 15, 14, 16, 18, 17, 19])
        result = self.rsi.calculate(data, period=3)
        
        self.assertIsNotNone(result.values)
        self.assertEqual(len(result.values), len(data))
        
        # RSI应该在0-100之间
        self.assertTrue(np.all(result.values >= 0))
        self.assertTrue(np.all(result.values <= 100))
        
    def test_rsi_signals(self):
        """测试RSI信号"""
        data = np.array([10, 12, 11, 13, 15, 14, 16, 18, 17, 19])
        result = self.rsi.calculate(data, period=3)
        
        self.assertIn('oversold', result.signals)
        self.assertIn('overbought', result.signals)
        self.assertIn('bullish', result.signals)
        self.assertIn('bearish', result.signals)
        
    def test_rsi_levels(self):
        """测试RSI水平"""
        data = np.array([10, 12, 11, 13, 15, 14, 16, 18, 17, 19])
        result = self.rsi.calculate(data, period=3)
        
        self.assertIn('oversold', result.levels)
        self.assertIn('overbought', result.levels)
        self.assertIn('neutral', result.levels)
        
        self.assertEqual(result.levels['oversold'], 30.0)
        self.assertEqual(result.levels['overbought'], 70.0)
        self.assertEqual(result.levels['neutral'], 50.0)
        
class TestStochastic(unittest.TestCase):
    """测试随机指标"""
    
    def setUp(self):
        self.stoch = Stochastic()
        
    def test_stochastic_calculation(self):
        """测试随机指标计算"""
        high = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        low = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        close = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        
        result = self.stoch.calculate(high, low, close, k_period=3, d_period=2)
        
        self.assertIsNotNone(result.values)
        self.assertIn('k', result.values)
        self.assertIn('d', result.values)
        
        # K值和D值应该在0-100之间
        self.assertTrue(np.all(result.values['k'] >= 0))
        self.assertTrue(np.all(result.values['k'] <= 100))
        self.assertTrue(np.all(result.values['d'] >= 0))
        self.assertTrue(np.all(result.values['d'] <= 100))
        
    def test_stochastic_signals(self):
        """测试随机指标信号"""
        high = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        low = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        close = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        
        result = self.stoch.calculate(high, low, close, k_period=3, d_period=2)
        
        self.assertIn('oversold', result.signals)
        self.assertIn('overbought', result.signals)
        self.assertIn('bullish_cross', result.signals)
        self.assertIn('bearish_cross', result.signals)
        
    def test_stochastic_levels(self):
        """测试随机指标水平"""
        high = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        low = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        close = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        
        result = self.stoch.calculate(high, low, close, k_period=3, d_period=2)
        
        self.assertIn('oversold', result.levels)
        self.assertIn('overbought', result.levels)
        self.assertIn('neutral', result.levels)
        
        self.assertEqual(result.levels['oversold'], 20.0)
        self.assertEqual(result.levels['overbought'], 80.0)
        self.assertEqual(result.levels['neutral'], 50.0)
        
if __name__ == '__main__':
    unittest.main()
