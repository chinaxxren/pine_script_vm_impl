"""
测试趋势指标
"""

import unittest
import numpy as np
from src.indicators.trend import MACD, ADX

class TestMACD(unittest.TestCase):
    """测试MACD指标"""
    
    def setUp(self):
        self.macd = MACD()
        
    def test_macd_calculation(self):
        """测试MACD计算"""
        data = np.array([10, 12, 11, 13, 15, 14, 16, 18, 17, 19,
                        20, 21, 22, 20, 19, 18, 17, 16, 15, 14])
        result = self.macd.calculate(data)
        
        self.assertIsNotNone(result.values)
        self.assertIn('macd_line', result.metadata)
        self.assertIn('signal_line', result.metadata)
        self.assertIn('histogram', result.metadata)
        
        # 检查数据长度
        self.assertEqual(len(result.metadata['macd_line']), len(data))
        self.assertEqual(len(result.metadata['signal_line']), len(data))
        self.assertEqual(len(result.metadata['histogram']), len(data))
        
        # 检查前几个值是否正确初始化
        self.assertAlmostEqual(result.metadata['macd_line'][0], 0)
        self.assertAlmostEqual(result.metadata['signal_line'][0], 0)
        self.assertAlmostEqual(result.metadata['histogram'][0], 0)
        
    def test_macd_signals(self):
        """测试MACD信号"""
        data = np.array([10, 12, 11, 13, 15, 14, 16, 18, 17, 19,
                        20, 21, 22, 20, 19, 18, 17, 16, 15, 14])
        result = self.macd.calculate(data)
        
        self.assertIn('crossover', result.signals)
        self.assertIn('crossunder', result.signals)
        
        # 检查信号长度
        self.assertEqual(len(result.signals['crossover']), len(data)-1)
        self.assertEqual(len(result.signals['crossunder']), len(data)-1)
        
    def test_macd_edge_cases(self):
        """测试MACD边界情况"""
        # 测试单一值
        single_value = np.array([10])
        result = self.macd.calculate(single_value)
        self.assertEqual(len(result.values), 1)
        self.assertAlmostEqual(result.values[0], 0)
        
        # 测试全相同值
        same_values = np.array([10] * 20)
        result = self.macd.calculate(same_values)
        self.assertTrue(np.allclose(result.values, 0))
        
        # 测试极端值
        extreme_values = np.array([1e6, 1e-6, 1e6, 1e-6] * 5)
        result = self.macd.calculate(extreme_values)
        self.assertFalse(np.any(np.isnan(result.values)))
        self.assertFalse(np.any(np.isinf(result.values)))
        
    def test_macd_trend_change(self):
        """测试MACD趋势变化"""
        # 创建一个明显的趋势变化序列
        uptrend = np.linspace(10, 20, 10)  # 上升趋势
        downtrend = np.linspace(20, 10, 10)  # 下降趋势
        data = np.concatenate([uptrend, downtrend])
        
        result = self.macd.calculate(data)
        
        # 在趋势变化点附近应该有信号
        crossover_count = sum(result.signals['crossover'])
        crossunder_count = sum(result.signals['crossunder'])
        self.assertGreater(crossover_count + crossunder_count, 0)
        
class TestADX(unittest.TestCase):
    """测试ADX指标"""
    
    def setUp(self):
        self.adx = ADX()
        
    def test_adx_calculation(self):
        """测试ADX计算"""
        high = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                        22, 23, 24, 23, 22, 21, 20, 19, 18, 17])
        low = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                       20, 21, 22, 21, 20, 19, 18, 17, 16, 15])
        close = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                         21, 22, 23, 22, 21, 20, 19, 18, 17, 16])
        
        result = self.adx.calculate(high, low, close)
        
        self.assertIsNotNone(result.values)
        self.assertIn('plus_di', result.metadata)
        self.assertIn('minus_di', result.metadata)
        
        # 检查数据长度
        self.assertEqual(len(result.values), len(high))
        self.assertEqual(len(result.metadata['plus_di']), len(high))
        self.assertEqual(len(result.metadata['minus_di']), len(high))
        
        # ADX应该在0-100之间
        self.assertTrue(np.all(result.values >= 0))
        self.assertTrue(np.all(result.values <= 100))
        
        # +DI和-DI应该在0-100之间
        self.assertTrue(np.all(result.metadata['plus_di'] >= 0))
        self.assertTrue(np.all(result.metadata['plus_di'] <= 100))
        self.assertTrue(np.all(result.metadata['minus_di'] >= 0))
        self.assertTrue(np.all(result.metadata['minus_di'] <= 100))
        
    def test_adx_signals(self):
        """测试ADX信号"""
        high = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                        22, 23, 24, 23, 22, 21, 20, 19, 18, 17])
        low = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                       20, 21, 22, 21, 20, 19, 18, 17, 16, 15])
        close = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                         21, 22, 23, 22, 21, 20, 19, 18, 17, 16])
        
        result = self.adx.calculate(high, low, close)
        
        self.assertIn('strong_trend', result.signals)
        self.assertIn('weak_trend', result.signals)
        self.assertIn('bullish', result.signals)
        self.assertIn('bearish', result.signals)
        
        # 检查信号长度
        self.assertEqual(len(result.signals['strong_trend']), len(high))
        self.assertEqual(len(result.signals['weak_trend']), len(high))
        self.assertEqual(len(result.signals['bullish']), len(high))
        self.assertEqual(len(result.signals['bearish']), len(high))
        
    def test_adx_levels(self):
        """测试ADX水平"""
        high = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        low = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        close = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        
        result = self.adx.calculate(high, low, close)
        
        self.assertIn('strong_trend', result.levels)
        self.assertIn('very_strong_trend', result.levels)
        
        self.assertEqual(result.levels['strong_trend'], 25.0)
        self.assertEqual(result.levels['very_strong_trend'], 50.0)
        
    def test_adx_edge_cases(self):
        """测试ADX边界情况"""
        # 测试单一值
        single_value = np.array([10])
        result = self.adx.calculate(single_value, single_value, single_value)
        self.assertEqual(len(result.values), 1)
        self.assertGreaterEqual(result.values[0], 0)
        self.assertLessEqual(result.values[0], 100)
        
        # 测试全相同值
        same_values = np.array([10] * 20)
        result = self.adx.calculate(same_values, same_values, same_values)
        self.assertTrue(np.all(result.values >= 0))
        self.assertTrue(np.all(result.values <= 100))
        
        # 测试极端值差异
        high = np.array([1e6, 1e6, 1e6] * 7)
        low = np.array([1e-6, 1e-6, 1e-6] * 7)
        close = np.array([1e3, 1e3, 1e3] * 7)
        result = self.adx.calculate(high, low, close)
        self.assertFalse(np.any(np.isnan(result.values)))
        self.assertFalse(np.any(np.isinf(result.values)))
        
    def test_adx_trend_change(self):
        """测试ADX趋势变化"""
        # 创建一个明显的趋势变化序列
        n_points = 20
        t = np.linspace(0, 4*np.pi, n_points)
        
        # 使用正弦波生成高低点，创造趋势变化
        high = 15 + 5*np.sin(t)
        low = 13 + 5*np.sin(t)
        close = 14 + 5*np.sin(t)
        
        result = self.adx.calculate(high, low, close)
        
        # 检查趋势信号
        strong_trend_count = sum(result.signals['strong_trend'])
        weak_trend_count = sum(result.signals['weak_trend'])
        bullish_count = sum(result.signals['bullish'])
        bearish_count = sum(result.signals['bearish'])
        
        # 应该至少有一些趋势信号
        self.assertGreater(strong_trend_count + weak_trend_count, 0)
        self.assertGreater(bullish_count + bearish_count, 0)
        
if __name__ == '__main__':
    unittest.main()
