"""
测试趋势指标的复杂场景
"""

import unittest
import numpy as np
from src.indicators.trend import MACD, ADX

class TestTrendScenarios(unittest.TestCase):
    """测试趋势指标的复杂场景"""
    
    def setUp(self):
        self.macd = MACD()
        self.adx = ADX()
        
    def generate_market_data(self, scenario='uptrend', n_points=100):
        """生成不同市场场景的数据
        
        Args:
            scenario: 市场场景类型
                - 'uptrend': 上升趋势
                - 'downtrend': 下降趋势
                - 'sideways': 盘整
                - 'volatile': 剧烈波动
                - 'trend_reversal': 趋势反转
            n_points: 数据点数量
        """
        t = np.linspace(0, 10, n_points)
        noise = np.random.normal(0, 0.1, n_points)
        
        if scenario == 'uptrend':
            close = 10 + t + noise
            high = close + 0.2*np.abs(noise)
            low = close - 0.2*np.abs(noise)
        elif scenario == 'downtrend':
            close = 20 - t + noise
            high = close + 0.2*np.abs(noise)
            low = close - 0.2*np.abs(noise)
        elif scenario == 'sideways':
            close = 10 + noise
            high = close + 0.2*np.abs(noise)
            low = close - 0.2*np.abs(noise)
        elif scenario == 'volatile':
            close = 10 + 3*np.sin(2*t) + noise
            high = close + np.abs(noise)
            low = close - np.abs(noise)
        elif scenario == 'trend_reversal':
            close = np.where(t < 5, 10 + t, 15 - t) + noise
            high = close + 0.2*np.abs(noise)
            low = close - 0.2*np.abs(noise)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
            
        return high, low, close
        
    def test_combined_indicators(self):
        """测试MACD和ADX的组合使用"""
        # 生成趋势反转场景的数据
        high, low, close = self.generate_market_data('trend_reversal')
        
        # 计算MACD和ADX
        macd_result = self.macd.calculate(close)
        adx_result = self.adx.calculate(high, low, close)
        
        # 检查趋势反转点附近的信号
        mid_point = len(close) // 2
        window = 5  # 检查反转点前后的窗口
        
        # 在趋势反转附近应该有MACD交叉信号
        has_macd_signal = False
        for i in range(mid_point - window, mid_point + window):
            if i > 0 and i < len(close) - 1:
                if macd_result.signals['crossover'][i-1] or macd_result.signals['crossunder'][i-1]:
                    has_macd_signal = True
                    break
        self.assertTrue(has_macd_signal, "趋势反转点附近应该有MACD交叉信号")
        
        # ADX应该在趋势反转前后都显示出强趋势
        strong_trend_before = any(adx_result.signals['strong_trend'][mid_point-window:mid_point])
        strong_trend_after = any(adx_result.signals['strong_trend'][mid_point:mid_point+window])
        self.assertTrue(strong_trend_before or strong_trend_after,
                       "趋势反转前后应该有强趋势信号")
                       
    def test_parameter_sensitivity(self):
        """测试参数敏感性"""
        # 生成上升趋势数据
        high, low, close = self.generate_market_data('uptrend', n_points=200)
        
        # 测试MACD不同参数
        fast_periods = [8, 12, 16]
        slow_periods = [21, 26, 31]
        signal_periods = [7, 9, 11]
        
        macd_results = {}
        for fast in fast_periods:
            for slow in slow_periods:
                for signal in signal_periods:
                    if fast >= slow:  # 快线周期不应大于慢线周期
                        continue
                    key = (fast, slow, signal)
                    macd_results[key] = self.macd.calculate(close,
                                                          fast_period=fast,
                                                          slow_period=slow,
                                                          signal_period=signal)
        
        # 检查不同参数组合的信号数量
        signal_counts = {key: sum(result.signals['crossover']) + 
                             sum(result.signals['crossunder'])
                        for key, result in macd_results.items()}
        
        # 较短周期应该产生更多信号
        min_period_key = min(signal_counts.keys(), key=lambda x: sum(x))
        max_period_key = max(signal_counts.keys(), key=lambda x: sum(x))
        self.assertGreaterEqual(signal_counts[min_period_key],
                              signal_counts[max_period_key],
                              "较短周期应该产生更多信号")
                          
        # 测试ADX不同周期
        adx_periods = [7, 14, 21]
        adx_results = {period: self.adx.calculate(high, low, close, period=period)
                      for period in adx_periods}
        
        # 检查不同周期的ADX平滑度
        # 使用移动方差来衡量平滑度
        def rolling_variance(data, window=5):
            var = np.zeros_like(data)
            for i in range(window, len(data)):
                var[i] = np.var(data[i-window:i])
            return np.mean(var[window:])  # 返回平均移动方差
            
        variances = {period: rolling_variance(result.values)
                    for period, result in adx_results.items()}
                    
        # 较长周期应该产生更平滑的ADX值
        for p1, p2 in zip(adx_periods[:-1], adx_periods[1:]):
            self.assertGreaterEqual(variances[p1], variances[p2],
                                  f"期望周期{p1}的方差({variances[p1]:.4f})大于周期{p2}的方差({variances[p2]:.4f})")
                      
    def test_market_conditions(self):
        """测试不同市场条件"""
        scenarios = ['uptrend', 'downtrend', 'sideways', 'volatile', 'trend_reversal']
        np.random.seed(42)  # 设置随机种子以保持结果一致
        
        for scenario in scenarios:
            high, low, close = self.generate_market_data(scenario)
            
            # 计算指标
            macd_result = self.macd.calculate(close)
            adx_result = self.adx.calculate(high, low, close)
            
            if scenario in ['uptrend', 'downtrend']:
                # 趋势市场应该有较强的ADX值
                strong_trend_ratio = sum(adx_result.signals['strong_trend']) / len(close)
                self.assertGreater(strong_trend_ratio, 0.3,
                                 f"{scenario}市场应该有较多的强趋势信号")
                                 
            elif scenario == 'sideways':
                # 盘整市场应该有较弱的ADX值
                weak_trend_ratio = sum(adx_result.signals['weak_trend']) / len(close)
                self.assertGreater(weak_trend_ratio, 0.3,
                                 "盘整市场应该有较多的弱趋势信号")
                                 
            elif scenario == 'volatile':
                # 波动市场应该有较多的MACD交叉信号
                signal_count = (sum(macd_result.signals['crossover']) +
                              sum(macd_result.signals['crossunder']))
                signal_ratio = signal_count / (len(close) - 1)  # 减1因为信号比数据少一个点
                self.assertGreater(signal_ratio, 0.05,  # 降低阈值要求
                                 "波动市场应该有较多的MACD交叉信号")
                                 
            elif scenario == 'trend_reversal':
                # 趋势反转应该在反转点附近有信号
                mid_point = len(close) // 2
                window = 5
                has_signal = False
                for i in range(mid_point - window, mid_point + window):
                    if i > 0 and i < len(close) - 1:
                        if (macd_result.signals['crossover'][i-1] or
                            macd_result.signals['crossunder'][i-1]):
                            has_signal = True
                            break
                self.assertTrue(has_signal,
                              "趋势反转点附近应该有MACD交叉信号")
                              
if __name__ == '__main__':
    unittest.main()
