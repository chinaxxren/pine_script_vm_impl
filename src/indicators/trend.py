"""
趋势指标
"""

import numpy as np
from .base import Indicator, IndicatorResult

class MACD(Indicator):
    """移动平均趋同/背离指标"""
    
    def __init__(self):
        super().__init__()
        self.description = "Moving Average Convergence/Divergence"
        self.category = "Trend"
        
    def calculate(self, data: np.ndarray,
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9) -> IndicatorResult:
        """计算MACD
        
        Args:
            data: 输入数据
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            
        Returns:
            IndicatorResult: 计算结果
        """
        self.validate_input(data)
        
        # 计算快线和慢线的EMA
        alpha_fast = 2.0 / (fast_period + 1)
        alpha_slow = 2.0 / (slow_period + 1)
        
        ema_fast = np.zeros_like(data)
        ema_slow = np.zeros_like(data)
        
        ema_fast[0] = data[0]
        ema_slow[0] = data[0]
        
        for i in range(1, len(data)):
            ema_fast[i] = alpha_fast * data[i] + (1 - alpha_fast) * ema_fast[i-1]
            ema_slow[i] = alpha_slow * data[i] + (1 - alpha_slow) * ema_slow[i-1]
            
        # 计算MACD线
        macd_line = ema_fast - ema_slow
        
        # 计算信号线
        alpha_signal = 2.0 / (signal_period + 1)
        signal_line = np.zeros_like(data)
        signal_line[0] = macd_line[0]
        
        for i in range(1, len(data)):
            signal_line[i] = alpha_signal * macd_line[i] + (1 - alpha_signal) * signal_line[i-1]
            
        # 计算柱状图
        histogram = macd_line - signal_line
        
        signals = self.generate_signals(macd_line, signal_line)
        levels = self.calculate_levels(histogram)
        
        metadata = {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }
        
        return IndicatorResult(values=histogram, signals=signals, levels=levels, metadata=metadata)
        
    def generate_signals(self, macd_line: np.ndarray, signal_line: np.ndarray) -> dict:
        """生成交易信号"""
        signals = {
            'crossover': [],   # MACD线上穿信号线
            'crossunder': []   # MACD线下穿信号线
        }
        
        for i in range(1, len(macd_line)):
            signals['crossover'].append(
                macd_line[i-1] < signal_line[i-1] and
                macd_line[i] > signal_line[i]
            )
            signals['crossunder'].append(
                macd_line[i-1] > signal_line[i-1] and
                macd_line[i] < signal_line[i]
            )
            
        return signals
        
    def calculate_levels(self, histogram: np.ndarray) -> dict:
        """计算重要水平"""
        pos_hist = histogram[histogram > 0]
        neg_hist = histogram[histogram < 0]
        
        levels = {'zero': 0.0}
        
        if len(pos_hist) > 0:
            levels['positive'] = np.percentile(pos_hist, 80)
        else:
            levels['positive'] = 0.0
            
        if len(neg_hist) > 0:
            levels['negative'] = np.percentile(neg_hist, 20)
        else:
            levels['negative'] = 0.0
            
        return levels

class ADX(Indicator):
    """平均趋向指数"""
    
    def __init__(self):
        super().__init__()
        self.description = "Average Directional Index"
        self.category = "Trend"
        
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int = 14) -> IndicatorResult:
        """计算ADX
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            period: 周期
            
        Returns:
            IndicatorResult: 计算结果
        """
        self.validate_input(high)
        self.validate_input(low)
        self.validate_input(close)
        
        # 确保数据长度足够
        if len(high) < period:
            period = len(high)
        
        # 计算真实波幅
        tr = np.zeros_like(high)
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
            
        # 计算方向变动
        pos_dm = np.zeros_like(high)
        neg_dm = np.zeros_like(high)
        
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                pos_dm[i] = up_move
            elif down_move > up_move and down_move > 0:
                neg_dm[i] = down_move
                
        # 计算移动平均
        def wilder_ma(data: np.ndarray, period: int) -> np.ndarray:
            result = np.zeros_like(data)
            result[period-1] = np.sum(data[:period])
            
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
                
            return result
            
        # 计算方向指标
        tr_ma = wilder_ma(tr, period)
        pos_dm_ma = wilder_ma(pos_dm, period)
        neg_dm_ma = wilder_ma(neg_dm, period)
        
        # 处理除零情况
        pos_di = np.zeros_like(tr_ma)
        neg_di = np.zeros_like(tr_ma)
        
        # 避免除以零
        mask = tr_ma != 0
        pos_di[mask] = 100 * pos_dm_ma[mask] / tr_ma[mask]
        neg_di[mask] = 100 * neg_dm_ma[mask] / tr_ma[mask]
        
        # 计算方向指标的绝对差异
        di_diff = np.abs(pos_di - neg_di)
        di_sum = pos_di + neg_di
        
        # 处理除零情况
        dx = np.zeros_like(di_sum)
        mask = di_sum != 0
        dx[mask] = 100 * di_diff[mask] / di_sum[mask]
        
        # 计算ADX
        adx = np.zeros_like(dx)
        adx[period-1] = np.mean(dx[:period])
        
        for i in range(period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        signals = self.generate_signals(adx)
        levels = self.calculate_levels(adx)
        
        metadata = {
            'pos_di': pos_di,
            'neg_di': neg_di,
            'dx': dx
        }
        
        return IndicatorResult(values=adx, signals=signals, levels=levels, metadata=metadata)
        
    def generate_signals(self, adx: np.ndarray) -> dict:
        """生成交易信号"""
        signals = {
            'strong_trend': [],  # ADX > 25表示强趋势
            'weak_trend': []     # ADX < 20表示弱趋势
        }
        
        for value in adx:
            signals['strong_trend'].append(value > 25)
            signals['weak_trend'].append(value < 20)
            
        return signals
        
    def calculate_levels(self, adx: np.ndarray) -> dict:
        """计算重要水平"""
        return {
            'strong_trend': 25.0,
            'weak_trend': 20.0
        }

class DI(Indicator):
    """方向指标"""
    
    def __init__(self):
        super().__init__()
        self.description = "Directional Indicator"
        self.category = "Trend"
        
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int = 14) -> IndicatorResult:
        """计算DI
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            period: 周期
            
        Returns:
            IndicatorResult: 计算结果
        """
        self.validate_input(high)
        self.validate_input(low)
        self.validate_input(close)
        
        # 计算真实波幅
        tr = np.zeros_like(high)
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
            
        # 计算方向变动
        pos_dm = np.zeros_like(high)
        neg_dm = np.zeros_like(high)
        
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                pos_dm[i] = up_move
            elif down_move > up_move and down_move > 0:
                neg_dm[i] = down_move
                
        # 计算移动平均
        def wilder_ma(data: np.ndarray, period: int) -> np.ndarray:
            result = np.zeros_like(data)
            result[period-1] = np.sum(data[:period])
            
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
                
            return result
            
        # 计算方向指标
        tr_ma = wilder_ma(tr, period)
        pos_dm_ma = wilder_ma(pos_dm, period)
        neg_dm_ma = wilder_ma(neg_dm, period)
        
        # 处理除零情况
        pos_di = np.zeros_like(tr_ma)
        neg_di = np.zeros_like(tr_ma)
        
        # 避免除以零
        mask = tr_ma != 0
        pos_di[mask] = 100 * pos_dm_ma[mask] / tr_ma[mask]
        neg_di[mask] = 100 * neg_dm_ma[mask] / tr_ma[mask]
        
        signals = self.generate_signals(pos_di, neg_di)
        levels = self.calculate_levels(pos_di, neg_di)
        
        metadata = {
            'pos_di': pos_di,
            'neg_di': neg_di
        }
        
        return IndicatorResult(values=pos_di - neg_di, signals=signals, levels=levels, metadata=metadata)
        
    def generate_signals(self, pos_di: np.ndarray, neg_di: np.ndarray) -> dict:
        """生成交易信号"""
        signals = {
            'bullish': [],  # +DI > -DI
            'bearish': []   # -DI > +DI
        }
        
        for i in range(len(pos_di)):
            signals['bullish'].append(pos_di[i] > neg_di[i])
            signals['bearish'].append(neg_di[i] > pos_di[i])
            
        return signals
        
    def calculate_levels(self, pos_di: np.ndarray, neg_di: np.ndarray) -> dict:
        """计算重要水平"""
        return {
            'strong_trend': 25.0,
            'weak_trend': 20.0
        }

class ATR(Indicator):
    """平均真实波幅"""
    
    def __init__(self):
        super().__init__()
        self.description = "Average True Range"
        self.category = "Trend"
        
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int = 14) -> IndicatorResult:
        """计算ATR
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            period: 周期
            
        Returns:
            IndicatorResult: 计算结果
        """
        self.validate_input(high)
        self.validate_input(low)
        self.validate_input(close)
        
        # 计算真实波幅
        tr = np.zeros_like(high)
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
            
        # 计算ATR
        atr = np.zeros_like(tr)
        atr[period-1] = np.mean(tr[:period])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            
        signals = self.generate_signals(atr)
        levels = self.calculate_levels(atr)
        
        metadata = {
            'tr': tr
        }
        
        return IndicatorResult(values=atr, signals=signals, levels=levels, metadata=metadata)
        
    def generate_signals(self, atr: np.ndarray) -> dict:
        """生成交易信号"""
        # 计算ATR的百分比变化
        atr_pct_change = np.zeros_like(atr)
        atr_pct_change[1:] = (atr[1:] - atr[:-1]) / atr[:-1] * 100
        
        signals = {
            'volatility_increase': [],  # ATR增加超过20%
            'volatility_decrease': []   # ATR减少超过20%
        }
        
        for value in atr_pct_change:
            signals['volatility_increase'].append(value > 20)
            signals['volatility_decrease'].append(value < -20)
            
        return signals
        
    def calculate_levels(self, atr: np.ndarray) -> dict:
        """计算重要水平"""
        return {
            'high_volatility': np.percentile(atr, 80),
            'low_volatility': np.percentile(atr, 20)
        }
