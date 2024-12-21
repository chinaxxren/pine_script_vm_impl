"""
使用Numba优化的技术指标计算函数
"""

from numba import jit, float64, int64
import numpy as np
from typing import Tuple, Optional, List

@jit(nopython=True)
def calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """计算简单移动平均"""
    result = np.zeros_like(data)
    for i in range(len(data)):
        if i < period:
            result[i] = np.mean(data[:i+1])
        else:
            result[i] = np.mean(data[i-period+1:i+1])
    return result

@jit(nopython=True)
def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """计算指数移动平均"""
    alpha = 2.0 / (period + 1)
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result

@jit(nopython=True)
def calculate_rsi(data: np.ndarray, period: int) -> np.ndarray:
    """计算相对强弱指标"""
    delta = np.zeros_like(data)
    delta[1:] = data[1:] - data[:-1]
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(data)
    avg_loss = np.zeros_like(data)
    
    # 第一个周期的平均值
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    # 后续周期的平均值
    for i in range(period+1, len(data)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-9, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi

@jit(nopython=True)
def calculate_bollinger_bands(data: np.ndarray,
                            period: int,
                            num_std: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算布林带"""
    middle = calculate_sma(data, period)
    std = np.zeros_like(data)
    
    for i in range(len(data)):
        if i < period:
            std[i] = np.std(data[:i+1])
        else:
            std[i] = np.std(data[i-period+1:i+1])
            
    upper = middle + num_std * std
    lower = middle - num_std * std
    
    return upper, middle, lower

@jit(nopython=True)
def calculate_macd(data: np.ndarray,
                  fast_period: int = 12,
                  slow_period: int = 26,
                  signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算MACD"""
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)
    
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

@jit(nopython=True)
def calculate_atr(high: np.ndarray,
                 low: np.ndarray,
                 close: np.ndarray,
                 period: int) -> np.ndarray:
    """计算平均真实范围"""
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
    return atr

@jit(nopython=True)
def calculate_keltner_channels(high: np.ndarray,
                             low: np.ndarray,
                             close: np.ndarray,
                             period: int = 20,
                             atr_multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算肯特纳通道"""
    middle = calculate_ema(close, period)
    atr = calculate_atr(high, low, close, period)
    
    upper = middle + atr_multiplier * atr
    lower = middle - atr_multiplier * atr
    
    return upper, middle, lower

@jit(nopython=True)
def calculate_stochastic(high: np.ndarray,
                        low: np.ndarray,
                        close: np.ndarray,
                        k_period: int = 14,
                        d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """计算随机指标"""
    k = np.zeros_like(close)
    
    for i in range(k_period-1, len(close)):
        high_window = high[i-k_period+1:i+1]
        low_window = low[i-k_period+1:i+1]
        highest_high = np.max(high_window)
        lowest_low = np.min(low_window)
        
        if highest_high == lowest_low:
            k[i] = 100
        else:
            k[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
    
    d = calculate_sma(k, d_period)
    return k, d

@jit(nopython=True)
def calculate_ichimoku(high: np.ndarray,
                      low: np.ndarray,
                      conversion_period: int = 9,
                      base_period: int = 26,
                      span_b_period: int = 52) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算一目均衡表"""
    conversion = np.zeros_like(high)
    base = np.zeros_like(high)
    span_b = np.zeros_like(high)
    
    for i in range(conversion_period-1, len(high)):
        high_window = high[i-conversion_period+1:i+1]
        low_window = low[i-conversion_period+1:i+1]
        conversion[i] = (np.max(high_window) + np.min(low_window)) / 2
        
    for i in range(base_period-1, len(high)):
        high_window = high[i-base_period+1:i+1]
        low_window = low[i-base_period+1:i+1]
        base[i] = (np.max(high_window) + np.min(low_window)) / 2
        
    for i in range(span_b_period-1, len(high)):
        high_window = high[i-span_b_period+1:i+1]
        low_window = low[i-span_b_period+1:i+1]
        span_b[i] = (np.max(high_window) + np.min(low_window)) / 2
        
    return conversion, base, span_b

@jit(nopython=True)
def calculate_pivot_points(high: np.ndarray,
                         low: np.ndarray,
                         close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray]:
    """计算枢轴点"""
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    
    return pivot, r1, r2, s1, s2

@jit(nopython=True)
def calculate_zigzag(data: np.ndarray,
                    deviation: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """计算之字形转向指标"""
    points = np.zeros_like(data)
    trends = np.zeros_like(data)  # 1 for up, -1 for down
    
    last_high = data[0]
    last_low = data[0]
    trend = 0  # 0 for undefined
    
    for i in range(1, len(data)):
        if trend == 0:
            if data[i] > last_high:
                trend = 1
                last_high = data[i]
                points[i] = data[i]
                trends[i] = 1
            elif data[i] < last_low:
                trend = -1
                last_low = data[i]
                points[i] = data[i]
                trends[i] = -1
        elif trend == 1:
            if data[i] > last_high:
                last_high = data[i]
                points[i] = data[i]
                trends[i] = 1
            elif data[i] < last_high * (1 - deviation):
                trend = -1
                last_low = data[i]
                points[i] = data[i]
                trends[i] = -1
        else:  # trend == -1
            if data[i] < last_low:
                last_low = data[i]
                points[i] = data[i]
                trends[i] = -1
            elif data[i] > last_low * (1 + deviation):
                trend = 1
                last_high = data[i]
                points[i] = data[i]
                trends[i] = 1
                
    return points, trends

@jit(nopython=True)
def calculate_parabolic_sar(high: np.ndarray,
                          low: np.ndarray,
                          acceleration: float = 0.02,
                          maximum: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """计算抛物线转向指标"""
    sar = np.zeros_like(high)
    trend = np.zeros_like(high)  # 1 for up, -1 for down
    ep = np.zeros_like(high)  # Extreme point
    acc = np.zeros_like(high)  # Current acceleration
    
    # 初始化
    trend[0] = 1 if high[1] > high[0] else -1
    sar[0] = low[0] if trend[0] == 1 else high[0]
    ep[0] = high[0] if trend[0] == 1 else low[0]
    acc[0] = acceleration
    
    # 计算SAR
    for i in range(1, len(high)):
        # 更新SAR
        sar[i] = sar[i-1] + acc[i-1] * (ep[i-1] - sar[i-1])
        
        # 确保SAR不会超出价格范围
        if trend[i-1] == 1:
            sar[i] = min(sar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
        else:
            sar[i] = max(sar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
            
        # 检查是否转向
        if (trend[i-1] == 1 and low[i] < sar[i]) or \
           (trend[i-1] == -1 and high[i] > sar[i]):
            trend[i] = -trend[i-1]
            sar[i] = ep[i-1]
            acc[i] = acceleration
            ep[i] = low[i] if trend[i] == 1 else high[i]
        else:
            trend[i] = trend[i-1]
            # 更新极值点和加速因子
            if trend[i] == 1:
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    acc[i] = min(acc[i-1] + acceleration, maximum)
                else:
                    ep[i] = ep[i-1]
                    acc[i] = acc[i-1]
            else:
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    acc[i] = min(acc[i-1] + acceleration, maximum)
                else:
                    ep[i] = ep[i-1]
                    acc[i] = acc[i-1]
                    
    return sar, trend

@jit(nopython=True)
def calculate_momentum(data: np.ndarray, period: int) -> np.ndarray:
    """计算动量指标"""
    momentum = np.zeros_like(data)
    for i in range(period, len(data)):
        momentum[i] = data[i] - data[i-period]
    return momentum

@jit(nopython=True)
def calculate_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """计算能量潮指标"""
    obv = np.zeros_like(close)
    obv[0] = volume[0]
    
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
            
    return obv

@jit(nopython=True)
def calculate_williams_r(high: np.ndarray,
                        low: np.ndarray,
                        close: np.ndarray,
                        period: int) -> np.ndarray:
    """计算威廉指标"""
    r = np.zeros_like(close)
    
    for i in range(period-1, len(close)):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            r[i] = -50
        else:
            r[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
            
    return r

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import threading

class ParallelIndicatorCalculator:
    """并行指标计算器"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or (threading.cpu_count() * 2)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._cache = {}
        self._cache_lock = threading.Lock()
        
    @lru_cache(maxsize=1000)
    def _cached_calculation(self, func, data_key: str, *args) -> np.ndarray:
        """缓存计算结果"""
        data = np.frombuffer(data_key, dtype=np.float64)
        return func(data, *args)
        
    def calculate_parallel(self,
                         func,
                         data_list: List[np.ndarray],
                         *args) -> List[np.ndarray]:
        """并行计算多个时间序列"""
        futures = []
        
        for data in data_list:
            # 创建数据键
            data_key = data.tobytes()
            
            # 检查缓存
            with self._cache_lock:
                if data_key in self._cache:
                    futures.append(self._cache[data_key])
                    continue
                    
            # 提交计算任务
            future = self.executor.submit(self._cached_calculation,
                                       func,
                                       data_key,
                                       *args)
                                       
            # 存入缓存
            with self._cache_lock:
                self._cache[data_key] = future
                
            futures.append(future)
            
        # 获取结果
        return [future.result() for future in futures]
        
    def clear_cache(self):
        """清理缓存"""
        with self._cache_lock:
            self._cache.clear()
        self._cached_calculation.cache_clear()
        
class MemoryOptimizedIndicator:
    """内存优化的指标计算"""
    
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
        
    def calculate_chunked(self,
                         func,
                         data: np.ndarray,
                         *args) -> np.ndarray:
        """分块计算指标"""
        result = np.zeros_like(data)
        
        # 计算需要的上下文大小
        context_size = max(args) if args else 0
        
        for i in range(0, len(data), self.chunk_size):
            # 确定分块的起始和结束位置
            start = max(0, i - context_size)
            end = min(len(data), i + self.chunk_size + context_size)
            
            # 计算当前分块
            chunk_result = func(data[start:end], *args)
            
            # 复制结果到对应位置
            result_start = i
            result_end = min(i + self.chunk_size, len(data))
            chunk_offset = max(0, i - start)
            result[result_start:result_end] = \
                chunk_result[chunk_offset:chunk_offset + (result_end - result_start)]
                
        return result
        
    def calculate_streaming(self,
                          func,
                          data: np.ndarray,
                          *args,
                          callback=None) -> np.ndarray:
        """流式计算指标"""
        result = np.zeros_like(data)
        context_size = max(args) if args else 0
        processed = 0
        
        while processed < len(data):
            # 确定当前分块
            end = min(processed + self.chunk_size, len(data))
            start = max(0, end - self.chunk_size - context_size)
            
            # 计算当前分块
            chunk_data = data[start:end]
            chunk_result = func(chunk_data, *args)
            
            # 更新结果
            result_start = processed
            result_end = end
            chunk_offset = processed - start
            result[result_start:result_end] = \
                chunk_result[chunk_offset:chunk_offset + (result_end - result_start)]
                
            # 回调进度
            if callback:
                callback(processed / len(data))
                
            processed = end
            
        return result
