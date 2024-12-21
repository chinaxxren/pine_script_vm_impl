from typing import Dict, Any, Tuple, Optional
import numpy as np
from functools import lru_cache
import hashlib

class IndicatorCache:
    """指标计算结果缓存"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        
    def _generate_key(self, data: np.ndarray, **kwargs) -> str:
        """生成缓存键
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            缓存键
        """
        # 计算数据的哈希值
        data_hash = hashlib.md5(data.tobytes()).hexdigest()
        
        # 将参数转换为排序后的字符串
        params = sorted(kwargs.items())
        param_str = ','.join(f"{k}={v}" for k, v in params)
        
        return f"{data_hash}:{param_str}"
        
    def get(self, data: np.ndarray, **kwargs) -> Optional[Any]:
        """获取缓存的结果
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            缓存的结果，如果没有缓存则返回None
        """
        key = self._generate_key(data, **kwargs)
        return self.cache.get(key)
        
    def set(self, data: np.ndarray, result: Any, **kwargs) -> None:
        """设置缓存
        
        Args:
            data: 输入数据
            result: 计算结果
            **kwargs: 其他参数
        """
        key = self._generate_key(data, **kwargs)
        
        # 如果缓存已满，删除最早的项
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        self.cache[key] = result
        
    def clear(self) -> None:
        """清除所有缓存"""
        self.cache.clear()
        
class OptimizedIndicator:
    """优化的指标计算基类"""
    
    def __init__(self):
        self.cache = IndicatorCache()
        
    @lru_cache(maxsize=128)
    def _calculate_moving_average(self, data: Tuple[float, ...], period: int) -> np.ndarray:
        """计算移动平均线（使用函数级缓存）
        
        Args:
            data: 输入数据
            period: 周期
            
        Returns:
            移动平均线
        """
        data_array = np.array(data)
        result = np.zeros_like(data_array)
        
        # 使用累积和优化计算
        cumsum = np.cumsum(data_array)
        cumsum[period:] = cumsum[period:] - cumsum[:-period]
        result[period-1:] = cumsum[period-1:] / period
        
        return result
        
    def _calculate_exponential_ma(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算指数移动平均线（使用向量化操作）
        
        Args:
            data: 输入数据
            period: 周期
            
        Returns:
            指数移动平均线
        """
        alpha = 2.0 / (period + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        
        # 使用向量化操作
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            
        return result
        
    def _calculate_true_range(self, high: np.ndarray, low: np.ndarray, 
                            close: np.ndarray) -> np.ndarray:
        """计算真实波幅（使用向量化操作）
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            
        Returns:
            真实波幅
        """
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr2[0] = tr1[0]
        tr3[0] = tr1[0]
        
        return np.maximum.reduce([tr1, tr2, tr3])
        
    def _calculate_momentum(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算动量（使用向量化操作）
        
        Args:
            data: 输入数据
            period: 周期
            
        Returns:
            动量
        """
        return data - np.roll(data, period)
        
    def _calculate_rate_of_change(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算变化率（使用向量化操作）
        
        Args:
            data: 输入数据
            period: 周期
            
        Returns:
            变化率
        """
        shifted_data = np.roll(data, period)
        shifted_data[:period] = data[:period]
        
        return (data - shifted_data) / shifted_data * 100
