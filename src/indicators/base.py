from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np
from .cache import OptimizedIndicator, IndicatorCache
from dataclasses import dataclass

class IndicatorResult:
    """指标计算结果"""
    
    def __init__(self, values: np.ndarray, signals: Dict[str, List[bool]] = None,
                 levels: Dict[str, float] = None, metadata: Dict[str, Any] = None):
        self.values = values
        self.signals = signals or {}
        self.levels = levels or {}
        self.metadata = metadata or {}

class Indicator(OptimizedIndicator):
    """指标基类"""
    
    def __init__(self):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.description = ""
        self.category = ""
        
    def calculate(self, *args, **kwargs) -> IndicatorResult:
        """计算指标值
        
        Returns:
            IndicatorResult: 计算结果
        """
        # 检查缓存
        cache_key = args[0] if args else kwargs.get('data')
        if cache_key is not None:
            cached_result = self.cache.get(cache_key, **kwargs)
            if cached_result is not None:
                return cached_result
                
        # 计算结果
        result = self._calculate(*args, **kwargs)
        
        # 缓存结果
        if cache_key is not None:
            self.cache.set(cache_key, result, **kwargs)
            
        return result
        
    def _calculate(self, *args, **kwargs) -> IndicatorResult:
        """实际的计算逻辑"""
        raise NotImplementedError
        
    def validate_input(self, data: np.ndarray) -> None:
        """验证输入数据
        
        Args:
            data: 输入数据
            
        Raises:
            ValueError: 数据无效
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be numpy array")
            
        if len(data) == 0:
            raise ValueError("Input data cannot be empty")
            
    def generate_signals(self, values: np.ndarray) -> Dict[str, List[bool]]:
        """生成交易信号
        
        Args:
            values: 指标值
            
        Returns:
            交易信号
        """
        return {}
        
    def calculate_levels(self, values: np.ndarray) -> Dict[str, float]:
        """计算重要水平
        
        Args:
            values: 指标值
            
        Returns:
            重要水平
        """
        return {}
