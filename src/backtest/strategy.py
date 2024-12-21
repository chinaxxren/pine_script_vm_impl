"""
交易策略基类
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from ..indicators import Indicator, IndicatorResult

class Strategy(ABC):
    """交易策略基类"""
    
    def __init__(self, name: str):
        """初始化
        
        Args:
            name: 策略名称
        """
        self.name = name
        self.indicators: Dict[str, Indicator] = {}
        self.indicator_results: Dict[str, IndicatorResult] = {}
        
    @abstractmethod
    def initialize(self) -> None:
        """初始化策略
        
        在这里添加指标和设置参数
        """
        pass
        
    @abstractmethod
    def on_bar(self, data: pd.DataFrame) -> Dict[str, Union[str, float]]:
        """处理每个K线数据
        
        Args:
            data: K线数据，包含 open, high, low, close, volume 等列
            
        Returns:
            Dict[str, Union[str, float]]: 交易信号，包含:
                - action: 交易动作，可选值: "buy", "sell", "hold"
                - price: 交易价格
                - volume: 交易数量
                - stop_loss: 止损价格
                - take_profit: 止盈价格
        """
        pass
        
    def add_indicator(self, name: str, indicator: Indicator) -> None:
        """添加指标
        
        Args:
            name: 指标名称
            indicator: 指标实例
        """
        self.indicators[name] = indicator
        
    def update_indicators(self, data: pd.DataFrame) -> None:
        """更新所有指标
        
        Args:
            data: K线数据
        """
        for name, indicator in self.indicators.items():
            self.indicator_results[name] = indicator.calculate(data)
            
    def get_indicator_result(self, name: str) -> Optional[IndicatorResult]:
        """获取指标计算结果
        
        Args:
            name: 指标名称
            
        Returns:
            Optional[IndicatorResult]: 指标计算结果
        """
        return self.indicator_results.get(name)
        
    def get_indicator_value(self, name: str) -> Optional[np.ndarray]:
        """获取指标值
        
        Args:
            name: 指标名称
            
        Returns:
            Optional[np.ndarray]: 指标值
        """
        result = self.get_indicator_result(name)
        if result is not None:
            return result.values
        return None
        
    def get_indicator_signals(self, name: str) -> Optional[Dict[str, List[bool]]]:
        """获取指标信号
        
        Args:
            name: 指标名称
            
        Returns:
            Optional[Dict[str, List[bool]]]: 指标信号
        """
        result = self.get_indicator_result(name)
        if result is not None:
            return result.signals
        return None
        
    def get_indicator_levels(self, name: str) -> Optional[Dict[str, float]]:
        """获取指标重要水平
        
        Args:
            name: 指标名称
            
        Returns:
            Optional[Dict[str, float]]: 指标重要水平
        """
        result = self.get_indicator_result(name)
        if result is not None:
            return result.levels
        return None
        
    def get_indicator_metadata(self, name: str) -> Optional[Dict]:
        """获取指标元数据
        
        Args:
            name: 指标名称
            
        Returns:
            Optional[Dict]: 指标元数据
        """
        result = self.get_indicator_result(name)
        if result is not None:
            return result.metadata
        return None
