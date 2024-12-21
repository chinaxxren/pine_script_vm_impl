"""
风险管理模块
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

@dataclass
class RiskMetrics:
    """风险指标"""
    value_at_risk: float  # 在险价值
    expected_shortfall: float  # 期望损失
    beta: float  # 贝塔系数
    correlation: float  # 相关系数
    volatility: float  # 波动率
    downside_volatility: float  # 下行波动率
    information_ratio: float  # 信息比率
    tracking_error: float  # 跟踪误差
    max_drawdown: float  # 最大回撤

class RiskManager:
    """风险管理器"""
    
    def __init__(self,
                confidence_level: float = 0.95,
                max_drawdown: float = 0.2,
                max_leverage: float = 2.0,
                position_size_limit: float = 0.1,
                risk_free_rate: float = 0.02):
        self.confidence_level = confidence_level
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        self.position_size_limit = position_size_limit
        self.risk_free_rate = risk_free_rate
        
    def calculate_metrics(self,
                       returns: pd.Series,
                       benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """计算风险指标"""
        if len(returns) == 0:
            return RiskMetrics(
                value_at_risk=0.0,
                expected_shortfall=0.0,
                beta=1.0,
                correlation=1.0,
                volatility=0.0,
                downside_volatility=0.0,
                information_ratio=0.0,
                tracking_error=0.0,
                max_drawdown=0.0
            )
            
        # 计算波动率
        volatility = returns.std() * np.sqrt(252)
        
        # 计算下行波动率
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # 计算VaR
        var = returns.quantile(1 - self.confidence_level)
        
        # 计算ES
        es = returns[returns <= var].mean()
        
        # 计算最大回撤
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # 计算贝塔和相关系数
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # 确保两个序列长度相同
            common_index = returns.index.intersection(benchmark_returns.index)
            returns = returns[common_index]
            benchmark_returns = benchmark_returns[common_index]
            
            # 计算相关系数
            correlation = returns.corr(benchmark_returns)
            
            # 计算贝塔
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 1.0
            
            # 计算跟踪误差
            tracking_diff = returns - benchmark_returns
            tracking_error = tracking_diff.std() * np.sqrt(252)
            
            # 计算信息比率
            excess_return = returns.mean() - benchmark_returns.mean()
            information_ratio = excess_return / tracking_error if tracking_error != 0 else 0
        else:
            correlation = 1.0
            beta = 1.0
            tracking_error = 0.0
            information_ratio = 0.0
            
        return RiskMetrics(
            value_at_risk=var,
            expected_shortfall=es,
            beta=beta,
            correlation=correlation,
            volatility=volatility,
            downside_volatility=downside_volatility,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            max_drawdown=max_drawdown
        )
        
    def check_risk_limits(self,
                       current_drawdown: float,
                       current_leverage: float,
                       position_sizes: Dict[str, float]) -> bool:
        """检查风险限制"""
        # 检查回撤限制
        if abs(current_drawdown) > self.max_drawdown:
            return False
            
        # 检查杠杆限制
        if current_leverage > self.max_leverage:
            return False
            
        # 检查持仓规模限制
        for size in position_sizes.values():
            if abs(size) > self.position_size_limit:
                return False
                
        return True
        
    def calculate_position_size(self,
                            price: float,
                            stop_loss: float,
                            account_balance: float) -> float:
        """计算仓位大小"""
        # 计算每点价值
        point_value = abs(price - stop_loss)
        
        # 计算风险金额
        risk_amount = account_balance * self.position_size_limit
        
        # 计算仓位大小
        if point_value > 0:
            position_size = risk_amount / point_value
        else:
            position_size = 0
            
        return position_size
