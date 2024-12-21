"""
绩效分析模块
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime

@dataclass
class TradeStats:
    """交易统计"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_duration: float

@dataclass
class PerformanceMetrics:
    """绩效指标"""
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    trade_stats: TradeStats

class PerformanceAnalyzer:
    """绩效分析器"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        
    def calculate_trade_stats(self, trades: List[Dict]) -> TradeStats:
        """计算交易统计"""
        if not trades:
            return TradeStats(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                avg_trade_duration=0.0
            )
            
        # 计算盈亏
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        # 计算胜率
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # 计算平均盈亏
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0
        
        # 计算获利因子
        total_profit = sum(t['pnl'] for t in winning_trades)
        total_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
        
        # 计算平均持仓时间
        durations = [
            (pd.to_datetime(t['exit_time']) - pd.to_datetime(t['entry_time'])).total_seconds() / 3600
            for t in trades
        ]
        avg_duration = np.mean(durations) if durations else 0
        
        return TradeStats(
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_trade_duration=avg_duration
        )
        
    def calculate_metrics(self,
                       trades: List[Dict],
                       equity_curve: pd.Series,
                       risk_free_rate: float = 0.02) -> PerformanceMetrics:
        """计算绩效指标"""
        # 计算基本回报
        final_capital = equity_curve.iloc[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # 计算年化回报
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # 计算每日收益率
        daily_returns = equity_curve.pct_change().dropna()
        
        # 计算夏普比率
        excess_returns = daily_returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_returns.std() \
            if len(daily_returns) > 0 else 0
            
        # 计算索提诺比率
        downside_returns = daily_returns[daily_returns < 0]
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std() \
            if len(downside_returns) > 0 else 0
            
        # 计算最大回撤
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # 计算最大回撤持续期
        underwater = drawdowns < 0
        underwater_periods = underwater.astype(int).groupby(
            underwater.astype(int).diff().ne(0).cumsum()
        ).sum()
        max_drawdown_duration = underwater_periods.max() if len(underwater_periods) > 0 else 0
        
        # 计算交易统计
        trade_stats = self.calculate_trade_stats(trades)
        
        return PerformanceMetrics(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            trade_stats=trade_stats
        )
