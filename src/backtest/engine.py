"""
回测引擎实现
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from ..trading.strategy import Strategy, Position
from .performance import PerformanceAnalyzer
from .risk import RiskManager

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 100000.0  # 初始资金
    commission_rate: float = 0.001  # 手续费率
    slippage: float = 0.001  # 滑点
    risk_free_rate: float = 0.02  # 无风险利率
    
@dataclass
class BacktestResult:
    """回测结果"""
    equity_curve: pd.Series  # 权益曲线
    trades: List[Dict]  # 交易记录
    performance_metrics: Dict  # 性能指标
    risk_metrics: Dict  # 风险指标
    
class BacktestEngine:
    """回测引擎"""
    
    def __init__(self,
                strategy: Strategy,
                data: pd.DataFrame,
                config: Optional[BacktestConfig] = None):
        """初始化回测引擎
        
        Args:
            strategy: 交易策略
            data: 历史数据
            config: 回测配置
        """
        self.strategy = strategy
        self.data = data.copy()
        self.config = config or BacktestConfig()
        
        # 初始化账户状态
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades = []
        self.equity_curve = []
        
        # 初始化分析器
        self.performance_analyzer = PerformanceAnalyzer(self.config.initial_capital)
        self.risk_manager = RiskManager()
        
    def run(self) -> BacktestResult:
        """运行回测"""
        logger.info("开始回测...")
        
        # 检查数据是否为空
        if len(self.data) == 0:
            logger.warning("回测数据为空")
            return BacktestResult(
                equity_curve=pd.Series([], dtype=float),
                trades=[],
                performance_metrics={},
                risk_metrics={}
            )
        
        # 遍历历史数据
        for i in range(len(self.data)):
            # 获取当前数据
            current_data = self.data.iloc[:i+1]
            current_bar = current_data.iloc[-1]
            
            # 更新持仓
            self._update_positions(current_bar)
            
            # 生成信号
            signals = self.strategy.generate_signals(current_data)
            
            # 执行交易
            if signals['signal'] != 0:
                self._execute_trade(current_bar, signals)
                
            # 更新权益
            self._update_equity(current_bar)
            
        # 平掉所有持仓
        self._close_all_positions(self.data.iloc[-1])
        
        # 计算回测结果
        result = self._calculate_results()
        
        logger.info("回测完成")
        return result
        
    def _update_positions(self, bar: pd.Series):
        """更新持仓状态"""
        for symbol, position in list(self.positions.items()):
            # 更新持仓盈亏
            position.update(bar['close'])
            
            # 检查是否触发止损止盈
            if position.status in ['stopped', 'profit']:
                self._close_position(position, bar)
                
    def _execute_trade(self, bar: pd.Series, signals: Dict):
        """执行交易"""
        # 检查是否有足够资金
        if self.capital <= 0:
            return
            
        # 计算交易数量
        price = bar['close']
        amount = self._calculate_position_size(price, signals['stop_loss'])
        side = 'buy' if signals['signal'] > 0 else 'sell'
        
        # 检查风险限制
        if not self._check_risk_limits(amount * price, side):
            return
            
        # 创建新持仓
        position = Position(
            symbol=bar.name,
            side=side,
            entry_price=price,
            amount=amount,
            stop_loss=signals['stop_loss'],
            take_profit=signals['take_profit']
        )
        
        # 更新账户
        cost = amount * price * (1 + self.config.commission_rate)
        self.capital -= cost
        self.positions[bar.name] = position
        
        # 记录交易
        self.trades.append({
            'symbol': bar.name,
            'side': position.side,
            'entry_time': bar.name,
            'entry_price': price,
            'amount': amount,
            'cost': cost,
            'stop_loss': signals['stop_loss'],
            'take_profit': signals['take_profit']
        })
        
    def _close_position(self, position: Position, bar: pd.Series):
        """平仓"""
        # 计算平仓价格
        exit_price = bar['close']
        
        # 计算交易成本
        cost = position.amount * exit_price * self.config.commission_rate
        
        # 更新账户
        self.capital += position.amount * exit_price - cost
        
        # 更新交易记录
        trade = self.trades[-1]
        trade.update({
            'exit_time': bar.name,
            'exit_price': exit_price,
            'pnl': position.pnl - cost,
            'status': position.status
        })
        
        # 删除持仓
        del self.positions[position.symbol]
        
    def _close_all_positions(self, bar: pd.Series):
        """平掉所有持仓"""
        for position in list(self.positions.values()):
            self._close_position(position, bar)
            
    def _calculate_position_size(self, price: float, stop_loss: float) -> float:
        """计算持仓规模"""
        risk_amount = self.capital * self.strategy.config.risk_per_trade
        point_value = abs(price - stop_loss)
        if point_value > 0:
            return risk_amount / point_value
        return 0
        
    def _check_risk_limits(self, exposure: float, side: str = 'buy') -> bool:
        """检查风险限制
        
        Args:
            exposure: 新增持仓的市值
            side: 交易方向，'buy' 或 'sell'
        """
        # 检查杠杆率
        total_exposure = sum(p.amount * p.entry_price for p in self.positions.values())
        if (total_exposure + exposure) / self.capital > self.strategy.config.max_leverage:
            return False
            
        # 检查最大持仓数，只统计同方向的持仓
        same_side_positions = sum(1 for p in self.positions.values() if p.side == side)
        if same_side_positions >= self.strategy.config.max_positions:
            return False
            
        return True
        
    def _update_equity(self, bar: pd.Series):
        """更新权益曲线"""
        equity = self.capital
        for position in self.positions.values():
            equity += position.pnl
        self.equity_curve.append({
            'timestamp': bar.name,
            'equity': equity
        })
        
    def _calculate_results(self) -> BacktestResult:
        """计算回测结果"""
        # 构建权益曲线
        equity_curve = pd.DataFrame(self.equity_curve)
        equity_curve.set_index('timestamp', inplace=True)
        equity_curve = equity_curve['equity']
        
        # 计算性能指标
        performance_metrics = self.performance_analyzer.calculate_metrics(
            self.trades,
            equity_curve,
            self.config.risk_free_rate
        )
        
        # 计算风险指标
        risk_metrics = self.risk_manager.calculate_metrics(
            equity_curve.pct_change().dropna()
        )
        
        # 将性能指标转换为字典
        metrics_dict = {
            'initial_capital': performance_metrics.initial_capital,
            'final_capital': performance_metrics.final_capital,
            'total_return': performance_metrics.total_return,
            'annual_return': performance_metrics.annual_return,
            'sharpe_ratio': performance_metrics.sharpe_ratio,
            'sortino_ratio': performance_metrics.sortino_ratio,
            'max_drawdown': performance_metrics.max_drawdown,
            'max_drawdown_duration': performance_metrics.max_drawdown_duration,
            'total_trades': performance_metrics.trade_stats.total_trades,
            'winning_trades': performance_metrics.trade_stats.winning_trades,
            'losing_trades': performance_metrics.trade_stats.losing_trades,
            'win_rate': performance_metrics.trade_stats.win_rate,
            'avg_win': performance_metrics.trade_stats.avg_win,
            'avg_loss': performance_metrics.trade_stats.avg_loss,
            'profit_factor': performance_metrics.trade_stats.profit_factor,
            'avg_trade_duration': performance_metrics.trade_stats.avg_trade_duration
        }
        
        return BacktestResult(
            equity_curve=equity_curve,
            trades=self.trades,
            performance_metrics=metrics_dict,
            risk_metrics=risk_metrics.__dict__
        )
        
class PortfolioBacktestEngine:
    """投资组合回测引擎"""
    
    def __init__(self,
                strategies: Dict[str, Strategy],
                data: Dict[str, pd.DataFrame],
                config: Optional[BacktestConfig] = None):
        """初始化投资组合回测引擎
        
        Args:
            strategies: 交易策略字典
            data: 历史数据字典
            config: 回测配置
        """
        self.strategies = strategies
        self.data = data
        self.config = config or BacktestConfig()
        
        # 创建单个回测引擎
        self.engines = {
            symbol: BacktestEngine(strategy, data[symbol], config)
            for symbol, strategy in strategies.items()
        }
        
    def run(self) -> Dict[str, BacktestResult]:
        """运行回测"""
        logger.info("开始投资组合回测...")
        
        # 运行每个回测引擎
        results = {}
        for symbol, engine in self.engines.items():
            logger.info(f"回测 {symbol}...")
            results[symbol] = engine.run()
            
        logger.info("投资组合回测完成")
        return results
