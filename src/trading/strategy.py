"""
交易策略实现
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class StrategyConfig:
    """策略配置"""
    symbol: str
    timeframe: str
    risk_per_trade: float = 0.02  # 每笔交易风险
    max_positions: int = 3  # 最大持仓数
    stop_loss_pct: float = 0.02  # 止损百分比
    take_profit_pct: float = 0.04  # 止盈百分比
    max_drawdown: float = 0.1  # 最大回撤
    position_sizing: str = 'risk'  # risk, equal, or kelly
    max_leverage: float = 2.0  # 最大杠杆率

class Position:
    """持仓管理"""
    
    def __init__(self,
                 symbol: str,
                 side: str,
                 entry_price: float,
                 amount: float,
                 stop_loss: float,
                 take_profit: float):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.amount = amount
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = datetime.now()
        self.pnl = 0.0
        self.status = 'open'
        
    def update(self, current_price: float):
        """更新持仓状态"""
        if self.side == 'buy':
            self.pnl = (current_price - self.entry_price) * self.amount
            if current_price <= self.stop_loss:
                self.status = 'stopped'
            elif current_price >= self.take_profit:
                self.status = 'profit'
        else:
            self.pnl = (self.entry_price - current_price) * self.amount
            if current_price >= self.stop_loss:
                self.status = 'stopped'
            elif current_price <= self.take_profit:
                self.status = 'profit'
                
class RiskManager:
    """风险管理"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.positions: List[Position] = []
        self.total_pnl = 0.0
        self.peak_equity = 0.0
        self.current_drawdown = 0.0
        
    def can_open_position(self, account_balance: float) -> bool:
        """检查是否可以开仓"""
        if len(self.positions) >= self.config.max_positions:
            return False
            
        if self.current_drawdown > self.config.max_drawdown:
            return False
            
        return True
        
    def calculate_position_size(self,
                              price: float,
                              stop_loss: float,
                              account_balance: float) -> float:
        """计算仓位大小"""
        risk_amount = account_balance * self.config.risk_per_trade
        
        if self.config.position_sizing == 'risk':
            # 基于风险的仓位
            price_risk = abs(price - stop_loss)
            return risk_amount / price_risk
        elif self.config.position_sizing == 'kelly':
            # 凯利公式
            win_rate = 0.5  # 这里可以使用历史胜率
            win_loss_ratio = self.config.take_profit_pct / self.config.stop_loss_pct
            kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
            return (account_balance * kelly_fraction) / price
        else:
            # 等额仓位
            return risk_amount / price
            
    def update_positions(self, current_price: float):
        """更新持仓状态"""
        active_positions = []
        closed_pnl = 0.0
        
        for position in self.positions:
            position.update(current_price)
            if position.status == 'open':
                active_positions.append(position)
            else:
                closed_pnl += position.pnl
                
        self.positions = active_positions
        self.total_pnl += closed_pnl
        
        # 更新回撤
        current_equity = self.total_pnl
        self.peak_equity = max(self.peak_equity, current_equity)
        self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
class Strategy(ABC):
    """策略基类"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.risk_manager = RiskManager(config)
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成交易信号"""
        pass
        
    def on_tick(self, tick: Dict[str, Any]):
        """处理实时行情"""
        current_price = float(tick['last'])
        self.risk_manager.update_positions(current_price)
        
class TrendFollowingStrategy(Strategy):
    """趋势跟踪策略"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.sma_short = 20
        self.sma_long = 50
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成交易信号"""
        # 创建数据副本
        df = data.copy()
        
        # 计算指标
        df.loc[:, 'sma_short'] = df['close'].rolling(self.sma_short).mean()
        df.loc[:, 'sma_long'] = df['close'].rolling(self.sma_long).mean()
        
        # 生成信号
        df.loc[:, 'signal'] = np.where(
            (df['sma_short'] > df['sma_long']), 1,
            np.where((df['sma_short'] < df['sma_long']), -1, 0)
        )
        
        # 计算止损和止盈
        df.loc[:, 'stop_loss'] = np.where(
            df['signal'] > 0,
            df['close'] * (1 - self.config.stop_loss_pct),
            df['close'] * (1 + self.config.stop_loss_pct)
        )
        
        df.loc[:, 'take_profit'] = np.where(
            df['signal'] > 0,
            df['close'] * (1 + self.config.take_profit_pct),
            df['close'] * (1 - self.config.take_profit_pct)
        )
        
        # 返回最新信号
        return df.iloc[-1][['signal', 'stop_loss', 'take_profit']].to_dict()
        
    def on_tick(self, tick: Dict[str, Any]):
        """处理实时行情"""
        pass

class MeanReversionStrategy(Strategy):
    """均值回归策略"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.lookback = 20
        self.std_dev = 2
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成交易信号"""
        # 创建数据副本
        df = data.copy()
        
        # 计算指标
        df.loc[:, 'sma'] = df['close'].rolling(self.lookback).mean()
        df.loc[:, 'std'] = df['close'].rolling(self.lookback).std()
        df.loc[:, 'upper'] = df['sma'] + self.std_dev * df['std']
        df.loc[:, 'lower'] = df['sma'] - self.std_dev * df['std']
        
        # 生成信号
        df.loc[:, 'signal'] = np.where(
            df['close'] < df['lower'], 1,
            np.where(df['close'] > df['upper'], -1, 0)
        )
        
        # 计算止损和止盈
        df.loc[:, 'stop_loss'] = np.where(
            df['signal'] > 0,
            df['close'] * (1 - self.config.stop_loss_pct),
            df['close'] * (1 + self.config.stop_loss_pct)
        )
        
        df.loc[:, 'take_profit'] = np.where(
            df['signal'] > 0,
            df['close'] * (1 + self.config.take_profit_pct),
            df['close'] * (1 - self.config.take_profit_pct)
        )
        
        # 返回最新信号
        return df.iloc[-1][['signal', 'stop_loss', 'take_profit']].to_dict()
        
    def on_tick(self, tick: Dict[str, Any]):
        """处理实时行情"""
        pass

class BreakoutStrategy(Strategy):
    """突破策略"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.lookback = 20
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成交易信号"""
        # 创建数据副本
        df = data.copy()
        
        # 计算指标
        df.loc[:, 'high_band'] = df['high'].rolling(self.lookback).max()
        df.loc[:, 'low_band'] = df['low'].rolling(self.lookback).min()
        
        # 生成信号
        df.loc[:, 'signal'] = np.where(
            df['close'] > df['high_band'], 1,
            np.where(df['close'] < df['low_band'], -1, 0)
        )
        
        # 计算止损和止盈
        df.loc[:, 'stop_loss'] = np.where(
            df['signal'] > 0,
            df['close'] * (1 - self.config.stop_loss_pct),
            df['close'] * (1 + self.config.stop_loss_pct)
        )
        
        df.loc[:, 'take_profit'] = np.where(
            df['signal'] > 0,
            df['close'] * (1 + self.config.take_profit_pct),
            df['close'] * (1 - self.config.take_profit_pct)
        )
        
        # 返回最新信号
        return df.iloc[-1][['signal', 'stop_loss', 'take_profit']].to_dict()
        
    def on_tick(self, tick: Dict[str, Any]):
        """处理实时行情"""
        pass
