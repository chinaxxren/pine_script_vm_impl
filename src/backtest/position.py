"""
仓位管理
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

class PositionSide(Enum):
    """仓位方向"""
    LONG = "long"    # 多仓
    SHORT = "short"  # 空仓
    
@dataclass
class Position:
    """仓位"""
    
    # 交易对
    symbol: str
    
    # 仓位方向
    side: PositionSide
    
    # 仓位数量
    volume: float
    
    # 开仓价格
    entry_price: float
    
    # 当前价格
    current_price: float
    
    # 止损价格
    stop_loss: Optional[float] = None
    
    # 止盈价格
    take_profit: Optional[float] = None
    
    # 开仓时间
    entry_time: datetime = datetime.now()
    
    # 最后更新时间
    update_time: datetime = datetime.now()
    
    def __post_init__(self):
        """初始化后检查"""
        if self.volume <= 0:
            raise ValueError("仓位数量必须大于0")
            
        if self.entry_price <= 0:
            raise ValueError("开仓价格必须大于0")
            
        if self.current_price <= 0:
            raise ValueError("当前价格必须大于0")
            
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError("止损价格必须大于0")
            
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError("止盈价格必须大于0")
            
    def update_price(self, price: float) -> None:
        """更新价格
        
        Args:
            price: 新价格
        """
        if price <= 0:
            raise ValueError("价格必须大于0")
            
        self.current_price = price
        self.update_time = datetime.now()
        
    def update_stop_loss(self, price: float) -> None:
        """更新止损价格
        
        Args:
            price: 新止损价格
        """
        if price <= 0:
            raise ValueError("止损价格必须大于0")
            
        self.stop_loss = price
        self.update_time = datetime.now()
        
    def update_take_profit(self, price: float) -> None:
        """更新止盈价格
        
        Args:
            price: 新止盈价格
        """
        if price <= 0:
            raise ValueError("止盈价格必须大于0")
            
        self.take_profit = price
        self.update_time = datetime.now()
        
    def get_unrealized_pnl(self) -> float:
        """获取未实现盈亏"""
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.volume
        else:
            return (self.entry_price - self.current_price) * self.volume
            
    def get_unrealized_pnl_ratio(self) -> float:
        """获取未实现盈亏比例"""
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price
            
    def get_duration(self) -> float:
        """获取持仓时间(小时)"""
        return (self.update_time - self.entry_time).total_seconds() / 3600
