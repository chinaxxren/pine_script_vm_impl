"""
订单类
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

class OrderType(Enum):
    """订单类型"""
    MARKET = "market"  # 市价单
    LIMIT = "limit"    # 限价单
    STOP = "stop"      # 止损单
    
class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"      # 等待成交
    FILLED = "filled"        # 已成交
    CANCELED = "canceled"    # 已取消
    REJECTED = "rejected"    # 已拒绝
    
class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"    # 买入
    SELL = "sell"  # 卖出
    
@dataclass
class Order:
    """订单"""
    
    # 订单ID
    order_id: str
    
    # 交易对
    symbol: str
    
    # 订单类型
    type: OrderType
    
    # 订单方向
    side: OrderSide
    
    # 订单数量
    volume: float
    
    # 订单价格
    price: float
    
    # 止损价格
    stop_loss: Optional[float] = None
    
    # 止盈价格
    take_profit: Optional[float] = None
    
    # 订单状态
    status: OrderStatus = OrderStatus.PENDING
    
    # 创建时间
    create_time: datetime = datetime.now()
    
    # 成交时间
    fill_time: Optional[datetime] = None
    
    # 成交价格
    fill_price: Optional[float] = None
    
    # 成交数量
    fill_volume: Optional[float] = None
    
    # 手续费
    fee: Optional[float] = None
    
    def __post_init__(self):
        """初始化后检查"""
        if self.volume <= 0:
            raise ValueError("订单数量必须大于0")
            
        if self.type != OrderType.MARKET and self.price <= 0:
            raise ValueError("限价单和止损单的价格必须大于0")
            
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError("止损价格必须大于0")
            
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError("止盈价格必须大于0")
            
    def is_filled(self) -> bool:
        """是否已成交"""
        return self.status == OrderStatus.FILLED
        
    def is_active(self) -> bool:
        """是否活跃"""
        return self.status == OrderStatus.PENDING
        
    def is_canceled(self) -> bool:
        """是否已取消"""
        return self.status == OrderStatus.CANCELED
        
    def is_rejected(self) -> bool:
        """是否已拒绝"""
        return self.status == OrderStatus.REJECTED
        
    def fill(self, price: float, volume: float, fee: float = 0) -> None:
        """成交
        
        Args:
            price: 成交价格
            volume: 成交数量
            fee: 手续费
        """
        if not self.is_active():
            raise ValueError("订单已经成交、取消或拒绝")
            
        if volume > self.volume:
            raise ValueError("成交数量不能大于订单数量")
            
        self.status = OrderStatus.FILLED
        self.fill_time = datetime.now()
        self.fill_price = price
        self.fill_volume = volume
        self.fee = fee
        
    def cancel(self) -> None:
        """取消订单"""
        if not self.is_active():
            raise ValueError("订单已经成交、取消或拒绝")
            
        self.status = OrderStatus.CANCELED
        
    def reject(self) -> None:
        """拒绝订单"""
        if not self.is_active():
            raise ValueError("订单已经成交、取消或拒绝")
            
        self.status = OrderStatus.REJECTED
