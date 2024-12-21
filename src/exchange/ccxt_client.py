"""
使用CCXT库实现交易所API集成
"""

import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import time
from functools import wraps
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning(f"尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (attempt + 1))
            raise last_error
        return wrapper
    return decorator

class ExchangeConfig:
    """交易所配置"""
    def __init__(self,
                 exchange_id: str,
                 api_key: Optional[str] = None,
                 secret: Optional[str] = None,
                 password: Optional[str] = None):
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.secret = secret
        self.password = password

class CCXTClient:
    """CCXT客户端封装"""
    
    def __init__(self, config: ExchangeConfig):
        """初始化CCXT客户端"""
        self.config = config
        self.exchange_class = getattr(ccxt, config.exchange_id)
        self.exchange = self.exchange_class({
            'apiKey': config.api_key,
            'secret': config.secret,
            'password': config.password,
            'enableRateLimit': True,
        })
        
    async def initialize(self):
        """初始化交易所连接"""
        if self.exchange.has['fetchMarkets']:
            await self.exchange.load_markets()
            logger.info(f"已加载 {len(self.exchange.markets)} 个市场")
            
    @retry_on_error()
    async def fetch_ohlcv(self,
                         symbol: str,
                         timeframe: str = '1m',
                         since: Optional[int] = None,
                         limit: Optional[int] = None) -> pd.DataFrame:
        """获取K线数据"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"获取K线数据失败: {str(e)}")
            raise
            
    @retry_on_error()
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """获取行情数据"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"获取行情数据失败: {str(e)}")
            raise
            
    @retry_on_error()
    async def fetch_order_book(self,
                             symbol: str,
                             limit: Optional[int] = None) -> Dict[str, Any]:
        """获取订单簿数据"""
        try:
            order_book = await self.exchange.fetch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            logger.error(f"获取订单簿数据失败: {str(e)}")
            raise
            
    @retry_on_error()
    async def create_order(self,
                          symbol: str,
                          type: str,
                          side: str,
                          amount: float,
                          price: Optional[float] = None,
                          params: Dict = {}) -> Dict[str, Any]:
        """创建订单"""
        try:
            order = await self.exchange.create_order(symbol, type, side, amount, price, params)
            return order
        except Exception as e:
            logger.error(f"创建订单失败: {str(e)}")
            raise
            
    @retry_on_error()
    async def cancel_order(self,
                          id: str,
                          symbol: Optional[str] = None,
                          params: Dict = {}) -> Dict[str, Any]:
        """取消订单"""
        try:
            result = await self.exchange.cancel_order(id, symbol, params)
            return result
        except Exception as e:
            logger.error(f"取消订单失败: {str(e)}")
            raise
            
    @retry_on_error()
    async def fetch_balance(self) -> Dict[str, Any]:
        """获取账户余额"""
        try:
            balance = await self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"获取账户余额失败: {str(e)}")
            raise
            
    @retry_on_error()
    async def fetch_positions(self,
                            symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """获取持仓信息"""
        try:
            positions = await self.exchange.fetch_positions(symbols)
            return positions
        except Exception as e:
            logger.error(f"获取持仓信息失败: {str(e)}")
            raise
            
    async def close(self):
        """关闭连接"""
        await self.exchange.close()
        
class DataStreamProcessor:
    """数据流处理器"""
    
    def __init__(self, client: CCXTClient):
        self.client = client
        self.callbacks = []
        
    def add_callback(self, callback):
        """添加数据处理回调"""
        self.callbacks.append(callback)
        
    async def process_ticker(self, symbol: str):
        """处理行情数据"""
        while True:
            try:
                ticker = await self.client.fetch_ticker(symbol)
                for callback in self.callbacks:
                    await callback(ticker)
            except Exception as e:
                logger.error(f"处理行情数据失败: {str(e)}")
            await asyncio.sleep(1)
            
    async def process_order_book(self, symbol: str, limit: Optional[int] = None):
        """处理订单簿数据"""
        while True:
            try:
                order_book = await self.client.fetch_order_book(symbol, limit)
                for callback in self.callbacks:
                    await callback(order_book)
            except Exception as e:
                logger.error(f"处理订单簿数据失败: {str(e)}")
            await asyncio.sleep(1)
            
class TradingBot:
    """交易机器人"""
    
    def __init__(self, client: CCXTClient):
        self.client = client
        self.positions = {}
        self.orders = {}
        
    async def update_positions(self):
        """更新持仓信息"""
        try:
            positions = await self.client.fetch_positions()
            self.positions = {p['symbol']: p for p in positions}
        except Exception as e:
            logger.error(f"更新持仓信息失败: {str(e)}")
            
    async def place_order(self,
                         symbol: str,
                         side: str,
                         amount: float,
                         order_type: str = 'limit',
                         price: Optional[float] = None):
        """下单"""
        try:
            order = await self.client.create_order(symbol, order_type, side, amount, price)
            self.orders[order['id']] = order
            return order
        except Exception as e:
            logger.error(f"下单失败: {str(e)}")
            raise
            
    async def cancel_order(self, order_id: str):
        """取消订单"""
        try:
            result = await self.client.cancel_order(order_id)
            if order_id in self.orders:
                del self.orders[order_id]
            return result
        except Exception as e:
            logger.error(f"取消订单失败: {str(e)}")
            raise
