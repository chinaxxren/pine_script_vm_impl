"""
WebSocket客户端实现
"""

import asyncio
import websockets
import json
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class WebSocketConfig:
    """WebSocket配置"""
    exchange: str
    api_key: Optional[str] = None
    secret: Optional[str] = None
    channels: List[str] = None
    symbols: List[str] = None

class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self, config: WebSocketConfig):
        self.config = config
        self.connections = {}
        self.callbacks = {}
        self.running = False
        self.reconnect_delay = 1
        self.max_reconnect_delay = 60
        self.heartbeat_interval = 30
        
    async def connect(self, url: str, channel: str):
        """建立WebSocket连接"""
        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    logger.info(f"已连接到 {url}")
                    self.connections[channel] = ws
                    
                    # 发送订阅消息
                    subscribe_msg = self._get_subscribe_message(channel)
                    await ws.send(json.dumps(subscribe_msg))
                    
                    # 启动心跳
                    heartbeat_task = asyncio.create_task(self._heartbeat(ws))
                    
                    # 处理消息
                    try:
                        while True:
                            message = await ws.recv()
                            await self._handle_message(channel, message)
                    except websockets.ConnectionClosed:
                        logger.warning(f"连接断开: {url}")
                        heartbeat_task.cancel()
                        
            except Exception as e:
                logger.error(f"WebSocket错误: {str(e)}")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2,
                                         self.max_reconnect_delay)
                
    def _get_subscribe_message(self, channel: str) -> Dict:
        """获取订阅消息"""
        if self.config.exchange == 'binance':
            return {
                "method": "SUBSCRIBE",
                "params": [
                    f"{symbol.lower()}@{channel}"
                    for symbol in self.config.symbols
                ],
                "id": int(time.time())
            }
        # 添加其他交易所的支持
        return {}
        
    async def _heartbeat(self, ws):
        """发送心跳"""
        while True:
            try:
                await ws.ping()
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"心跳错误: {str(e)}")
                break
                
    async def _handle_message(self, channel: str, message: str):
        """处理WebSocket消息"""
        try:
            data = json.loads(message)
            if channel in self.callbacks:
                for callback in self.callbacks[channel]:
                    await callback(data)
        except Exception as e:
            logger.error(f"处理消息错误: {str(e)}")
            
    def add_callback(self, channel: str, callback: Callable):
        """添加回调函数"""
        if channel not in self.callbacks:
            self.callbacks[channel] = []
        self.callbacks[channel].append(callback)
        
    async def start(self):
        """启动WebSocket管理器"""
        self.running = True
        tasks = []
        
        for channel in self.config.channels:
            url = self._get_websocket_url(channel)
            task = asyncio.create_task(self.connect(url, channel))
            tasks.append(task)
            
        await asyncio.gather(*tasks)
        
    def _get_websocket_url(self, channel: str) -> str:
        """获取WebSocket URL"""
        if self.config.exchange == 'binance':
            return f"wss://stream.binance.com:9443/ws"
        # 添加其他交易所的支持
        return ""
        
    async def stop(self):
        """停止WebSocket管理器"""
        self.running = False
        for ws in self.connections.values():
            await ws.close()
            
class MarketDepthAnalyzer:
    """市场深度分析器"""
    
    def __init__(self, depth_levels: int = 10, window_size: int = 100):
        self.depth_levels = depth_levels
        self.window_size = window_size
        self.order_book_history = deque(maxlen=window_size)
        self.imbalance_history = deque(maxlen=window_size)
        self.spread_history = deque(maxlen=window_size)
        
    def update(self, order_book: Dict):
        """更新订单簿数据"""
        self.order_book_history.append(order_book)
        
        # 计算买卖压力比
        bid_volume = sum(float(price) * float(amount)
                        for price, amount in order_book['bids'][:self.depth_levels])
        ask_volume = sum(float(price) * float(amount)
                        for price, amount in order_book['asks'][:self.depth_levels])
                        
        imbalance = bid_volume / (bid_volume + ask_volume) if bid_volume + ask_volume > 0 else 0.5
        self.imbalance_history.append(imbalance)
        
        # 计算买卖价差
        best_bid = float(order_book['bids'][0][0])
        best_ask = float(order_book['asks'][0][0])
        spread = (best_ask - best_bid) / best_bid
        self.spread_history.append(spread)
        
    def get_volume_profile(self) -> pd.DataFrame:
        """获取成交量分布"""
        if not self.order_book_history:
            return pd.DataFrame()
            
        prices = []
        volumes = []
        
        for book in self.order_book_history:
            for price, amount in book['bids'][:self.depth_levels]:
                prices.append(float(price))
                volumes.append(float(amount))
            for price, amount in book['asks'][:self.depth_levels]:
                prices.append(float(price))
                volumes.append(-float(amount))  # 卖单用负值表示
                
        df = pd.DataFrame({'price': prices, 'volume': volumes})
        return df.groupby('price').sum()
        
    def get_imbalance_indicators(self) -> Dict[str, float]:
        """获取订单簿不平衡指标"""
        if not self.imbalance_history:
            return {}
            
        imbalances = np.array(self.imbalance_history)
        return {
            'current_imbalance': imbalances[-1],
            'mean_imbalance': np.mean(imbalances),
            'std_imbalance': np.std(imbalances),
            'min_imbalance': np.min(imbalances),
            'max_imbalance': np.max(imbalances)
        }
        
    def get_spread_analysis(self) -> Dict[str, float]:
        """获取价差分析"""
        if not self.spread_history:
            return {}
            
        spreads = np.array(self.spread_history)
        return {
            'current_spread': spreads[-1],
            'mean_spread': np.mean(spreads),
            'std_spread': np.std(spreads),
            'min_spread': np.min(spreads),
            'max_spread': np.max(spreads)
        }
        
    def get_liquidity_score(self) -> float:
        """计算流动性得分"""
        if not self.order_book_history:
            return 0.0
            
        # 使用价差和深度计算流动性得分
        spread_score = 1 - np.mean(self.spread_history)
        
        # 计算深度得分
        latest_book = self.order_book_history[-1]
        total_volume = sum(float(amount) for _, amount in
                          latest_book['bids'][:self.depth_levels] +
                          latest_book['asks'][:self.depth_levels])
                          
        # 归一化深度得分
        depth_score = min(1.0, total_volume / 1000)  # 假设1000是基准深度
        
        # 综合得分
        return (spread_score + depth_score) / 2
        
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常情况"""
        anomalies = []
        
        if not self.order_book_history:
            return anomalies
            
        # 检查价差异常
        spreads = np.array(self.spread_history)
        mean_spread = np.mean(spreads)
        std_spread = np.std(spreads)
        
        if spreads[-1] > mean_spread + 2 * std_spread:
            anomalies.append({
                'type': 'spread_anomaly',
                'value': spreads[-1],
                'threshold': mean_spread + 2 * std_spread
            })
            
        # 检查订单簿不平衡
        imbalances = np.array(self.imbalance_history)
        if imbalances[-1] > 0.7 or imbalances[-1] < 0.3:
            anomalies.append({
                'type': 'imbalance_anomaly',
                'value': imbalances[-1],
                'threshold': 0.7 if imbalances[-1] > 0.7 else 0.3
            })
            
        return anomalies
