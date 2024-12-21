"""
测试CCXT客户端
"""

import pytest
import asyncio
import os
from src.exchange.ccxt_client import CCXTClient, ExchangeConfig, DataStreamProcessor, TradingBot
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def config():
    """创建测试配置"""
    return ExchangeConfig(
        exchange_id='binance',
        api_key=os.getenv('BINANCE_API_KEY'),
        secret=os.getenv('BINANCE_SECRET')
    )

@pytest.fixture
async def client(config):
    """创建测试客户端"""
    client = CCXTClient(config)
    await client.initialize()
    yield client
    await client.close()

@pytest.mark.asyncio
async def test_fetch_ohlcv(client):
    """测试获取K线数据"""
    symbol = 'BTC/USDT'
    timeframe = '1h'
    limit = 10
    
    df = await client.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= limit
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
@pytest.mark.asyncio
async def test_fetch_ticker(client):
    """测试获取行情数据"""
    symbol = 'BTC/USDT'
    
    ticker = await client.fetch_ticker(symbol)
    
    assert isinstance(ticker, dict)
    assert 'symbol' in ticker
    assert 'last' in ticker
    
@pytest.mark.asyncio
async def test_fetch_order_book(client):
    """测试获取订单簿数据"""
    symbol = 'BTC/USDT'
    limit = 5
    
    order_book = await client.fetch_order_book(symbol, limit)
    
    assert isinstance(order_book, dict)
    assert 'bids' in order_book
    assert 'asks' in order_book
    assert len(order_book['bids']) <= limit
    assert len(order_book['asks']) <= limit
    
@pytest.mark.asyncio
async def test_data_stream_processor(client):
    """测试数据流处理器"""
    received_data = []
    
    async def callback(data):
        received_data.append(data)
        
    processor = DataStreamProcessor(client)
    processor.add_callback(callback)
    
    symbol = 'BTC/USDT'
    
    # 运行处理器一小段时间
    task = asyncio.create_task(processor.process_ticker(symbol))
    await asyncio.sleep(2)
    task.cancel()
    
    assert len(received_data) > 0
    
@pytest.mark.asyncio
async def test_trading_bot(client):
    """测试交易机器人"""
    bot = TradingBot(client)
    
    # 测试更新持仓
    await bot.update_positions()
    assert isinstance(bot.positions, dict)
    
    # 测试下单和取消订单（使用测试模式）
    if client.exchange.has['createOrder']:
        order = await bot.place_order(
            symbol='BTC/USDT',
            side='buy',
            amount=0.001,
            order_type='limit',
            price=30000.0
        )
        
        assert isinstance(order, dict)
        assert 'id' in order
        
        # 测试取消订单
        result = await bot.cancel_order(order['id'])
        assert isinstance(result, dict)
        
@pytest.mark.asyncio
async def test_error_handling(client):
    """测试错误处理"""
    # 测试无效的交易对
    with pytest.raises(Exception):
        await client.fetch_ticker('INVALID/PAIR')
        
    # 测试无效的时间框架
    with pytest.raises(Exception):
        await client.fetch_ohlcv('BTC/USDT', timeframe='invalid')
