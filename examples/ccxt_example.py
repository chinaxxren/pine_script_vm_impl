"""
CCXT集成示例
"""

import asyncio
import os
from src.exchange.ccxt_client import CCXTClient, ExchangeConfig, DataStreamProcessor, TradingBot
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def print_ticker(ticker):
    """打印行情数据"""
    logger.info(f"Ticker: {ticker['symbol']} - Last: {ticker['last']}")

async def print_order_book(order_book):
    """打印订单簿数据"""
    logger.info(f"Bids: {order_book['bids'][:3]}")
    logger.info(f"Asks: {order_book['asks'][:3]}")

async def main():
    # 创建交易所配置
    config = ExchangeConfig(
        exchange_id='binance',
        api_key=os.getenv('BINANCE_API_KEY'),
        secret=os.getenv('BINANCE_SECRET')
    )
    
    # 创建CCXT客户端
    client = CCXTClient(config)
    await client.initialize()
    
    try:
        # 获取BTC/USDT的K线数据
        symbol = 'BTC/USDT'
        ohlcv = await client.fetch_ohlcv(symbol, timeframe='1h', limit=10)
        logger.info(f"K线数据:\n{ohlcv}")
        
        # 创建数据流处理器
        processor = DataStreamProcessor(client)
        processor.add_callback(print_ticker)
        processor.add_callback(print_order_book)
        
        # 创建交易机器人
        bot = TradingBot(client)
        
        # 更新持仓信息
        await bot.update_positions()
        logger.info(f"当前持仓: {bot.positions}")
        
        # 模拟下单（注意：这里使用的是测试模式）
        if client.exchange.has['createOrder']:
            order = await bot.place_order(
                symbol='BTC/USDT',
                side='buy',
                amount=0.001,
                order_type='limit',
                price=30000.0
            )
            logger.info(f"下单结果: {order}")
            
            # 取消订单
            if order:
                result = await bot.cancel_order(order['id'])
                logger.info(f"取消订单结果: {result}")
                
        # 启动数据流处理
        await asyncio.gather(
            processor.process_ticker(symbol),
            processor.process_order_book(symbol)
        )
        
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
    finally:
        await client.close()

if __name__ == '__main__':
    asyncio.run(main())
