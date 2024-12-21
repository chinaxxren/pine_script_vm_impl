"""
测试优化后的指标计算性能
"""

import numpy as np
import time
from src.indicators.optimized import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_bollinger_bands, calculate_macd,
    calculate_atr, calculate_keltner_channels,
    calculate_stochastic, calculate_ichimoku,
    calculate_pivot_points, calculate_zigzag,
    calculate_parabolic_sar, calculate_momentum,
    calculate_obv, calculate_williams_r,
    ParallelIndicatorCalculator, MemoryOptimizedIndicator
)

def generate_test_data(size: int = 1000000) -> np.ndarray:
    """生成测试数据"""
    return np.random.randn(size).cumsum()

def test_new_indicators():
    """测试新增指标"""
    # 生成测试数据
    data = generate_test_data()
    high = data + np.random.rand(len(data))
    low = data - np.random.rand(len(data))
    close = data
    volume = np.abs(np.random.randn(len(data))) * 1000
    
    # 测试抛物线转向指标
    start = time.time()
    sar, trend = calculate_parabolic_sar(high, low)
    sar_time = time.time() - start
    print(f"SAR计算时间: {sar_time:.4f}秒")
    
    # 测试动量指标
    start = time.time()
    momentum = calculate_momentum(data, 10)
    momentum_time = time.time() - start
    print(f"动量指标计算时间: {momentum_time:.4f}秒")
    
    # 测试能量潮指标
    start = time.time()
    obv = calculate_obv(close, volume)
    obv_time = time.time() - start
    print(f"OBV计算时间: {obv_time:.4f}秒")
    
    # 测试威廉指标
    start = time.time()
    williams = calculate_williams_r(high, low, close, 14)
    williams_time = time.time() - start
    print(f"威廉指标计算时间: {williams_time:.4f}秒")

def test_parallel_calculation():
    """测试并行计算"""
    # 生成多个测试数据
    data_list = [generate_test_data(100000) for _ in range(10)]
    
    # 创建并行计算器
    calculator = ParallelIndicatorCalculator()
    
    # 测试并行SMA计算
    start = time.time()
    results = calculator.calculate_parallel(calculate_sma, data_list, 20)
    parallel_time = time.time() - start
    print(f"并行SMA计算时间: {parallel_time:.4f}秒")
    
    # 测试串行SMA计算
    start = time.time()
    serial_results = [calculate_sma(data, 20) for data in data_list]
    serial_time = time.time() - start
    print(f"串行SMA计算时间: {serial_time:.4f}秒")
    print(f"并行加速比: {serial_time/parallel_time:.2f}x")
    
    # 测试缓存效果
    start = time.time()
    cached_results = calculator.calculate_parallel(calculate_sma, data_list, 20)
    cached_time = time.time() - start
    print(f"缓存SMA计算时间: {cached_time:.4f}秒")
    
    # 清理缓存
    calculator.clear_cache()

def test_memory_optimization():
    """测试内存优化"""
    # 生成大量测试数据
    data = generate_test_data(5000000)
    
    # 创建内存优化计算器
    optimizer = MemoryOptimizedIndicator(chunk_size=100000)
    
    # 测试分块计算
    start = time.time()
    chunked_result = optimizer.calculate_chunked(calculate_sma, data, 20)
    chunked_time = time.time() - start
    print(f"分块SMA计算时间: {chunked_time:.4f}秒")
    
    # 测试流式计算
    def progress_callback(progress):
        if int(progress * 100) % 20 == 0:
            print(f"处理进度: {progress*100:.0f}%")
            
    start = time.time()
    streaming_result = optimizer.calculate_streaming(
        calculate_sma, data, 20,
        callback=progress_callback
    )
    streaming_time = time.time() - start
    print(f"流式SMA计算时间: {streaming_time:.4f}秒")
    
def test_performance():
    """测试性能"""
    print("\n=== 测试基本指标 ===")
    # 生成测试数据
    data = generate_test_data()
    high = data + np.random.rand(len(data))
    low = data - np.random.rand(len(data))
    close = data
    
    # 预热JIT编译
    _ = calculate_sma(data[:1000], 20)
    _ = calculate_ema(data[:1000], 20)
    _ = calculate_rsi(data[:1000], 14)
    
    # 测试SMA
    start = time.time()
    sma = calculate_sma(data, 20)
    sma_time = time.time() - start
    print(f"SMA计算时间: {sma_time:.4f}秒")
    
    # 测试EMA
    start = time.time()
    ema = calculate_ema(data, 20)
    ema_time = time.time() - start
    print(f"EMA计算时间: {ema_time:.4f}秒")
    
    # 测试RSI
    start = time.time()
    rsi = calculate_rsi(data, 14)
    rsi_time = time.time() - start
    print(f"RSI计算时间: {rsi_time:.4f}秒")
    
    # 测试布林带
    start = time.time()
    upper, middle, lower = calculate_bollinger_bands(data, 20, 2.0)
    bb_time = time.time() - start
    print(f"布林带计算时间: {bb_time:.4f}秒")
    
    # 测试MACD
    start = time.time()
    macd, signal, hist = calculate_macd(data)
    macd_time = time.time() - start
    print(f"MACD计算时间: {macd_time:.4f}秒")
    
    print("\n=== 测试新增指标 ===")
    test_new_indicators()
    
    print("\n=== 测试并行计算 ===")
    test_parallel_calculation()
    
    print("\n=== 测试内存优化 ===")
    test_memory_optimization()
    
if __name__ == '__main__':
    test_performance()
