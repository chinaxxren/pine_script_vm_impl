"""
优化器测试
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.optimization.optimizer import StrategyOptimizer, OptimizationConfig
from src.trading.strategy import TrendFollowingStrategy, StrategyConfig
from src.backtest.engine import BacktestEngine, BacktestConfig

class TestOptimizer(unittest.TestCase):
    """优化器测试"""
    
    def setUp(self):
        """准备测试数据"""
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        self.data = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 100,
            'high': np.random.randn(len(dates)).cumsum() + 102,
            'low': np.random.randn(len(dates)).cumsum() + 98,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # 创建回测配置
        self.backtest_config = BacktestConfig(
            initial_capital=100000.0,
            commission_rate=0.001,
            slippage=0.001
        )
        
        # 创建策略配置
        self.strategy_config = StrategyConfig(
            symbol='TEST',
            timeframe='1d',
            risk_per_trade=0.02,
            max_positions=3
        )
        
    def test_grid_search(self):
        """测试网格搜索"""
        # 创建优化配置
        opt_config = OptimizationConfig(
            method='grid',
            parameters={
                'fast_length': range(5, 15, 2),  # [5, 7, 9, 11, 13]
                'slow_length': range(20, 40, 5)  # [20, 25, 30, 35]
            },
            metric='sharpe_ratio',
            maximize=True
        )
        
        # 创建优化器
        optimizer = StrategyOptimizer(
            strategy_class=TrendFollowingStrategy,
            data=self.data,
            backtest_config=self.backtest_config,
            strategy_config=self.strategy_config,
            optimization_config=opt_config
        )
        
        # 运行优化
        results = optimizer.optimize()
        
        # 验证结果
        self.assertTrue(len(results) > 0)
        self.assertEqual(len(results), 20)  # 5 * 4 种参数组合
        
        # 验证最优结果
        best_params = results[0]
        self.assertIn('parameters', best_params)
        self.assertIn('metrics', best_params)
        self.assertIn('fast_length', best_params['parameters'])
        self.assertIn('slow_length', best_params['parameters'])
        self.assertIn('sharpe_ratio', best_params['metrics'])
        
    def test_random_search(self):
        """测试随机搜索"""
        # 创建优化配置
        opt_config = OptimizationConfig(
            method='random',
            parameters={
                'fast_length': (5, 15),
                'slow_length': (20, 40)
            },
            n_trials=20,
            metric='total_return',
            maximize=True
        )
        
        # 创建优化器
        optimizer = StrategyOptimizer(
            strategy_class=TrendFollowingStrategy,
            data=self.data,
            backtest_config=self.backtest_config,
            strategy_config=self.strategy_config,
            optimization_config=opt_config
        )
        
        # 运行优化
        results = optimizer.optimize()
        
        # 验证结果
        self.assertEqual(len(results), 20)  # 20 次随机尝试
        
        # 验证参数范围
        for result in results:
            fast_length = result['parameters']['fast_length']
            slow_length = result['parameters']['slow_length']
            self.assertTrue(5 <= fast_length <= 15)
            self.assertTrue(20 <= slow_length <= 40)
            
    def test_bayesian_optimization(self):
        """测试贝叶斯优化"""
        # 创建优化配置
        opt_config = OptimizationConfig(
            method='bayesian',
            parameters={
                'fast_length': (5, 15),
                'slow_length': (20, 40),
                'risk_per_trade': (0.01, 0.05)
            },
            n_trials=10,
            metric='sortino_ratio',
            maximize=True
        )
        
        # 创建优化器
        optimizer = StrategyOptimizer(
            strategy_class=TrendFollowingStrategy,
            data=self.data,
            backtest_config=self.backtest_config,
            strategy_config=self.strategy_config,
            optimization_config=opt_config
        )
        
        # 运行优化
        results = optimizer.optimize()
        
        # 验证结果
        self.assertEqual(len(results), 10)  # 10 次贝叶斯优化尝试
        
        # 验证优化过程是否收敛
        metrics = [r['metrics']['sortino_ratio'] for r in results]
        self.assertTrue(max(metrics) >= metrics[0])  # 最好的结果应该不差于第一次尝试
        
    def test_walk_forward_optimization(self):
        """测试向前优化"""
        # 创建优化配置
        opt_config = OptimizationConfig(
            method='grid',
            parameters={
                'fast_length': range(5, 15, 5),
                'slow_length': range(20, 40, 10)
            },
            metric='sharpe_ratio',
            maximize=True,
            walk_forward=True,
            train_ratio=0.7
        )
        
        # 创建优化器
        optimizer = StrategyOptimizer(
            strategy_class=TrendFollowingStrategy,
            data=self.data,
            backtest_config=self.backtest_config,
            strategy_config=self.strategy_config,
            optimization_config=opt_config
        )
        
        # 运行优化
        results = optimizer.optimize()
        
        # 验证结果
        self.assertTrue(len(results) > 0)
        
        # 验证训练集和测试集的结果
        for result in results:
            self.assertIn('train_metrics', result)
            self.assertIn('test_metrics', result)
            self.assertIn('sharpe_ratio', result['train_metrics'])
            self.assertIn('sharpe_ratio', result['test_metrics'])
            
    def test_constraint_handling(self):
        """测试约束处理"""
        # 创建带有约束的优化配置
        opt_config = OptimizationConfig(
            method='grid',
            parameters={
                'fast_length': range(5, 15, 2),
                'slow_length': range(20, 40, 5)
            },
            constraints=[
                lambda params: params['fast_length'] < params['slow_length']  # 快线长度必须小于慢线长度
            ],
            metric='sharpe_ratio',
            maximize=True
        )
        
        # 创建优化器
        optimizer = StrategyOptimizer(
            strategy_class=TrendFollowingStrategy,
            data=self.data,
            backtest_config=self.backtest_config,
            strategy_config=self.strategy_config,
            optimization_config=opt_config
        )
        
        # 运行优化
        results = optimizer.optimize()
        
        # 验证约束条件
        for result in results:
            params = result['parameters']
            self.assertLess(params['fast_length'], params['slow_length'])
            
    def test_multi_objective_optimization(self):
        """测试多目标优化"""
        # 创建多目标优化配置
        opt_config = OptimizationConfig(
            method='grid',
            parameters={
                'fast_length': range(5, 15, 5),
                'slow_length': range(20, 40, 10)
            },
            metrics=['sharpe_ratio', 'max_drawdown'],
            maximize=[True, False]  # 最大化夏普比率，最小化最大回撤
        )
        
        # 创建优化器
        optimizer = StrategyOptimizer(
            strategy_class=TrendFollowingStrategy,
            data=self.data,
            backtest_config=self.backtest_config,
            strategy_config=self.strategy_config,
            optimization_config=opt_config
        )
        
        # 运行优化
        results = optimizer.optimize()
        
        # 验证帕累托前沿
        for result in results:
            self.assertIn('sharpe_ratio', result['metrics'])
            self.assertIn('max_drawdown', result['metrics'])
            
        # 验证非支配解
        def is_dominated(a, b):
            """检查解 a 是否被解 b 支配"""
            return (b['metrics']['sharpe_ratio'] >= a['metrics']['sharpe_ratio'] and
                    b['metrics']['max_drawdown'] <= a['metrics']['max_drawdown'])
                    
        for i, result_a in enumerate(results):
            dominated = False
            for j, result_b in enumerate(results):
                if i != j and is_dominated(result_a, result_b):
                    dominated = True
                    break
            self.assertFalse(dominated)  # 所有解都应该是非支配的
            
if __name__ == '__main__':
    unittest.main()
