"""
回测引擎测试用例
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    PortfolioBacktestEngine
)
from src.trading.strategy import (
    Strategy,
    StrategyConfig,
    TrendFollowingStrategy,
    MeanReversionStrategy,
    BreakoutStrategy
)

class TestBacktestEngine(unittest.TestCase):
    """回测引擎测试"""
    
    def setUp(self):
        """测试数据准备"""
        # 创建测试数据
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        self.test_data = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 100,
            'high': np.random.randn(len(dates)).cumsum() + 102,
            'low': np.random.randn(len(dates)).cumsum() + 98,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # 创建回测配置
        self.config = BacktestConfig(
            initial_capital=100000.0,
            commission_rate=0.001,
            slippage=0.001,
            risk_free_rate=0.02
        )
        
        # 创建策略配置
        self.strategy_config = StrategyConfig(
            symbol='BTC/USDT',
            timeframe='1d',
            risk_per_trade=0.02,
            max_positions=3,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            max_drawdown=0.1
        )
        
    def test_trend_following_strategy(self):
        """测试趋势跟踪策略"""
        # 创建策略
        strategy = TrendFollowingStrategy(self.strategy_config)
        
        # 创建回测引擎
        engine = BacktestEngine(strategy, self.test_data, self.config)
        
        # 运行回测
        result = engine.run()
        
        # 验证结果
        self.assertIsInstance(result, BacktestResult)
        self.assertIsNotNone(result.equity_curve)
        self.assertTrue(len(result.trades) > 0)
        
        # 检查性能指标
        self.assertIn('total_return', result.performance_metrics)
        self.assertIn('sharpe_ratio', result.performance_metrics)
        
        # 检查风险指标
        self.assertIn('max_drawdown', result.risk_metrics)
        self.assertIn('value_at_risk', result.risk_metrics)
        
    def test_mean_reversion_strategy(self):
        """测试均值回归策略"""
        # 创建策略
        strategy = MeanReversionStrategy(self.strategy_config)
        
        # 创建回测引擎
        engine = BacktestEngine(strategy, self.test_data, self.config)
        
        # 运行回测
        result = engine.run()
        
        # 验证结果
        self.assertIsInstance(result, BacktestResult)
        self.assertTrue(len(result.trades) > 0)
        
        # 检查交易记录
        for trade in result.trades:
            self.assertIn('entry_time', trade)
            self.assertIn('exit_time', trade)
            self.assertIn('pnl', trade)
            self.assertIn('side', trade)
            
    def test_breakout_strategy(self):
        """测试突破策略"""
        # 创建策略
        strategy = BreakoutStrategy(self.strategy_config)
        
        # 创建回测引擎
        engine = BacktestEngine(strategy, self.test_data, self.config)
        
        # 运行回测
        result = engine.run()
        
        # 验证结果
        self.assertIsInstance(result, BacktestResult)
        self.assertTrue(len(result.trades) > 0)
        
    def test_portfolio_backtest(self):
        """测试投资组合回测"""
        # 创建多个策略
        strategies = {
            'BTC/USDT': TrendFollowingStrategy(self.strategy_config),
            'ETH/USDT': MeanReversionStrategy(self.strategy_config),
            'LTC/USDT': BreakoutStrategy(self.strategy_config)
        }
        
        # 创建多个数据集
        data = {
            'BTC/USDT': self.test_data.copy(),
            'ETH/USDT': self.test_data.copy() * 1.2,
            'LTC/USDT': self.test_data.copy() * 0.8
        }
        
        # 创建投资组合回测引擎
        engine = PortfolioBacktestEngine(strategies, data, self.config)
        
        # 运行回测
        results = engine.run()
        
        # 验证结果
        self.assertEqual(len(results), 3)
        for symbol, result in results.items():
            self.assertIsInstance(result, BacktestResult)
            self.assertTrue(len(result.trades) > 0)
            
    def test_risk_management(self):
        """测试风险管理"""
        # 创建策略
        strategy = TrendFollowingStrategy(self.strategy_config)
        
        # 创建回测引擎
        engine = BacktestEngine(strategy, self.test_data, self.config)
        
        # 运行回测
        result = engine.run()
        
        # 验证风险指标
        self.assertIn('value_at_risk', result.risk_metrics)
        self.assertIn('expected_shortfall', result.risk_metrics)
        self.assertIn('beta', result.risk_metrics)
        self.assertIn('volatility', result.risk_metrics)
        
    def test_performance_metrics(self):
        """测试性能指标计算"""
        # 创建策略
        strategy = TrendFollowingStrategy(self.strategy_config)
        
        # 创建回测引擎
        engine = BacktestEngine(strategy, self.test_data, self.config)
        
        # 运行回测
        result = engine.run()
        
        # 验证性能指标
        metrics = result.performance_metrics
        self.assertGreaterEqual(metrics['total_trades'], 0)
        self.assertGreaterEqual(metrics['winning_trades'], 0)
        self.assertGreaterEqual(metrics['losing_trades'], 0)
        self.assertGreaterEqual(metrics['win_rate'], 0)
        self.assertGreaterEqual(metrics['profit_factor'], 0)
        
    def test_edge_cases(self):
        """测试边缘情况"""
        # 创建空数据
        empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        # 创建策略
        strategy = TrendFollowingStrategy(self.strategy_config)
        
        # 创建回测引擎
        engine = BacktestEngine(strategy, empty_data, self.config)
        
        # 运行回测
        result = engine.run()
        
        # 验证结果
        self.assertEqual(len(result.trades), 0)
        self.assertEqual(len(result.equity_curve), 0)
        
    def test_position_sizing(self):
        """测试仓位管理"""
        strategy_config = StrategyConfig(
            symbol='TEST',
            timeframe='1d',
            risk_per_trade=0.02,
            max_positions=3,
            position_sizing='risk'
        )
        strategy = TrendFollowingStrategy(strategy_config)
        
        engine = BacktestEngine(strategy, self.test_data, self.config)
        result = engine.run()
        
        # 验证持仓限制
        long_positions = [0]  # 多头持仓数量
        short_positions = [0]  # 空头持仓数量
        
        for trade in result.trades:
            if trade['side'] == 'buy':
                if 'exit_time' not in trade:  # 开仓
                    long_positions.append(long_positions[-1] + 1)
                    short_positions.append(short_positions[-1])
                else:  # 平仓
                    long_positions.append(long_positions[-1] - 1)
                    short_positions.append(short_positions[-1])
            else:  # sell
                if 'exit_time' not in trade:  # 开仓
                    short_positions.append(short_positions[-1] + 1)
                    long_positions.append(long_positions[-1])
                else:  # 平仓
                    short_positions.append(short_positions[-1] - 1)
                    long_positions.append(long_positions[-1])
                
        self.assertLessEqual(max(long_positions), strategy_config.max_positions)
        self.assertLessEqual(max(short_positions), strategy_config.max_positions)
        
        # 验证风险限制
        for trade in result.trades:
            if 'stop_loss' in trade:  # 只检查开仓交易
                potential_loss = abs(trade['entry_price'] - trade['stop_loss']) * trade['amount']
                self.assertLessEqual(
                    potential_loss / self.config.initial_capital,
                    strategy_config.risk_per_trade * 1.2  # 允许20%的误差，考虑滑点和手续费的影响
                )
            
    def test_commission_and_slippage(self):
        """测试手续费和滑点"""
        # 创建高手续费和滑点的配置
        high_cost_config = BacktestConfig(
            initial_capital=100000.0,
            commission_rate=0.01,  # 1%手续费
            slippage=0.01,  # 1%滑点
            risk_free_rate=0.02
        )
        
        # 创建策略
        strategy = TrendFollowingStrategy(self.strategy_config)
        
        # 运行两次回测，比较结果
        engine1 = BacktestEngine(strategy, self.test_data, self.config)
        result1 = engine1.run()
        
        engine2 = BacktestEngine(strategy, self.test_data, high_cost_config)
        result2 = engine2.run()
        
        # 验证高成本回测的收益更低
        self.assertGreater(
            result1.performance_metrics['total_return'],
            result2.performance_metrics['total_return']
        )
        
if __name__ == '__main__':
    unittest.main()
