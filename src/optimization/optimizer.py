"""
策略优化器
"""

from typing import Dict, List, Optional, Tuple, Callable, Type, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit

from ..trading.strategy import Strategy
from ..backtest.engine import BacktestEngine, BacktestConfig

@dataclass
class OptimizationConfig:
    """优化配置"""
    method: str  # 优化方法：'grid', 'random', 'bayesian'
    parameters: Dict  # 参数范围
    metric: Union[str, List[str]]  # 优化指标
    maximize: Union[bool, List[bool]]  # 是否最大化
    n_trials: Optional[int] = None  # 试验次数（随机搜索和贝叶斯优化）
    walk_forward: bool = False  # 是否使用向前优化
    train_ratio: float = 0.7  # 训练集比例
    constraints: Optional[List[Callable]] = None  # 参数约束条件
    
class StrategyOptimizer:
    """策略优化器"""
    
    def __init__(self,
                strategy_class: Type[Strategy],
                data: pd.DataFrame,
                backtest_config: BacktestConfig,
                strategy_config: Dict,
                optimization_config: OptimizationConfig):
        """初始化优化器
        
        Args:
            strategy_class: 策略类
            data: 历史数据
            backtest_config: 回测配置
            strategy_config: 策略配置
            optimization_config: 优化配置
        """
        self.strategy_class = strategy_class
        self.data = data
        self.backtest_config = backtest_config
        self.strategy_config = strategy_config
        self.optimization_config = optimization_config
        
    def optimize(self) -> List[Dict]:
        """运行优化
        
        Returns:
            List[Dict]: 优化结果列表，按性能排序
        """
        if self.optimization_config.method == 'grid':
            return self._grid_search()
        elif self.optimization_config.method == 'random':
            return self._random_search()
        elif self.optimization_config.method == 'bayesian':
            return self._bayesian_optimization()
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_config.method}")
            
    def _grid_search(self) -> List[Dict]:
        """网格搜索
        
        Returns:
            List[Dict]: 优化结果列表
        """
        # 生成参数网格
        param_grid = self._generate_param_grid()
        results = []
        
        for params in param_grid:
            # 检查参数约束
            if not self._check_constraints(params):
                continue
                
            # 运行回测
            result = self._run_backtest(params)
            results.append(result)
            
        return self._sort_results(results)
        
    def _random_search(self) -> List[Dict]:
        """随机搜索
        
        Returns:
            List[Dict]: 优化结果列表
        """
        results = []
        n_trials = self.optimization_config.n_trials or 100
        
        for _ in range(n_trials):
            # 随机生成参数
            params = self._generate_random_params()
            
            # 检查参数约束
            if not self._check_constraints(params):
                continue
                
            # 运行回测
            result = self._run_backtest(params)
            results.append(result)
            
        return self._sort_results(results)
        
    def _bayesian_optimization(self) -> List[Dict]:
        """贝叶斯优化
        
        Returns:
            List[Dict]: 优化结果列表
        """
        from skopt import gp_minimize
        from skopt.space import Real, Integer
        
        # 定义参数空间
        space = []
        param_names = []
        for name, range_ in self.optimization_config.parameters.items():
            param_names.append(name)
            if isinstance(range_, tuple):
                if isinstance(range_[0], int):
                    space.append(Integer(range_[0], range_[1]))
                else:
                    space.append(Real(range_[0], range_[1]))
                    
        # 定义目标函数
        def objective(x):
            params = dict(zip(param_names, x))
            if not self._check_constraints(params):
                return float('inf') if self.optimization_config.maximize else float('-inf')
            result = self._run_backtest(params)
            return -result['metrics'][self.optimization_config.metric] if self.optimization_config.maximize else result['metrics'][self.optimization_config.metric]
            
        # 运行优化
        n_trials = self.optimization_config.n_trials or 50
        res = gp_minimize(objective, space, n_calls=n_trials)
        
        # 转换结果
        results = []
        for x, value in zip(res.x_iters, res.func_vals):
            params = dict(zip(param_names, x))
            result = self._run_backtest(params)
            results.append(result)
            
        return self._sort_results(results)
        
    def _generate_param_grid(self) -> List[Dict]:
        """生成参数网格"""
        from itertools import product
        
        # 获取参数名和值范围
        param_names = []
        param_values = []
        for name, values in self.optimization_config.parameters.items():
            param_names.append(name)
            if isinstance(values, range):
                param_values.append(list(values))
            else:
                param_values.append(values)
                
        # 生成所有组合
        param_grid = []
        for combo in product(*param_values):
            param_grid.append(dict(zip(param_names, combo)))
            
        return param_grid
        
    def _generate_random_params(self) -> Dict:
        """生成随机参数"""
        params = {}
        for name, range_ in self.optimization_config.parameters.items():
            if isinstance(range_, tuple):
                if isinstance(range_[0], int):
                    params[name] = np.random.randint(range_[0], range_[1] + 1)
                else:
                    params[name] = np.random.uniform(range_[0], range_[1])
            elif isinstance(range_, range):
                params[name] = np.random.choice(range_)
            else:
                params[name] = np.random.choice(range_)
        return params
        
    def _check_constraints(self, params: Dict) -> bool:
        """检查参数约束"""
        if self.optimization_config.constraints is None:
            return True
            
        return all(constraint(params) for constraint in self.optimization_config.constraints)
        
    def _run_backtest(self, params: Dict) -> Dict:
        """运行回测
        
        Args:
            params: 策略参数
            
        Returns:
            Dict: 回测结果
        """
        # 更新策略配置
        strategy_config = self.strategy_config.copy()
        strategy_config.update(params)
        
        if self.optimization_config.walk_forward:
            return self._run_walk_forward(strategy_config)
        else:
            # 创建策略和回测引擎
            strategy = self.strategy_class(strategy_config)
            engine = BacktestEngine(strategy, self.data, self.backtest_config)
            
            # 运行回测
            result = engine.run()
            
            return {
                'parameters': params,
                'metrics': result.performance_metrics
            }
            
    def _run_walk_forward(self, strategy_config: Dict) -> Dict:
        """运行向前优化
        
        Args:
            strategy_config: 策略配置
            
        Returns:
            Dict: 回测结果
        """
        # 创建时间序列分割器
        tscv = TimeSeriesSplit(n_splits=5)
        
        train_metrics = []
        test_metrics = []
        
        for train_idx, test_idx in tscv.split(self.data):
            # 分割数据
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]
            
            # 在训练集上训练策略
            strategy = self.strategy_class(strategy_config)
            engine = BacktestEngine(strategy, train_data, self.backtest_config)
            train_result = engine.run()
            
            # 在测试集上验证策略
            strategy = self.strategy_class(strategy_config)
            engine = BacktestEngine(strategy, test_data, self.backtest_config)
            test_result = engine.run()
            
            train_metrics.append(train_result.performance_metrics)
            test_metrics.append(test_result.performance_metrics)
            
        # 计算平均指标
        train_metrics_avg = self._average_metrics(train_metrics)
        test_metrics_avg = self._average_metrics(test_metrics)
        
        return {
            'parameters': strategy_config,
            'train_metrics': train_metrics_avg,
            'test_metrics': test_metrics_avg
        }
        
    def _average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """计算平均指标"""
        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            avg_metrics[key] = np.mean(values)
        return avg_metrics
        
    def _sort_results(self, results: List[Dict]) -> List[Dict]:
        """对结果排序"""
        metric = self.optimization_config.metric
        maximize = self.optimization_config.maximize
        
        if isinstance(metric, str):
            metrics = [metric]
            maximize = [maximize]
        else:
            metrics = metric
            
        def sort_key(result):
            if 'test_metrics' in result:
                metrics_dict = result['test_metrics']
            else:
                metrics_dict = result['metrics']
                
            values = []
            for m, max_ in zip(metrics, maximize):
                value = metrics_dict[m]
                values.append(value if max_ else -value)
            return tuple(values)
            
        return sorted(results, key=sort_key, reverse=True)
