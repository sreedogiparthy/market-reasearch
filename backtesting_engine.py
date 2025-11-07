import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional
import warnings
warnings.filterwarnings('ignore')

class BacktestingEngine:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.strategies = {}
        self._register_default_strategies()
        
    def _register_default_strategies(self):
        """Register default trading strategies"""
        self.add_strategy('moving_average_crossover', self._ma_crossover_strategy)
        
    @staticmethod
    def _ma_crossover_strategy(data, fast_period=20, slow_period=50):
        """
        Moving Average Crossover Strategy
        Generates signals (1 for buy, -1 for sell, 0 for hold) based on MA crossovers
        """
        if len(data) < slow_period:
            return pd.Series(0, index=data.index)
            
        # Calculate moving averages
        fast_ma = data['close'].rolling(window=fast_period).mean()
        slow_ma = data['close'].rolling(window=slow_period).mean()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1  # Buy signal when fast MA crosses above slow MA
        signals[fast_ma < slow_ma] = -1  # Sell signal when fast MA crosses below slow MA
        
        # Ensure we don't have signals in the warmup period
        signals[:slow_period] = 0
        
        return signals
        
        
    def add_strategy(self, name, strategy_function):
        """Add a trading strategy"""
        self.strategies[name] = strategy_function
        
    def backtest_strategy(self, data, strategy_name, **params):
        """Backtest a specific strategy"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not found")
            
        strategy = self.strategies[strategy_name]
        signals = strategy(data, **params)
        
        # Calculate returns
        returns = data['close'].pct_change().shift(-1)
        strategy_returns = returns * signals.shift(1)
        
        # Calculate performance metrics
        total_return = strategy_returns.sum()
        sharpe_ratio = self.calculate_sharpe(strategy_returns)
        max_drawdown = self.calculate_max_drawdown(strategy_returns)
        win_rate = self.calculate_win_rate(strategy_returns)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'signals': signals,
            'returns': strategy_returns
        }
    
    def calculate_sharpe(self, returns, risk_free_rate=0.05):
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_win_rate(self, returns):
        """Calculate win rate"""
        if len(returns) == 0:
            return 0
        winning_trades = returns[returns > 0]
        total_trades = returns[returns != 0]
        return len(winning_trades) / len(total_trades) if len(total_trades) > 0 else 0
    
    def optimize_parameters(self, data, strategy_name, parameter_grid=None):
        """Optimize strategy parameters using grid search"""
        if parameter_grid is None:
            # Default parameter grid for common strategies
            parameter_grid = {
                'fast_period': [10, 20, 50],
                'slow_period': [50, 100, 200],
                'rsi_period': [14, 21, 28]
            }
        
        best_score = -np.inf
        best_params = {}
        
        # Simple grid search implementation
        from itertools import product
        
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))
            
            try:
                result = self.backtest_strategy(data, strategy_name, **params)
                score = result['sharpe_ratio']  # Use Sharpe ratio as optimization criterion
                
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception as e:
                continue
                
        return best_params
    
    def walk_forward_optimization(self, data, strategy_name, optimization_period=126, test_period=63):
        """Perform walk-forward optimization"""
        results = []
        
        for i in range(0, len(data) - optimization_period - test_period, test_period):
            # Split data
            train_data = data.iloc[i:i+optimization_period]
            test_data = data.iloc[i+optimization_period:i+optimization_period+test_period]
            
            # Optimize parameters on training data
            best_params = self.optimize_parameters(train_data, strategy_name)
            
            # Test on out-of-sample data
            test_result = self.backtest_strategy(test_data, strategy_name, **best_params)
            test_result['period'] = test_data.index[0].strftime('%Y-%m-%d')
            test_result['params'] = best_params
            results.append(test_result)
            
        return pd.DataFrame(results)
    
    def monte_carlo_backtest(self, data, strategy_name, num_simulations=1000, **params):
        """Run Monte Carlo simulation on strategy"""
        original_results = self.backtest_strategy(data, strategy_name, **params)
        simulated_returns = []
        
        for _ in range(num_simulations):
            # Create random walk by shuffling returns
            shuffled_returns = original_results['returns'].sample(frac=1).reset_index(drop=True)
            simulated_cumulative = (1 + shuffled_returns).cumprod() - 1
            simulated_returns.append(simulated_cumulative.iloc[-1] if len(shuffled_returns) > 0 else 0)
            
        return {
            'original': original_results,
            'simulated_returns': np.array(simulated_returns),
            'success_probability': np.mean(np.array(simulated_returns) > original_results['total_return'])
        }