class BacktestingEngine:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.strategies = {}
        
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
        returns = data['Close'].pct_change().shift(-1)
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
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_win_rate(self, returns):
        """Calculate win rate"""
        winning_trades = returns[returns > 0]
        return len(winning_trades) / len(returns[returns != 0])
    
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
            results.append(test_result)
            
        return pd.DataFrame(results)