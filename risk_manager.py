import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class RiskManager:
    def __init__(self, account_size=10000, risk_per_trade=0.01):
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade
        
    def calculate_position_size(self, entry_price, stop_loss, portfolio_risk=None):
        """Calculate position size based on risk parameters"""
        if portfolio_risk is None:
            portfolio_risk = self.risk_per_trade
            
        risk_amount = self.account_size * portfolio_risk
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
            
        position_size = risk_amount / price_risk
        return int(position_size)
    
    def calculate_risk_reward(self, entry_price, stop_loss, take_profit):
        """Calculate risk-reward ratio"""
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return 0
            
        return reward / risk
    
    def kelly_criterion(self, win_rate, avg_win, avg_loss):
        """Calculate optimal position size using Kelly Criterion"""
        if avg_loss == 0:
            return 0
            
        win_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / win_ratio
        return max(0, kelly * 0.5)  # Use half-kelly for safety
    
    def monte_carlo_simulation(self, returns, num_simulations=1000, periods=252):
        """Run Monte Carlo simulation for portfolio returns"""
        simulated_returns = []
        
        for _ in range(num_simulations):
            random_returns = np.random.choice(returns, size=periods, replace=True)
            cumulative_return = np.prod(1 + random_returns) - 1
            simulated_returns.append(cumulative_return)
            
        return np.array(simulated_returns)