import logging
from typing import Dict, List, Optional
import pandas as pd

class RiskManager:
    """Manages risk for the trading bot"""
    
    def __init__(self, config: dict):
        """
        Initialize the risk manager with configuration
        
        Args:
            config (dict): Configuration dictionary with risk parameters
        """
        self.config = config.get('risk', {})
        self.logger = logging.getLogger(__name__)
        
        # Set default values if not provided in config
        self.max_position_size = self.config.get('max_position_size', 0.1)  # 10% of portfolio per position
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.02)  # 2% max risk per trade
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.05)  # 5% stop loss
        self.take_profit_pct = self.config.get('take_profit_pct', 0.10)  # 10% take profit
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, account_balance: float) -> float:
        """
        Calculate position size based on risk parameters
        
        Args:
            entry_price: Entry price of the trade
            stop_loss: Stop loss price
            account_balance: Current account balance
            
        Returns:
            Number of shares to buy/sell
        """
        try:
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)
            
            if risk_per_share <= 0:
                self.logger.warning("Invalid stop loss or entry price")
                return 0.0
                
            # Calculate position size based on risk
            risk_amount = account_balance * self.max_portfolio_risk
            position_size = risk_amount / risk_per_share
            
            # Apply position size limits
            max_position_value = account_balance * self.max_position_size
            max_position_size = max_position_value / entry_price
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}", exc_info=True)
            return 0.0
    
    def validate_trade(self, symbol: str, price: float, quantity: float, 
                      portfolio: Dict, open_positions: Dict) -> bool:
        """
        Validate if a trade meets risk parameters
        
        Args:
            symbol: Stock symbol
            price: Current price
            quantity: Number of shares to trade
            portfolio: Current portfolio holdings
            open_positions: Current open positions
            
        Returns:
            bool: True if trade is valid, False otherwise
        """
        try:
            # Check if we already have too many positions
            max_positions = self.config.get('max_positions', 10)
            if len(open_positions) >= max_positions:
                self.logger.warning(f"Maximum number of positions ({max_positions}) reached")
                return False
                
            # Check position size
            position_value = price * abs(quantity)
            portfolio_value = portfolio.get('total_value', 0)
            
            if portfolio_value <= 0:
                self.logger.warning("Invalid portfolio value")
                return False
                
            position_pct = position_value / portfolio_value
            if position_pct > self.max_position_size:
                self.logger.warning(f"Position size {position_pct:.1%} exceeds maximum {self.max_position_size:.1%}")
                return False
                
            # Check if we already have a position in this symbol
            if symbol in open_positions:
                self.logger.info(f"Already have a position in {symbol}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {e}", exc_info=True)
            return False
    
    def calculate_stop_loss(self, entry_price: float, is_long: bool = True) -> float:
        """
        Calculate stop loss price
        
        Args:
            entry_price: Entry price of the trade
            is_long: Whether this is a long position (True) or short position (False)
            
        Returns:
            Stop loss price
        """
        if is_long:
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, is_long: bool = True) -> float:
        """
        Calculate take profit price
        
        Args:
            entry_price: Entry price of the trade
            is_long: Whether this is a long position (True) or short position (False)
            
        Returns:
            Take profit price
        """
        if is_long:
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)
