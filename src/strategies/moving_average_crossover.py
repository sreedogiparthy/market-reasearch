import pandas as pd
import logging

class EnhancedMovingAverageCrossover:
    """Enhanced Moving Average Crossover Strategy with RSI and Volume Filters"""
    
    def __init__(self, short_window=20, long_window=50, rsi_period=14, volume_threshold=1.2):
        """
        Initialize the strategy with parameters
        
        Args:
            short_window (int): Window for short-term moving average
            long_window (int): Window for long-term moving average
            rsi_period (int): Period for RSI calculation
            volume_threshold (float): Volume threshold as a multiple of average volume
        """
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.volume_threshold = volume_threshold
        self.logger = logging.getLogger(__name__)
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI for a price series"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def generate_signals(self, data):
        """
        Generate trading signals based on moving average crossover with RSI and volume filters
        
        Args:
            data (pd.DataFrame): DataFrame with 'Close' and 'Volume' columns
            
        Returns:
            pd.Series: Series of signals (1 for buy, -1 for sell, 0 for hold)
        """
        try:
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            
            # Calculate moving averages
            signals['short_mavg'] = data['Close'].rolling(window=self.short_window, min_periods=1).mean()
            signals['long_mavg'] = data['Close'].rolling(window=self.long_window, min_periods=1).mean()
            
            # Calculate RSI
            signals['rsi'] = self.calculate_rsi(data['Close'], self.rsi_period)
            
            # Calculate volume moving average
            signals['volume_ma'] = data['Volume'].rolling(window=20).mean()
            
            # Generate signals
            signals['signal'] = 0  # Default to no position
            
            # Long signal: short MA crosses above long MA, RSI not overbought, volume above threshold
            signals.loc[
                (signals['short_mavg'] > signals['long_mavg']) & 
                (signals['short_mavg'].shift(1) <= signals['long_mavg'].shift(1)) &
                (signals['rsi'] < 70) &  # Not overbought
                (data['Volume'] > signals['volume_ma'] * self.volume_threshold),  # Volume spike
                'signal'
            ] = 1
            
            # Short signal: short MA crosses below long MA, RSI not oversold, volume above threshold
            signals.loc[
                (signals['short_mavg'] < signals['long_mavg']) & 
                (signals['short_mavg'].shift(1) >= signals['long_mavg'].shift(1)) &
                (signals['rsi'] > 30) &  # Not oversold
                (data['Volume'] > signals['volume_ma'] * self.volume_threshold),  # Volume spike
                'signal'
            ] = -1
            
            return signals['signal']
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}", exc_info=True)
            # Return a series of zeros with the same index as input data
            return pd.Series(0, index=data.index)
