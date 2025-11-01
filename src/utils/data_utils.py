import yfinance as yf
import pandas as pd
import logging
from typing import Dict, Optional, List
from datetime import datetime

class DataFetcher:
    """Handles data fetching and preparation for the trading bot"""
    
    def __init__(self, config: dict):
        """
        Initialize the data fetcher with configuration
        
        Args:
            config (dict): Configuration dictionary with data-related settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def fetch_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for the given symbols
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping symbols to their historical data DataFrames
        """
        data = {}
        for symbol in symbols:
            try:
                self.logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if df.empty:
                    self.logger.warning(f"No data found for {symbol}")
                    continue
                    
                # Basic data cleaning
                df = self._clean_data(df, symbol)
                data[symbol] = df
                
                self.logger.info(f"Downloaded {len(df)} rows of clean data for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error downloading data for {symbol}: {e}", exc_info=True)
                
        return data
    
    def _clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and prepare the raw data"""
        try:
            # Make a copy to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)
            
            # Handle missing values
            df = df.ffill()  # Forward fill
            df = df.bfill()  # Backward fill any remaining NaNs
            
            # Ensure all required columns are present
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column {col} not found in data for {symbol}")
            
            # Ensure no NaN values in critical columns
            if df[required_columns].isnull().any().any():
                self.logger.warning(f"NaN values found in data for {symbol} after cleaning")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error cleaning data for {symbol}: {e}", exc_info=True)
            raise
    
    def prepare_data(self, data: Dict[str, pd.DataFrame], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for backtesting or live trading
        
        Args:
            data: Dictionary of raw data DataFrames
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary of prepared data
        """
        prepared_data = {}
        for symbol, df in data.items():
            try:
                # Filter by date range
                mask = (df.index >= start_date) & (df.index <= end_date)
                df = df.loc[mask].copy()
                
                if df.empty:
                    self.logger.warning(f"No data in date range for {symbol}")
                    continue
                
                # Add any additional preparation steps here
                prepared_data[symbol] = df
                
            except Exception as e:
                self.logger.error(f"Error preparing data for {symbol}: {e}", exc_info=True)
                
        return prepared_data
