import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple
from ta.utils import dropna
from ta import add_all_ta_features

# Local imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.stock_config import load_stock_config, get_company_ticker

# Load stock configuration
STOCK_CONFIG = load_stock_config()


def get_stock_data(stock_group: str = "indian_it", period: str = "6mo", interval: str = "1d") -> Tuple[Dict, Dict]:
    """
    Fetch stock data using yfinance with basic technical indicators
    """
    stocks_data = {}
    metadata = {}
    
    if stock_group not in STOCK_CONFIG:
        print(f"Stock group '{stock_group}' not found. Using 'indian_it' as default.")
        stock_group = "indian_it"
    
    stock_list = STOCK_CONFIG[stock_group]
    
    for company, info in stock_list.items():
        if isinstance(info, str):
            ticker = info
            sector = "Unknown"
        else:
            ticker = info.get('symbol', '')
            sector = info.get('sector', 'Unknown')
        
        if not ticker:
            print(f"No ticker symbol for {company}")
            continue
            
        try:
            print(f"Fetching {company} ({ticker})...")
            
            # Fetch data
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval, auto_adjust=True)
            
            if df.empty:
                print(f"No data for {company}")
                continue
            
            # Calculate technical indicators using ta package
            try:
                # Ensure data is clean and has the required columns
                df = df.copy()
                # Convert column names to lowercase for consistency
                df.columns = [col.lower() for col in df.columns]
                
                # Create a new DataFrame with just the OHLCV data we need
                ohlcv = df[['open', 'high', 'low', 'close', 'volume']].copy()
                ohlcv = dropna(ohlcv)
                
                # Add technical indicators to the OHLCV data
                df_ta = add_all_ta_features(
                    ohlcv,
                    open="open",
                    high="high",
                    low="low",
                    close="close",
                    volume="volume",
                    fillna=True
                )
                
                # Select only the columns we need
                keep_columns = [
                    'open', 'high', 'low', 'close', 'volume',
                    'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
                    'momentum_rsi', 'momentum_macd', 'momentum_macd_signal', 'momentum_macd_diff',
                    'volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi', 'volatility_bbli',
                    'volume_obv', 'volume_adi', 'volume_cmf', 'volume_fi'
                ]
                
                # Filter columns to keep only those that exist in df_ta
                existing_columns = [col for col in keep_columns if col in df_ta.columns]
                df_ta = df_ta[existing_columns]
                
                # Calculate daily change
                df_ta = df_ta.assign(daily_change=df_ta['close'].pct_change() * 100)
                
                # Use the new DataFrame with technical indicators
                df = df_ta
                
                # Keep only the most relevant indicators to avoid too many columns
                keep_columns = [
                    'open', 'high', 'low', 'close', 'volume', 'daily_change',
                    'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
                    'momentum_rsi', 'momentum_macd', 'momentum_macd_signal', 'momentum_macd_diff',
                    'volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi', 'volatility_bbli',
                    'volume_obv', 'volume_adi', 'volume_cmf', 'volume_fi'
                ]
                
                # Filter columns to keep
                df = df[[col for col in keep_columns if col in df.columns]]
                
            except Exception as ind_error:
                print(f"Indicator error for {company}: {ind_error}")
                # Continue with basic data if ta indicators fail
                df.columns = [col.lower() for col in df.columns]
                df['daily_change'] = df['close'].pct_change() * 100
            
            stocks_data[company] = df
            metadata[company] = {
                'ticker': ticker,
                'sector': sector,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"Error processing {company}: {e}")
    
    return stocks_data, metadata

def get_fundamental_data(symbol: str, stock_group: str = "indian_it") -> Dict[str, Any]:
    """
    Get fundamental data with fallback to yfinance if API fails
    
    Args:
        symbol: Company symbol or name
        stock_group: Stock group to look up (default: "indian_it")
        
    Returns:
        Dict containing fundamental data
    """
    try:
        # Use the get_company_ticker function to resolve the symbol
        ticker = get_company_ticker(symbol, stock_group=stock_group)
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', info.get('forwardPE', 0)),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'profit_margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
            'analyst_target': info.get('targetMeanPrice', 0)
        }
    except Exception as e:
        print(f"Error getting fundamental data for {symbol}: {e}")
        return {}

def get_company_news(symbol: str, days: int = 7, stock_group: str = "indian_it") -> List[Dict]:
    """
    Get company news with fallback to yfinance if API fails
    
    Args:
        symbol: Company symbol or name
        days: Number of days of news to fetch
        stock_group: Stock group to look up (default: "indian_it")
        
    Returns:
        List of news items with headlines and URLs
    """
    try:
        # Use the get_company_ticker function to resolve the symbol
        ticker = get_company_ticker(symbol, stock_group=stock_group)
        stock = yf.Ticker(ticker)
        news = stock.news
        return [{'headline': item.get('title', ''), 'url': item.get('link', '')} 
               for item in news][:3]  # Return only top 3 news items
    except Exception as e:
        print(f"Error getting news for {symbol}: {e}")
        return []
