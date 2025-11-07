# Commented out agno imports as they're not available
# from agno.agent import Agent
# from agno.models.groq import Groq
# from agno.tools.duckduckgo import DuckDuckGoTools
# from agno.tools.yfinance import YFinanceTools
# from agno.tools.pandas import PandasTools
import os
import json
import time
import warnings
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import os
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import style
import mplfinance as mpf
from ta import add_all_ta_features
from ta.utils import dropna

# Suppress warnings
warnings.filterwarnings('ignore', message='Series.__setitem__ treating keys as positions is deprecated')

# Initialize API clients
finnhub_client = None
alpha_vantage_client = None

# Import local modules
# Note: tech_indicators.py, risk_manager.py, and backtesting_engine.py are not found
# from risk_manager import RiskManager
# from backtesting_engine import BacktestingEngine
from option_chain_analysis import OptionsAnalyzer

# Set style for plots
style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 8)

load_dotenv()

# Initialize API clients
try:
        # Initialize API clients if available
    try:
        if os.getenv('FINNHUB_API_KEY'):
            import finnhub
            finnhub_client = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))
    except ImportError:
        print("Warning: finnhub-python not installed. Some features may be limited.")
    
    try:
        if os.getenv('ALPHA_VANTAGE_API_KEY'):
            from alpha_vantage.timeseries import TimeSeries
            from alpha_vantage.techindicators import TechIndicators
            from alpha_vantage.fundamentaldata import FundamentalData
            ts = TimeSeries(key=os.getenv('ALPHA_VANTAGE_API_KEY'), output_format='pandas')
            ti = TechIndicators(key=os.getenv('ALPHA_VANTAGE_API_KEY'), output_format='pandas')
            fd = FundamentalData(key=os.getenv('ALPHA_VANTAGE_API_KEY'), output_format='pandas')
    except ImportError:
        print("Warning: alpha_vantage not installed. Some features may be limited.")
    
except Exception as e:
    print(f"Error initializing API clients: {e}")
    raise

def load_stock_config():
    """Load stock configurations from JSON file"""
    try:
        config_path = Path(__file__).parent / 'config' / 'stocks.json'
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Normalize configuration to ensure consistent structure
        normalized_config = {}
        
        for group_name, stocks in config_data.items():
            normalized_config[group_name] = {}
            
            for company, stock_info in stocks.items():
                if isinstance(stock_info, str):
                    # Convert simple string format to object format
                    normalized_config[group_name][company] = {
                        "symbol": stock_info,
                        "sector": "Unknown"
                    }
                elif isinstance(stock_info, dict):
                    normalized_config[group_name][company] = stock_info
                else:
                    print(f"Warning: Invalid format for {company} in {group_name}")
        
        return normalized_config
        
    except Exception as e:
        print(f"Error loading stock config: {e}")
        return {
            "indian_it": {
                "TCS": {"symbol": "TCS.NS", "sector": "IT Services"},
                "Infosys": {"symbol": "INFY.NS", "sector": "IT Services"}
            }
        }

# Load stock configurations
STOCK_CONFIG = load_stock_config()

def get_stock_data(stock_group="indian_it", period="6mo", interval="1d"):
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

def generate_simple_analysis(stock_data):
    """Generate technical analysis using ta indicators"""
    analysis = {}
    
    for company, data in stock_data.items():
        try:
            if data.empty or len(data) < 2:
                analysis[company] = {"error": "Insufficient data"}
                continue
            
            # Get the latest row safely
            latest_row = data.iloc[-1]
            prev_row = data.iloc[-2] if len(data) > 1 else latest_row
            
            # Extract scalar values safely - using lowercase column names
            current_price = float(latest_row['close']) if 'close' in latest_row and not pd.isna(latest_row['close']) else 0.0
            prev_price = float(prev_row['close']) if 'close' in prev_row and not pd.isna(prev_row['close']) else current_price
            
            # Calculate price change
            price_change_pct = 0.0
            if prev_price != 0:
                price_change_pct = ((current_price - prev_price) / prev_price) * 100
            
            # Trend analysis using simple comparisons
            trend = "Neutral"
            try:
                sma_fast = float(latest_row.get('trend_sma_fast', current_price)) if 'trend_sma_fast' in latest_row and not pd.isna(latest_row['trend_sma_fast']) else current_price
                sma_slow = float(latest_row.get('trend_sma_slow', current_price)) if 'trend_sma_slow' in latest_row and not pd.isna(latest_row['trend_sma_slow']) else current_price
                
                if sma_fast > sma_slow:
                    trend = "Bullish"
                elif sma_fast < sma_slow:
                    trend = "Bearish"
            except Exception as e:
                print(f"Trend analysis error for {company}: {e}")
                trend = "Unknown"
            
            # RSI analysis
            rsi_signal = "Neutral"
            rsi_value = None
            try:
                rsi_value = float(latest_row['momentum_rsi']) if 'momentum_rsi' in latest_row and not pd.isna(latest_row['momentum_rsi']) else None
                if rsi_value is not None:
                    if rsi_value > 70:
                        rsi_signal = "Overbought"
                    elif rsi_value < 30:
                        rsi_signal = "Oversold"
            except Exception as e:
                print(f"RSI analysis error for {company}: {e}")
                rsi_signal = "Unknown"
            
            analysis[company] = {
                'Price': round(current_price, 2),
                'Change (%)': round(price_change_pct, 2),
                'Trend': trend,
                'RSI': round(rsi_value, 2) if rsi_value else None,
                'RSI Signal': rsi_signal
            }
            
        except Exception as e:
            print(f"Error analyzing {company}: {e}")
            analysis[company] = {"error": str(e)}
    
    return analysis

def plot_stock_data(stock_data, company):
    """Generate simple stock plots"""
    try:
        df = stock_data[company]
        if df.empty or len(df) < 30:
            return None
            
        # Create simple plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Price chart
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=2)
        if 'trend_sma_fast' in df.columns:
            ax1.plot(df.index, df['trend_sma_fast'], label='Fast SMA', alpha=0.7)
        if 'trend_sma_slow' in df.columns:
            ax1.plot(df.index, df['trend_sma_slow'], label='Slow SMA', alpha=0.7)
        ax1.set_title(f'{company} Price Chart')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RSI chart if available
        if 'momentum_rsi' in df.columns:
            ax2.plot(df.index, df['momentum_rsi'], label='RSI', color='purple')
            ax2.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought')
            ax2.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold')
            ax2.set_title('RSI (14)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        plot_path = f'plots/{company.replace(" ", "_").replace("&", "and")}_analysis.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"Plot error for {company}: {e}")
        return None

def get_fundamental_data(symbol: str) -> Dict[str, Any]:
    """Get fundamental data with fallback to yfinance if API fails"""
    try:
        # Map company names to their correct Yahoo Finance symbols
        symbol_map = {
            'TCS': 'TCS.NS',
            'Infosys': 'INFY.NS',
            'Wipro': 'WIPRO.NS',
            'HCL': 'HCLTECH.NS',
            'HCL Tech': 'HCLTECH.NS',
            'Tech Mahindra': 'TECHM.NS',
            'Tech': 'TECHM.NS'
        }
        
        ticker = symbol_map.get(symbol, f"{symbol}.NS")
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

def get_analyst_recommendations(symbol: str) -> List[Dict]:
    """Get analyst recommendations with fallback to empty list"""
    if finnhub_client is None:
        return []
    try:
        return finnhub_client.recommendation_trends(f"{symbol}.NS")
    except Exception as e:
        print(f"Error getting recommendations for {symbol}: {e}")
        return []

def get_company_news(symbol: str, days: int = 7) -> List[Dict]:
    """Get recent company news with fallback to empty list"""
    try:
        symbol_map = {
            'TCS': 'TCS.NS',
            'Infosys': 'INFY.NS',
            'Wipro': 'WIPRO.NS',
            'HCL': 'HCLTECH.NS',
            'HCL Tech': 'HCLTECH.NS',
            'Tech Mahindra': 'TECHM.NS',
            'Tech': 'TECHM.NS'
        }
        ticker = symbol_map.get(symbol, f"{symbol}.NS")
        stock = yf.Ticker(ticker)
        news = stock.news
        return [{'headline': item.get('title', ''), 'url': item.get('link', '')} 
               for item in news][:3]  # Return only top 3 news items
    except Exception as e:
        print(f"Error getting news for {symbol}: {e}")
        return []

def get_company_ticker(company_name, stock_group="indian_it"):
    """Get the ticker symbol for a company from the config"""
    if stock_group in STOCK_CONFIG:
        # First try exact match
        if company_name in STOCK_CONFIG[stock_group]:
            if isinstance(STOCK_CONFIG[stock_group][company_name], dict):
                return STOCK_CONFIG[stock_group][company_name].get('symbol', f"{company_name.split()[0]}.NS")
            return STOCK_CONFIG[stock_group][company_name]
        
        # Try case-insensitive match
        company_lower = company_name.lower()
        for name, info in STOCK_CONFIG[stock_group].items():
            if name.lower() == company_lower:
                if isinstance(info, dict):
                    return info.get('symbol', f"{name.split()[0]}.NS")
                return info
    
    # Default to first part of company name with .NS suffix
    return f"{company_name.split()[0]}.NS"

def analyze_stocks(stock_group="indian_it"):
    """Main function to analyze stocks"""
    print(f"Fetching data for {stock_group}...")
    stock_data, metadata = get_stock_data(stock_group)
    
    if not stock_data:
        print("No data available")
        return
    
    print("Performing analysis...")
    analysis = generate_simple_analysis(stock_data)
    
    # Generate plots for top stocks
    successful_stocks = [k for k, v in analysis.items() if 'error' not in v]
    top_companies = successful_stocks[:3]  # Just take first 3
    
    plot_paths = {}
    for company in top_companies:
        plot_path = plot_stock_data(stock_data, company)
        if plot_path:
            plot_paths[company] = plot_path
    
    # Initialize sectors dictionary
    sectors = {}
    for company, data in stock_data.items():
        sector = metadata.get(company, {}).get('sector', 'Unknown')
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(company)

    # Print analysis results in a tabular format
    print("\n" + "="*120)
    print(f"{'STOCK ANALYSIS REPORT':^120}")
    print("="*120)
    print(f"{'Company':<20} | {'Price':>10} | {'Change %':>10} | {'Trend':<15} | {'RSI':>8} | {'Signal':<10} | {'PE':>8} | {'Target':>10} | {'News'}")
    print("-" * 120)
    
    # Initialize strategy lists
    long_term_stocks = []
    intraday_stocks = []
    swing_stocks = []
    options_stocks = []
    
    for company, data in stock_data.items():
        try:
            # Get the latest data point (most recent)
            if data.empty:
                print(f"No data available for {company}")
                continue
                
            latest = data.iloc[-1]
            
            # Get price and change - using lowercase column names
            close_price = latest.get('close')
            if pd.isna(close_price) or close_price is None:
                print(f"No price data for {company}")
                continue
                
            # Format price with 2 decimal places
            price = f"{float(close_price):.2f}"
            
            # Calculate daily change if not available
            if 'daily_change' in latest:
                daily_change = latest['daily_change']
            elif len(data) > 1:
                # Calculate daily change from previous close if not available
                prev_close = data.iloc[-2]['close'] if len(data) > 1 else close_price
                daily_change = ((close_price - prev_close) / prev_close) * 100 if prev_close else 0
            else:
                daily_change = 0
                
            if pd.isna(daily_change):
                daily_change = 0
                
            change_str = f"{daily_change:+.2f}%"
            
            # Get technical indicators with proper defaults
            sma_fast = latest.get('trend_sma_fast', 0) if 'trend_sma_fast' in latest else 0
            sma_slow = latest.get('trend_sma_slow', 0) if 'trend_sma_slow' in latest else 0
            trend = 'Bullish' if sma_fast > sma_slow else 'Bearish'
            
            rsi = latest.get('momentum_rsi')
            rsi = float(rsi) if rsi is not None and not pd.isna(rsi) else None
            rsi_str = f"{rsi:.2f}" if rsi is not None else 'N/A'
            rsi_signal = (
                'Overbought' if rsi and rsi > 70 
                else 'Oversold' if rsi and rsi < 30 
                else 'Neutral'
            )


            # Get additional data
            try:
                ticker = get_company_ticker(company, stock_group)
                fundamental = get_fundamental_data(company)  # Use full company name
                
                # Safely get P/E ratio
                pe_ratio = None
                if fundamental and 'pe_ratio' in fundamental and fundamental['pe_ratio'] is not None:
                    try:
                        pe_ratio = float(fundamental['pe_ratio'])
                        pe = f"{pe_ratio:.2f}"
                    except (ValueError, TypeError):
                        pe = 'N/A'
                else:
                    pe = 'N/A'
                
                # Safely get target price
                target_price = None
                if fundamental and 'analyst_target' in fundamental and fundamental['analyst_target'] is not None:
                    try:
                        target_price = float(fundamental['analyst_target'])
                        target = f"{target_price:.2f}"
                    except (ValueError, TypeError):
                        target = 'N/A'
                else:
                    target = 'N/A'
                
                # Get latest news
                news = get_company_news(company)  # Use full company name
                news_snippet = news[0]['headline'][:30] + '...' if news and len(news) > 0 else 'No recent news'
            
            except Exception as e:
                print(f"Error getting fundamental data for {company}: {e}")
                pe = 'N/A'
                target = 'N/A'
                fundamental = {}
                news = []
                news_snippet = 'Error fetching news'
            
            # Print the row with proper alignment
            try:
                print(f"{company[:18]:<20} | {price:>10} | {change_str:>10} | {trend:<15} | {rsi_str:>8} | {rsi_signal:<10} | {pe:>8} | {target:>10} | {news_snippet[:30]}")
            except Exception as e:
                print(f"Error printing row for {company}: {e}")
                continue
            
            # Store additional data for analysis
            data['fundamental'] = fundamental if fundamental else {}
            data['news'] = news if news else []
            
        except Exception as e:
            print(f"Error processing {company}: {e}")
            continue 

        # Get recommendations if Finnhub client is available
        if finnhub_client:
            data['recommendations'] = get_analyst_recommendations(company)
        else:
            data['recommendations'] = []
        
        # Strategy classification
        stock_info = {
            'name': company,
            'price': float(price.replace(',', '')),
            'change': daily_change,
            'trend': trend,
            'rsi': rsi if rsi != 'N/A' else None,
            'rsi_signal': rsi_signal
        }
        
        # Long-term criteria: Strong uptrend and not overbought
        if trend == 'Bullish' and (rsi is None or (rsi > 40 and rsi < 70)):
            long_term_stocks.append(stock_info)
            
        # Intraday criteria: High volatility and good volume
        if abs(change) > 1.0 and 'volume' in data and data['volume'] > 100000:  # Example threshold
            intraday_stocks.append(stock_info)
            
        # Swing trading criteria: Starting a new trend or pullback in a trend
        if (trend == 'Bullish' and rsi_signal == 'Oversold') or \
           (trend == 'Bearish' and rsi_signal == 'Overbought'):
            swing_stocks.append(stock_info)
            
        # Options criteria: High volatility and approaching key levels
        if rsi is not None and (rsi > 70 or rsi < 30):
            options_stocks.append(stock_info)
    
    print("-" * 100)
    
    # Print strategy recommendations
    def print_strategy(name, stocks, sort_key=None, reverse=False):
        if not stocks:
            print(f"\nNo {name} opportunities found.")
            return
            
        if sort_key:
            stocks = sorted(stocks, key=lambda x: x.get(sort_key, 0), reverse=reverse)
            
        print(f"\n{name.upper()} OPPORTUNITIES (Top {min(3, len(stocks))}):")
        print("-" * 100)
        for stock in stocks[:3]:  # Show top 3 for each strategy
            print(f"- {stock['name']:<15} | Price: {stock['price']:<10} | Change: {stock['change']:+.2f}% | "
                  f"Trend: {stock['trend']:<8} | RSI: {stock['rsi']:.2f} ({stock['rsi_signal']})")
    
    # Print recommendations for each strategy
    print_strategy("Long-term Investments", long_term_stocks, 'change', True)
    print_strategy("Intraday Trading", intraday_stocks, 'change', True)
    print_strategy("Swing Trading", swing_stocks, 'rsi')
    print_strategy("Options Trading", options_stocks, 'rsi', True)
    
    # Prepare analysis prompt for AI (kept for potential future use)
    prompt = f"""
    Analyze the {stock_group.replace('_', ' ').title()} sector based on this technical data:

    Sector Breakdown:
    """
    
    for sector, companies in sectors.items():
        prompt += f"- {sector}: {', '.join(companies)}\n"
    
    prompt += """
    
    Provide a technical analysis covering:
    1. Current price levels and changes
    2. Trend direction based on moving averages
    3. RSI levels and overbought/oversold conditions
    4. Overall market sentiment
    5. Top performing stocks
    6. Trading recommendations
    """
def analyze_options(stock_symbol: str, expiry_date: str = None) -> Dict:
    """Analyze options chain for a given stock"""
    try:
        stock = yf.Ticker(f"{stock_symbol}.NS")
        
        # Get options expiry dates if not provided
        if not expiry_date:
            expiries = stock.options
            if not expiries:
                return {"error": "No options data available for this stock"}
            expiry_date = expiries[0]  # Use nearest expiry by default
        
        # Get options chain
        opt = stock.option_chain(expiry_date)
        
        # Analyze calls and puts
        def analyze_side(df, option_type: str) -> List[Dict]:
            if df.empty:
                return []
                
            # Calculate key metrics
            df['intrinsic'] = np.where(
                option_type == 'call',
                np.maximum(0, stock.info['regularMarketPrice'] - df['strike']),
                np.maximum(0, df['strike'] - stock.info['regularMarketPrice'])
            )
            df['extrinsic'] = df['lastPrice'] - df['intrinsic']
            df['extrinsic_ratio'] = df['extrinsic'] / df['lastPrice'] * 100
            
            # Find interesting opportunities
            # 1. High open interest and volume
            # 2. High implied volatility relative to historical
            # 3. Good risk/reward ratios
            
            return df.sort_values('volume', ascending=False).head(5).to_dict('records')
        
        return {
            'stock': stock_symbol,
            'current_price': stock.info.get('regularMarketPrice', 0),
            'expiry': expiry_date,
            'calls': analyze_side(opt.calls, 'call'),
            'puts': analyze_side(opt.puts, 'put'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        return {"error": f"Error analyzing options: {str(e)}"}

def print_options_analysis(analysis: Dict):
    """Print options analysis in a readable format"""
    if 'error' in analysis:
        print(f"\nError: {analysis['error']}")
        return
        
    print(f"\n{'='*80}")
    print(f"OPTIONS ANALYSIS: {analysis['stock']} (Current: {analysis['current_price']:.2f}) - Expiry: {analysis['expiry']}")
    print(f"{'='*80}")
    
    # Print top calls
    print("\nTOP CALL OPPORTUNITIES:")
    print("-" * 80)
    print(f"{'Strike':<8} | {'Last':<8} | {'Bid':<8} | {'Ask':<8} | {'OI':<8} | {'Volume':<8} | {'IV':<8} | {'Intrinsic':<10} | {'Extrinsic':<10}")
    print("-" * 80)
    for call in analysis.get('calls', [])[:5]:  # Show top 5
        print(f"{call['strike']:<8.2f} | {call['lastPrice']:<8.2f} | {call['bid']:<8.2f} | "
              f"{call['ask']:<8.2f} | {call['openInterest']:<8} | {call['volume']:<8} | "
              f"{call['impliedVolatility']*100 if 'impliedVolatility' in call else 'N/A':<8.2f} | "
              f"{call.get('intrinsic', 0):<10.2f} | {call.get('extrinsic', 0):<10.2f}")
    
    # Print top puts
    print("\nTOP PUT OPPORTUNITIES:")
    print("-" * 80)
    print(f"{'Strike':<8} | {'Last':<8} | {'Bid':<8} | {'Ask':<8} | {'OI':<8} | {'Volume':<8} | {'IV':<8} | {'Intrinsic':<10} | {'Extrinsic':<10}")
    print("-" * 80)
    for put in analysis.get('puts', [])[:5]:  # Show top 5
        print(f"{put['strike']:<8.2f} | {put['lastPrice']:<8.2f} | {put['bid']:<8.2f} | "
              f"{put['ask']:<8.2f} | {put['openInterest']:<8} | {put['volume']:<8} | "
              f"{put['impliedVolatility']*100 if 'impliedVolatility' in put else 'N/A':<8.2f} | "
              f"{put.get('intrinsic', 0):<10.2f} | {put.get('extrinsic', 0):<10.2f}")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock and Options Analysis Tool')
    parser.add_argument('--stocks', action='store_true', help='Run stock analysis (default)')
    parser.add_argument('--options', type=str, help='Run options analysis for a specific stock (e.g., TCS, INFY)')
    parser.add_argument('--expiry', type=str, help='Options expiry date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    if args.options:
        # Run options analysis
        print(f"\nAnalyzing options for {args.options}...")
        analysis = analyze_options(args.options, args.expiry)
        print_options_analysis(analysis)
    else:
        # Default to stock analysis
        analyze_stocks("indian_it")
