from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.pandas import PandasTools
import optuna  # for hyperparameter optimization
from backtesting import Backtest  # for backtesting framework
import quantstats as qs  # for performance analytics
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
from datetime import datetime, timedelta
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import style

# Import local modules
from tech_indicators import IntradayAnalyzer
from risk_manager import RiskManager
from backtesting_engine import BacktestingEngine
from option_chain_analysis import OptionsAnalyzer

# Set style for plots
style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 8)

load_dotenv()

# Initialize LLM with Groq
llm = Groq(id="llama-3.3-70b-versatile", temperature=0.7)

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
            
            # Calculate basic indicators safely
            try:
                # Simple moving averages
                if len(df) >= 20:
                    df['SMA_20'] = df['Close'].rolling(window=20).mean()
                if len(df) >= 50:
                    df['SMA_50'] = df['Close'].rolling(window=50).mean()
                if len(df) >= 200:
                    df['SMA_200'] = df['Close'].rolling(window=200).mean()
                
                # RSI
                if len(df) >= 15:
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                
                # Basic price change
                df['Daily_Change'] = df['Close'].pct_change() * 100
                
            except Exception as ind_error:
                print(f"Indicator error for {company}: {ind_error}")
                # Continue with basic data
            
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
    """Generate technical analysis without complex pandas_ta indicators"""
    analysis = {}
    
    for company, data in stock_data.items():
        try:
            if data.empty or len(data) < 2:
                analysis[company] = {"error": "Insufficient data"}
                continue
            
            # Get the latest row safely
            latest_row = data.iloc[-1]
            prev_row = data.iloc[-2] if len(data) > 1 else latest_row
            
            # Extract scalar values safely
            current_price = float(latest_row['Close']) if not pd.isna(latest_row['Close']) else 0.0
            prev_price = float(prev_row['Close']) if not pd.isna(prev_row['Close']) else current_price
            
            # Calculate price change
            price_change_pct = 0.0
            if prev_price != 0:
                price_change_pct = ((current_price - prev_price) / prev_price) * 100
            
            # Trend analysis using simple comparisons
            trend = "Neutral"
            try:
                sma_20 = float(latest_row.get('SMA_20', current_price)) if not pd.isna(latest_row.get('SMA_20')) else current_price
                sma_50 = float(latest_row.get('SMA_50', current_price)) if not pd.isna(latest_row.get('SMA_50')) else current_price
                sma_200 = float(latest_row.get('SMA_200', current_price)) if not pd.isna(latest_row.get('SMA_200')) else current_price
                
                if sma_20 > sma_50 and sma_50 > sma_200:
                    trend = "Bullish"
                elif sma_20 < sma_50 and sma_50 < sma_200:
                    trend = "Bearish"
            except:
                trend = "Unknown"
            
            # RSI analysis
            rsi_signal = "Neutral"
            rsi_value = 50.0
            try:
                rsi_value = float(latest_row.get('RSI', 50)) if not pd.isna(latest_row.get('RSI')) else 50.0
                if rsi_value > 70:
                    rsi_signal = "Overbought"
                elif rsi_value < 30:
                    rsi_signal = "Oversold"
            except:
                rsi_signal = "Unknown"
            
            analysis[company] = {
                'Price': round(current_price, 2),
                'Change (%)': round(price_change_pct, 2),
                'Trend': trend,
                'RSI': round(rsi_value, 2),
                'RSI Signal': rsi_signal,
                '20-Day MA': round(float(latest_row.get('SMA_20', current_price)), 2) if not pd.isna(latest_row.get('SMA_20')) else round(current_price, 2),
                '50-Day MA': round(float(latest_row.get('SMA_50', current_price)), 2) if not pd.isna(latest_row.get('SMA_50')) else round(current_price, 2),
                '200-Day MA': round(float(latest_row.get('SMA_200', current_price)), 2) if not pd.isna(latest_row.get('SMA_200')) else round(current_price, 2),
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
        ax1.plot(df.index, df['Close'], label='Close Price', linewidth=2)
        if 'SMA_20' in df.columns:
            ax1.plot(df.index, df['SMA_20'], label='20-Day MA', alpha=0.7)
        if 'SMA_50' in df.columns:
            ax1.plot(df.index, df['SMA_50'], label='50-Day MA', alpha=0.7)
        ax1.set_title(f'{company} Price Chart')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RSI chart if available
        if 'RSI' in df.columns:
            ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
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

def analyze_stocks(stock_group="indian_it"):
    """
    Analyze stocks with simple technical indicators
    """
    print(f"Fetching data for {stock_group}...")
    stock_data, metadata = get_stock_data(stock_group=stock_group)
    
    if not stock_data:
        return "No data available", {}, {}, {}
    
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
    
    # Prepare analysis prompt
    sectors = {}
    for company, data in metadata.items():
        sector = data.get('sector', 'Other')
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(company)
    
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