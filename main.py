import os
import sys
import json
import time
import warnings
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

# Data processing
import pandas as pd
import numpy as np
import yfinance as yf

# Visualization
import matplotlib.pyplot as plt
from matplotlib import style

# Local modules
from risk_manager import RiskManager
from backtesting_engine import BacktestingEngine

# Configuration
from dotenv import load_dotenv

# Import from our new modules
from config.stock_config import load_stock_config, get_company_ticker
from data.fetcher import get_stock_data, get_fundamental_data, get_company_news
from analysis.technical import generate_simple_analysis
from analysis.fundamental import get_analyst_recommendations
from visualization.plotter import plot_stock_data

# Suppress warnings
warnings.filterwarnings('ignore', message='Series.__setitem__ treating keys as positions is deprecated')

# Initialize API clients
finnhub_client = None
alpha_vantage_client = None

# Import local modules
# Note: tech_indicators.py, risk_manager.py, and backtesting_engine.py are not found
from risk_manager import RiskManager
from backtesting_engine import BacktestingEngine
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

def get_company_ticker(company_name: str, stock_group: str = "indian_it") -> str:
    """
    Get the ticker symbol for a company from the config
    
    Args:
        company_name: Name of the company
        stock_group: Stock group to look up (default: "indian_it")
        
    Returns:
        str: Ticker symbol for the company
    """
    if stock_group not in STOCK_CONFIG:
        return f"{company_name.split()[0]}.NS"  # Default format if group not found
        
    # Try exact match first
    if company_name in STOCK_CONFIG[stock_group]:
        info = STOCK_CONFIG[stock_group][company_name]
        if isinstance(info, dict):
            return info.get('symbol', f"{company_name.split()[0]}.NS")
        return info
    
    # Try case-insensitive match
    company_lower = company_name.lower()
    for name, info in STOCK_CONFIG[stock_group].items():
        if name.lower() == company_lower:
            if isinstance(info, dict):
                return info.get('symbol', f"{name.split()[0]}.NS")
            return info
            
    # Default return if no match found
    return f"{company_name.split()[0]}.NS"

def analyze_stocks(stock_group: str = "indian_it") -> tuple[dict, dict, dict, dict]:
    """
    Analyze stocks with integrated risk management and technical analysis.
    
    Args:
        stock_group: Name of the stock group to analyze (default: "indian_it")
        
    Returns:
        tuple: (analysis_results, plot_paths, metadata, risk_assessment)
    """
    # Initialize components
    risk_mgr = RiskManager(account_size=10000, risk_per_trade=0.01)
    backtester = BacktestingEngine()
    
    # Fetch and prepare data
    stock_data, metadata = get_stock_data(stock_group)
    
    if not stock_data:
        print("No data available")
        return None, None, None, None
    
    print(f"Analyzing {len(stock_data)} stocks in {stock_group}...")
    
    # Initialize results
    analysis_results = {}
    plot_paths = {}
    risk_assessment = {}
    
    # Process each stock
    for company, data in stock_data.items():
        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                print(f"Skipping {company}: Invalid or empty data")
                continue
                
            # 1. Technical Analysis
            analysis = generate_simple_analysis({company: data})
            analysis_results[company] = analysis.get(company, {})
            
            # 2. Risk Assessment
            if not data.empty:
                latest = data.iloc[-1]
                
                # Calculate position size based on ATR for stop loss
                atr = data.get('volatility_atr', pd.Series([0] * len(data))).iloc[-1]
                stop_loss = latest['close'] - (2 * atr) if atr > 0 else latest['close'] * 0.95
                position_size = risk_mgr.calculate_position_size(
                    entry_price=latest['close'],
                    stop_loss=stop_loss
                )
                
                risk_assessment[company] = {
                    'position_size': position_size,
                    'stop_loss': stop_loss,
                    'risk_reward': risk_mgr.calculate_risk_reward(
                        entry_price=latest['close'],
                        stop_loss=stop_loss,
                        take_profit=latest['close'] * 1.02  # 2% target
                    )
                }
                
            # 3. Generate plots for top stocks
            if len(plot_paths) < 3:  # Only plot top 3 stocks
                try:
                    # Use the ticker symbol for plotting instead of company name
                    ticker = get_company_ticker(company, stock_group)
                    plot_path = plot_stock_data({ticker: data}, ticker)
                    if plot_path:
                        plot_paths[company] = plot_path
                except Exception as e:
                    print(f"Error generating plot for {company}: {str(e)}")
                    plot_paths[company] = None
                    
            # 4. Run backtesting on the strategy
            try:
                backtest_results = backtester.backtest_strategy(
                    data,
                    strategy_name='moving_average_crossover',
                    fast_period=20,
                    slow_period=50
                )
                if 'backtest' not in analysis_results[company]:
                    analysis_results[company]['backtest'] = {}
                analysis_results[company]['backtest'].update({
                    'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
                    'max_drawdown': backtest_results.get('max_drawdown', 0),
                    'win_rate': backtest_results.get('win_rate', 0)
                })
            except Exception as e:
                print(f"Backtesting failed for {company}: {str(e)}")
                
        except Exception as e:
            print(f"Error processing {company}: {str(e)}")
            continue
    
    # Group stocks by sector for reporting
    sectors = {}
    for company, data in stock_data.items():
        sector = metadata.get(company, {}).get('sector', 'Unknown')
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(company)
    
    # Print analysis summary
    print("\n" + "="*120)
    print(f"{'STOCK ANALYSIS REPORT':^120}")
    print("="*120)
    print(f"{'Company':<20} | {'Price':>10} | {'Change %':>10} | {'Trend':<15} | {'RSI':>8} | {'Signal':<10} | {'PE':>8} | {'Target':>10} | {'News'}")
    print("-" * 120)
    
    for company, data in analysis_results.items():
        if not data or 'error' in data:
            continue
            
        # Get latest price data
        latest = stock_data[company].iloc[-1] if company in stock_data else {}
        prev_close = stock_data[company].iloc[-2] if company in stock_data and len(stock_data[company]) > 1 else latest
        
        # Calculate price change
        if 'close' in latest and 'close' in prev_close:
            price_change = ((latest['close'] - prev_close['close']) / prev_close['close']) * 100
            change_str = f"{price_change:+.2f}%"
            change_color = "green" if price_change >= 0 else "red"
        else:
            change_str = "N/A"
            change_color = "white"
        
        # Get analysis metrics
        rsi = data.get('rsi', 50)
        trend = data.get('trend', 'Neutral')
        pe = data.get('pe_ratio', 'N/A')
        target = data.get('target_price', 'N/A')
        
        # Get news snippet
        news_snippet = data.get('news', [{}])[0].get('headline', 'No recent news')[:30] + '...' \
                      if data.get('news') else 'No news available'
        
        # Print row
        print(f"{company:<20} | {latest.get('close', 'N/A'):>10.2f} | {change_str:>10} | {trend:<15} | "
              f"{rsi:>8.2f} | {data.get('signal', 'Neutral'):<10} | {pe:>8} | {target:>10} | {news_snippet}")
    
    return analysis_results, plot_paths, metadata, risk_assessment

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Stock and Options Analysis Tool')
    parser.add_argument('--group', type=str, help='Run stock analysis for a specific group (e.g., indian_it)')
    parser.add_argument('--options', type=str, help='Run options analysis for a specific stock (e.g., TCS, INFY)')
    parser.add_argument('--expiry', type=str, help='Options expiry date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    try:
        if args.group:
            results = analyze_stocks(args.group)
            if results is not None:
                analysis_results, plot_paths, metadata, risk_assessment = results
                print("\nAnalysis completed successfully!")
                print(f"Generated {len(plot_paths)} plots")
        elif args.options:
            analysis = analyze_options(args.options, args.expiry)
            if analysis:
                print_options_analysis(analysis)
        else:
            results = analyze_stocks()
            if results is not None:
                analysis_results, plot_paths, metadata, risk_assessment = results
                print("\nAnalysis completed successfully!")
                print(f"Generated {len(plot_paths)} plots")
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
