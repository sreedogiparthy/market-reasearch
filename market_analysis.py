import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
from typing import Dict, Any, Optional
from ai_analyzer import AIAnalyzer

class MarketAnalyzer:
    def __init__(self, enable_ai: bool = True):
        self.load_config()
        self.setup_plotting()
        self.ai_analyzer = AIAnalyzer() if enable_ai else None
    
    def load_config(self):
        """Load configuration from file"""
        try:
            config_path = Path(__file__).parent / 'config' / 'stocks.json'
            with open(config_path, 'r') as f:
                self.stock_config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            # Default configuration if file not found
            self.stock_config = {
                "nifty_50": {
                    "RELIANCE": "RELIANCE.NS",
                    "HDFCBANK": "HDFCBANK.NS",
                    "INFY": "INFY.NS",
                    "TCS": "TCS.NS",
                    "HINDUNILVR": "HINDUNILVR.NS"
                }
            }
    
    def setup_plotting(self):
        """Configure plotting settings"""
        # Use a built-in style that's available by default
        plt.style.use('ggplot')
        plt.rcParams['figure.figsize'] = (15, 8)
        plt.rcParams['axes.grid'] = True
    
    def _get_max_intraday_period(self, interval: str) -> str:
        """
        Get the maximum allowed period for intraday data based on interval
        
        Args:
            interval: Data interval (1m, 5m, 15m, 30m, 60m, 90m, 1h)
            
        Returns:
            str: Maximum allowed period for the given interval
        """
        if interval == '1m':
            return '7d'  # 1-minute data only available for last 7 days
        elif interval in ['2m', '5m', '15m', '30m', '60m', '90m']:
            return '60d'  # Other intraday intervals for last 60 days
        elif interval == '1h':
            return '730d'  # 1-hour data for up to 2 years
        return '10y'  # Default for daily+ intervals

    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock data using yfinance with proper handling of intraday data limitations
        
        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with stock data and indicators
        """
        try:
            # For intraday intervals, adjust period if it's too long
            if any(x in interval for x in ['m', 'h']):
                max_period = self._get_max_intraday_period(interval)
                if period != max_period:
                    print(f"ℹ️  Adjusting period to {max_period} for {interval} interval (Yahoo Finance limitation)")
                    period = max_period
            
            # Fetch data from Yahoo Finance
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval, auto_adjust=True)
            
            if df.empty:
                print(f"❌ No data available for {symbol}")
                return pd.DataFrame()
                
            # Calculate basic indicators
            if not df.empty:
                # Moving Averages (only if we have enough data points)
                min_periods = min(20, len(df) // 2)  # Use at least 20 periods or half the data, whichever is smaller
                df['SMA_20'] = df['Close'].rolling(window=20, min_periods=min_periods).mean()
                df['SMA_50'] = df['Close'].rolling(window=50, min_periods=min_periods).mean()
                
                # RSI (only if we have enough data points)
                if len(df) > 14:
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                
                # Daily Returns
                df['Daily_Return'] = df['Close'].pct_change() * 100
                
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {str(e)}")
            if "data not available" in str(e):
                print(f"   This is likely because {interval} interval data is not available for the requested period.")
                print(f"   Try a shorter period or different interval.")
            return pd.DataFrame()
    
    def _get_technical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract technical indicators for AI analysis"""
        latest = df.iloc[-1]
        prev_close = df.iloc[-2]['Close'] if len(df) > 1 else latest['Close']
        price_change_pct = ((latest['Close'] - prev_close) / prev_close) * 100
        
        # Determine trend
        trend = "Neutral"
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            trend = "Bullish" if latest['SMA_20'] > latest['SMA_50'] else "Bearish"
        
        # Determine momentum
        momentum = "Neutral"
        if 'RSI' in df.columns:
            if latest['RSI'] > 70:
                momentum = "Overbought"
            elif latest['RSI'] < 30:
                momentum = "Oversold"
        
        return {
            'current_price': latest['Close'],
            'price_change_pct': price_change_pct,
            'sma_20': latest.get('SMA_20', None),
            'sma_50': latest.get('SMA_50', None),
            'rsi': latest.get('RSI', None),
            'trend': trend,
            'momentum': momentum,
            'volume': latest.get('Volume', None)
        }
    
    def analyze_stock(self, symbol: str, period: str = "1y", interval: str = "1d", 
                     generate_trade_idea: bool = False) -> Dict[str, Any]:
        """
        Analyze a single stock with optional AI-powered insights
        
        Args:
            symbol: Stock symbol
            period: Data period to analyze
            interval: Data interval
            generate_trade_idea: Whether to generate a detailed trade idea
            
        Returns:
            Dictionary containing analysis results
        """
        print(f"\n{'='*50}")
        print(f"ANALYZING {symbol}")
        print(f"{'='*50}")
        
        # Get stock data
        df = self.get_stock_data(symbol, period, interval)
        if df is None or df.empty:
            print(f"❌ No data available for {symbol}")
            return {}
            
        # Calculate technical indicators if not already present
        if 'SMA_20' not in df.columns or 'SMA_50' not in df.columns or 'RSI' not in df.columns:
            df = self._calculate_indicators(df)
        
        # Get technical summary
        tech_summary = self._get_technical_summary(df)
        
        # Prepare result dictionary
        result = {
            'symbol': symbol,
            'data': df,
            'current_price': tech_summary['current_price'],
            'change_pct': tech_summary['price_change_pct'],
            'sma_20': tech_summary['sma_20'],
            'sma_50': tech_summary['sma_50'],
            'rsi': tech_summary['rsi'],
            'trend': tech_summary['trend'],
            'momentum': tech_summary['momentum'],
            'volume': tech_summary['volume']
        }
        
        # Print basic analysis
        print(f"Current Price: {tech_summary['current_price']:.2f}")
        print(f"Change: {tech_summary['price_change_pct']:.2f}%")
        print(f"20-Day MA: {tech_summary['sma_20'] or 'N/A'}")
        print(f"50-Day MA: {tech_summary['sma_50'] or 'N/A'}")
        print(f"RSI: {tech_summary['rsi'] or 'N/A'}")
        print(f"Trend: {tech_summary['trend']}")
        print(f"Momentum: {tech_summary['momentum']}")
        
        # Generate and plot technical indicators
        self.plot_stock(df, symbol)
        
        # AI Analysis if enabled
        ai_analysis = {}
        if self.ai_analyzer:
            print("\n🤖 AI Analysis:")
            try:
                # Add delay to avoid rate limiting
                if hasattr(self, 'last_analysis_time'):
                    time_since_last = time.time() - self.last_analysis_time
                    if time_since_last < 1:  # 1 second between API calls
                        time.sleep(1 - time_since_last)
                
                self.last_analysis_time = time.time()
                
                # Get AI analysis
                ai_result = self.ai_analyzer.analyze_market_conditions(symbol, tech_summary)
                ai_analysis = {
                    'analysis': ai_result.get('analysis', ''),
                    'recommendation': ai_result.get('recommendation', 'N/A'),
                    'confidence': ai_result.get('confidence', 0)
                }
                
                # Print the analysis
                if 'analysis' in ai_result:
                    print("\n" + ai_result['analysis'])
                
                if 'recommendation' in ai_result:
                    print(f"\nRecommendation: {ai_result['recommendation']} (Confidence: {ai_result.get('confidence', 0) * 100:.0f}%)")
                
                # Generate trade idea if requested
                if generate_trade_idea:
                    print("\n🔍 Generating detailed trade idea...")
                    time.sleep(1)  # Add delay
                    trade_idea = self.ai_analyzer.generate_trade_idea(symbol, tech_summary, ai_result)
                    if trade_idea and 'trade_idea' in trade_idea:
                        print("\n📈 Trade Idea:")
                        print(trade_idea['trade_idea'])
                        ai_analysis['trade_idea'] = trade_idea['trade_idea']
                    
            except Exception as e:
                print(f"⚠️ AI Analysis failed: {str(e)}")
        
        # Add AI analysis to result
        result['ai_analysis'] = ai_analysis
        
        # Add recommendation and confidence to top level for easy access in summary
        result['recommendation'] = ai_analysis.get('recommendation', 'N/A')
        result['confidence'] = ai_analysis.get('confidence', 0)
        
        return result
    
    def plot_stock(self, df, symbol):
        """Plot stock data with indicators"""
        if df.empty:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
        
        # Price and Moving Averages
        ax1.plot(df.index, df['Close'], label='Close', alpha=0.7)
        
        if 'SMA_20' in df.columns:
            ax1.plot(df.index, df['SMA_20'], label='20-Day MA', alpha=0.7)
        if 'SMA_50' in df.columns:
            ax1.plot(df.index, df['SMA_50'], label='50-Day MA', alpha=0.7)
            
        ax1.set_title(f'{symbol} Price and Moving Averages')
        ax1.legend()
        
        # RSI
        if 'RSI' in df.columns:
            ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
            ax2.axhline(70, color='red', linestyle='--', alpha=0.3)
            ax2.axhline(30, color='green', linestyle='--', alpha=0.3)
            ax2.set_ylim(0, 100)
            ax2.set_title('RSI')
        
        plt.tight_layout()
        plt.show()
    
    def _generate_summary_table(self, interval: str = '1d'):
        """
        Generate a summary table of all analyzed stocks
        
        Args:
            interval: Data interval (e.g., '1d', '1h', '15m') to determine analysis type
        """
        if not hasattr(self, 'analysis_results') or not self.analysis_results:
            return
        
        # Determine analysis type based on interval
        interval = interval.lower()
        if 'm' in interval:  # Minutes (e.g., 15m, 1h)
            analysis_type = 'INTRADAY'
            time_frame = f"{interval.upper()} Timeframe"
        elif 'h' in interval:  # Hours (e.g., 4h)
            analysis_type = 'SWING'
            time_frame = f"{interval.upper()} Timeframe"
        else:  # Default to daily/long-term
            analysis_type = 'LONG-TERM'
            time_frame = 'Daily'
            
        print("\n" + "="*120)
        print(f"📊 {analysis_type} TRADING REPORT - {time_frame}".center(120))
        print("="*120)
        
        # Define table headers and format
        headers = ["Symbol", "Price", "Change %", "20D MA", "50D MA", "RSI", "Trend", "Momentum", "Rec.", "Conf."]
        row_format = "{:<8} {:<10} {:<10} {:<10} {:<10} {:<8} {:<12} {:<12} {:<8} {:<8}"
        
        # Print table header
        print(row_format.format(*headers))
        print("-" * 120)
        
        # Print each stock's data
        for result in self.analysis_results:
            symbol = result.get('symbol', 'N/A')
            current_price = f"{result.get('current_price', 0):.2f}"
            change_pct = f"{result.get('change_pct', 0):.2f}%"
            sma_20 = f"{result.get('sma_20', 0):.2f}" if result.get('sma_20') else 'N/A'
            sma_50 = f"{result.get('sma_50', 0):.2f}" if result.get('sma_50') else 'N/A'
            rsi = f"{result.get('rsi', 0):.2f}" if result.get('rsi') else 'N/A'
            trend = result.get('trend', 'N/A')
            momentum = result.get('momentum', 'N/A')
            
            # Get AI recommendation if available
            ai_analysis = result.get('ai_analysis', {})
            recommendation = ai_analysis.get('recommendation', 'N/A')
            confidence = f"{ai_analysis.get('confidence', 0) * 100:.0f}%" if 'confidence' in ai_analysis else 'N/A'
            
            # Color code the recommendation
            if 'Buy' in str(recommendation):
                recommendation = f"\033[92m{recommendation}\033[0m"  # Green for Buy
            elif 'Sell' in str(recommendation):
                recommendation = f"\033[91m{recommendation}\033[0m"  # Red for Sell
                
            print(row_format.format(
                symbol, current_price, change_pct, sma_20, sma_50, rsi, 
                trend, momentum, recommendation, confidence
            ))
        
        print("\n💡 Key:")
        print("- Trend: Bullish/Bearish/Neutral")
        print("- Momentum: Overbought/Oversold/Neutral")
        print("- Rec.: AI Recommendation")
        print("- Conf.: AI Confidence Level")
        print("=" * 120)
    
    def analyze_group(self, group_name: str, period: str = "1y", interval: str = "1d", 
                     generate_trade_ideas: bool = False) -> dict:
        """
        Analyze a group of stocks with optional AI-powered insights
        
        Args:
            group_name: Name of the stock group to analyze
            period: Data period to analyze
            interval: Data interval
            generate_trade_ideas: Whether to generate detailed trade ideas
            
        Returns:
            Dictionary containing analysis results for all stocks in the group
        """
        if group_name not in self.stock_config:
            print(f"Group '{group_name}' not found. Available groups: {', '.join(self.stock_config.keys())}")
            return {}
            
        print(f"\n{'='*50}")
        print(f"ANALYZING {group_name.upper()} STOCKS")
        print(f"{'='*50}")
        
        # Initialize list to store analysis results
        self.analysis_results = []
        results = {}
        
        for name, symbol in self.stock_config[group_name].items():
            if isinstance(symbol, dict):  # Handle nested config
                symbol = symbol.get('symbol', name)
                
            result = self.analyze_stock(
                symbol, 
                period=period, 
                interval=interval,
                generate_trade_idea=generate_trade_ideas
            )
            
            if result and 'data' in result and not result['data'].empty:
                self.plot_stock(result['data'], name)
                results[name] = result
                self.analysis_results.append(result)
                
                # Add some spacing between stock analyses
                print("\n" + "-" * 50 + "\n")
        
        # Generate and display summary table with the specified interval
        self._generate_summary_table(interval=interval)
        
        print(f"\n✅ Analysis of {len(results)} stocks completed!")
        return results

def main():
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Market Analysis Tool with AI Insights')
    parser.add_argument('group', nargs='?', default='nifty_50',
                      help='Stock group to analyze (default: nifty_50)')
    parser.add_argument('--period', '-p', default='1y',
                      help='Data period to analyze (default: 1y)')
    parser.add_argument('--interval', '-i', default='1d',
                      help='Data interval (default: 1d)')
    parser.add_argument('--no-ai', action='store_true',
                      help='Disable AI analysis')
    parser.add_argument('--trade-ideas', '-t', action='store_true',
                      help='Generate detailed trade ideas')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    print("🚀 Starting Market Analysis with AI Insights...")
    print(f"📊 Group: {args.group.upper()}")
    print(f"⏱️  Period: {args.period}, Interval: {args.interval}")
    print(f"🤖 AI Analysis: {'Enabled' if not args.no_ai else 'Disabled'}")
    print(f"💡 Trade Ideas: {'Enabled' if args.trade_ideas else 'Disabled'}")
    print("-" * 50 + "\n")
    
    try:
        # Create analyzer instance with AI enabled/disabled
        analyzer = MarketAnalyzer(enable_ai=not args.no_ai)
        
        # Analyze the group with the specified parameters
        results = analyzer.analyze_group(
            group_name=args.group,
            period=args.period,
            interval=args.interval,
            generate_trade_ideas=args.trade_ideas
        )
        
        print("\n" + "="*50)
        print("✅ Analysis Complete!")
        print(f"📈 Analyzed {len(results)} stocks")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    main()
