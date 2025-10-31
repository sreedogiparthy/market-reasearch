from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import style

# Set style for plots
style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 8)

load_dotenv()

# Initialize LLM with Groq
llm = Groq(id="llama-3.3-70b-versatile", temperature=0.7)

def load_stock_config():
    """Load stock configurations from JSON file with enhanced error handling"""
    try:
        config_path = Path(__file__).parent / 'config' / 'stocks.json'
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Normalize the configuration to ensure consistent structure
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
                    # Ensure required fields exist
                    normalized_stock = {
                        "symbol": stock_info.get("symbol", ""),
                        "sector": stock_info.get("sector", "Unknown")
                    }
                    normalized_config[group_name][company] = normalized_stock
                else:
                    print(f"Warning: Invalid format for {company} in {group_name}")
        
        return normalized_config
        
    except Exception as e:
        print(f"Error loading stock config: {e}")
        # Fallback to default configuration
        return {
            "indian_it": {
                "TCS": {"symbol": "TCS.NS", "sector": "IT Services"},
                "Infosys": {"symbol": "INFY.NS", "sector": "IT Services"}
            }
        }
# Load stock configurations
STOCK_CONFIG = load_stock_config()

def get_stock_data(stock_group="indian_it", period="1y", interval="1d"):
    """
    Fetch stock data using yfinance with technical indicators and robust error handling
    """
    stocks_data = {}
    metadata = {}
    
    if stock_group not in STOCK_CONFIG:
        print(f"Stock group '{stock_group}' not found in configuration. Using 'indian_it' as default.")
        stock_group = "indian_it"
    
    stock_list = STOCK_CONFIG[stock_group]
    
    for company, info in stock_list.items():
        # Handle both string and object formats
        if isinstance(info, str):
            ticker = info
            sector = "Unknown"
        else:
            ticker = info.get('symbol', '')
            sector = info.get('sector', 'Unknown')
        
        if not ticker:
            print(f"No ticker symbol found for {company}")
            continue
            
        try:
            print(f"Fetching data for {company} ({ticker})...")
            
            # Fetch data
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            if df.empty:
                print(f"No data found for {company} ({ticker})")
                continue
            
            # Check if we have enough data for technical indicators
            if len(df) < 50:
                print(f"Warning: Insufficient data for {company} ({ticker}) - only {len(df)} records")
                # Store basic data without technical indicators
                stocks_data[company] = df
                metadata[company] = {
                    'ticker': ticker,
                    'sector': sector,
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'warning': 'Insufficient data for full technical analysis'
                }
                continue
                
            # Calculate technical indicators only if we have enough data
            try:
                # RSI (Relative Strength Index)
                df['RSI'] = ta.rsi(df['Close'], length=14)
                
                # MACD (Moving Average Convergence Divergence)
                macd = ta.macd(df['Close'])
                if macd is not None:
                    df = pd.concat([df, macd], axis=1)
                
                # Bollinger Bands
                bbands = ta.bbands(df['Close'], length=20, std=2)
                if bbands is not None:
                    df = pd.concat([df, bbands], axis=1)
                
                # Moving Averages
                df['SMA_20'] = ta.sma(df['Close'], length=20)
                df['SMA_50'] = ta.sma(df['Close'], length=50)
                df['SMA_200'] = ta.sma(df['Close'], length=200)
                
                # Volume Weighted Average Price (VWAP)
                df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
                
            except Exception as indicator_error:
                print(f"Error calculating indicators for {company}: {indicator_error}")
                # Continue with basic data
                
            # Store the data and metadata
            stocks_data[company] = df
            metadata[company] = {
                'ticker': ticker,
                'sector': sector,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"Error processing {company} ({ticker}): {str(e)}")
    
    return stocks_data, metadata

def generate_technical_analysis(stock_data):
    """Generate technical analysis for stocks with proper error handling"""
    analysis = {}
    
    for company, data in stock_data.items():
        try:
            if data.empty or len(data) < 2:
                analysis[company] = {"error": "Insufficient data"}
                continue
                
            latest = data.iloc[-1]
            prev_day = data.iloc[-2]
            
            # Calculate price change - handle NaN values
            price_change = 0.0
            if not pd.isna(latest['Close']) and not pd.isna(prev_day['Close']) and prev_day['Close'] != 0:
                price_change = ((latest['Close'] - prev_day['Close']) / prev_day['Close']) * 100
            
            # Determine trend based on moving averages with NaN checks
            trend = "Neutral"
            if (not pd.isna(latest.get('SMA_20', np.nan)) and 
                not pd.isna(latest.get('SMA_50', np.nan)) and 
                not pd.isna(latest.get('SMA_200', np.nan))):
                if latest['SMA_20'] > latest['SMA_50'] > latest['SMA_200']:
                    trend = "Bullish"
                elif latest['SMA_20'] < latest['SMA_50'] < latest['SMA_200']:
                    trend = "Bearish"
            
            # RSI analysis with NaN check
            rsi_signal = "Neutral"
            rsi_value = latest.get('RSI', 50)
            if not pd.isna(rsi_value):
                if rsi_value > 70:
                    rsi_signal = "Overbought"
                elif rsi_value < 30:
                    rsi_signal = "Oversold"
            
            # MACD analysis with proper series comparison
            macd_signal = "Neutral"
            try:
                current_macd = latest.get('MACD_12_26_9', 0)
                current_signal = latest.get('MACDs_12_26_9', 0)
                prev_macd = data['MACD_12_26_9'].iloc[-2] if len(data) > 1 else current_macd
                prev_signal = data['MACDs_12_26_9'].iloc[-2] if len(data) > 1 else current_signal
                
                if (not pd.isna(current_macd) and not pd.isna(current_signal) and
                    not pd.isna(prev_macd) and not pd.isna(prev_signal)):
                    if current_macd > current_signal and prev_macd <= prev_signal:
                        macd_signal = "Bullish Crossover"
                    elif current_macd < current_signal and prev_macd >= prev_signal:
                        macd_signal = "Bearish Crossover"
            except (KeyError, IndexError):
                macd_signal = "Data Unavailable"
            
            analysis[company] = {
                'Price': round(latest['Close'], 2) if not pd.isna(latest['Close']) else 0,
                'Change (%)': round(price_change, 2),
                'Trend': trend,
                'RSI': round(rsi_value, 2) if not pd.isna(rsi_value) else 0,
                'RSI Signal': rsi_signal,
                'MACD Signal': macd_signal,
                '20-Day MA': round(latest.get('SMA_20', 0), 2) if not pd.isna(latest.get('SMA_20', np.nan)) else 0,
                '50-Day MA': round(latest.get('SMA_50', 0), 2) if not pd.isna(latest.get('SMA_50', np.nan)) else 0,
                '200-Day MA': round(latest.get('SMA_200', 0), 2) if not pd.isna(latest.get('SMA_200', np.nan)) else 0,
                'VWAP': round(latest.get('VWAP', 0), 2) if not pd.isna(latest.get('VWAP', np.nan)) else 0
            }
            
        except Exception as e:
            print(f"Error analyzing {company}: {str(e)}")
            analysis[company] = {"error": str(e)}
    
    return analysis

def plot_stock_data(stock_data, company):
    """Generate technical analysis plots for a stock"""
    try:
        df = stock_data[company].copy()
        if df.empty:
            return None
            
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot candlesticks
        mpf.plot(df[-90:], type='candle', style='charles', ax=ax1, 
                volume=ax2, mav=(20, 50, 200), returnfig=False)
        
        # Plot RSI
        ax3.set_title('RSI (14)')
        ax3.plot(df.index[-90:], df['RSI'][-90:], label='RSI', color='purple')
        ax3.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax3.fill_between(df.index[-90:], 70, 30, color='gray', alpha=0.1)
        ax3.legend()
        
        plt.suptitle(f'{company} Technical Analysis', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('plots', exist_ok=True)
        plot_path = f'plots/{company.replace(" ", "_")}_analysis.png'
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    except Exception as e:
        print(f"Error generating plot for {company}: {str(e)}")
        return None

def analyze_stocks(stock_group="indian_it"):
    """
    Analyze stocks with technical indicators
    
    Args:
        stock_group (str): Key from STOCK_CONFIG to get the list of stocks
        
    Returns:
        tuple: (prompt, analysis, plot_paths, metadata)
    """
    print(f"Fetching market data for {stock_group}...")
    stock_data, metadata = get_stock_data(stock_group=stock_group, period="1y", interval="1d")
    
    if not stock_data:
        return "Error: Could not fetch stock data. Please try again later.", {}, {}
    
    print("Performing technical analysis...")
    analysis = generate_technical_analysis(stock_data)
    
    # Generate plots for top 3 stocks by market cap
    top_companies = sorted(
        [k for k in analysis.keys() if 'error' not in analysis[k]],
        key=lambda x: analysis[x].get('Price', 0),
        reverse=True
    )[:3]
    
    plot_paths = {}
    for company in top_companies:
        plot_path = plot_stock_data(stock_data, company)
        if plot_path:
            plot_paths[company] = plot_path
    
    # Get sector information from metadata
    sectors = {}
    for company, data in metadata.items():
        sector = data.get('sector', 'Other')
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(company)
    
    # Prepare analysis prompt
    prompt = f"""
    Perform a comprehensive technical analysis of the {stock_group.replace('_', ' ').title()} sector based on the provided data.
    
    Sector Breakdown:
    """
    
    # Add sector information
    for sector, companies in sectors.items():
        prompt += f"- {sector}: {', '.join(companies)}\n"
    
    prompt += """
    
    For each stock, analyze and include:
    1. Current price and daily change
    2. Trend analysis using moving averages (20, 50, 200-day)
    3. RSI (Relative Strength Index) and its implications
    4. MACD (Moving Average Convergence Divergence) signals
    5. Support and resistance levels from Bollinger Bands
    6. Volume analysis and VWAP
    
    Then provide:
    1. Overall market sentiment (Bullish/Bearish/Neutral)
    2. Top 3 stocks with strongest technical setup
    3. Stocks showing reversal or continuation patterns
    4. Key support and resistance levels to watch
    5. Trading recommendations based on technicals
    
    Use tables to present the data clearly and include specific price levels.
    """
    
    return prompt, analysis, plot_paths, metadata

# Initialize agents
web_agent = Agent(
    name="web_agent",
    role="Search the web and answer questions based on the search results.",
    model=llm,
    tools=[DuckDuckGoTools()],
    instructions="""You are a web agent that can search the web and answer questions about Indian markets. 
    Always include sources and focus on Indian stock market context.
    Provide recent and relevant information.""",
    markdown=True,
)

finance_agent = Agent(
    name="finance_agent",
    role="Get Financial Data for Indian Markets",
    model=llm,
    tools=[YFinanceTools()],
    instructions="""You are a finance agent that can get financial data for Indian companies. 
    Always include sources. Use tables to display the data. 
    Focus on key metrics like current price, 52-week high/low, PE ratio, and market cap.
    For Indian stocks, use the .NS suffix (e.g., TCS.NS for TCS).""",
    markdown=True,
)

market_research_agent = Agent(
    name="market_research_agent",
    role="Indian Market Research Agent",
    model=llm,
    tools=[web_agent, finance_agent],
    instructions="""You are a market research agent focused on Indian markets. 
    Analyze and provide insights on Indian IT and semiconductor companies. 
    Always include sources and use tables to display financial data. 
    Provide both fundamental and technical analysis where relevant.
    Focus on recent trends and data.""",
    markdown=True,
)

def main(stock_group=None):
    """
    Main function to run the market analysis
    
    Args:
        stock_group (str, optional): Specific stock group to analyze. If None, analyzes all groups.
    """
   # Debug the configuration first
    debug_stock_data(stock_group or "indian_it")
    
    if stock_group is None:
        # Analyze all stock groups
        for group in STOCK_CONFIG.keys():
            analyze_and_display(group)
    else:
        analyze_and_display(stock_group)

def analyze_and_display(stock_group):
    """
    Analyze and display results for a specific stock group
    
    Args:
        stock_group (str): The stock group to analyze
    """
    print(f"\n{'='*80}")
    print(f"Analyzing {stock_group.replace('_', ' ').title()} with Technical Indicators...\n")
    
    try:
        # Get analysis with technical indicators
        analysis_prompt, analysis_data, plot_paths, metadata = analyze_stocks(stock_group=stock_group)
        
        if not analysis_data:
            print(f"No analysis data available for {stock_group}. Skipping...")
            return
        
        # Add technical data to the prompt
        analysis_prompt += "\n\nTechnical Data Summary:\n"
        
        # Create a DataFrame with analysis data
        analysis_df = pd.DataFrame(analysis_data).T
        
        # Add sector information to the DataFrame
        sectors = []
        for company in analysis_df.index:
            sectors.append(metadata.get(company, {}).get('sector', 'N/A'))
        analysis_df['Sector'] = sectors
        
        # Reorder columns to show sector first
        cols = ['Sector'] + [col for col in analysis_df.columns if col != 'Sector']
        analysis_df = analysis_df[cols]
        
        # Convert DataFrame to markdown and add to prompt
        analysis_prompt += analysis_df.to_markdown()
        
        # Get analysis from the market research agent
        print("\nGenerating comprehensive market analysis...\n")
        market_research_agent.print_response(analysis_prompt)
        
        # Show plot paths if available
        if plot_paths:
            print("\nTechnical analysis plots saved in the 'plots' directory:")
            for company, path in plot_paths.items():
                print(f"- {company}: {path}")
                
    except Exception as e:
        print(f"Error analyzing {stock_group}: {str(e)}")
    
    # Show available stock groups at the end of all analyses
    if stock_group == list(STOCK_CONFIG.keys())[-1]:  # Only show once after last group
        print("\n" + "="*80)
        print("\nAvailable stock groups:")
        for group in STOCK_CONFIG.keys():
            print(f"- {group} ({len(STOCK_CONFIG[group])} stocks)")
        print("\nYou can analyze a specific group by running: python main.py <group_name>")
        print("Example: python main.py indian_banks")

class AnalysisType(Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"

@dataclass
class StockAnalysis:
    symbol: str
    company: str
    sector: str
    technical_indicators: Dict
    fundamental_data: Optional[Dict] = None
    sentiment_score: Optional[float] = None
    risk_level: str = "medium"

class ResilientStockFetcher:
    """Enhanced stock data fetcher with retry logic and caching"""
    
    def __init__(self, max_retries: int = 3, cache_duration: int = 300):
        self.max_retries = max_retries
        self.cache_duration = cache_duration
        self._cache = {}
    
    async def fetch_stock_data_with_retry(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch stock data with exponential backoff retry logic"""
        for attempt in range(self.max_retries):
            try:
                if symbol in self._cache:
                    cached_data, timestamp = self._cache[symbol]
                    if time.time() - timestamp < self.cache_duration:
                        return cached_data
                
                data = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: yf.download(symbol, period=period, progress=False)
                )
                
                if not data.empty:
                    self._cache[symbol] = (data, time.time())
                    return data
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"Failed to fetch {symbol} after {self.max_retries} attempts: {e}")
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

class PortfolioAnalyzer:
    """Analyze portfolio risk and correlations"""
    
    def __init__(self, stock_data: Dict):
        self.stock_data = stock_data
    
    def calculate_correlations(self) -> pd.DataFrame:
        """Calculate correlation matrix between stocks"""
        closes = pd.DataFrame()
        for company, data in self.stock_data.items():
            if not data.empty:
                closes[company] = data['Close']
        
        return closes.corr()
    
    def portfolio_risk_analysis(self, weights: Optional[Dict] = None) -> Dict:
        """Analyze portfolio risk metrics"""
        if weights is None:
            # Equal weighting if not specified
            n_stocks = len(self.stock_data)
            weights = {company: 1/n_stocks for company in self.stock_data.keys()}
        
        returns = self.calculate_portfolio_returns(weights)
        
        risk_metrics = {
            'annual_return': returns.mean() * 252,
            'annual_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'max_drawdown': self.calculate_max_drawdown(returns),
            'var_95': self.calculate_var(returns, 0.95)
        }
        
        return risk_metrics

# Add advanced technical indicators
def calculate_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced technical indicators"""
    try:
        # Ichimoku Cloud
        ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])
        df = pd.concat([df, ichimoku], axis=1)
        
        # Fibonacci Retracement
        high_52w = df['High'].max()
        low_52w = df['Low'].min()
        diff = high_52w - low_52w
        
        df['Fib_23.6'] = high_52w - diff * 0.236
        df['Fib_38.2'] = high_52w - diff * 0.382
        df['Fib_61.8'] = high_52w - diff * 0.618
        
        # Volume-based indicators
        df['Volume_SMA'] = ta.sma(df['Volume'], length=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Volatility measures
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        
        return df
        
    except Exception as e:
        print(f"Error calculating advanced indicators: {e}")
        return df

def generate_trading_signals(df: pd.DataFrame) -> Dict:
    """Generate comprehensive trading signals"""
    latest = df.iloc[-1]
    
    signals = {
        'trend_strength': calculate_trend_strength(df),
        'momentum': calculate_momentum_score(df),
        'volatility_alert': latest['Volatility'] > df['Volatility'].quantile(0.8),
        'volume_breakout': latest['Volume_Ratio'] > 2.0,
        'support_resistance': identify_support_resistance(df)
    }
    
    return signals

async def main_async(stock_group: Optional[str] = None):
    """Enhanced main function with async support"""
    config_manager = ConfigManager()
    app_config = config_manager.load_config("app_config.json")
    
    analyzer = StockMarketAnalyzer(app_config)
    
    try:
        if stock_group:
            await analyzer.analyze_stock_group(stock_group)
        else:
            for group in STOCK_CONFIG.keys():
                await analyzer.analyze_stock_group(group)
                
    except Exception as e:
        print(f"Analysis failed: {e}")
        # Implement fallback strategy
        await analyzer.run_basic_analysis()

class StockMarketAnalyzer:
    """Main analyzer class with enhanced capabilities"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.fetcher = ResilientStockFetcher()
        self.portfolio_analyzer = None
    
    async def analyze_stock_group(self, stock_group: str):
        """Analyze a specific stock group"""
        print(f"\n📊 Analyzing {stock_group.replace('_', ' ').title()}...")
        
        # Fetch and analyze data
        stock_data = await self.fetch_stock_data(stock_group)
        analysis_results = self.analyze_stocks(stock_data)
        
        # Generate insights
        insights = self.generate_insights(analysis_results)
        
        # Create reports and visualizations
        await self.generate_reports(stock_group, analysis_results, insights)

# Add caching and performance monitoring
import functools
import time
from concurrent.futures import ThreadPoolExecutor

def timer_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class PerformanceMonitor:
    """Monitor system performance and resource usage"""
    
    def __init__(self):
        self.metrics = {}
    
    def track_metric(self, name: str, value: float):
        self.metrics[name] = value

def debug_stock_data(stock_group="indian_it"):
    """Debug function to check stock data structure"""
    print(f"\n🔍 Debugging {stock_group}...")
    
    if stock_group not in STOCK_CONFIG:
        print(f"Stock group '{stock_group}' not found!")
        return
    
    stock_list = STOCK_CONFIG[stock_group]
    print(f"Found {len(stock_list)} stocks in {stock_group}")
    
    for company, info in stock_list.items():
        print(f"\n{company}:")
        if isinstance(info, str):
            print(f"  Ticker: {info}")
            print(f"  Type: Simple string")
        elif isinstance(info, dict):
            print(f"  Ticker: {info.get('symbol', 'MISSING')}")
            print(f"  Sector: {info.get('sector', 'Unknown')}")
            print(f"  Type: Object")
        else:
            print(f"  Type: UNKNOWN - {type(info)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # If a stock group is provided as a command-line argument
        stock_group = sys.argv[1]
        if stock_group not in STOCK_CONFIG:
            print(f"Error: Stock group '{stock_group}' not found.")
            print("\nAvailable stock groups:")
            for group in STOCK_CONFIG.keys():
                print(f"- {group} ({len(STOCK_CONFIG[group])} stocks)")
            sys.exit(1)
        main(stock_group)
    else:
        # If no arguments, analyze all stock groups
        main()
