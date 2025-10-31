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
    """Load stock configurations from JSON file"""
    try:
        config_path = Path(__file__).parent / 'config' / 'stocks.json'
        with open(config_path, 'r') as f:
            return json.load(f)
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
    Fetch stock data using yfinance with technical indicators
    
    Args:
        stock_group (str): Key from STOCK_CONFIG to get the list of stocks
        period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        tuple: (stocks_data, metadata) where:
            - stocks_data: dict containing data and technical indicators for each stock
            - metadata: dict containing additional information about the stocks
    """
    stocks_data = {}
    metadata = {}
    
    if stock_group not in STOCK_CONFIG:
        print(f"Stock group '{stock_group}' not found in configuration. Using 'indian_it' as default.")
        stock_group = "indian_it"
    
    stock_list = STOCK_CONFIG[stock_group]
    
    for company, info in stock_list.items():
        ticker = info['symbol']
        try:
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
                
            # Calculate technical indicators
            # RSI (Relative Strength Index)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            # MACD (Moving Average Convergence Divergence)
            macd = ta.macd(df['Close'])
            df = pd.concat([df, macd], axis=1)
            
            # Bollinger Bands
            bbands = ta.bbands(df['Close'], length=20, std=2)
            df = pd.concat([df, bbands], axis=1)
            
            # Moving Averages
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['SMA_200'] = ta.sma(df['Close'], length=200)
            
            # Volume Weighted Average Price (VWAP)
            df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Store the data and metadata
            stocks_data[company] = df
            metadata[company] = {
                'ticker': ticker,
                'sector': info.get('sector', 'N/A'),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"Error processing {company} ({ticker}): {str(e)}")
    
    return stocks_data, metadata

def generate_technical_analysis(stock_data):
    """Generate technical analysis for stocks"""
    analysis = {}
    
    for company, data in stock_data.items():
        try:
            latest = data.iloc[-1]
            prev_day = data.iloc[-2] if len(data) > 1 else latest
            
            # Calculate price change
            price_change = ((latest['Close'] - prev_day['Close']) / prev_day['Close']) * 100
            
            # Determine trend based on moving averages
            trend = "Bullish" if latest['SMA_20'] > latest['SMA_50'] > latest['SMA_200'] else "Bearish"
            
            # RSI analysis
            rsi_signal = ""
            if latest['RSI'] > 70:
                rsi_signal = "Overbought"
            elif latest['RSI'] < 30:
                rsi_signal = "Oversold"
            else:
                rsi_signal = "Neutral"
                
            # MACD analysis
            macd_signal = ""
            if latest['MACD_12_26_9'] > latest['MACDs_12_26_9'] and \
               data['MACD_12_26_9'].iloc[-2] <= data['MACDs_12_26_9'].iloc[-2]:
                macd_signal = "Bullish Crossover"
            elif latest['MACD_12_26_9'] < latest['MACDs_12_26_9'] and \
                 data['MACD_12_26_9'].iloc[-2] >= data['MACDs_12_26_9'].iloc[-2]:
                macd_signal = "Bearish Crossover"
            else:
                macd_signal = "Neutral"
            
            analysis[company] = {
                'Price': round(latest['Close'], 2),
                'Change (%)': round(price_change, 2),
                'Trend': trend,
                'RSI': round(latest['RSI'], 2),
                'RSI Signal': rsi_signal,
                'MACD Signal': macd_signal,
                '20-Day MA': round(latest['SMA_20'], 2),
                '50-Day MA': round(latest['SMA_50'], 2),
                '200-Day MA': round(latest['SMA_200'], 2),
                'VWAP': round(latest['VWAP'], 2)
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
