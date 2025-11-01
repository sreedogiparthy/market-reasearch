# run_trading_bot.py
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from trading_bot import (
    EnhancedTradingBot, 
    EnhancedMovingAverageCrossover,
    Backtester,
    TradeSignal
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', mode='w'),  # Overwrite log file
        logging.StreamHandler()
    ]
)

# Set specific log levels for noisy libraries
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def run_backtest():
    """Run backtest with the enhanced strategy"""
    print("🧪 RUNNING BACKTEST...")
    
    # Define configuration
    config = {
        'strategy': {
            'long_window': 50,
            'short_window': 20,
            'rsi_period': 14,
            'volume_threshold': 1.2
        },
        'backtest': {
            'initial_capital': 100000,
            'commission': 0.001,
            'slippage': 0.0005,
            'position_size': 0.1,
            'max_positions': 5
        },
        'data': {
            'source': 'yfinance',
            'interval': '1d'
        }
    }
    
    backtester = Backtester(config=config)
    
    # Test symbols - trying both Indian and US markets
    test_symbols = [
        # Indian stocks (NSE)
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS',
        # US stocks
        'AAPL', 'MSFT', 'GOOGL', 'AMZN'
    ]
    
    # Check which symbols have data available
    print("\n🔍 Checking data availability for symbols...")
    valid_symbols = []
    
    for symbol in test_symbols:
        try:
            print(f"Checking {symbol}...", end=' ')
            data = yf.download(symbol, start='2022-01-01', end='2023-01-01', progress=False)
            if len(data) > 50:  # Need at least 50 data points
                valid_symbols.append(symbol)
                print(f"✅ ({len(data)} days of data)")
            else:
                print(f"❌ (Insufficient data: {len(data)} days)")
        except Exception as e:
            print(f"❌ (Error: {str(e)})")
    
    if not valid_symbols:
        print("\n❌ No valid symbols with sufficient data found!")
        print("Please check your internet connection or try different symbols.")
        return None
    
    # Use only the first 4 valid symbols for backtesting
    symbols = valid_symbols[:4]
    print(f"\n📊 Using symbols: {', '.join(symbols)}")
    
    # Use a more recent date range with more data
    start_date = '2022-01-01'
    end_date = '2023-01-01'  # 1 year of data
    
    # Create enhanced strategy
    strategy = EnhancedMovingAverageCrossover(
        short_window=20,
        long_window=50,
        rsi_period=14,
        volume_threshold=1.2
    )
    
    try:
        # Run backtest
        print(f"\n{'='*50}")
        print(f"Starting backtest from {start_date} to {end_date}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Strategy: {strategy.__class__.__name__}")
        print('='*50 + '\n')
        
        results = backtester.run_backtest(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        if results is None:
            print("\n❌ Backtest failed to produce results")
            return None
            
        # Generate and print report
        if hasattr(backtester, 'generate_report'):
            report = backtester.generate_report()
            print("\n" + "="*50)
            print("📊 BACKTEST REPORT")
            print("="*50)
            print(report)
        
        # Plot results if method exists
        if hasattr(backtester, 'plot_results'):
            try:
                backtester.plot_results('backtest_results.png')
                print("\n✅ Backtest results saved to 'backtest_results.png'")
            except Exception as e:
                print(f"\n⚠️  Could not generate plot: {str(e)}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during backtest: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_live_trading():
    """Run the trading bot in live/paper trading mode"""
    print("🚀 STARTING LIVE TRADING...")
    
    # Initialize trading bot
    bot = EnhancedTradingBot()
    
    try:
        # Run for a specific duration (e.g., 1 hour for demo)
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < 3600:  # 1 hour
            try:
                bot.monitor_market(interval=300)  # 5-minute intervals
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"Trading error: {e}")
                time.sleep(60)
    except Exception as e:
        logging.error(f"Fatal error in live trading: {e}")
    finally:
        logging.info("Trading bot stopped")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'live':
        run_live_trading()
    else:
        run_backtest()