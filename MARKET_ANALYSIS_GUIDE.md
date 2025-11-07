# 📈 Market Analysis Tool - User Guide

Welcome to the Market Analysis Tool! This guide will help you get started with analyzing stocks, generating reports, and making data-driven trading decisions using our powerful Python-based platform.

## 🏗 Architecture Overview

The tool is built with a modular architecture for maintainability and extensibility:

```
market-research/
├── config/           # Configuration files
├── data/             # Data fetching and processing
├── analysis/         # Analysis modules
├── visualization/    # Plotting and reporting
└── backtesting_engine.py  # Strategy backtesting
```

## 🚀 Quick Start Guide

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/market-research.git
cd market-research

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
1. Create a `.env` file with your API keys:
   ```env
   FINNHUB_API_KEY=your_finnhub_key
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   ```

2. Configure stock groups in `config/stocks.json`

### 3. Run Your First Analysis
```bash
# Make the script executable
chmod +x run.sh

# Run the complete workflow (recommended)
./run.sh

# Or run the main application directly
python main.py --group indian_it
```

## 🔍 Understanding the Analysis

### 1. Technical Analysis
- **Trend Indicators**: SMA, EMA, MACD, ADX, Ichimoku Cloud
- **Momentum Oscillators**: RSI, Stochastic, CCI, ROC
- **Volatility Measures**: Bollinger Bands, ATR, Keltner Channels
- **Volume Analysis**: OBV, VWAP, MFI, Accumulation/Distribution
- **Pattern Recognition**: Support/Resistance, Chart Patterns

### 2. Fundamental Analysis
- **Valuation Metrics**: P/E, P/B, P/S, EV/EBITDA ratios
   - P/E Ratio: Price-to-Earnings ratio
   - P/B Ratio: Price-to-Book value
   - Dividend Yield: Annual dividend/current price
   - Market Cap: Company valuation
- **Growth Metrics**: Revenue growth, EPS growth
- **Profitability**: ROE, ROA, Gross/Net Margins
- **Analyst Coverage**: Ratings, price targets, recommendations

### 3. Backtesting & Strategy
- **Moving Average Crossover**: Tested strategy included
- **Custom Strategy Support**: Implement your own strategies
- **Performance Metrics**:
  - Total Return
  - Annualized Return
  - Volatility
  - Sharpe Ratio
  - Max Drawdown
  - Win Rate
  - Profit Factor

### 4. Risk Management
- Position Sizing
- Stop-Loss Strategies
- Risk-Reward Analysis
- Portfolio Optimization

## 📊 Understanding the Analysis

### Key Components

1. **Price Action**
   - Candlestick patterns
   - Support and resistance levels
   - Volume analysis
   - Price trends and channels

2. **Technical Indicators**
   - Trend indicators (SMA, EMA, MACD, ADX)
   - Momentum oscillators (RSI, Stochastic, CCI)
   - Volatility measures (Bollinger Bands, ATR)
   - Volume indicators (OBV, VWAP, MFI)

3. **AI Insights**
   - Market sentiment analysis
   - Pattern recognition
   - Trade ideas and setups
   - Risk assessment

### 📊 Understanding the Output

#### 1. Console Output
- **Stock Table**: Overview of all analyzed stocks
  - Current price and daily change
  - Key technical indicators
  - Trend and momentum signals
  - Volume analysis

#### 2. Generated Plots (in `plots/` directory)
- **Price Chart**: Candlestick chart with moving averages
- **RSI Chart**: Relative Strength Index with overbought/oversold levels
- **MACD Chart**: Moving Average Convergence Divergence
- **Volume Profile**: Trading volume with moving average

#### 3. Backtest Results
- **Equity Curve**: Account value over time
- **Trade History**: List of all trades with entry/exit points
- **Performance Metrics**: Detailed statistics about strategy performance

### 🛠 Troubleshooting

#### Common Issues
1. **API Rate Limits**:
   - If you see rate limit errors, consider upgrading your API plan or reducing the number of requests
   - Implement proper error handling and retries in your code

2. **Missing Data**:
   - Check if the stock symbol is correct
   - Verify your internet connection
   - Some historical data might not be available for certain timeframes

3. **Backtesting Errors**:
   - Ensure sufficient historical data is available for the backtest period
   - Verify that the strategy parameters are valid
   - Check for look-ahead bias in your strategy

#### Getting Help
- Check the project's [Issues](https://github.com/yourusername/market-research/issues) page
- Review the [documentation](https://github.com/yourusername/market-research/wiki)
- For bugs or feature requests, please open a new issue

#### 2. Technical Analysis
- Price action analysis
- Indicator signals
- Volume analysis
- Trend analysis

#### 3. Trade Ideas
- Entry/Exit points
- Stop loss levels
- Take profit targets
- Risk-reward ratios

## 🛠 Advanced Usage

### 1. Custom Stock Groups
Edit `config/stocks.json` to create custom watchlists:

```json
{
  "my_watchlist": {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon"
  },
  "indian_banks": {
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "SBI"
  }
}
```

### 2. Custom Indicators
Add custom technical indicators by extending the `TechnicalIndicators` class:

```python
from market_analysis.indicators import TechnicalIndicators
import pandas as pd

class CustomIndicators(TechnicalIndicators):
    @staticmethod
    def super_trend(high, low, close, period=10, multiplier=3):
        """Calculate SuperTrend indicator"""
        # Implementation here
        pass

# Usage
df = pd.DataFrame(...)  # Your OHLCV data
df['supertrend'] = CustomIndicators.super_trend(
    df['high'], df['low'], df['close']
)
```

### 3. Backtesting Strategies
```python
from market_analysis.backtesting import BacktestEngine
from market_analysis.strategies import MovingAverageCrossover

# Define strategy parameters
strategy = MovingAverageCrossover(
    fast_period=10,
    slow_period=30,
    initial_capital=10000.0
)

# Run backtest
backtest = BacktestEngine(
    strategy=strategy,
    data=historical_data,
    commission=0.001  # 0.1% commission
)

results = backtest.run()
backtest.plot_results()
```

### 4. Generating Custom Reports
```python
from market_analysis import MarketAnalyzer
from market_analysis.reporting import PDFReport, HTMLReport

# Initialize analyzer
analyzer = MarketAnalyzer(
    symbol="AAPL",
    period="1y",
    interval="1d"
)

# Generate analysis
analysis = analyzer.analyze()

# Create PDF report
pdf_report = PDFReport(analysis)
pdf_report.generate("aapl_analysis.pdf")

# Create HTML report
html_report = HTMLReport(analysis)
html_report.generate("aapl_analysis.html")
```

## 📊 Interpreting Technical Indicators

### 1. Trend Analysis
- **Moving Averages**: Identify trends and potential reversals
  - Golden Cross (Bullish): 50-day MA crosses above 200-day MA
  - Death Cross (Bearish): 50-day MA crosses below 200-day MA

- **Ichimoku Cloud**: Comprehensive trend analysis
  - Price above cloud = Uptrend
  - Price below cloud = Downtrend
  - Cloud color change signals potential trend reversal

### 2. Momentum Indicators
- **RSI (Relative Strength Index)**
  - 30-70 = Normal range
  - <30 = Oversold (Potential buy)
  - >70 = Overbought (Potential sell)

- **MACD (Moving Average Convergence Divergence)**
  - Bullish: MACD line crosses above signal line
  - Bearish: MACD line crosses below signal line
  - Divergences can signal potential reversals

### 3. Volatility Indicators
- **Bollinger Bands**
  - Price near upper band = Overbought
  - Price near lower band = Oversold
  - Band width indicates volatility

- **ATR (Average True Range)**
  - Higher values indicate higher volatility
  - Useful for setting stop-loss levels

## 🚀 Best Practices

### 1. Risk Management
- Never risk more than 1-2% of capital on a single trade
- Always use stop-loss orders
- Maintain a risk-reward ratio of at least 1:2
- Diversify across sectors and asset classes

### 2. Trading Strategies
- **Trend Following**: Trade in the direction of the dominant trend
- **Mean Reversion**: Bet on price returning to its average
- **Breakout Trading**: Enter on breakouts from key levels
- **Swing Trading**: Capture short- to medium-term price movements

### 3. Common Pitfalls to Avoid
- Overtrading
- Revenge trading
- Ignoring risk management
- Chasing performance
- Letting emotions drive decisions

## 🔍 Troubleshooting

### Common Issues

1. **API Rate Limits**
   - Symptom: `API rate limit exceeded`
   - Solution: Wait for rate limit reset or upgrade your API plan

2. **Data Fetching Errors**
   - Symptom: `Failed to fetch data for symbol XYZ`
   - Solution:
     - Check internet connection
     - Verify symbol is correct and exists
     - Try again later if it's a temporary issue

3. **Missing Dependencies**
   - Symptom: `ModuleNotFoundError`
   - Solution: Run `pip install -r requirements.txt`

4. **Chart Rendering Issues**
   - Symptom: Charts not displaying properly
   - Solution:
     - Update plotly: `pip install --upgrade plotly`
     - Check browser compatibility

## 📚 Resources

### Learning Resources
- [Technical Analysis Explained](https://www.investopedia.com/technical-analysis-4689657)
- [Python for Finance](https://www.oreilly.com/library/view/python-for-finance/9781492024323/)
- [Algorithmic Trading](https://www.quantstart.com/articles/)

### API Documentation
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Alpha Vantage API](https://www.alphavantage.co/documentation/)
- [GROQ API](https://console.groq.com/docs/introduction)

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Bugs**
   - Open an issue with detailed reproduction steps

2. **Suggest Enhancements**
   - Create a feature request issue
   - Discuss in the project's discussions

3. **Submit Pull Requests**
   - Fork the repository
   - Create a feature branch
   - Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and informational purposes only. It is not intended as financial advice. Trading in financial markets involves risk, and you should carefully consider your investment objectives, level of experience, and risk appetite before making any investment decisions. Past performance is not indicative of future results.
