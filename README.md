# 📈 Market Research & Analysis Tool

A comprehensive Python-based solution for technical and fundamental analysis of global markets with AI-powered insights, risk assessment, backtesting, and automated reporting.

## 🚀 Key Features

- **Dynamic Stock Configuration**: Manage stock symbols and metadata through `config/stocks.json`
- **Modular Architecture**: Clean separation of concerns with dedicated modules for data, analysis, visualization, and backtesting
- **Comprehensive Analysis**: Technical indicators, fundamental data, news integration, and strategy backtesting
- **Extensible Design**: Easy to add new analysis methods, data sources, and trading strategies
- **Automated Testing**: Comprehensive test suite with integration tests for all major components

## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/market-research.git
   cd market-research
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏗 Project Structure

```
market-research/
├── config/                    # Configuration files
│   ├── __init__.py
│   ├── stock_config.py        # Stock configuration and symbol mapping
│   └── stocks.json            # Stock symbols and metadata
│
├── data/                      # Data handling
│   ├── __init__.py
│   └── fetcher.py             # Data fetching and processing
│
├── analysis/                  # Analysis modules
│   ├── __init__.py
│   ├── technical.py           # Technical analysis
│   └── fundamental.py         # Fundamental analysis
│
├── visualization/             # Visualization tools
│   ├── __init__.py
│   └── plotter.py             # Plot generation
│
├── backtesting_engine.py      # Strategy backtesting framework
├── main.py                    # Main application entry point
├── test_integration.py        # Integration tests
├── run.sh                     # Setup and execution script
└── requirements.txt           # Python dependencies
```

## 🚦 Running the Application

### Using the run.sh script (recommended):
```bash
# Make the script executable if needed
chmod +x run.sh

# Run the complete workflow
./run.sh
```

### Running directly:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the main application
python main.py --group indian_it
```

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest test_integration.py -v

# Run a specific test
python -m pytest test_integration.py::test_specific_test -v
```

## 📝 Configuration

1. **Stock Configuration**: Edit `config/stocks.json` to add or modify stock groups and symbols
2. **Environment Variables**: Create a `.env` file with your API keys:
   ```
   FINNHUB_API_KEY=your_api_key_here
   ALPHA_VANTAGE_API_KEY=your_api_key_here
   ```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/yourusername/market-research/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/market-research/actions)
[![Documentation Status](https://readthedocs.org/projects/market-research/badge/?version=latest)](https://market-research.readthedocs.io/)

## 🚀 Key Features

### 📊 Technical Analysis
- **Comprehensive Indicators**: RSI, MACD, Bollinger Bands, Ichimoku Cloud, ATR, VWAP, Moving Averages, and more
- **Multiple Timeframes**: Analyze data across different time periods
- **Custom Strategies**: Implement and test your own trading strategies

### 📈 Backtesting
- **Strategy Testing**: Test trading strategies with historical data
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate, and more
- **Custom Parameters**: Fine-tune strategy parameters for optimization

### 📊 Visualization
- **Interactive Plots**: Generate and save visualizations of technical indicators
- **Customizable Charts**: Tailor plots to your analysis needs
- **Multiple Output Formats**: Save plots as PNG or view interactively
- **Multiple Timeframes**: Analyze data from 1-minute to monthly intervals
- **Customizable Settings**: Adjust all technical parameters to fit your trading style

### 🤖 AI-Powered Insights
- **Intelligent Analysis**: Get AI-generated market insights and trade ideas
- **Sentiment Analysis**: Gauge market sentiment from news and social media
- **Pattern Recognition**: Automated detection of chart patterns and key levels

### 📈 Advanced Charting
- **Interactive Visuals**: Plotly-powered interactive charts
- **Multiple Indicators**: Combine multiple indicators on a single chart
- **Export Options**: Save charts as PNG, JPG, or interactive HTML

### ⚡ Performance & Risk
- **Risk Metrics**: Value at Risk (VaR), Maximum Drawdown, Sharpe Ratio
- **Portfolio Analysis**: Correlation matrix and efficient frontier
- **Backtesting**: Test strategies on historical data

### 📦 Data Management
- **Multiple Data Sources**: yfinance, Alpha Vantage, and custom APIs
- **Caching**: Local storage of historical data for faster access
- **Export Formats**: JSON, CSV, Excel, and PDF reports

## 🛠 Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/market-research.git
   cd market-research
   ```

2. **Set up a virtual environment**:
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # Install core requirements
   pip install -r requirements.txt
   
   # For development (optional)
   pip install -r requirements-dev.txt
   ```

4. **Configure environment variables**:
   ```bash
   # Copy the sample environment file
   cp .env.sample .env
   ```
   
   Edit `.env` and add your API keys:
   ```
   # Required
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   
   # Optional
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   FINNHUB_API_KEY=your_finnhub_key
   ```

## ⚙️ Configuration

The application is highly configurable through the `config/app_config.json` file. Here's an overview of the main configuration sections:

### Core Configuration

```json
{
  "analysis_settings": {
    "default_period": "1y",
    "technical_indicators": ["RSI", "MACD", "BBANDS", "SMA", "EMA", "VWAP", "ATR", "ICHIMOKU"],
    "risk_free_rate": 0.05,
    "cache_duration": 300,
    "max_retries": 3,
    "backtest_periods": ["1m", "5m", "15m", "1h", "1d", "1w"]
  },
  "data_sources": {
    "default": "yfinance",
    "yfinance": {
      "tickers": ["AAPL", "MSFT", "GOOGL"],
      "interval": "1d"
    },
    "alpha_vantage": {
      "enabled": false,
      "api_key": "${ALPHA_VANTAGE_API_KEY}"
    }
  },
  "plot_settings": {
    "style": "plotly_white",
    "figsize": [1200, 800],
    "template": "plotly_white",
    "save_directory": "reports/charts",
    "export_formats": ["html", "png"]
  },
  "risk_management": {
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "stop_loss_pct": 2.0,
    "take_profit_pct": 4.0,
    "max_position_size": 10.0,
    "max_portfolio_risk": 1.0
  },
  "reporting": {
    "enabled": true,
    "format": "html",
    "save_directory": "reports",
    "email_alerts": false,
    "slack_webhook": ""
  }
}
```

### Configuration Management

Use the built-in `ConfigManager` to handle configurations programmatically:

```python
from config import ConfigManager

# Initialize with custom config directory
config_manager = ConfigManager("config")

# Load and validate configuration
try:
    # Load main config
    config = config_manager.load_config("app_config.json")
    
    # Access nested values with dot notation
    rsi_overbought = config_manager.get_config_value(
        config, "risk_management.rsi_overbought", default=70
    )
    
    # Update and save configuration
    config_manager.update_config("risk_management.stop_loss_pct", 2.5)
    config_manager.save_config("app_config.json")
    
    # Create a backup
    config_manager.backup_config("app_config.json")
    
except ConfigError as e:
    print(f"Configuration error: {e}")
    # Handle error or use defaults
```

### Environment Variables

Configuration can be overridden using environment variables with the `MARKET_` prefix:

```bash
# Example: Override RSI thresholds
export MARKET_RISK_MANAGEMENT__RSI_OVERBOUGHT=75
export MARKET_RISK_MANAGEMENT__RSI_OVERSOLD=25
```

## 🚀 Usage

### Command Line Interface

```bash
# Analyze default stock group with default settings
python market_analysis.py

# Analyze specific stock group (defined in config/stocks.json)
python market_analysis.py nifty_50

# Analyze single stock
python market_analysis.py --symbol AAPL

# Custom time period and interval
python market_analysis.py --period 1y --interval 1d

# Enable AI-powered analysis and trade ideas
python market_analysis.py --enable-ai --trade-ideas

# Run backtest on historical data
python market_analysis.py --backtest --start-date 2023-01-01 --end-date 2023-12-31

# Generate report in specific format
python market_analysis.py --format html --output-dir reports/
```

### Python API

```python
from market_analysis import MarketAnalyzer

# Initialize analyzer
analyzer = MarketAnalyzer(
    symbol="AAPL",
    period="1y",
    interval="1d",
    enable_ai=True
)

# Fetch and analyze data
analysis = analyzer.analyze()

# Generate report
report = analyzer.generate_report(format="html")

# Get trading signals
signals = analyzer.get_signals()

# Plot interactive chart
fig = analyzer.plot_chart()
fig.show()

# Save analysis to file
analyzer.save_report("analysis_report.html")
```

### Example Workflow

1. **Data Collection**:
   ```python
   from market_analysis import DataFetcher
   
   # Fetch historical data
   df = DataFetcher.fetch(
       symbol="AAPL",
       period="1y",
       interval="1d"
   )
   ```

2. **Technical Analysis**:
   ```python
   from market_analysis.indicators import TechnicalIndicators
   
   # Calculate RSI
   df['RSI'] = TechnicalIndicators.rsi(df['close'])
   
   # Calculate Moving Averages
   df['SMA_20'] = TechnicalIndicators.sma(df['close'], window=20)
   df['EMA_50'] = TechnicalIndicators.ema(df['close'], span=50)
   ```

3. **Generate Report**:
   ```python
   from market_analysis.reporting import ReportGenerator
   
   # Create report generator
   reporter = ReportGenerator(analysis)
   
   # Generate HTML report
   html_report = reporter.generate_html()
   
   # Save to file
   with open("analysis_report.html", "w") as f:
       f.write(html_report)
   ```

## 📊 Technical Indicators

The tool supports a comprehensive set of technical indicators across different categories:

### 📈 Trend Indicators
| Indicator | Description | Common Parameters |
|-----------|-------------|-------------------|
| **SMA** | Simple Moving Average | Periods: 20, 50, 200 |
| **EMA** | Exponential Moving Average | Periods: 9, 21, 50, 200 |
| **MACD** | Moving Average Convergence Divergence | Fast: 12, Slow: 26, Signal: 9 |
| **Ichimoku Cloud** | Comprehensive trend analysis | Conversion: 9, Base: 26, Span B: 52 |
| **ADX** | Average Directional Index | Period: 14 |
| **Parabolic SAR** | Stop and Reverse indicator | Step: 0.02, Max: 0.2 |

### 🚀 Momentum Indicators
| Indicator | Description | Common Parameters |
|-----------|-------------|-------------------|
| **RSI** | Relative Strength Index | Period: 14 |
| **Stochastic** | Stochastic Oscillator | %K: 14, %D: 3, Smoothing: 3 |
| **CCI** | Commodity Channel Index | Period: 20 |
| **ROC** | Rate of Change | Period: 12 |
| **Williams %R** | Williams Percent Range | Period: 14 |
| **Awesome Oscillator** | Market momentum | Fast: 5, Slow: 34 |

### 📉 Volatility Indicators
| Indicator | Description | Common Parameters |
|-----------|-------------|-------------------|
| **Bollinger Bands** | Volatility bands | Period: 20, Std Dev: 2 |
| **ATR** | Average True Range | Period: 14 |
| **Keltner Channels** | Volatility-based envelopes | EMA: 20, ATR: 10, Multiplier: 2 |
| **Donchian Channels** | Price channel indicator | Period: 20 |
| **Chaikin Volatility** | Volatility indicator | Period: 10 |

### 📊 Volume Indicators
| Indicator | Description | Common Parameters |
|-----------|-------------|-------------------|
| **VWAP** | Volume Weighted Average Price | Session-based |
| **OBV** | On-Balance Volume | - |
| **CMF** | Chaikin Money Flow | Period: 20 |
| **MFI** | Money Flow Index | Period: 14 |
| **Volume SMA** | Volume Moving Average | Periods: 20, 50 |

### 🎯 Custom Indicators
| Indicator | Description | Common Parameters |
|-----------|-------------|-------------------|
| **Fibonacci Retracement** | Support/Resistance levels | High/Low points |
| **Pivot Points** | Key price levels | Daily/Weekly/Monthly |
| **Volume Profile** | Volume at price | Session-based |
| **Market Profile** | Time-price opportunities | - |
| **Order Flow** | Buy/Sell pressure | Tick data |

### Example: Using Indicators

```python
from market_analysis.indicators import TechnicalIndicators
import pandas as pd

# Sample price data
data = pd.DataFrame({
    'open': [100, 101, 102, 101, 103],
    'high': [102, 103, 104, 103, 105],
    'low': [99, 100, 101, 100, 102],
    'close': [101, 102, 103, 102, 104],
    'volume': [1000, 1200, 1500, 1300, 2000]
})

# Calculate RSI
data['RSI'] = TechnicalIndicators.rsi(data['close'], period=14)

# Calculate Bollinger Bands
data[['BB_upper', 'BB_middle', 'BB_lower']] = TechnicalIndicators.bollinger_bands(
    data['close'], 
    window=20, 
    window_dev=2
)

# Calculate MACD
macd_line, signal_line, macd_hist = TechnicalIndicators.macd(
    data['close'],
    fast_period=12,
    slow_period=26,
    signal_period=9
)

data['MACD'] = macd_line
data['MACD_Signal'] = signal_line
data['MACD_Hist'] = macd_hist
```

## 📊 Report Output

The tool generates comprehensive reports with the following components:

### 1. Executive Summary
- Market overview and key findings
- Top trading opportunities
- Risk assessment
- AI-generated insights

### 2. Technical Analysis
- Price action analysis
- Indicator signals
- Support/Resistance levels
- Volume analysis

### 3. Charts & Visualizations
- Interactive price charts with indicators
- Multiple timeframe analysis
- Custom drawing tools
- Export options

### 4. Risk Management
- Position sizing
- Stop-loss levels
- Risk-reward ratios
- Portfolio allocation

### 5. Backtest Results (if enabled)
- Performance metrics
- Trade statistics
- Equity curve
- Drawdown analysis

### Example Report Structure

```
reports/
├── AAPL_20231106/
│   ├── analysis_report.html     # Interactive HTML report
│   ├── summary.json            # JSON data export
│   └── charts/                 # Individual chart files
│       ├── price_chart.html
│       ├── rsi_chart.html
│       └── macd_chart.html
└── portfolio_analysis_20231106.pdf  # Portfolio report
```

## 📱 Sample Output

### Console Output
```
=== Market Analysis Report ===
Symbol: AAPL
Date: 2023-11-06
Last Price: $181.37 (+1.23%)

📈 Technical Indicators:
- RSI(14): 62.4 (Neutral)
- MACD: 1.23 (Bullish)
- SMA(50): $175.42 (Support)
- SMA(200): $165.89 (Support)

🎯 Key Levels:
- Support: $178.50, $175.00
- Resistance: $182.00, $185.00

🤖 AI Analysis:
- Bullish momentum is building with increasing volume
- Watch for breakout above $182.00 resistance
- Consider long entry on pullback to $179.00 with stop at $176.50

💡 Trade Idea:
- Entry: $179.00
- Stop Loss: $176.50 (-1.4%)
- Take Profit: $185.00 (+3.4%)
- Risk/Reward: 1:2.4

📊 Report saved to: reports/AAPL_20231106/
```

### Chart Example
![Sample Chart](https://via.placeholder.com/1200x600/2c3e50/ffffff?text=Interactive+Price+Chart+with+Indicators)

## Configuration

You can modify the following in `main.py`:
- `INDIAN_IT_STOCKS`: Add or remove stocks to analyze
- `get_stock_data()` parameters: Adjust the time period and interval
- `generate_technical_analysis()`: Modify technical indicator parameters

## Dependencies

### Core Dependencies
- Python 3.8+
- pandas: Data manipulation and analysis
- yfinance: Market data from Yahoo Finance
- pandas_ta: Technical analysis indicators
- mplfinance: Financial market data visualization
- matplotlib: Plotting library
- python-dotenv: Environment variable management
- groq: Groq API client for AI-powered insights

### Development Dependencies
- pytest: Testing framework
- black: Code formatter
- mypy: Static type checking
- flake8: Linting
- pytest-cov: Test coverage reporting

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- [pandas-ta](https://github.com/twopirllc/pandas-ta) for technical analysis indicators
- [yfinance](https://github.com/ranaroussis/yfinance) for market data
- [mplfinance](https://github.com/matplotlib/mplfinance) for financial charting
- [Groq](https://groq.com/) for AI-powered insights

## Disclaimer

This tool is for educational and informational purposes only. It should not be considered financial advice. Always do your own research and consult with a qualified financial advisor before making any investment decisions.