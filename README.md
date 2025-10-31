# Market Research Tool

A Python-based tool for analyzing Indian stock market data with advanced technical indicators and AI-powered insights.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Features

- **Technical Analysis**: Comprehensive technical indicators including RSI, MACD, Bollinger Bands, Ichimoku Cloud, and moving averages
- **Visual Charts**: Automated generation of candlestick charts with volume and technical indicators
- **AI-Powered Insights**: Integration with Groq's LLM for intelligent market analysis
- **Multiple Timeframes**: Analyze daily, weekly, or monthly data
- **Top Stock Identification**: Automatically identifies and highlights stocks with strong technical setups
- **Robust Configuration**: Flexible and type-safe configuration management with validation
- **Portfolio Analysis**: Advanced portfolio risk metrics and correlation analysis

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/market-research.git
   cd market-research
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies using `uv` (recommended) or `pip`:
   ```bash
   uv add -r requirements.txt
   # or
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   - Copy `.env.sample` to `.env`
   - Add your Groq API key:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```

## Configuration

The application uses a JSON-based configuration system with built-in validation. The main configuration file is located at `config/app_config.json`.

### Configuration Structure

```json
{
  "analysis_settings": {
    "default_period": "1y",
    "technical_indicators": ["RSI", "MACD", "BBANDS", "SMA", "VWAP"],
    "risk_free_rate": 0.05,
    "cache_duration": 300,
    "max_retries": 3
  },
  "plot_settings": {
    "style": "fivethirtyeight",
    "figsize": [15, 8],
    "dpi": 100,
    "save_directory": "plots"
  },
  "risk_settings": {
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "high_volatility_threshold": 0.02,
    "volume_spike_threshold": 2.0
  }
}
```

### Configuration Management

The `ConfigManager` class provides a robust way to handle configurations:

```python
from config import ConfigManager

# Initialize config manager
config_manager = ConfigManager("config")

# Load configuration
try:
    config = config_manager.load_config("app_config.json")
    
    # Access nested values safely
    rsi_overbought = config_manager.get_config_value(
        config, "risk_settings.rsi_overbought", default=70
    )
    
    # Save configuration (creates backup automatically)
    config_manager.save_config(config, "app_config_backup.json")
    
except ConfigError as e:
    print(f"Configuration error: {e}")
```

## Usage

### Basic Usage

Run the analysis for all stock groups:
```bash
python main.py
```

### Advanced Usage

Analyze a specific stock group (e.g., nifty_50):
```bash
python main.py nifty_50
```

### Command Line Options

```bash
# Run with custom config file
python main.py --config custom_config.json

# Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
python main.py --log-level DEBUG

# Run in backtest mode
python main.py --backtest --start-date 2023-01-01 --end-date 2023-12-31
python main.py nifty_50

```

The script will:
1. Fetch the latest market data for major Indian IT stocks
2. Calculate technical indicators
3. Generate analysis reports
4. Save visual charts in the `plots` directory

## Technical Indicators

The tool supports a wide range of technical indicators:

### Trend Indicators
- **Moving Averages (SMA, EMA)**: 20, 50, and 200-day moving averages
- **MACD (Moving Average Convergence Divergence)**: Identifies trend changes and momentum
- **Ichimoku Cloud**: Comprehensive trend analysis with support/resistance levels
- **Parabolic SAR**: Identifies potential reversals in price movement

### Momentum Indicators
- **RSI (Relative Strength Index)**: Measures momentum and identifies overbought/oversold conditions (14-period default)
- **Stochastic Oscillator**: Identifies overbought/oversold conditions
- **CCI (Commodity Channel Index)**: Identifies cyclical trends
- **ROC (Rate of Change)**: Measures the percentage change in price

### Volatility Indicators
- **Bollinger Bands**: Shows volatility and potential price targets
- **ATR (Average True Range)**: Measures market volatility
- **Keltner Channels**: Volatility-based envelopes set above and below a moving average

### Volume Indicators
- **VWAP (Volume Weighted Average Price)**: Helps identify the true average price
- **OBV (On-Balance Volume)**: Measures buying and selling pressure
- **Volume SMA**: Simple moving average of volume

### Custom Indicators
- **Fibonacci Retracement**: Identifies potential support and resistance levels
- **Pivot Points**: Calculates support and resistance levels
- **Volume Profile**: Analyzes trading activity over specific price levels

## Output

The script generates:
- Console output with detailed technical analysis
- Visual charts in the `plots` directory including:
  - Candlestick charts with volume
  - RSI indicators
  - Moving averages
  - MACD signals

## Example Output

```
Analyzing Indian IT Sector with Technical Indicators...

Fetching market data...
Performing technical analysis...
Generating comprehensive market analysis...

[AI Analysis Output]
...

Technical analysis plots saved in the 'plots' directory:
- TCS: plots/TCS_analysis.png
- Infosys: plots/Infosys_analysis.png
- HCL Tech: plots/HCL_Tech_analysis.png
```

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