# Indian Market Research Tool

A Python-based tool for analyzing Indian stock market data with advanced technical indicators and AI-powered insights.

## Features

- **Technical Analysis**: Comprehensive technical indicators including RSI, MACD, Bollinger Bands, and moving averages
- **Visual Charts**: Automated generation of candlestick charts with volume and technical indicators
- **AI-Powered Insights**: Integration with Groq's LLM for intelligent market analysis
- **Multiple Timeframes**: Analyze daily, weekly, or monthly data
- **Top Stock Identification**: Automatically identifies and highlights stocks with strong technical setups

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/indian-market-research.git
   cd indian-market-research
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
   - Copy `.env.example` to `.env`
   - Add your Groq API key:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```

## Usage

Run the analysis:
```bash
python main.py
```

The script will:
1. Fetch the latest market data for major Indian IT stocks
2. Calculate technical indicators
3. Generate analysis reports
4. Save visual charts in the `plots` directory

## Technical Indicators

- **RSI (Relative Strength Index)**: Measures momentum and identifies overbought/oversold conditions
- **MACD (Moving Average Convergence Divergence)**: Identifies trend changes and momentum
- **Bollinger Bands**: Shows volatility and potential price targets
- **Moving Averages (20, 50, 200-day)**: Identifies trends and support/resistance levels
- **VWAP (Volume Weighted Average Price)**: Helps identify the true average price

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

- Python 3.8+
- See `requirements.txt` for complete list of Python packages

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is for educational and informational purposes only. It should not be considered financial advice. Always do your own research and consult with a qualified financial advisor before making any investment decisions.