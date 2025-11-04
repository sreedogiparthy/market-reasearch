# Market Analysis Tool with AI Insights

A powerful Python-based tool for technical analysis of stocks with AI-powered insights using GROQ's language models.

## 📋 Features

- **Multiple Timeframe Analysis**: Analyze stocks across different timeframes (intraday, daily, weekly)
- **Technical Indicators**: Built-in calculation of key indicators (SMA, RSI, etc.)
- **AI-Powered Analysis**: Get AI-generated market insights and trade ideas
- **Visual Charts**: Interactive price charts with technical indicators
- **Multi-Stock Analysis**: Analyze groups of stocks with a single command
- **Customizable**: Configure stock groups and analysis parameters

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Environment**:
   - Create a `.env` file with your GROQ API key:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     GROQ_MODEL=llama-3.1-8b-instant
     ```

3. **Run Analysis**:
   ```bash
   # Basic analysis of NIFTY 50 stocks (daily data)
   python market_analysis.py nifty_50
   
   # Intraday analysis (15-minute candles)
   python market_analysis.py nifty_50 --interval 15m
   
   # With AI-powered trade ideas
   python market_analysis.py nifty_50 --trade-ideas
   
   # Custom period and interval
   python market_analysis.py nifty_50 --period 30d --interval 1h
   ```

## 📊 Sample Output

### 1. Stock Analysis Summary
```
========================================================================================================================
                                       📊 INTRADAY TRADING REPORT - 15M Timeframe                                        
========================================================================================================================
Symbol   Price      Change %   20D MA     50D MA     RSI      Trend        Momentum     Rec.     Conf.   
------------------------------------------------------------------------------------------------------------------------
RELIANCE.NS 1473.10    0.07%      1482.68    1486.43    30.19    Bearish      Neutral      Consider Buy 80%     
HDFCBANK.NS 985.70     0.17%      991.85     991.47     31.89    Bullish      Neutral      Consider Buy 80%     
LT.NS    3922.50    -0.13%     3939.91    3974.80    24.49    Bearish      Oversold     Consider Buy 80%     
```

### 2. AI Analysis Example
```
🤖 AI Analysis:

📊 Technical Summary for LT.NS:
Current Price: 3922.5
20-Day MA: 3939.91
50-Day MA: 3974.80
RSI: 24.49
Trend: Bearish
Momentum: Oversold

Analysis:
- Strong oversold conditions (RSI 24.49)
- Bearish trend but potential for reversal
- Support at 3840, Resistance at 3950

Recommendation: Consider Buy (Confidence: 80%)
```

### 3. Sample Chart
![Sample Chart](https://via.placeholder.com/800x400.png?text=Price+Chart+with+Indicators)
*(Actual chart will show price, moving averages, and RSI)*

## 🛠 Configuration

### Stock Groups
Edit `config/stocks.json` to add or modify stock groups:
```json
{
  "nifty_50": {
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "INFY": "INFY.NS",
    "TCS": "TCS.NS"
  },
  "banking": {
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS"
  }
}
```

### Analysis Parameters
- `--period`: Data period (default: 1y)
  - Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
- `--interval`: Data interval (default: 1d)
  - Options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
- `--trade-ideas`: Enable AI-generated trade ideas
- `--no-ai`: Disable AI analysis

## 📈 Technical Indicators
- **SMA (Simple Moving Average)**: 20 & 50 periods
- **RSI (Relative Strength Index)**: 14 periods
- **Price Change %**: Daily/interval change
- **Trend**: Bullish/Bearish based on MA crossovers
- **Momentum**: Overbought/Oversold based on RSI

## 🤖 AI Analysis Features
- Market condition analysis
- Support/Resistance levels
- Trading recommendations
- Risk assessment
- Trade ideas (when enabled)

## 📝 Notes
- For intraday analysis, the tool automatically adjusts the period based on Yahoo Finance's limitations
- AI analysis requires a valid GROQ API key
- All recommendations are for educational purposes only

## 📄 License
This project is for educational purposes only. Not financial advice.
