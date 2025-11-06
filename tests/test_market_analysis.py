"""
Comprehensive test suite for market analysis functionality.
Tests cover all features enabled in app_config.json
"""
import os
import json
import pytest
import pandas as pd
import yfinance as yf
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "app_config.json"

# Load the app configuration
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

# Test data
TEST_SYMBOL = "AAPL"  # Using Apple as test symbol
TEST_EXPIRY = None    # Will use nearest expiry

# Fixtures
@pytest.fixture(scope="module")
def stock_data():
    """Fixture to get stock data for testing"""
    ticker = yf.Ticker(TEST_SYMBOL)
    return ticker.history(period=CONFIG["analysis_settings"]["default_period"])

# Test cases for technical analysis
def test_technical_indicators(stock_data):
    """Test that all configured technical indicators can be calculated"""
    # Test each indicator in the configuration
    for indicator in CONFIG["analysis_settings"]["technical_indicators"]:
        if indicator == "RSI":
            from ta.momentum import RSIIndicator
            rsi = RSIIndicator(close=stock_data["Close"])
            rsi_values = rsi.rsi()
            assert not rsi_values.empty, f"RSI calculation failed for {TEST_SYMBOL}"
            
        elif indicator == "MACD":
            from ta.trend import MACD
            macd = MACD(close=stock_data["Close"])
            assert not macd.macd().empty, f"MACD calculation failed for {TEST_SYMBOL}"
            
        elif indicator == "BBANDS":
            from ta.volatility import BollingerBands
            bb = BollingerBands(close=stock_data["Close"])
            assert not bb.bollinger_hband().empty, "Bollinger Bands calculation failed"
            
        elif indicator == "SMA":
            from ta.trend import SMAIndicator
            sma = SMAIndicator(close=stock_data["Close"], window=20)
            assert not sma.sma_indicator().empty, "SMA calculation failed"
            
        elif indicator == "VWAP":
            from ta.volume import VolumeWeightedAveragePrice
            vwap = VolumeWeightedAveragePrice(
                high=stock_data["High"],
                low=stock_data["Low"],
                close=stock_data["Close"],
                volume=stock_data["Volume"],
                window=14
            )
            assert not vwap.volume_weighted_average_price().empty, "VWAP calculation failed"
            
        elif indicator == "ATR":
            from ta.volatility import AverageTrueRange
            atr = AverageTrueRange(
                high=stock_data["High"],
                low=stock_data["Low"],
                close=stock_data["Close"]
            )
            assert not atr.average_true_range().empty, "ATR calculation failed"
            
        elif indicator == "ICHIMOKU":
            from ta.trend import IchimokuIndicator
            ichimoku = IchimokuIndicator(
                high=stock_data["High"],
                low=stock_data["Low"]
            )
            assert not ichimoku.ichimoku_a().empty, "Ichimoku calculation failed"

# Test cases for options analysis
@pytest.mark.skipif(not CONFIG["analysis_settings"]["options_analysis"], 
                  reason="Options analysis is disabled in config")
def test_options_analysis():
    """Test options analysis functionality"""
    ticker = yf.Ticker(TEST_SYMBOL)
    
    # Get options chain
    expirations = ticker.options
    assert len(expirations) > 0, f"No option expirations found for {TEST_SYMBOL}"
    
    # Get nearest expiration if not specified
    expiry = expirations[0] if TEST_EXPIRY is None else TEST_EXPIRY
    
    # Get options chain
    opt_chain = ticker.option_chain(expiry)
    
    # Test calls and puts
    assert not opt_chain.calls.empty, f"No call options found for {TEST_SYMBOL}"
    assert not opt_chain.puts.empty, f"No put options found for {TEST_SYMBOL}"
    
    # Test PCR if enabled in config
    if CONFIG["options_settings"].get("pcr_analysis", False):
        pcr = opt_chain.puts["volume"].sum() / opt_chain.calls["volume"].sum()
        assert isinstance(pcr, float), "Put-Call Ratio calculation failed"

# Test cases for risk management
@pytest.mark.skipif(not CONFIG["analysis_settings"]["risk_management"], 
                  reason="Risk management is disabled in config")
def test_risk_management(stock_data):
    """Test risk management calculations"""
    risk_settings = CONFIG["risk_settings"]
    
    # Test position sizing
    if risk_settings.get("position_sizing_method") == "kelly":
        # Simple Kelly Criterion test
        win_rate = 0.6  # Example win rate
        win_loss_ratio = 1.5  # Example win/loss ratio
        kelly_f = win_rate - ((1 - win_rate) / win_loss_ratio)
        assert 0 <= kelly_f <= 1, "Invalid Kelly fraction"
    
    # Test stop loss calculation
    if risk_settings.get("stop_loss_method") == "atr":
        from ta.volatility import AverageTrueRange
        atr = AverageTrueRange(
            high=stock_data["High"],
            low=stock_data["Low"],
            close=stock_data["Close"]
        ).average_true_range().iloc[-1]
        assert atr > 0, "ATR-based stop loss calculation failed"

# Test cases for backtesting
@pytest.mark.skipif(not CONFIG["analysis_settings"]["backtesting"], 
                  reason="Backtesting is disabled in config")
def test_backtesting(stock_data):
    """Test backtesting functionality"""
    # Simple backtest: Buy and hold strategy
    initial_capital = CONFIG["backtest_settings"]["initial_capital"]
    commission = CONFIG["backtest_settings"]["commission"]
    
    # Calculate returns
    returns = stock_data["Close"].pct_change().dropna()
    
    # Apply commission (simplified)
    returns = returns - (2 * commission)  # Round-trip commission
    
    # Calculate equity curve
    equity = (1 + returns).cumprod() * initial_capital
    
    # Basic assertions
    assert len(equity) > 0, "Backtest failed to generate equity curve"
    assert equity.iloc[-1] != initial_capital, "Equity curve shows no change"

# Test cases for intraday analysis
@pytest.mark.skipif(not CONFIG["analysis_settings"]["intraday_analysis"], 
                  reason="Intraday analysis is disabled in config")
def test_intraday_analysis():
    """Test intraday data fetching and analysis"""
    # Fetch 1-day intraday data (5-minute intervals)
    ticker = yf.Ticker(TEST_SYMBOL)
    intraday_data = ticker.history(period="1d", interval="5m")
    
    # Basic assertions
    assert not intraday_data.empty, f"Failed to fetch intraday data for {TEST_SYMBOL}"
    assert len(intraday_data) > 1, "Insufficient intraday data points"
    
    # Check required columns
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_columns:
        assert col in intraday_data.columns, f"Missing required column: {col}"

# Test configuration validation
def test_config_validation():
    """Validate the application configuration"""
    # Check required sections
    required_sections = ["analysis_settings", "options_settings", 
                        "risk_settings", "backtest_settings"]
    for section in required_sections:
        assert section in CONFIG, f"Missing required config section: {section}"
    
    # Check required analysis settings
    required_analysis_settings = ["default_period", "technical_indicators", 
                                 "options_analysis", "intraday_analysis", 
                                 "risk_management", "backtesting"]
    for setting in required_analysis_settings:
        assert setting in CONFIG["analysis_settings"], f"Missing analysis setting: {setting}"
    
    # Check that at least one technical indicator is configured
    assert len(CONFIG["analysis_settings"]["technical_indicators"]) > 0, \
        "No technical indicators configured"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
