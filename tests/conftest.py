"""Shared test configuration and fixtures"""
import pytest
import yfinance as yf
import pandas as pd
from pathlib import Path
import json

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "app_config.json"

# Load the app configuration
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

# Test configuration
TEST_SYMBOL = "AAPL"  # Using Apple as test symbol
TEST_EXPIRY = None    # Will use nearest expiry

@pytest.fixture(scope="module")
def stock_data():
    """Fixture to get stock data for testing"""
    ticker = yf.Ticker(TEST_SYMBOL)
    return ticker.history(period=CONFIG["analysis_settings"]["default_period"])

@pytest.fixture(scope="module")
def options_chain():
    """Fixture to get options chain data for testing"""
    if not CONFIG["analysis_settings"]["options_analysis"]:
        pytest.skip("Options analysis is disabled in config")
        
    ticker = yf.Ticker(TEST_SYMBOL)
    expirations = ticker.options
    if not expirations:
        pytest.skip(f"No option expirations found for {TEST_SYMBOL}")
        
    expiry = expirations[0] if TEST_EXPIRY is None else TEST_EXPIRY
    opt_chain = ticker.option_chain(expiry)
    return opt_chain

@pytest.fixture(scope="module")
def intraday_data():
    """Fixture to get intraday data for testing"""
    if not CONFIG["analysis_settings"]["intraday_analysis"]:
        pytest.skip("Intraday analysis is disabled in config")
        
    ticker = yf.Ticker(TEST_SYMBOL)
    data = ticker.history(period="1d", interval="5m")
    if data.empty:
        pytest.skip(f"No intraday data available for {TEST_SYMBOL}")
    return data

@pytest.fixture(scope="module")
def config():
    """Fixture to access the app configuration"""
    return CONFIG
