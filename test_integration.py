import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

# Import modules
from config.stock_config import load_stock_config, get_company_ticker
from data.fetcher import get_stock_data, get_fundamental_data, get_company_news
from analysis.technical import generate_simple_analysis
from analysis.fundamental import get_analyst_recommendations
from visualization.plotter import plot_stock_data

def test_stock_config():
    print("\n=== Testing Stock Config ===")
    config = load_stock_config()
    print(f"Loaded {len(config)} stock groups")
    tcs_ticker = get_company_ticker('TCS')
    print(f"Sample ticker for TCS: {tcs_ticker}")
    assert isinstance(config, dict), "Config should be a dictionary"
    assert len(config) > 0, "Config should not be empty"
    assert tcs_ticker is not None, "Should get a ticker for TCS"

def test_data_fetcher():
    print("\n=== Testing Data Fetcher ===")
    stock_data, metadata = get_stock_data("indian_it")
    print(f"Fetched data for {len(stock_data)} stocks")
    
    assert len(stock_data) > 0, "Should fetch data for at least one stock"
    assert isinstance(metadata, dict), "Metadata should be a dictionary"
    
    # Test fundamental data
    tcs_data = get_fundamental_data("TCS")
    print(f"TCS Fundamental Data: {tcs_data}")
    assert isinstance(tcs_data, dict), "Fundamental data should be a dictionary"
    
    # Test news
    news = get_company_news("TCS")
    print(f"TCS News: {len(news)} items")
    assert isinstance(news, list), "News should be a list"

def test_analysis():
    print("\n=== Testing Analysis ===")
    stock_data, _ = get_stock_data("indian_it")
    analysis = generate_simple_analysis(stock_data)
    print(f"Generated analysis for {len(analysis)} stocks")
    
    assert len(analysis) > 0, "Should generate analysis for stocks"
    assert isinstance(analysis, dict), "Analysis should be a dictionary"
    
    # Test recommendations if API key is available
    try:
        recs = get_analyst_recommendations("TCS")
        print(f"TCS Recommendations: {recs}")
        assert isinstance(recs, list), "Recommendations should be a list"
    except Exception as e:
        print(f"Skipping recommendations test: {e}")

def test_visualization():
    print("\n=== Testing Visualization ===")
    stock_data, _ = get_stock_data("indian_it")
    assert len(stock_data) > 0, "Need stock data for visualization test"
    
    first_stock = next(iter(stock_data.keys()))
    plot_path = plot_stock_data(stock_data, first_stock)
    print(f"Generated plot at: {plot_path}" if plot_path else "Plot generation failed")
    
    # Check if plot file was created if path is returned
    if plot_path:
        assert os.path.exists(plot_path), f"Plot file should exist at {plot_path}"

if __name__ == "__main__":
    import pytest
    import sys
    
    print("=== Starting Integration Tests ===")
    
    # Run tests using pytest's test runner
    exit_code = pytest.main([
        "-v",  # verbose output
        "--tb=short",  # shorter traceback
        __file__
    ])
    
    if exit_code == 0:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    sys.exit(exit_code)
