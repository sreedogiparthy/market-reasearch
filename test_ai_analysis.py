"""Test script for AI-powered market analysis"""
from market_analysis import MarketAnalyzer

def main():
    print("🚀 Testing AI-Powered Market Analysis...")
    
    # Initialize with AI enabled
    analyzer = MarketAnalyzer(enable_ai=True)
    
    # Test with a single stock
    print("\n" + "="*50)
    print("ANALYZING RELIANCE.NS")
    print("="*50)
    
    result = analyzer.analyze_stock(
        symbol="RELIANCE.NS",
        period="1mo",  # Last month
        interval="1d",  # Daily data
        generate_trade_idea=True
    )
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main()
