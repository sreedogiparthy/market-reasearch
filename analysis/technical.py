import pandas as pd
from typing import Dict, Any

def generate_simple_analysis(stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """
    Generate technical analysis for stock data
    
    Args:
        stock_data: Dictionary of company names to their respective DataFrames
        
    Returns:
        Dict containing technical analysis for each company
    """
    """Generate technical analysis using ta indicators"""
    analysis = {}
    
    for company, data in stock_data.items():
        try:
            if data.empty or len(data) < 2:
                analysis[company] = {"error": "Insufficient data"}
                continue
            
            # Get the latest row safely
            latest_row = data.iloc[-1]
            prev_row = data.iloc[-2] if len(data) > 1 else latest_row
            
            # Extract scalar values safely - using lowercase column names
            current_price = float(latest_row['close']) if 'close' in latest_row and not pd.isna(latest_row['close']) else 0.0
            prev_price = float(prev_row['close']) if 'close' in prev_row and not pd.isna(prev_row['close']) else current_price
            
            # Calculate price change
            price_change_pct = 0.0
            if prev_price != 0:
                price_change_pct = ((current_price - prev_price) / prev_price) * 100
            
            # Trend analysis using simple comparisons
            trend = "Neutral"
            try:
                sma_fast = float(latest_row.get('trend_sma_fast', current_price)) if 'trend_sma_fast' in latest_row and not pd.isna(latest_row['trend_sma_fast']) else current_price
                sma_slow = float(latest_row.get('trend_sma_slow', current_price)) if 'trend_sma_slow' in latest_row and not pd.isna(latest_row['trend_sma_slow']) else current_price
                
                if sma_fast > sma_slow:
                    trend = "Bullish"
                elif sma_fast < sma_slow:
                    trend = "Bearish"
            except Exception as e:
                print(f"Trend analysis error for {company}: {e}")
                trend = "Unknown"
            
            # RSI analysis
            rsi_signal = "Neutral"
            rsi_value = None
            try:
                rsi_value = float(latest_row['momentum_rsi']) if 'momentum_rsi' in latest_row and not pd.isna(latest_row['momentum_rsi']) else None
                if rsi_value is not None:
                    if rsi_value > 70:
                        rsi_signal = "Overbought"
                    elif rsi_value < 30:
                        rsi_signal = "Oversold"
            except Exception as e:
                print(f"RSI analysis error for {company}: {e}")
                rsi_signal = "Unknown"
            
            analysis[company] = {
                'Price': round(current_price, 2),
                'Change (%)': round(price_change_pct, 2),
                'Trend': trend,
                'RSI': round(rsi_value, 2) if rsi_value else None,
                'RSI Signal': rsi_signal
            }
            
        except Exception as e:
            print(f"Error analyzing {company}: {e}")
            analysis[company] = {"error": str(e)}
    
    return analysis

