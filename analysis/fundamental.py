from typing import List, Dict, Optional

def get_analyst_recommendations(symbol: str) -> List[Dict]:
    """Get analyst recommendations with fallback to empty list"""
    if finnhub_client is None:
        return []
    try:
        return finnhub_client.recommendation_trends(f"{symbol}.NS")
    except Exception as e:
        print(f"Error getting recommendations for {symbol}: {e}")
        return []
