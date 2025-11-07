# config/stock_config.py
import json
from pathlib import Path
from typing import Dict, Any

def load_stock_config() -> Dict[str, Any]:
    """Load stock configurations from JSON file"""
    try:
        config_path = Path(__file__).parent.parent / 'config' / 'stocks.json'
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Normalize configuration to ensure consistent structure
        normalized_config = {}
        
        for group_name, stocks in config_data.items():
            normalized_config[group_name] = {}
            
            for company, stock_info in stocks.items():
                if isinstance(stock_info, str):
                    # Convert simple string format to object format
                    normalized_config[group_name][company] = {
                        "symbol": stock_info,
                        "sector": "Unknown"
                    }
                elif isinstance(stock_info, dict):
                    normalized_config[group_name][company] = stock_info
                else:
                    print(f"Warning: Invalid format for {company} in {group_name}")
        
        return normalized_config
    except Exception as e:
        print(f"Error loading stock config: {e}")
        return {}

def get_company_ticker(company_name: str, stock_group: str = "indian_it") -> str:
    """
    Get the ticker symbol for a company from the config.
    
    Args:
        company_name: Name of the company
        stock_group: Stock group to look up (default: "indian_it")
        
    Returns:
        str: Ticker symbol for the company
    """
    config = load_stock_config()
    
    if stock_group not in config:
        return f"{company_name.split()[0]}.NS"  # Default format if group not found
        
    # Try exact match first
    if company_name in config[stock_group]:
        info = config[stock_group][company_name]
        if isinstance(info, dict):
            return info.get('symbol', f"{company_name.split()[0]}.NS")
        return info
    
    # Try case-insensitive match
    company_lower = company_name.lower()
    for name, info in config[stock_group].items():
        if name.lower() == company_lower:
            if isinstance(info, dict):
                return info.get('symbol', f"{name.split()[0]}.NS")
            return info
            
    # Default return if no match found
    return f"{company_name.split()[0]}.NS"