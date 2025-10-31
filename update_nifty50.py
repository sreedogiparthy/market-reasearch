import json
import csv
from pathlib import Path

def load_existing_config():
    """Load existing stock configuration"""
    config_path = Path(__file__).parent / 'config' / 'stocks.json'
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config):
    """Save updated configuration back to file"""
    config_path = Path(__file__).parent / 'config' / 'stocks.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def process_nifty50_csv(csv_path):
    """Process Nifty 50 CSV and return a dictionary of stocks"""
    nifty50_stocks = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('Name', '').strip('"')
            if not name:
                continue
                
            # Skip header row if it exists
            if name == 'Name':
                continue
                
            # Extract the ticker symbol from the URL or name
            # This is a simple approach, you might need to adjust based on actual data
            ticker = name.replace(' ', '-').upper() + '.NS'
            
            # Add to nifty50_stocks
            nifty50_stocks[name] = {
                'symbol': ticker,
                'sector': 'Nifty 50'  # You can update this with actual sector if available
            }
    
    return nifty50_stocks

def main():
    # Load existing config
    config = load_existing_config()
    
    # Process Nifty 50 CSV
    csv_path = Path('/Users/sreenivasdogiparthy/Downloads/Nifty 50.csv')
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return
    
    print(f"Processing {csv_path}...")
    nifty50_stocks = process_nifty50_csv(csv_path)
    
    # Add to config
    config['nifty_50'] = nifty50_stocks
    
    # Save updated config
    save_config(config)
    print(f"Added {len(nifty50_stocks)} Nifty 50 stocks to config.")
    print("\nFirst few stocks added:")
    for name, info in list(nifty50_stocks.items())[:5]:
        print(f"- {name}: {info['symbol']}")
    print("\nRun 'python main.py nifty_50' to analyze Nifty 50 stocks.")

if __name__ == "__main__":
    main()
