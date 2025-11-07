import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Any

def plot_stock_data(stock_data: Dict[str, pd.DataFrame], company: str) -> Optional[str]:
    """Generate simple stock plots"""
    try:
        df = stock_data[company]
        if df.empty or len(df) < 30:
            return None
            
        # Create simple plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Price chart
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=2)
        if 'trend_sma_fast' in df.columns:
            ax1.plot(df.index, df['trend_sma_fast'], label='Fast SMA', alpha=0.7)
        if 'trend_sma_slow' in df.columns:
            ax1.plot(df.index, df['trend_sma_slow'], label='Slow SMA', alpha=0.7)
        ax1.set_title(f'{company} Price Chart')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RSI chart if available
        if 'momentum_rsi' in df.columns:
            ax2.plot(df.index, df['momentum_rsi'], label='RSI', color='purple')
            ax2.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought')
            ax2.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold')
            ax2.set_title('RSI (14)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        plot_path = f'plots/{company.replace(" ", "_").replace("&", "and")}_analysis.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"Plot error for {company}: {e}")
        return None
