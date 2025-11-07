"""
Generate comprehensive market analysis reports based on app_config.json
"""
import os
import json
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from ta import add_all_ta_features
from ta.trend import MACD, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
import numpy as np

# Set up paths
PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "config" / "app_config.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
oshow = True  # Set to False to display plots

# Ensure reports directory exists
REPORTS_DIR.mkdir(exist_ok=True)

# Load configuration
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

# Constants
SYMBOL = "AAPL"  # Default symbol, can be made configurable
TODAY = datetime.now().strftime("%Y-%m-%d")

class MarketAnalyzer:
    """Main class for market analysis and report generation"""
    
    def __init__(self, symbol=SYMBOL):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self.stock_data = None
        self.indicators = {}
        self.report = {
            'metadata': {
                'symbol': symbol,
                'date': TODAY,
                'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            'technical_analysis': {},
            'options_analysis': {},
            'risk_analysis': {},
            'recommendations': []
        }
    
    def fetch_data(self):
        """Fetch required market data"""
        print(f"Fetching data for {self.symbol}...")
        period = CONFIG["analysis_settings"]["default_period"]
        self.stock_data = self.ticker.history(period=period)
        
        # Add technical indicators
        self._add_technical_indicators()
        
        # Add options data if enabled
        if CONFIG["analysis_settings"]["options_analysis"]:
            self._add_options_data()
    
    def _add_technical_indicators(self):
        """Add technical indicators to the stock data"""
        print("Calculating technical indicators...")
        
        # Add all TA indicators if available
        try:
            self.stock_data = add_all_ta_features(
                self.stock_data, 
                open="Open", 
                high="High", 
                low="Low", 
                close="Close", 
                volume="Volume"
            )
        except Exception as e:
            print(f"Could not add all TA indicators: {e}")
            
        # Manually add indicators that might be missing
        if "RSI" in CONFIG["analysis_settings"]["technical_indicators"]:
            self.indicators['rsi'] = RSIIndicator(close=self.stock_data["Close"]).rsi()
        
        if "MACD" in CONFIG["analysis_settings"]["technical_indicators"]:
            macd = MACD(close=self.stock_data["Close"])
            self.indicators['macd'] = macd.macd()
            self.indicators['macd_signal'] = macd.macd_signal()
        
        if "BBANDS" in CONFIG["analysis_settings"]["technical_indicators"]:
            bb = BollingerBands(close=self.stock_data["Close"])
            self.indicators['bb_high'] = bb.bollinger_hband()
            self.indicators['bb_mid'] = bb.bollinger_mavg()
            self.indicators['bb_low'] = bb.bollinger_lband()
    
    def _add_options_data(self):
        """Add options data to the analysis"""
        print("Fetching options data...")
        try:
            # Get nearest expiration
            expirations = self.ticker.options
            if not expirations:
                print("No option expirations found")
                return
                
            expiry = expirations[0]  # Nearest expiration
            opt_chain = self.ticker.option_chain(expiry)
            
            # Calculate PCR
            total_put_vol = opt_chain.puts["volume"].sum()
            total_call_vol = opt_chain.calls["volume"].sum()
            pcr = total_put_vol / total_call_vol if total_call_vol > 0 else 0
            
            self.report['options_analysis'] = {
                'expiration': expiry,
                'put_call_ratio': round(pcr, 2),
                'total_put_volume': int(total_put_vol),
                'total_call_volume': int(total_call_vol),
                'sentiment': 'Bearish' if pcr > 1.0 else 'Bullish' if pcr < 0.7 else 'Neutral'
            }
            
        except Exception as e:
            print(f"Error fetching options data: {e}")
    
    def generate_technical_analysis(self):
        """Generate technical analysis report"""
        print("Generating technical analysis...")
        latest = self.stock_data.iloc[-1]
        
        # Price action
        price_change = (latest["Close"] - self.stock_data["Close"].iloc[0]) / self.stock_data["Close"].iloc[0] * 100
        price_trend = "Up" if price_change > 0 else "Down"
        
        # Volume analysis
        vol_avg = self.stock_data["Volume"].mean()
        vol_trend = "Above Average" if latest["Volume"] > vol_avg else "Below Average"
        
        # RSI analysis
        rsi_signal = ""
        if 'rsi' in self.indicators:
            rsi = self.indicators['rsi'].iloc[-1]
            if rsi > 70:
                rsi_signal = "Overbought"
            elif rsi < 30:
                rsi_signal = "Oversold"
            else:
                rsi_signal = "Neutral"
        
        self.report['technical_analysis'] = {
            'current_price': round(latest["Close"], 2),
            'price_change_pct': round(price_change, 2),
            'price_trend': price_trend,
            'volume_trend': vol_trend,
            'rsi_signal': rsi_signal,
            'indicators': {}
        }
        
        # Add indicator values
        for name, values in self.indicators.items():
            if not values.empty:
                self.report['technical_analysis']['indicators'][name] = round(float(values.iloc[-1]), 2)
    
    def generate_risk_analysis(self):
        """Generate risk analysis report"""
        print("Generating risk analysis...")
        risk_settings = CONFIG["risk_settings"]
        
        # Calculate ATR for stop loss
        atr = AverageTrueRange(
            high=self.stock_data["High"],
            low=self.stock_data["Low"],
            close=self.stock_data["Close"]
        ).average_true_range().iloc[-1]
        
        current_price = self.stock_data["Close"].iloc[-1]
        
        # Position sizing
        account_size = risk_settings["account_size"]
        risk_per_trade = account_size * risk_settings["risk_per_trade"]
        
        # Calculate position size based on ATR stop
        position_size = risk_per_trade / (atr * 2)  # Using 2x ATR as stop distance
        
        self.report['risk_analysis'] = {
            'atr': round(atr, 2),
            'suggested_stop_loss': round(current_price - (atr * 2), 2),
            'position_size': round(position_size, 2),
            'risk_per_trade': risk_per_trade,
            'max_portfolio_risk': account_size * risk_settings["max_portfolio_risk"]
        }
    
    def generate_recommendations(self):
        """Generate trading recommendations"""
        print("Generating recommendations...")
        ta = self.report['technical_analysis']
        risk = self.report['risk_analysis']
        
        # Price trend recommendation
        if ta['price_trend'] == 'Up':
            self.report['recommendations'].append(
                f"The price trend is UP by {ta['price_change_pct']}% in the given period. "
                "Consider looking for buying opportunities on pullbacks."
            )
        else:
            self.report['recommendations'].append(
                f"The price trend is DOWN by {abs(ta['price_change_pct'])}% in the given period. "
                "Be cautious with long positions."
            )
        
        # RSI recommendation
        if 'rsi' in ta['indicators']:
            rsi = ta['indicators']['rsi']
            if rsi > 70:
                self.report['recommendations'].append(
                    f"RSI at {rsi:.1f} indicates overbought conditions. "
                    "Consider taking profits or waiting for a pullback before entering new long positions."
                )
            elif rsi < 30:
                self.report['recommendations'].append(
                    f"RSI at {rsi:.1f} indicates oversold conditions. "
                    "This could be a buying opportunity if other indicators align."
                )
        
        # Options recommendation if available
        if self.report['options_analysis']:
            pcr = self.report['options_analysis']['put_call_ratio']
            if pcr > 1.5:
                self.report['recommendations'].append(
                    f"High Put/Call Ratio of {pcr:.2f} indicates bearish sentiment. "
                    "Traders are buying more puts than calls, which could indicate a potential reversal."
                )
            elif pcr < 0.7:
                self.report['recommendations'].append(
                    f"Low Put/Call Ratio of {pcr:.2f} indicates bullish sentiment. "
                    "Traders are buying more calls than puts, which could indicate continued upward momentum."
                )
        
        # Risk management recommendation
        self.report['recommendations'].append(
            f"Based on your risk settings, consider a position size of {risk['position_size']:.2f} shares "
            f"with a stop loss at ${risk['suggested_stop_loss']:.2f} to limit risk to ${risk['risk_per_trade']:.2f} per trade."
        )
    
    def generate_charts(self):
        """Generate interactive charts"""
        print("Generating charts...")
        
        # Create main price chart with volume
        fig = go.Figure()
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=self.stock_data.index,
            open=self.stock_data['Open'],
            high=self.stock_data['High'],
            low=self.stock_data['Low'],
            close=self.stock_data['Close'],
            name='Price'
        ))
        
        # Add volume
        fig.add_trace(go.Bar(
            x=self.stock_data.index,
            y=self.stock_data['Volume'],
            name='Volume',
            yaxis='y2',
            marker_color='rgba(100, 100, 200, 0.6)'
        ))
        
        # Add Bollinger Bands if available
        if 'bb_high' in self.indicators:
            fig.add_trace(go.Scatter(
                x=self.stock_data.index,
                y=self.indicators['bb_high'],
                name='BB Upper',
                line=dict(color='rgba(200, 100, 100, 0.7)')
            ))
            fig.add_trace(go.Scatter(
                x=self.stock_data.index,
                y=self.indicators['bb_mid'],
                name='BB Middle',
                line=dict(color='rgba(100, 200, 100, 0.7)')
            ))
            fig.add_trace(go.Scatter(
                x=self.stock_data.index,
                y=self.indicators['bb_low'],
                name='BB Lower',
                line=dict(color='rgba(100, 100, 200, 0.7)')
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{self.symbol} Price Chart',
            yaxis_title='Price',
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        # Save the figure
        chart_path = REPORTS_DIR / f'{self.symbol}_price_chart.html'
        fig.write_html(str(chart_path))
        self.report['charts'] = {
            'price_chart': str(chart_path.relative_to(PROJECT_ROOT))
        }
        
        # Create RSI chart if available
        if 'rsi' in self.indicators:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=self.stock_data.index,
                y=self.indicators['rsi'],
                name='RSI',
                line=dict(color='blue')
            ))
            
            # Add overbought/oversold lines
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                             annotation_text="Overbought", annotation_position="top right")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                             annotation_text="Oversold", annotation_position="bottom right")
            
            fig_rsi.update_layout(
                title=f'{self.symbol} RSI',
                yaxis_title='RSI',
                height=400
            )
            
            rsi_path = REPORTS_DIR / f'{self.symbol}_rsi.html'
            fig_rsi.write_html(str(rsi_path))
            self.report['charts']['rsi_chart'] = str(rsi_path.relative_to(PROJECT_ROOT))
    
    def save_report(self):
        """Save the analysis report to a file"""
        # Save JSON report
        report_path = REPORTS_DIR / f'{self.symbol}_report_{TODAY}.json'
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=2)
        
        # Save HTML report
        html_path = REPORTS_DIR / f'{self.symbol}_report_{TODAY}.html'
        self._generate_html_report(html_path)
        
        print(f"\nReports generated successfully:")
        print(f"- JSON Report: {report_path}")
        print(f"- HTML Report: {html_path}")
        print(f"- Charts saved in: {REPORTS_DIR}")
    
    def _generate_html_report(self, output_path):
        """Generate an HTML version of the report"""
        # Prepare variables for the template
        symbol = self.report['metadata']['symbol']
        date = self.report['metadata']['analysis_date']
        price = self.report['technical_analysis']['current_price']
        price_change = self.report['technical_analysis']['price_change_pct']
        price_class = 'positive' if price_change >= 0 else 'negative'
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Market Analysis Report - {symbol} - {date}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .card {{ background: #f9f9f9; border-left: 4px solid #3498db; margin: 10px 0; padding: 10px 15px; }}
                .recommendation {{ background: #e8f4fc; border-left: 4px solid #3498db; padding: 10px; margin: 10px 0; }}
                .positive {{ color: #27ae60; }}
                .negative {{ color: #e74c3c; }}
                .chart-container {{ margin: 30px 0; }}
                .summary {{ font-size: 1.1em; }}
            </style>
        </head>
        <body>
            <h1>Market Analysis Report</h1>
            <div class="summary">
                <h2>{symbol} - {date}</h2>
                <p>Current Price: <strong>${price:,.2f}</strong> 
                (<span class="{price_class}">
                {price_change:+.2f}%</span>)</p>
            </div>
            
            <h2>Technical Analysis</h2>
            <div class="card">
                <h3>Key Indicators</h3>
                <ul>
        """
        
        # Add technical indicators
        for name, value in self.report['technical_analysis'].get('indicators', {}).items():
            html += f"<li><strong>{name.upper()}:</strong> {value:.2f}</li>\n"
        
        html += """
                </ul>
            </div>
            
            <div class="chart-container">
                <h3>Price Chart</h3>
                <iframe src="{}" width="100%" height="600" frameborder="0"></iframe>
            </div>
        """.format(
            self.report['charts'].get('price_chart', '').replace('\\', '/')
        )
        
        # Add RSI chart if available
        if 'rsi_chart' in self.report['charts']:
            html += """
            <div class="chart-container">
                <h3>RSI</h3>
                <iframe src="{}" width="100%" height="450" frameborder="0"></iframe>
            </div>
            """.format(
                self.report['charts']['rsi_chart'].replace('\\', '/')
            )
        
        # Add options analysis if available
        if self.report['options_analysis']:
            html += """
            <h2>Options Analysis</h2>
            <div class="card">
                <h3>Put/Call Ratio: {pcr:.2f} - <span style="color: {color};">{sentiment}</span></h3>
                <p>Total Put Volume: {put_vol:,}</p>
                <p>Total Call Volume: {call_vol:,}</p>
                <p>Next Expiration: {expiry}</p>
            </div>
            """.format(
                pcr=self.report['options_analysis']['put_call_ratio'],
                sentiment=self.report['options_analysis']['sentiment'],
                color='red' if self.report['options_analysis']['put_call_ratio'] > 1.0 else 'green',
                put_vol=self.report['options_analysis']['total_put_volume'],
                call_vol=self.report['options_analysis']['total_call_volume'],
                expiry=self.report['options_analysis']['expiration']
            )
        
        # Add risk analysis
        html += """
        <h2>Risk Analysis</h2>
        <div class="card">
            <h3>Position Sizing</h3>
            <p>Account Size: ${account_size:,.2f}</p>
            <p>Risk per Trade: ${risk_per_trade:,.2f} ({risk_pct}% of account)</p>
            <p>Max Portfolio Risk: ${max_risk:,.2f} ({max_risk_pct}% of account)</p>
            <p>Suggested Stop Loss: ${stop_loss:,.2f} (2x ATR: {atr:.2f})</p>
            <p>Recommended Position Size: {position_size:,.2f} shares</p>
        </div>
        """.format(
            account_size=CONFIG["risk_settings"]["account_size"],
            risk_pct=CONFIG["risk_settings"]["risk_per_trade"] * 100,
            risk_per_trade=self.report['risk_analysis']['risk_per_trade'],
            max_risk=self.report['risk_analysis']['max_portfolio_risk'],
            max_risk_pct=CONFIG["risk_settings"]["max_portfolio_risk"] * 100,
            stop_loss=self.report['risk_analysis']['suggested_stop_loss'],
            atr=self.report['risk_analysis']['atr'],
            position_size=self.report['risk_analysis']['position_size']
        )
        
        # Add recommendations
        html += """
        <h2>Trading Recommendations</h2>
        """
        
        for rec in self.report['recommendations']:
            html += f"""
            <div class="recommendation">
                <p>{rec}</p>
            </div>
            """
        
        # Close HTML
        html += """
            <div style="margin-top: 40px; font-size: 0.9em; color: #7f8c8d; text-align: center;">
                <p>Report generated on {date} by Market Analysis Tool</p>
            </div>
        </body>
        </html>
        """.format(date=self.report['metadata']['analysis_date'])
        
        with open(output_path, 'w') as f:
            f.write(html)

def main():
    """Main function to generate the report"""
    print("=== Market Analysis Report Generator ===")
    
    # Initialize analyzer
    analyzer = MarketAnalyzer()
    
    # Fetch and process data
    analyzer.fetch_data()
    
    # Generate analysis
    analyzer.generate_technical_analysis()
    analyzer.generate_risk_analysis()
    analyzer.generate_recommendations()
    
    # Generate charts
    analyzer.generate_charts()
    
    # Save the report
    analyzer.save_report()
    
    print("\nReport generation complete!")

if __name__ == "__main__":
    main()
