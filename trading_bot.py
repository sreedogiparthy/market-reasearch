# trading_bot.py
import time
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class TradeSignal:
    symbol: str
    action: str  # 'BUY', 'SELL', or 'HOLD'
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    indicators: Optional[dict] = None
    timestamp: str = datetime.now().isoformat()

class TradingBot:
    def __init__(self, config_path: str = 'config/app_config.json'):
        self.config = self._load_config(config_path)
        self.active_positions = {}  # Track open positions
        self.watchlist = self._load_watchlist()
        
    def _load_config(self, config_path: str) -> dict:
        """Load trading configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return {
                'analysis_settings': {
                    'default_period': '1d',
                    'interval': '5m',
                    'rsi_period': 14,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30,
                    'atr_period': 14,
                    'atr_multiplier': 2.0
                }
            }
    
    def _load_watchlist(self) -> List[Dict]:
        """Load watchlist from stocks.json"""
        try:
            with open('config/stocks.json', 'r') as f:
                stocks = json.load(f)
                # Flatten all stocks from different groups
                watchlist = []
                for group in stocks.values():
                    for name, info in group.items():
                        watchlist.append({
                            'name': name,
                            'symbol': info['symbol'] if isinstance(info, dict) else info,
                            'sector': info.get('sector', 'Unknown') if isinstance(info, dict) else 'Unknown'
                        })
                return watchlist
        except Exception as e:
            logging.error(f"Error loading watchlist: {e}")
            return []

    def get_historical_data(self, symbol: str, period: str = '1d', interval: str = '5m') -> pd.DataFrame:
        """Fetch historical price data"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)
            if df.empty:
                logging.warning(f"No data for {symbol}")
                return pd.DataFrame()
            return df
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        if df.empty:
            return df
            
        # RSI
        df['rsi'] = ta.rsi(df['Close'], length=self.config['analysis_settings'].get('rsi_period', 14))
        
        # ATR for volatility and stop loss
        atr = ta.atr(df['High'], df['Low'], df['Close'], 
                    length=self.config['analysis_settings'].get('atr_period', 14))
        df['atr'] = atr
        
        # Moving Averages
        df['sma20'] = ta.sma(df['Close'], length=20)
        df['sma50'] = ta.sma(df['Close'], length=50)
        
        # MACD
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)
        
        return df

    def generate_signal(self, symbol: str) -> Optional[TradeSignal]:
        """Generate trading signal for a symbol"""
        try:
            # Get data
            df = self.get_historical_data(
                symbol,
                period=self.config['analysis_settings'].get('default_period', '1d'),
                interval=self.config['analysis_settings'].get('interval', '5m')
            )
            
            if df.empty:
                return None
                
            # Calculate indicators
            df = self.calculate_indicators(df)
            if df.empty:
                return None
                
            latest = df.iloc[-1]
            
            # Initialize signal
            signal = TradeSignal(
                symbol=symbol,
                action='HOLD',
                price=latest['Close'],
                indicators={}
            )
            
            # RSI Analysis
            rsi = latest.get('rsi', 50)
            if pd.notna(rsi):
                signal.indicators['rsi'] = rsi
                if rsi < self.config['analysis_settings'].get('rsi_oversold', 30):
                    signal.action = 'BUY'
                    signal.confidence += 0.3
                elif rsi > self.config['analysis_settings'].get('rsi_overbought', 70):
                    signal.action = 'SELL'
                    signal.confidence += 0.3
            
            # Moving Average Crossover
            if 'sma20' in latest and 'sma50' in latest:
                sma20 = latest['sma20']
                sma50 = latest['sma50']
                signal.indicators.update({'sma20': sma20, 'sma50': sma50})
                
                if pd.notna(sma20) and pd.notna(sma50):
                    if sma20 > sma50 and signal.action == 'BUY':
                        signal.confidence += 0.2
                    elif sma20 < sma50 and signal.action == 'SELL':
                        signal.confidence += 0.2
            
            # MACD Signal
            if 'MACD_12_26_9' in latest and 'MACDs_12_26_9' in latest:
                macd = latest['MACD_12_26_9']
                signal_line = latest['MACDs_12_26_9']
                signal.indicators.update({'macd': macd, 'signal': signal_line})
                
                if pd.notna(macd) and pd.notna(signal_line):
                    if macd > signal_line and signal.action == 'BUY':
                        signal.confidence += 0.2
                    elif macd < signal_line and signal.action == 'SELL':
                        signal.confidence += 0.2
            
            # Set stop loss and take profit based on ATR
            atr = latest.get('atr')
            if pd.notna(atr):
                atr_multiplier = self.config['analysis_settings'].get('atr_multiplier', 2.0)
                if signal.action == 'BUY':
                    signal.stop_loss = latest['Close'] - (atr * atr_multiplier)
                    signal.take_profit = latest['Close'] + (atr * atr_multiplier * 1.5)
                elif signal.action == 'SELL':
                    signal.stop_loss = latest['Close'] + (atr * atr_multiplier)
                    signal.take_profit = latest['Close'] - (atr * atr_multiplier * 1.5)
            
            # Only return signals with sufficient confidence
            if signal.confidence >= 0.5 and signal.action != 'HOLD':
                return signal
            return None
                
        except Exception as e:
            logging.error(f"Error generating signal for {symbol}: {e}")
            return None

    def monitor_market(self, interval: int = 300):
        """Monitor market and generate signals"""
        try:
            logging.info("Starting market monitoring...")
            
            while True:
                signals = []
                logging.info(f"Checking market at {datetime.now()}")
                
                for stock in self.watchlist:
                    symbol = stock['symbol']
                    try:
                        signal = self.generate_signal(symbol)
                        if signal and signal.action != 'HOLD':
                            signals.append(signal)
                            logging.info(f"Signal generated for {symbol}: {signal.action} at {signal.price}")
                    except Exception as e:
                        logging.error(f"Error processing {symbol}: {e}")
                
                # Process signals if any
                if signals:
                    self.process_signals(signals)
                    logging.info(f"Processed {len(signals)} signals. Next check in {interval} seconds...")
                
                # Wait for the next interval
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logging.info("Market monitoring stopped by user")
        except Exception as e:
            logging.error(f"Error in market monitoring: {e}")
        finally:
            logging.info("Market monitoring stopped")

    def log_signal(self, signal: TradeSignal):
        """Log trading signal"""
        log_msg = (
            f"\n{'='*50}\n"
            f"🚀 {signal.action} Signal for {signal.symbol}\n"
            f"📈 Price: {signal.price:.2f}\n"
            f"🎯 Stop Loss: {signal.stop_loss:.2f}\n"
            f"🎯 Take Profit: {signal.take_profit:.2f}\n"
            f"📊 Confidence: {signal.confidence*100:.1f}%\n"
            f"📊 Indicators: {json.dumps(signal.indicators, indent=2, default=str)}\n"
            f"⏰ {signal.timestamp}\n"
            f"{'='*50}\n"
        )
        logging.info(log_msg)

    def process_signals(self, signals: List[TradeSignal]):
        """Process generated trading signals"""
        for signal in signals:
            try:
                if signal.action == 'BUY' and signal.symbol not in self.active_positions:
                    # Execute buy order
                    self.execute_order(signal)
                    self.active_positions[signal.symbol] = {
                        'entry_price': signal.price,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'entry_time': datetime.now().isoformat()
                    }
                    logging.info(f"Opened position in {signal.symbol} at {signal.price}")
                    
                elif signal.action == 'SELL' and signal.symbol in self.active_positions:
                    # Execute sell order
                    position = self.active_positions[signal.symbol]
                    pnl = (signal.price - position['entry_price']) / position['entry_price'] * 100
                    self.execute_order(signal)
                    del self.active_positions[signal.symbol]
                    logging.info(
                        f"Closed position in {signal.symbol} at {signal.price} "
                        f"(PnL: {pnl:.2f}%)"
                    )
                    
            except Exception as e:
                logging.error(f"Error processing signal for {signal.symbol}: {e}")

    def execute_order(self, signal: TradeSignal):
        """Execute trading order (placeholder for actual trading API)"""
        # In a real implementation, this would connect to your broker's API
        order_details = {
            'symbol': signal.symbol,
            'action': signal.action,
            'price': signal.price,
            'timestamp': datetime.now().isoformat(),
            'status': 'EXECUTED'
        }
        logging.info(f"Order executed: {order_details}")

class RiskManager:
    def __init__(self, max_portfolio_risk: float = 0.02, max_position_size: float = 0.1):
        self.max_portfolio_risk = max_portfolio_risk  # 2% max portfolio risk per trade
        self.max_position_size = max_position_size    # 10% max per position
    
    def calculate_position_size(self, signal: TradeSignal, portfolio_value: float) -> float:
        """Calculate position size based on risk"""
        risk_per_share = abs(signal.price - signal.stop_loss) if signal.stop_loss else 0
        if risk_per_share <= 0:
            return 0
            
        max_risk_amount = portfolio_value * self.max_portfolio_risk
        position_size = max_risk_amount / risk_per_share
        
        # Cap position size
        max_position_value = portfolio_value * self.max_position_size
        max_shares = max_position_value / signal.price
        
        return min(position_size, max_shares)

class PaperTrading:
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.commission = commission
    
    def execute_paper_trade(self, signal: TradeSignal, shares: int):
        commission_rate = self.commission
        
        if signal.action == 'BUY':
            cost = shares * signal.price
            commission_cost = cost * commission_rate
            total_cost = cost + commission_cost
            
            if total_cost <= self.capital:
                self.capital -= total_cost
                self.positions[signal.symbol] = {
                    'shares': shares,
                    'entry_price': signal.price,
                    'entry_time': datetime.now(),
                    'total_cost': total_cost
                }
                
                self.trade_history.append({
                    'symbol': signal.symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': signal.price,
                    'commission': commission_cost,
                    'total_cost': total_cost,
                    'timestamp': datetime.now()
                })
                
        elif signal.action == 'SELL' and signal.symbol in self.positions:
            position = self.positions[signal.symbol]
            proceeds = shares * signal.price
            commission_cost = proceeds * commission_rate
            net_proceeds = proceeds - commission_cost
            
            # Calculate PnL
            pnl = net_proceeds - position['total_cost']
            
            trade_record = {
                'symbol': signal.symbol,
                'action': 'SELL',
                'shares': shares,
                'entry_price': position['entry_price'],
                'exit_price': signal.price,
                'commission': commission_cost,
                'proceeds': net_proceeds,
                'pnl': pnl,
                'timestamp': datetime.now()
            }
            self.trade_history.append(trade_record)
            
            # Update capital and remove position
            self.capital += net_proceeds
            del self.positions[signal.symbol]
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        total_value = self.capital
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total_value += position['shares'] * current_prices[symbol]
        return total_value

def validate_signal(self, signal: TradeSignal, df: pd.DataFrame) -> bool:
    """Validate signal with additional checks"""
    
    # Volume check
    if 'Volume' in df.columns:
        avg_volume = df['Volume'].tail(20).mean()
        current_volume = df['Volume'].iloc[-1]
        if current_volume < avg_volume * 0.7:  # Low volume
            return False
    
    # Trend confirmation
    if len(df) >= 50:
        price_trend = df['Close'].tail(20).mean() > df['Close'].tail(50).mean()
        if signal.action == 'BUY' and not price_trend:
            return False
        if signal.action == 'SELL' and price_trend:
            return False
    
    # Multiple timeframe confirmation
    hourly_data = self.get_historical_data(signal.symbol, period='5d', interval='1h')
    if not hourly_data.empty:
        hourly_rsi = ta.rsi(hourly_data['Close'], length=14).iloc[-1]
        if signal.action == 'BUY' and hourly_rsi > 60:
            return False
        if signal.action == 'SELL' and hourly_rsi < 40:
            return False
    
    return True
class PerformanceTracker:
    def __init__(self):
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'largest_win': 0,
            'largest_loss': 0
        }
    
    def update_metrics(self, trade_record: dict):
        self.metrics['total_trades'] += 1
        self.metrics['total_pnl'] += trade_record['pnl']
        
        if trade_record['pnl'] > 0:
            self.metrics['winning_trades'] += 1
            self.metrics['largest_win'] = max(self.metrics['largest_win'], trade_record['pnl'])
        else:
            self.metrics['losing_trades'] += 1
            self.metrics['largest_loss'] = min(self.metrics['largest_loss'], trade_record['pnl'])
    
    def get_performance_summary(self):
        win_rate = (self.metrics['winning_trades'] / self.metrics['total_trades'] * 100 
                   if self.metrics['total_trades'] > 0 else 0)
        
        return {
            'win_rate': f"{win_rate:.1f}%",
            'total_pnl': f"${self.metrics['total_pnl']:.2f}",
            'profit_factor': abs(self.metrics['largest_win'] / self.metrics['largest_loss']) 
                           if self.metrics['largest_loss'] != 0 else float('inf'),
            'total_trades': self.metrics['total_trades']
        }
def get_market_regime(self) -> str:
    """Detect current market regime"""
    try:
        # Use SPY as market proxy
        spy_data = self.get_historical_data('SPY', period='3mo', interval='1d')
        if spy_data.empty:
            return "UNKNOWN"
        
        # Calculate volatility
        returns = spy_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Calculate trend
        sma_20 = spy_data['Close'].rolling(20).mean().iloc[-1]
        sma_50 = spy_data['Close'].rolling(50).mean().iloc[-1]
        
        if volatility > 0.25:
            return "HIGH_VOLATILITY"
        elif sma_20 > sma_50:
            return "BULL_MARKET"
        else:
            return "BEAR_MARKET"
            
    except Exception as e:
        logging.error(f"Error detecting market regime: {e}")
        return "UNKNOWN"

class EnhancedTradingBot(TradingBot):
    def __init__(self, config_path: str = 'config/app_config.json'):
        super().__init__(config_path)
        self.risk_manager = RiskManager()
        self.paper_trader = PaperTrading()
        self.performance_tracker = PerformanceTracker()
        self.consecutive_losses = 0

    def calculate_daily_pnl(self) -> float:
        """Calculate today's PnL"""
        try:
            today = datetime.now().date()
            today_trades = [t for t in self.paper_trader.trade_history 
                          if t['timestamp'].date() == today]
            return sum(trade['pnl'] for trade in today_trades)
        except Exception as e:
            logging.error(f"Error calculating daily PnL: {e}")
            return 0.0
    
    def calculate_positions_value(self) -> float:
        """Calculate current value of all positions"""
        total_value = 0.0
        for symbol, position in self.paper_trader.positions.items():
            # Get current price (simplified - in real implementation, fetch current price)
            current_price = position['entry_price']  # Placeholder
            total_value += position['shares'] * current_price
        return total_value
    
    def validate_signal(self, signal: TradeSignal, df: pd.DataFrame) -> bool:
        """Validate signal with additional checks"""
        # Volume check
        if 'Volume' in df.columns and len(df) >= 20:
            avg_volume = df['Volume'].tail(20).mean()
            current_volume = df['Volume'].iloc[-1]
            if current_volume < avg_volume * 0.7:  # Low volume
                return False
        
        # Trend confirmation
        if len(df) >= 50:
            price_trend = df['Close'].tail(20).mean() > df['Close'].tail(50).mean()
            if signal.action == 'BUY' and not price_trend:
                return False
            if signal.action == 'SELL' and price_trend:
                return False
        
        return True
    
    def get_market_regime(self) -> str:
        """Detect current market regime"""
        try:
            # Use Nifty 50 as market proxy for Indian stocks
            nifty_data = self.get_historical_data('^NSEI', period='3mo', interval='1d')
            if nifty_data.empty:
                return "UNKNOWN"
            
            # Calculate volatility
            returns = nifty_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate trend
            if len(nifty_data) >= 50:
                sma_20 = nifty_data['Close'].rolling(20).mean().iloc[-1]
                sma_50 = nifty_data['Close'].rolling(50).mean().iloc[-1]
                
                if volatility > 0.25:
                    return "HIGH_VOLATILITY"
                elif sma_20 > sma_50:
                    return "BULL_MARKET"
                else:
                    return "BEAR_MARKET"
            return "UNKNOWN"
                
        except Exception as e:
            logging.error(f"Error detecting market regime: {e}")
            return "UNKNOWN"

    def should_trade(self) -> bool:
        """Check if trading should proceed"""
        # Check market hours
        if self.config['trading_settings'].get('trading_hours_only', True):
            now = datetime.now().time()
            market_open = datetime.strptime(
                self.config['trading_settings'].get('market_open', '09:30'), 
                '%H:%M'
            ).time()
            market_close = datetime.strptime(
                self.config['trading_settings'].get('market_close', '16:00'), 
                '%H:%M'
            ).time()
            
            if not (market_open <= now <= market_close):
                return False
        
        # Check daily loss limit
        daily_pnl = self.calculate_daily_pnl()
        max_daily_loss = self.config['risk_management'].get('max_daily_loss', 0.05)
        if daily_pnl < -max_daily_loss:
            logging.warning("Daily loss limit reached. Stopping trading for today.")
            return False
            
        # Check consecutive losses
        max_consecutive = self.config['risk_management'].get('max_consecutive_losses', 3)
        if self.consecutive_losses >= max_consecutive:
            logging.warning("Max consecutive losses reached. Taking a break.")
            return False
            
        return True
    
    def process_signals(self, signals: List[TradeSignal]):
        """Enhanced signal processing"""
        if not self.should_trade():
            return
            
        market_regime = self.get_market_regime()
        logging.info(f"Current market regime: {market_regime}")
        
        for signal in signals:
            try:
                # Adjust strategy based on market regime
                if market_regime == "HIGH_VOLATILITY":
                    signal.confidence *= 0.8  # Reduce confidence in high volatility
                
                # Validate signal
                df = self.get_historical_data(signal.symbol, period='1d', interval='5m')
                if not self.validate_signal(signal, df):
                    continue
                
                # Calculate position size
                portfolio_value = self.paper_trader.capital + self.calculate_positions_value()
                shares = self.risk_manager.calculate_position_size(signal, portfolio_value)
                
                if shares > 0:
                    self.paper_trader.execute_paper_trade(signal, int(shares))
                    self.log_signal(signal, shares)
                    
            except Exception as e:
                logging.error(f"Error processing signal for {signal.symbol}: {e}")


class EnhancedMovingAverageCrossover:
    """Enhanced Moving Average Crossover Strategy with RSI and Volume Filters"""
    
    def __init__(self, short_window=20, long_window=50, rsi_period=14, volume_threshold=1.2):
        """
        Initialize the strategy with parameters
        
        Args:
            short_window (int): Window for short-term moving average
            long_window (int): Window for long-term moving average
            rsi_period (int): Period for RSI calculation
            volume_threshold (float): Volume threshold as a multiple of average volume
        """
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.volume_threshold = volume_threshold
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI for a price series"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def generate_signals(self, data):
        """
        Generate trading signals based on moving average crossover with RSI and volume filters
        
        Args:
            data (pd.DataFrame): DataFrame with 'Close' and 'Volume' columns
            
        Returns:
            pd.Series: Series of signals (1 for buy, -1 for sell, 0 for hold)
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate moving averages
        signals['short_mavg'] = data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        signals['long_mavg'] = data['Close'].rolling(window=self.long_window, min_periods=1).mean()
        
        # Calculate RSI
        signals['rsi'] = self.calculate_rsi(data['Close'], self.rsi_period)
        
        # Calculate volume moving average
        signals['volume_ma'] = data['Volume'].rolling(window=20).mean()
        
        # Generate signals
        signals['signal'] = 0  # Default to no position
        
        # Long signal: short MA crosses above long MA, RSI not overbought, volume above threshold
        signals.loc[
            (signals['short_mavg'] > signals['long_mavg']) & 
            (signals['short_mavg'].shift(1) <= signals['long_mavg'].shift(1)) &
            (signals['rsi'] < 70) &  # Not overbought
            (data['Volume'] > signals['volume_ma'] * self.volume_threshold),  # Volume spike
            'signal'
        ] = 1
        
        # Short signal: short MA crosses below long MA, RSI not oversold, volume above threshold
        signals.loc[
            (signals['short_mavg'] < signals['long_mavg']) & 
            (signals['short_mavg'].shift(1) >= signals['long_mavg'].shift(1)) &
            (signals['rsi'] > 30) &  # Not oversold
            (data['Volume'] > signals['volume_ma'] * self.volume_threshold),  # Volume spike
            'signal'
        ] = -1
        
        return signals['signal']


class Backtester:
    def __init__(self, config_path: str = None, config=None):
        if config_path:
            self.config = self._load_config(config_path)
        else:
            # Initialize with default configuration
            self.config = {
                'strategy': {},
                'backtest': {
                    'initial_capital': 100000,
                    'commission': 0.001,
                    'slippage': 0.0005,
                    'position_size': 0.1,  # 10% of portfolio per trade
                    'max_positions': 5     # Maximum number of concurrent positions
                },
                'data': {
                    'source': 'yfinance',  # Default data source
                    'interval': '1d'       # Default interval
                }
            }
            
            # Update with user-provided config if any
            if config:
                if 'strategy' in config:
                    self.config['strategy'].update(config['strategy'])
                if 'backtest' in config:
                    self.config['backtest'].update(config['backtest'])
                if 'data' in config:
                    self.config['data'].update(config['data'])
        
        # Initialize instance variables
            if 'backtest' in config:
                self.config['backtest'].update(config['backtest'])
            if 'data' in config:
                self.config['data'].update(config['data'])
        
        # Set strategy parameters
        self.strategy_config = self.config['strategy']
        self.long_window = self.strategy_config.get('long_window', 50)
        self.short_window = self.strategy_config.get('short_window', 20)
        
        # Initialize other instance variables
        self.results = None
        self.trades = []
        self.portfolio_history = []
        self.positions = {}  # Track current positions
        self.portfolio_value = self.config['backtest']['initial_capital']
        self.commission = self.config['backtest']['commission']
        self.slippage = self.config['backtest']['slippage']
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize data storage
        self.historical_data = {}
        self.signals = {}
        self.returns = {}

    def _filter_symbol_data(self, symbol, prepared_data, current_date):
        """Helper method to filter and validate symbol data"""
        try:
            if symbol not in prepared_data:
                logging.warning(f"  {symbol} - Not found in prepared_data")
                return None

            symbol_data = prepared_data[symbol]
            
            # Ensure we're working with a DataFrame and not a Series
            if isinstance(symbol_data, pd.Series):
                symbol_data = symbol_data.to_frame().T
                
            # Make sure we have a valid index
            if not hasattr(symbol_data.index, 'tz_localize'):
                symbol_data.index = pd.to_datetime(symbol_data.index)
            
            # Filter data up to current date
            mask = symbol_data.index <= current_date
            filtered_data = symbol_data.loc[mask] if hasattr(mask, '__getitem__') else symbol_data
            
            # Ensure we have enough data points for indicators
            if len(filtered_data) < self.long_window:
                logging.debug(f"  {symbol} - Insufficient data: {len(filtered_data)} rows (need {self.long_window})")
                return None
                
            # Clean any NaN values
            filtered_data = filtered_data.ffill().bfill()
            
            # Only keep rows where we have all required data
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in filtered_data.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in filtered_data.columns]
                logging.warning(f"  {symbol} - Missing required columns: {missing_cols}")
                return None
            
            # Ensure all required columns are numeric
            for col in required_cols:
                filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')
            
            # Drop any remaining NaN values
            filtered_data = filtered_data.dropna(subset=required_cols)
            
            if len(filtered_data) < self.long_window:
                logging.debug(f"  {symbol} - Insufficient data after cleaning: {len(filtered_data)} rows")
                return None
                
            logging.debug(f"  {symbol} - {len(filtered_data)} valid rows up to {current_date}")
            return filtered_data
            
        except Exception as e:
            logging.error(f"Error processing {symbol} on {current_date}: {str(e)}", exc_info=True)
            return None

    def _load_config(self, config_path: str) -> dict:
        """Load backtesting configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return {
                'backtest': {
                    'initial_capital': 100000,
                    'commission': 0.001,
                    'slippage': 0.0005,
                    'start_date': '2023-01-01',
                    'end_date': datetime.now().strftime('%Y-%m-%d')
                }
            }

    def validate_data_structure(self, data: pd.DataFrame):
        """Validate the data structure and log details"""
        logging.info("=== DATA STRUCTURE VALIDATION ===")
        logging.info(f"Data shape: {data.shape}")
        logging.info(f"Index names: {data.index.names}")
        logging.info(f"Columns: {data.columns.tolist()}")
        
        if isinstance(data.index, pd.MultiIndex):
            symbols = data.index.get_level_values('Symbol').unique()
            dates = data.index.get_level_values('Date').unique()
            logging.info(f"Unique symbols: {len(symbols)}")
            logging.info(f"Unique dates: {len(dates)}")
            logging.info(f"Date range: {min(dates)} to {max(dates)}")
            
            # Check for NaN values
            nan_counts = data.isna().sum()
            if nan_counts.sum() > 0:
                logging.warning("NaN values found:")
                for col, count in nan_counts[nan_counts > 0].items():
                    logging.warning(f"  {col}: {count} NaN values")
        
        logging.info("=== END VALIDATION ===")

    def run_backtest(self, strategy, symbols, start_date, end_date):
        """Run the backtest with the given strategy and symbols"""
        try:
            # Get historical data
            logging.info(f"Starting backtest from {start_date} to {end_date}")
            data = self._get_historical_data(symbols, start_date, end_date)
            
            # Check if we got any data
            if not data or not isinstance(data, dict) or len(data) == 0:
                raise ValueError("No valid historical data available for backtesting")
                
            logging.info(f"Downloaded data for {len(data)} symbols")
            
            # Prepare the data for backtesting
            prepared_data = self._prepare_data(data, start_date, end_date)
            
            # Check if we have any prepared data
            if not prepared_data or len(prepared_data) == 0:
                raise ValueError("Failed to prepare data for backtesting")
                
            logging.info(f"Prepared data for {len(prepared_data)} symbols")
            
            # Initialize portfolio
            initial_capital = self.config['backtest']['initial_capital']
            portfolio = {
                'cash': initial_capital,
                'positions': {},
                'value': initial_capital,
                'history': []
            }
            
            # Get all unique dates across all symbols
            all_dates = set()
            for symbol_data in prepared_data.values():
                all_dates.update(symbol_data.index)
            all_dates = sorted(all_dates)
            
            # Main backtest loop
            for current_date in tqdm(all_dates, desc="Running backtest"):
                current_data = {}
                
                # Get data for each symbol up to current date
                for symbol in symbols:
                    if symbol in prepared_data:
                        symbol_data = prepared_data[symbol]
                        mask = symbol_data.index <= current_date
                        current_data[symbol] = symbol_data[mask]
                
                # Skip if no data for current date
                if not current_data:
                    continue
                    
                # Generate signals
                signals = strategy.generate_signals(current_data)
                
                # Execute trades
                for symbol, signal in signals.items():
                    if symbol in current_data and not current_data[symbol].empty:
                        current_price = current_data[symbol].iloc[-1]['Close']
                        self._execute_trade(symbol, signal, current_price, portfolio, current_date)
                
                # Update portfolio value
                self._update_portfolio_value(portfolio, current_date, current_data)
            
            # Store results
            self.results = portfolio
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Error in run_backtest: {str(e)}", exc_info=True)
            raise

    def _get_historical_data(self, symbols, start_date, end_date):
        """Download historical data from Yahoo Finance"""
        import yfinance as yf
        
        data = {}
        for symbol in symbols:
            try:
                self.logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
                # Convert dates to datetime if they're strings
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                
                # Calculate start date with buffer for indicators
                buffer_start = start_date - pd.Timedelta(days=100)
                
                # Download the data
                df = yf.download(
                    symbol,
                    start=buffer_start,
                    end=end_date,
                    progress=False,
                    auto_adjust=True  # Adjust for corporate actions
                )
                
                # Filter for the requested date range
                df = df[df.index >= start_date]
                df = df[df.index <= end_date]
                
                if not df.empty:
                    data[symbol] = df
                    self.logger.info(f"Downloaded {len(df)} rows of clean data for {symbol}")
                else:
                    self.logger.warning(f"No data available for {symbol} after filtering")
            except Exception as e:
                self.logger.error(f"Error downloading data for {symbol}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        return data

    def _prepare_data(self, data, start_date, end_date):
        """Prepare and clean the data for backtesting"""
        prepared_data = {}
        
        for symbol, df in data.items():
            try:
                if df is None or df.empty:
                    self.logger.warning(f"No data available for {symbol}")
                    continue
                    
                # Basic cleaning
                df = df.copy()
                
                # Ensure the index is a DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                # Sort by date
                df = df.sort_index()
                
                # Handle missing values
                df = df.ffill()  # Forward fill missing values
                df = df.bfill()  # Backward fill any remaining missing values
                
                # Ensure we have enough data
                if len(df) < self.long_window:
                    self.logger.warning(f"Insufficient data for {symbol}: {len(df)} rows, need at least {self.long_window}")
                    continue
                
                # Ensure we have the required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    self.logger.warning(f"Missing required columns for {symbol}: {missing_columns}")
                    continue
                
                prepared_data[symbol] = df
                
            except Exception as e:
                logging.error(f"Unexpected error processing {symbol}: {e}", exc_info=True)
        
        return prepared_data

def run_complete_system():
    """Run the complete trading system with backtesting and live trading"""
    
    # 1. Run Backtest First
    print("🧪 RUNNING BACKTEST...")
    
    # Define strategy parameters
    strategy_config = {
        'short_window': 20,
        'long_window': 50,
        'rsi_period': 14,
        'volume_threshold': 1.2
    }
    
    # Initialize backtester with config
    backtester = Backtester(config={
        'strategy': strategy_config,
        'backtest': {
            'initial_capital': 100000,
            'commission': 0.001,
            'slippage': 0.0005
        }
    })
    
    # Initialize strategy with the same parameters
    strategy = EnhancedMovingAverageCrossover(**strategy_config)
    
    symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS']
    
    try:
        # Run backtest
        print("⏳ Running backtest...")
        results = backtester.run_backtest(
            strategy=strategy,
            symbols=symbols,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        # Generate and display report
        print("\n" + "="*50)
        print(backtester.generate_report())
        print("="*50 + "\n")
        
        # Save results plot
        plot_file = 'backtest_results.png'
        backtester.plot_results(plot_file)
        print(f"📊 Results plot saved to {plot_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during backtest: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # You can choose to run backtest only or live trading
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'backtest':
        run_complete_system()
    else:
        # Run enhanced trading bot by default
        bot = EnhancedTradingBot()
        try:
            bot.monitor_market(interval=300)
        except KeyboardInterrupt:
            logging.info("Trading bot stopped by user")