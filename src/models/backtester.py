import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import os

from src.strategies.moving_average_crossover import EnhancedMovingAverageCrossover
from src.utils.data_utils import DataFetcher
from src.models.risk_manager import RiskManager

class Backtester:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, config: dict):
        """
        Initialize the backtester with configuration
        
        Args:
            config (dict): Configuration dictionary with backtest parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_fetcher = DataFetcher(config.get('data', {}))
        self.risk_manager = RiskManager(config.get('risk', {}))
        
        # Backtest parameters
        self.initial_capital = config.get('backtest', {}).get('initial_capital', 100000)
        self.commission = config.get('backtest', {}).get('commission', 0.001)  # 0.1% commission
        self.slippage = config.get('backtest', {}).get('slippage', 0.0005)  # 0.05% slippage
        
        # Results storage
        self.results = {}
        self.equity_curve = None
        self.trades = []
    
    def run_backtest(self, strategy, symbols: List[str], 
                    start_date: str, end_date: str) -> dict:
        """
        Run a backtest on the given symbols with the specified strategy
        
        Args:
            strategy: The trading strategy to use
            symbols: List of stock symbols to backtest on
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary containing backtest results
        """
        try:
            self.logger.info(f"Starting backtest from {start_date} to {end_date}")
            
            # Fetch and prepare data
            raw_data = self.data_fetcher.fetch_historical_data(symbols, start_date, end_date)
            if not raw_data:
                raise ValueError("No data available for the given symbols and date range")
                
            prepared_data = self.data_fetcher.prepare_data(raw_data, start_date, end_date)
            
            # Initialize portfolio
            portfolio = {
                'cash': self.initial_capital,
                'positions': {symbol: 0 for symbol in symbols},
                'total_value': self.initial_capital,
                'shares': {symbol: 0 for symbol in symbols},
                'trades': []
            }
            
            # Get all trading days
            all_dates = []
            for symbol, data in prepared_data.items():
                all_dates.extend(data.index.tolist())
            all_dates = sorted(list(set(all_dates)))
            
            # Main backtest loop
            for current_date in tqdm(all_dates, desc="Running backtest"):
                for symbol, data in prepared_data.items():
                    if current_date not in data.index:
                        continue
                        
                    # Get current price and data
                    current_data = data.loc[:current_date].copy()
                    current_price = current_data['Close'].iloc[-1]
                    
                    # Generate signals
                    signals = strategy.generate_signals(current_data)
                    current_signal = signals.iloc[-1] if not signals.empty else 0
                    
                    # Execute trades based on signals
                    self._execute_trade(
                        symbol=symbol,
                        price=current_price,
                        signal=current_signal,
                        date=current_date,
                        portfolio=portfolio
                    )
                
                # Update portfolio value at the end of each day
                self._update_portfolio_value(portfolio, prepared_data, current_date)
            
            # Calculate performance metrics
            self._calculate_performance_metrics(portfolio)
            
            # Generate reports
            self._generate_reports(portfolio, start_date, end_date)
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error in run_backtest: {e}", exc_info=True)
            raise
    
    def _execute_trade(self, symbol: str, price: float, signal: int, 
                      date: datetime, portfolio: dict) -> None:
        """
        Execute a trade based on the signal
        
        Args:
            symbol: Stock symbol
            price: Current price
            signal: Trading signal (1 for buy, -1 for sell, 0 for hold)
            date: Current date
            portfolio: Current portfolio state
        """
        try:
            if signal == 0:  # No signal
                return
                
            current_shares = portfolio['shares'].get(symbol, 0)
            
            # Buy signal
            if signal > 0 and current_shares <= 0:
                # Calculate position size based on risk
                stop_loss = price * 0.95  # Example: 5% stop loss
                position_size = self.risk_manager.calculate_position_size(
                    entry_price=price,
                    stop_loss=stop_loss,
                    account_balance=portfolio['cash']
                )
                
                if position_size <= 0:
                    return
                    
                # Calculate order value with slippage and commission
                order_value = position_size * price
                commission = order_value * self.commission
                slippage = order_value * self.slippage
                total_cost = order_value + commission + slippage
                
                if total_cost > portfolio['cash']:
                    self.logger.warning(f"Insufficient cash to buy {symbol}")
                    return
                
                # Execute buy order
                portfolio['cash'] -= total_cost
                portfolio['shares'][symbol] = position_size
                portfolio['positions'][symbol] = {
                    'entry_price': price,
                    'entry_date': date,
                    'stop_loss': stop_loss,
                    'take_profit': price * 1.10  # Example: 10% take profit
                }
                
                # Record trade
                trade = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'date': date,
                    'price': price,
                    'shares': position_size,
                    'value': order_value,
                    'commission': commission,
                    'slippage': slippage
                }
                portfolio['trades'].append(trade)
                self.trades.append(trade)
                
                self.logger.info(
                    f"{date.strftime('%Y-%m-%d')} - Bought {position_size:.2f} shares of {symbol} "
                    f"at {price:.2f} (Value: ${order_value:,.2f})"
                )
            
            # Sell signal
            elif signal < 0 and current_shares > 0:
                # Calculate order value with slippage and commission
                order_value = current_shares * price
                commission = order_value * self.commission
                slippage = order_value * self.slippage
                proceeds = order_value - commission - slippage
                
                # Execute sell order
                portfolio['cash'] += proceeds
                portfolio['shares'][symbol] = 0
                
                # Calculate P&L
                entry_price = portfolio['positions'][symbol]['entry_price']
                pnl = (price - entry_price) * current_shares
                pnl_pct = (price / entry_price - 1) * 100
                
                # Record trade
                trade = {
                    'symbol': symbol,
                    'action': 'SELL',
                    'date': date,
                    'price': price,
                    'shares': current_shares,
                    'value': order_value,
                    'commission': commission,
                    'slippage': slippage,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                }
                portfolio['trades'].append(trade)
                self.trades.append(trade)
                
                self.logger.info(
                    f"{date.strftime('%Y-%m-%d')} - Sold {current_shares:.2f} shares of {symbol} "
                    f"at {price:.2f} (Value: ${order_value:,.2f}, P&L: ${pnl:,.2f} [{pnl_pct:+.2f}%])"
                )
                
                # Clear position
                portfolio['positions'].pop(symbol, None)
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}", exc_info=True)
    
    def _update_portfolio_value(self, portfolio: dict, data: dict, current_date: datetime) -> None:
        """Update the total portfolio value based on current positions"""
        try:
            total_value = portfolio['cash']
            
            for symbol, shares in portfolio['shares'].items():
                if shares > 0 and symbol in data and current_date in data[symbol].index:
                    current_price = data[symbol].loc[current_date, 'Close']
                    total_value += shares * current_price
            
            portfolio['total_value'] = total_value
            
            # Record daily value for equity curve
            if not hasattr(self, 'equity_curve'):
                self.equity_curve = pd.DataFrame(columns=['date', 'value'])
                
            self.equity_curve = self.equity_curve._append(
                {'date': current_date, 'value': total_value},
                ignore_index=True
            )
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {e}", exc_info=True)
    
    def _calculate_performance_metrics(self, portfolio: dict) -> None:
        """Calculate performance metrics for the backtest"""
        try:
            if not self.trades:
                self.logger.warning("No trades were executed during the backtest")
                return
                
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(self.trades)
            
            # Calculate basic metrics
            total_return = (portfolio['total_value'] / self.initial_capital - 1) * 100
            num_trades = len(trades_df) // 2  # Round-trip trades
            win_rate = (trades_df[trades_df['action'] == 'SELL']['pnl'] > 0).mean() * 100
            avg_trade_return = trades_df[trades_df['action'] == 'SELL']['pnl_pct'].mean()
            max_drawdown = self._calculate_max_drawdown()
            
            # Store results
            self.results = {
                'initial_capital': self.initial_capital,
                'final_value': portfolio['total_value'],
                'total_return_pct': total_return,
                'num_trades': num_trades,
                'win_rate_pct': win_rate,
                'avg_trade_return_pct': avg_trade_return,
                'max_drawdown_pct': max_drawdown,
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'trades': trades_df.to_dict('records'),
                'equity_curve': self.equity_curve.to_dict('records')
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}", exc_info=True)
            raise
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve"""
        try:
            if self.equity_curve is None or len(self.equity_curve) == 0:
                return 0.0
                
            values = self.equity_curve['value'].values
            peak = values[0]
            max_drawdown = 0.0
            
            for value in values[1:]:
                if value > peak:
                    peak = value
                else:
                    drawdown = (peak - value) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                        
            return max_drawdown * 100  # Convert to percentage
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}", exc_info=True)
            return 0.0
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio"""
        try:
            if self.equity_curve is None or len(self.equity_curve) < 2:
                return 0.0
                
            # Calculate daily returns
            returns = self.equity_curve['value'].pct_change().dropna()
            
            # Skip if not enough data
            if len(returns) < 2:
                return 0.0
                
            # Calculate annualized Sharpe ratio (252 trading days in a year)
            excess_returns = returns - (risk_free_rate / 252)
            sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(252)
            
            return sharpe_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}", exc_info=True)
            return 0.0
    
    def _generate_reports(self, portfolio: dict, start_date: str, end_date: str) -> None:
        """Generate performance reports and visualizations"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs('backtest_results', exist_ok=True)
            
            # Save results to CSV
            results_file = f'backtest_results/backtest_{start_date}_to_{end_date}.csv'
            pd.DataFrame([self.results]).to_csv(results_file, index=False)
            
            # Save trades to CSV
            trades_file = f'backtest_results/trades_{start_date}_to_{end_date}.csv'
            pd.DataFrame(self.trades).to_csv(trades_file, index=False)
            
            # Generate equity curve plot
            self._plot_equity_curve(start_date, end_date)
            
            # Print summary
            self._print_summary()
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}", exc_info=True)
    
    def _plot_equity_curve(self, start_date: str, end_date: str) -> None:
        """Plot the equity curve"""
        try:
            if self.equity_curve is None or len(self.equity_curve) == 0:
                self.logger.warning("No equity curve data to plot")
                return
                
            plt.figure(figsize=(12, 6))
            plt.plot(self.equity_curve['date'], self.equity_curve['value'])
            plt.title(f'Equity Curve ({start_date} to {end_date})')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            
            # Save the plot
            plot_file = f'backtest_results/equity_curve_{start_date}_to_{end_date}.png'
            plt.savefig(plot_file)
            plt.close()
            
            self.logger.info(f"Saved equity curve plot to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting equity curve: {e}", exc_info=True)
    
    def _print_summary(self) -> None:
        """Print a summary of the backtest results"""
        if not self.results:
            self.logger.warning("No results to display")
            return
            
        print("\n" + "="*50)
        print("BACKTEST SUMMARY")
        print("="*50)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${self.results['final_value']:,.2f}")
        print(f"Total Return: {self.results['total_return_pct']:.2f}%")
        print(f"Number of Trades: {self.results['num_trades']}")
        print(f"Win Rate: {self.results['win_rate_pct']:.1f}%")
        print(f"Average Trade Return: {self.results['avg_trade_return_pct']:.2f}%")
        print(f"Max Drawdown: {self.results['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        print("="*50 + "\n")
