import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
from typing import Tuple, Dict, Optional, Union, List
from datetime import datetime, timedelta
import ta  # Technical analysis library
import json
from dataclasses import dataclass

@dataclass
class OptionContract:
    """Data class to store option contract details"""
    symbol: str
    expiry: str
    strike: float
    option_type: str  # 'call' or 'put'
    last_price: float
    volume: int
    open_interest: int
    implied_vol: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    bid: float
    ask: float
    last_trade_date: str
    in_the_money: bool
    theoretical_value: float

@dataclass
class MarketCondition:
    """Data class to store market condition analysis"""
    trend: str
    momentum: str
    volatility: str
    volume: str
    rsi: float
    macd_signal: str
    bollinger_signal: str
    support_level: float
    resistance_level: float
    vix_level: float
    vix_signal: str
    market_sentiment: str

class OptionsAnalyzer:
    """
    A comprehensive options analysis tool that provides:
    - Options chain data fetching with advanced filtering
    - Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
    - Put-Call Ratio analysis with sentiment
    - Volatility surface and skew analysis
    - Technical indicators and market condition analysis
    - Trade recommendations with confidence scoring
    - Risk assessment and management
    """
    
    def __init__(self, risk_free_rate: float = 0.05, lookback_days: int = 90):
        """
        Initialize OptionsAnalyzer with configuration
        
        Args:
            risk_free_rate: Annual risk-free interest rate (default: 0.05 or 5%)
            lookback_days: Number of days to look back for technical analysis (default: 90)
        """
        self.risk_free_rate = risk_free_rate
        self.lookback_days = lookback_days
        self.indicators = [
            'rsi', 'macd', 'bollinger', 'stoch', 'atr',
            'obv', 'adx', 'ichimoku', 'vwap', 'sma', 'ema'
        ]
        
    def get_historical_data(self, ticker: str, period: str = '1y') -> pd.DataFrame:
        """
        Fetch historical price data with technical indicators
        
        Args:
            ticker: Stock ticker symbol
            period: Data period to fetch (default: '1y')
            
        Returns:
            DataFrame with OHLCV data and technical indicators
        """
        try:
            # Fetch historical data
            df = yf.Ticker(ticker).history(period=period, interval='1d')
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_low'] = bollinger.bollinger_lband()
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(
            df['High'], df['Low'], df['Close']).average_true_range()
            
        # OBV
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            df['Close'], df['Volume']).on_balance_volume()
            
        # ADX
        df['adx'] = ta.trend.ADXIndicator(
            df['High'], df['Low'], df['Close']).adx()
            
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(
            high=df['High'], low=df['Low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        
        # VWAP
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            high=df['High'], low=df['Low'], 
            close=df['Close'], volume=df['Volume']
        ).volume_weighted_average_price()
        
        # Moving Averages
        df['sma_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        df['sma_200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
        df['ema_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=df['High'], low=df['Low'], close=df['Close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        return df

    def analyze_market_condition(self, ticker: str) -> MarketCondition:
        """
        Analyze overall market condition for the given ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            MarketCondition object with analysis results
        """
        try:
            # Get historical data with indicators
            df = self.get_historical_data(ticker, f'{self.lookback_days}d')
            if df.empty:
                return MarketCondition(
                    trend='Neutral', momentum='Neutral', volatility='Medium',
                    volume='Average', rsi=50, macd_signal='Neutral',
                    bollinger_signal='Neutral', support_level=0,
                    resistance_level=0, vix_level=0, vix_signal='Neutral',
                    market_sentiment='Neutral'
                )
            
            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Trend analysis
            trend = 'Neutral'
            if latest['sma_50'] > latest['sma_200'] and latest['Close'] > latest['sma_50']:
                trend = 'Strong Uptrend'
            elif latest['sma_50'] > latest['sma_200']:
                trend = 'Mild Uptrend'
            elif latest['sma_50'] < latest['sma_200'] and latest['Close'] < latest['sma_50']:
                trend = 'Strong Downtrend'
            elif latest['sma_50'] < latest['sma_200']:
                trend = 'Mild Downtrend'
            
            # Momentum
            momentum = 'Neutral'
            if latest['rsi'] > 70:
                momentum = 'Overbought'
            elif latest['rsi'] < 30:
                momentum = 'Oversold'
                
            # Volatility
            atr_percent = (latest['atr'] / latest['Close']) * 100
            if atr_percent > 5:
                volatility = 'High'
            elif atr_percent > 2:
                volatility = 'Medium'
            else:
                volatility = 'Low'
                
            # Volume
            avg_volume = df['Volume'].mean()
            volume = 'Average'
            if latest['Volume'] > avg_volume * 1.5:
                volume = 'High'
            elif latest['Volume'] < avg_volume * 0.5:
                volume = 'Low'
                
            # MACD Signal
            macd_signal = 'Neutral'
            if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                macd_signal = 'Bullish Crossover'
            elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                macd_signal = 'Bearish Crossover'
                
            # Bollinger Bands
            bollinger_signal = 'Neutral'
            if latest['Close'] < latest['bb_low']:
                bollinger_signal = 'Oversold (Below Lower Band)'
            elif latest['Close'] > latest['bb_high']:
                bollinger_signal = 'Overbought (Above Upper Band)'
                
            # Support/Resistance (simplified)
            support = df['Low'].rolling(20).min().iloc[-1]
            resistance = df['High'].rolling(20).max().iloc[-1]
            
            # VIX (Market Fear Index)
            try:
                vix = yf.Ticker('^VIX').history(period='1d')['Close'].iloc[-1]
                vix_signal = 'High Fear' if vix > 30 else 'Greed' if vix < 15 else 'Neutral'
            except:
                vix = 0
                vix_signal = 'N/A'
                
            # Overall Market Sentiment
            sentiment_score = 0
            sentiment_score += 1 if trend in ['Strong Uptrend', 'Mild Uptrend'] else -1
            sentiment_score += 1 if momentum == 'Oversold' else -1 if momentum == 'Overbought' else 0
            sentiment_score += 1 if macd_signal == 'Bullish Crossover' else -1 if macd_signal == 'Bearish Crossover' else 0
            sentiment_score += 1 if bollinger_signal == 'Oversold' else -1 if bollinger_signal == 'Overbought' else 0
            
            if sentiment_score >= 2:
                market_sentiment = 'Bullish'
            elif sentiment_score <= -2:
                market_sentiment = 'Bearish'
            else:
                market_sentiment = 'Neutral'
            
            return MarketCondition(
                trend=trend,
                momentum=momentum,
                volatility=volatility,
                volume=volume,
                rsi=round(latest['rsi'], 2),
                macd_signal=macd_signal,
                bollinger_signal=bollinger_signal,
                support_level=round(support, 2),
                resistance_level=round(resistance, 2),
                vix_level=round(vix, 2) if vix > 0 else 0,
                vix_signal=vix_signal,
                market_sentiment=market_sentiment
            )
            
        except Exception as e:
            print(f"Error in analyze_market_condition: {str(e)}")
            return MarketCondition(
                trend='Neutral', momentum='Neutral', volatility='Medium',
                volume='Average', rsi=50, macd_signal='Neutral',
                bollinger_signal='Neutral', support_level=0,
                resistance_level=0, vix_level=0, vix_signal='Neutral',
                market_sentiment='Neutral'
            )

    def get_options_chain(
        self, 
        ticker: str, 
        expiry_date: Optional[str] = None,
        min_volume: int = 100,
        min_open_interest: int = 100,
        moneyness: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, str, Dict]:
        """
        Fetch and filter options chain data with enhanced metrics
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            expiry_date: Expiration date in 'YYYY-MM-DD' format (default: nearest expiry)
            min_volume: Minimum volume filter (default: 100)
            min_open_interest: Minimum open interest filter (default: 100)
            moneyness: Percentage range for near-the-money options (default: 0.2 for 20%)
            
        Returns:
            Tuple of (filtered calls DataFrame, filtered puts DataFrame, 
                     expiration date used, market condition)
        """
        try:
            # Get market condition first
            market_condition = self.analyze_market_condition(ticker)
            
            # Get stock data
            stock = yf.Ticker(ticker)
            stock_info = stock.history(period='1d')
            if stock_info.empty:
                raise ValueError(f"No data available for {ticker}")
                
            spot_price = stock_info['Close'].iloc[-1]
            
            # Get expiration dates
            if expiry_date is None:
                expiry_dates = stock.options
                if not expiry_dates:
                    raise ValueError(f"No expiry dates available for {ticker}")
                expiry_date = expiry_dates[0]  # Default to nearest expiry
            
            # Get option chain
            opt_chain = stock.option_chain(expiry_date)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Add moneyness and filter near-the-money options
            moneyness_range = (1 - moneyness, 1 + moneyness)
            
            def process_chain(chain, is_call: bool):
                chain = chain.copy()
                chain['moneyness'] = chain['strike'] / spot_price
                
                # Filter for near-the-money options
                if is_call:
                    chain = chain[chain['moneyness'].between(*moneyness_range)]
                else:
                    chain = chain[chain['moneyness'].between(2 - moneyness_range[1], 2 - moneyness_range[0])]
                
                # Filter for minimum volume and open interest
                chain = chain[(chain['volume'] >= min_volume) & 
                             (chain['openInterest'] >= min_open_interest)]
                
                # Calculate Greeks for each option
                greeks = []
                for _, row in chain.iterrows():
                    try:
                        greek = self.calculate_greeks(
                            option_type='call' if is_call else 'put',
                            S=spot_price,
                            K=row['strike'],
                            T=(pd.to_datetime(expiry_date) - pd.Timestamp.now()).days / 365.25,
                            sigma=row['impliedVolatility'],
                            r=self.risk_free_rate,
                            option_price=row['lastPrice']
                        )
                        greeks.append(greek)
                    except Exception as e:
                        print(f"Error calculating Greeks: {str(e)}")
                        greeks.append({k: 0 for k in ['delta', 'gamma', 'theta', 'vega', 'rho', 'price']})
                
                # Add Greeks to the chain
                greeks_df = pd.DataFrame(greeks)
                for col in greeks_df.columns:
                    chain[col] = greeks_df[col].values
                
                # Calculate probability of being in the money
                if 'delta' in chain.columns:
                    chain['prob_itm'] = chain['delta'].abs()
                
                return chain
            
            # Process calls and puts
            filtered_calls = process_chain(calls, is_call=True)
            filtered_puts = process_chain(puts, is_call=False)
            
            return filtered_calls, filtered_puts, expiry_date, market_condition
            
        except Exception as e:
            print(f"Error in get_options_chain: {str(e)}")
            return pd.DataFrame(), pd.DataFrame(), '', market_condition
    
    def calculate_greeks(self, option_type: str, S: float, K: float, T: float, 
                        sigma: float, r: Optional[float] = None, 
                        option_price: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate option Greeks using Black-Scholes model
        
        Args:
            option_type: 'call' or 'put'
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            sigma: Implied volatility (annualized)
            r: Risk-free rate (default: uses instance risk_free_rate)
            option_price: Optional market price (for implied volatility calculation)
            
        Returns:
            Dictionary containing Greeks and theoretical price
        """
        if r is None:
            r = self.risk_free_rate
            
        option_type = option_type.lower()
        if option_type not in ['call', 'put']:
            raise ValueError("option_type must be either 'call' or 'put'")
            
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        if option_type == 'call':
            delta = norm.cdf(d1)
            price = S * delta - K * np.exp(-r * T) * norm.cdf(d2)
            theta = (-S * norm.pdf(d1) * sigma) / (2 * sqrt_T) - r * K * np.exp(-r * T) * norm.cdf(d2)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            delta = -norm.cdf(-d1)
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            theta = (-S * norm.pdf(d1) * sigma) / (2 * sqrt_T) + r * K * np.exp(-r * T) * norm.cdf(-d2)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
        gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
        vega = S * norm.pdf(d1) * sqrt_T
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta / 365,  # Convert to daily theta
            'rho': rho,
            'price': price,
            'implied_vol': sigma  # Will be updated if calculating implied vol
        }
    
    def analyze_pcr(self, calls: pd.DataFrame, puts: pd.DataFrame) -> Dict[str, Union[float, str]]:
        """
        Analyze Put-Call Ratio (PCR) for given options data with enhanced metrics
        
        Args:
            calls: DataFrame containing call options data
            puts: DataFrame containing put options data
            
        Returns:
            Dictionary with PCR analysis results
        """
        if calls.empty or puts.empty:
            return {
                'put_call_ratio_oi': 0,
                'put_call_ratio_volume': 0,
                'total_call_oi': 0,
                'total_put_oi': 0,
                'total_call_volume': 0,
                'total_put_volume': 0,
                'sentiment_oi': 'Neutral',
                'sentiment_volume': 'Neutral',
                'pcr_rank': 0.5,
                'skew_analysis': 'Neutral',
                'unusual_activity': 'None',
                'confidence_score': 0
            }
        
        # Basic PCR calculations
        total_call_oi = calls['openInterest'].sum()
        total_put_oi = puts['openInterest'].sum()
        total_call_vol = calls['volume'].sum()
        total_put_vol = puts['volume'].sum()
        
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        
        # Advanced metrics
        call_volume_avg = calls['volume'].mean() if not calls.empty else 0
        put_volume_avg = puts['volume'].mean() if not puts.empty else 0
        call_oi_avg = calls['openInterest'].mean() if not calls.empty else 0
        put_oi_avg = puts['openInterest'].mean() if not puts.empty else 0
        
        # Volume and OI ratios
        volume_ratio = put_volume_avg / call_volume_avg if call_volume_avg > 0 else 0
        oi_ratio = put_oi_avg / call_oi_avg if call_oi_avg > 0 else 0
        
        # Sentiment scoring (0-1 scale)
        def get_sentiment_score(pcr_value):
            if pcr_value > 1.5:
                return 0.9  # Strongly Bearish
            elif pcr_value > 1.1:
                return 0.7  # Bearish
            elif pcr_value > 0.9:
                return 0.5  # Neutral
            elif pcr_value > 0.5:
                return 0.3  # Bullish
            else:
                return 0.1  # Strongly Bullish
        
        def get_sentiment_label(score):
            if score >= 0.8:
                return 'Extremely Bearish'
            elif score >= 0.6:
                return 'Bearish'
            elif score > 0.4:
                return 'Neutral'
            elif score > 0.2:
                return 'Bullish'
            else:
                return 'Extremely Bullish'
        
        # Calculate confidence score (0-1)
        confidence = min(1.0, (total_call_vol + total_put_vol) / 10000)  # Normalize by volume
        
        # Skew analysis
        skew = 'Neutral'
        if pcr_oi > 1.2 and pcr_vol > 1.2:
            skew = 'Strong Put Skew (Bearish)'
        elif pcr_oi < 0.8 and pcr_vol < 0.8:
            skew = 'Strong Call Skew (Bullish)'
        elif pcr_oi > 1.0 and pcr_vol > 1.0:
            skew = 'Moderate Put Skew (Slightly Bearish)'
        elif pcr_oi < 1.0 and pcr_vol < 1.0:
            skew = 'Moderate Call Skew (Slightly Bullish)'
        
        # Unusual activity detection
        unusual_activity = []
        if volume_ratio > 2.0:
            unusual_activity.append('Heavy Put Volume')
        if oi_ratio > 2.0:
            unusual_activity.append('High Put Open Interest')
        if not unusual_activity:
            unusual_activity = ['None']
        
        # Calculate overall sentiment score (weighted average)
        oi_score = get_sentiment_score(pcr_oi)
        vol_score = get_sentiment_score(pcr_vol)
        overall_score = (oi_score * 0.6) + (vol_score * 0.4)  # Weight OI more than volume
        
        return {
            'put_call_ratio_oi': round(pcr_oi, 2),
            'put_call_ratio_volume': round(pcr_vol, 2),
            'total_call_oi': int(total_call_oi),
            'total_put_oi': int(total_put_oi),
            'total_call_volume': int(total_call_vol),
            'total_put_volume': int(total_put_vol),
            'sentiment_oi': get_sentiment_label(oi_score),
            'sentiment_volume': get_sentiment_label(vol_score),
            'pcr_rank': round(overall_score, 2),
            'skew_analysis': skew,
            'unusual_activity': ', '.join(unusual_activity),
            'confidence_score': round(confidence, 2)
        }
    
    def calculate_iv_surface(self, ticker: str, expiry_date: Optional[str] = None) -> Dict:
        """
        Calculate implied volatility surface for a given ticker
        
        Args:
            ticker: Stock ticker symbol
            expiry_date: Expiration date in 'YYYY-MM-DD' format (default: all available)
            
        Returns:
            Dictionary containing IV surface data
        """
        stock = yf.Ticker(ticker)
        spot_price = stock.history(period='1d')['Close'].iloc[-1]
        
        if expiry_date is None:
            expiry_dates = stock.options
        else:
            expiry_dates = [expiry_date]
        
        iv_surface = {}
        
        for expiry in expiry_dates:
            try:
                calls, puts, _ = self.get_options_chain(ticker, expiry)
                
                # Calculate time to expiration in years
                expiry_dt = pd.to_datetime(expiry)
                T = (expiry_dt - pd.Timestamp.now()).days / 365.25
                
                # Calculate IV for calls and puts at different moneyness levels
                for moneyness in [0.8, 0.9, 1.0, 1.1, 1.2]:
                    strike = spot_price * moneyness
                    
                    # Find nearest strike
                    calls_strike = calls.iloc[(calls['strike'] - strike).abs().argsort()[:1]]
                    puts_strike = puts.iloc[(puts['strike'] - strike).abs().argsort()[:1]]
                    
                    if not calls_strike.empty and not puts_strike.empty:
                        call_iv = calls_strike['impliedVolatility'].values[0]
                        put_iv = puts_strike['impliedVolatility'].values[0]
                        
                        if expiry not in iv_surface:
                            iv_surface[expiry] = {}
                            
                        iv_surface[expiry][moneyness] = {
                            'call_iv': call_iv,
                            'put_iv': put_iv,
                            'strike': strike,
                            'moneyness': moneyness
                        }
                        
            except Exception as e:
                print(f"Error processing {expiry}: {str(e)}")
                continue
                
        return iv_surface
    
    def analyze_skew(self, ticker: str, expiry_date: Optional[str] = None) -> Dict:
        """
        Analyze volatility skew for a given ticker and expiration
        
        Args:
            ticker: Stock ticker symbol
            expiry_date: Expiration date in 'YYYY-MM-DD' format (default: nearest)
            
        Returns:
            Dictionary containing skew analysis
        """
        calls, puts, expiry = self.get_options_chain(ticker, expiry_date)
        stock = yf.Ticker(ticker)
        spot_price = stock.history(period='1d')['Close'].iloc[-1]
        
        # Calculate time to expiration in years
        expiry_dt = pd.to_datetime(expiry)
        T = (expiry_dt - pd.Timestamp.now()).days / 365.25
        
        # Calculate moneyness for each option
        calls['moneyness'] = calls['strike'] / spot_price
        puts['moneyness'] = puts['strike'] / spot_price
        
        # Group by moneyness ranges
        bins = [0, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, np.inf]
        labels = ['<0.8', '0.8-0.9', '0.9-0.95', '0.95-1.0', '1.0-1.05', '1.05-1.1', '1.1-1.2', '>1.2']
        
        calls['moneyness_group'] = pd.cut(calls['moneyness'], bins=bins, labels=labels)
        puts['moneyness_group'] = pd.cut(puts['moneyness'], bins=bins, labels=labels)
        
        # Calculate average IV by moneyness group
        call_iv = calls.groupby('moneyness_group')['impliedVolatility'].mean()
        put_iv = puts.groupby('moneyness_group')['impliedVolatility'].mean()
        
        # Calculate skew (slope of IV vs moneyness)
        def calculate_skew(iv_series):
            x = np.arange(len(iv_series))
            slope = np.polyfit(x, iv_series, 1)[0] if len(iv_series) > 1 else 0
            return slope
        
        call_skew = calculate_skew(call_iv.values)
        put_skew = calculate_skew(put_iv.values)
        
        return {
            'expiry': expiry,
            'spot_price': spot_price,
            'call_skew': call_skew,
            'put_skew': put_skew,
            'call_iv_by_moneyness': call_iv.to_dict(),
            'put_iv_by_moneyness': put_iv.to_dict(),
            'skew_interpretation': {
                'call_skew': 'Positive skew (OTM calls > ITM calls)' if call_skew > 0 else 'Negative skew (OTM calls < ITM calls)',
                'put_skew': 'Positive skew (OTM puts > ITM puts)' if put_skew > 0 else 'Negative skew (OTM puts < ITM puts)'
            }
        }
