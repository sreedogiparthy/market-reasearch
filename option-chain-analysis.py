class OptionsAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.05
        
    def get_options_chain(self, symbol, expiration=None):
        """Fetch options chain data"""
        stock = yf.Ticker(symbol)
        options_dates = stock.options
        
        if not expiration:
            expiration = options_dates[0]  # Nearest expiration
            
        chain = stock.option_chain(expiration)
        return chain.calls, chain.puts, expiration
    
    def calculate_greeks(self, option_type, S, K, T, r, sigma):
        """Calculate Black-Scholes Greeks"""
        from scipy.stats import norm
        import math
        
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type == 'call':
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
            theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) 
                    - r * K * math.exp(-r * T) * norm.cdf(d2))
            vega = S * norm.pdf(d1) * math.sqrt(T)
            rho = K * T * math.exp(-r * T) * norm.cdf(d2)
        else:  # put
            delta = norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
            theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) 
                    + r * K * math.exp(-r * T) * norm.cdf(-d2))
            vega = S * norm.pdf(d1) * math.sqrt(T)
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)
            
        return {
            'delta': delta,
            'gamma': gamma, 
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def analyze_pcr(self, calls, puts):
        """Analyze Put-Call Ratio"""
        total_call_oi = calls.openInterest.sum()
        total_put_oi = puts.openInterest.sum()
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        return {
            'put_call_ratio': pcr,
            'total_call_oi': total_call_oi,
            'total_put_oi': total_put_oi,
            'sentiment': 'Bearish' if pcr > 1 else 'Bullish'
        }