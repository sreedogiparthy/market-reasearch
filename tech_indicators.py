class IntradayAnalyzer:
    def __init__(self):
        self.volume_profile_data = {}
        
    def calculate_volume_profile(self, df, price_bins=50):
        """Calculate volume profile for intraday analysis"""
        price_min, price_max = df['Low'].min(), df['High'].max()
        bin_size = (price_max - price_min) / price_bins
        
        volume_profile = {}
        for i in range(price_bins):
            price_level = price_min + i * bin_size
            next_level = price_level + bin_size
            
            # Volume at this price level
            mask = (df['Low'] >= price_level) & (df['High'] <= next_level)
            volume_at_level = df[mask]['Volume'].sum()
            
            volume_profile[price_level] = volume_at_level
            
        self.volume_profile_data = volume_profile
        return volume_profile
    
    def generate_renko_charts(self, df, brick_size=2.0):
        """Generate Renko charts"""
        renko_data = []
        current_brick = df['Close'].iloc[0]
        direction = 0  # 0: flat, 1: up, -1: down
        
        for close in df['Close']:
            if direction == 0:
                if close >= current_brick + brick_size:
                    renko_data.append({'price': current_brick + brick_size, 'direction': 1})
                    current_brick += brick_size
                    direction = 1
                elif close <= current_brick - brick_size:
                    renko_data.append({'price': current_brick - brick_size, 'direction': -1})
                    current_brick -= brick_size
                    direction = -1
            elif direction == 1:
                if close >= current_brick + brick_size:
                    while close >= current_brick + brick_size:
                        renko_data.append({'price': current_brick + brick_size, 'direction': 1})
                        current_brick += brick_size
                elif close <= current_brick - brick_size:
                    renko_data.append({'price': current_brick - brick_size, 'direction': -1})
                    current_brick -= brick_size
                    direction = -1
            else:  # direction == -1
                if close <= current_brick - brick_size:
                    while close <= current_brick - brick_size:
                        renko_data.append({'price': current_brick - brick_size, 'direction': -1})
                        current_brick -= brick_size
                elif close >= current_brick + brick_size:
                    renko_data.append({'price': current_brick + brick_size, 'direction': 1})
                    current_brick += brick_size
                    direction = 1
                    
        return pd.DataFrame(renko_data)