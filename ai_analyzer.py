import os
import json
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Try to import groq, but make it optional
try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

class AIAnalyzer:
    """AI-powered market analysis using GROQ"""
    
    def __init__(self, api_key: str = None):
        """Initialize the AI analyzer with an optional API key.
        
        Args:
            api_key: Optional GROQ API key. If not provided, will try to load from .env file.
        """
        if not GROQ_AVAILABLE:
            raise ImportError("groq package is not installed. Please install it with: pip install groq")
            
        load_dotenv()
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        
        if not self.api_key or self.api_key == "your_actual_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY not found or using placeholder. Please update the .env file with your actual API key.\n"
                "You can get an API key from https://console.groq.com/keys"
            )
            
        self.client = groq.Client(api_key=self.api_key)
        self.last_api_call = 0
        self.min_call_interval = 1.0  # Minimum seconds between API calls
        # Use a production model from GROQ
        self.model = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')
        print(f"Using GROQ model: {self.model}")  # Debug log
    
    def _rate_limit(self):
        """Ensure we don't exceed API rate limits"""
        elapsed = time.time() - self.last_api_call
        if elapsed < self.min_call_interval:
            time.sleep(self.min_call_interval - elapsed)
        self.last_api_call = time.time()
    
    def analyze_market_conditions(self, symbol: str, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI analysis of market conditions"""
        try:
            # Prepare the technical data for the prompt
            tech_summary = (
                f"Symbol: {symbol}\n"
                f"Current Price: {technical_data.get('current_price', 'N/A')}\n"
                f"20-Day MA: {technical_data.get('sma_20', 'N/A')}\n"
                f"50-Day MA: {technical_data.get('sma_50', 'N/A')}\n"
                f"RSI: {technical_data.get('rsi', 'N/A')}\n"
                f"Trend: {technical_data.get('trend', 'N/A')}\n"
                f"Momentum: {technical_data.get('momentum', 'N/A')}"
            )
            
            print("\n📊 Technical Summary for AI Analysis:")
            print("-" * 50)
            print(tech_summary)
            print("-" * 50)
            
            # Create the prompt for GROQ
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert financial analyst with deep knowledge of technical analysis 
                    and market behavior. Provide concise, actionable insights based on the technical data."""
                },
                {
                    "role": "user",
                    "content": f"""Please analyze this stock's technical indicators and provide insights:
                    
                    {tech_summary}
                    
                    Focus on:
                    1. Current market condition
                    2. Key technical observations
                    3. Short-term outlook (1-5 days)
                    4. Support and resistance levels
                    5. Trading recommendation (Buy/Sell/Hold)"""
                }
            ]
            
            # Add rate limiting
            self._rate_limit()
            
            print("\n🤖 Calling GROQ API...")
            print(f"Using model: {self.model}")
            
            try:
                # Call GROQ API with the current model
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=0.3,
                    max_tokens=500,
                    top_p=1,
                    stream=False,
                    stop=None,
                )
                
                print(f"API Response Status: {response}")
                
                # Extract and format the response
                if not hasattr(response, 'choices') or not response.choices:
                    raise ValueError("No choices in API response")
                    
                if not hasattr(response.choices[0], 'message') or not hasattr(response.choices[0].message, 'content'):
                    raise ValueError("Invalid response format from API")
                
                analysis = response.choices[0].message.content
                
                if not analysis:
                    raise ValueError("Empty analysis in API response")
                
                print("✅ Successfully received AI Analysis")
                
                return {
                    "analysis": analysis,
                    "recommendation": self._extract_recommendation(analysis),
                    "confidence": 0.8  # Placeholder for confidence score
                }
                
            except Exception as api_error:
                print(f"\n❌ Error in GROQ API call: {str(api_error)}")
                print(f"Error type: {type(api_error).__name__}")
                if hasattr(api_error, 'response') and api_error.response:
                    print(f"Response status: {api_error.response.status_code}")
                    print(f"Response body: {api_error.response.text}")
                raise
            
        except Exception as e:
            error_msg = f"AI Analysis failed: {str(e)}"
            print(f"\n❌ {error_msg}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            return {
                "error": error_msg,
                "recommendation": "No recommendation available",
                "confidence": 0.0
            }
    
    def _extract_recommendation(self, analysis: str) -> str:
        """Extract a clear recommendation from the analysis text"""
        # This is a simple implementation - could be enhanced with more sophisticated NLP
        analysis_lower = analysis.lower()
        if any(word in analysis_lower for word in ["buy", "bullish", "oversold"]):
            return "Consider Buy"
        elif any(word in analysis_lower for word in ["sell", "bearish", "overbought"]):
            return "Consider Sell"
        return "Hold or Wait"
    
    def generate_trade_idea(self, symbol: str, analysis: str) -> Dict[str, Any]:
        """Generate a detailed trade idea based on analysis"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert financial advisor. Generate a detailed trade idea 
                    based on the provided analysis. Include entry, stop-loss, and take-profit levels.
                    
                    Format your response in a clear, structured way with the following sections:
                    
                    [Trade Setup]:
                    - Type: (Swing/Position/Intraday)
                    - Timeframe: (e.g., 1-5 days, 1-2 weeks, etc.)
                    
                    [Entry]:
                    - Ideal entry price range
                    - Conditions for entry
                    
                    [Stop-Loss]:
                    - Price level
                    - Rationale for placement
                    
                    [Take-Profit Targets]:
                    - Target 1: (price, % gain)
                    - Target 2: (price, % gain)
                    - Final Target: (price, % gain)
                    
                    [Risk Management]:
                    - Position sizing suggestion
                    - Risk-reward ratio
                    - Key levels to watch
                    
                    [Additional Notes]:
                    - Any important considerations
                    - Upcoming events that might impact the trade"""
                },
                {
                    "role": "user",
                    "content": f"""Based on this analysis for {symbol}, please provide a detailed trade idea:
                    
                    {analysis}"""
                }
            ]
            
            # Add rate limiting
            self._rate_limit()
            
            response = self.client.chat.completions.create(
                messages=messages,
                model="mixtral-8x7b-32768",
                temperature=0.3,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            return {
                "trade_idea": response.choices[0].message.content
            }
            
        except Exception as e:
            return {"error": f"Failed to generate trade idea: {str(e)}"}
