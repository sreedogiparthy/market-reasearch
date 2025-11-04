"""Test GROQ API connection"""
import os
from dotenv import load_dotenv
import groq

def test_connection(api_key: str):
    """Test GROQ API connection with the provided API key"""
    try:
        client = groq.Client(api_key=api_key)
        
        # Simple test message
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"}
        ]
        
        print("Sending test request to GROQ API...")
        response = client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",
            max_tokens=20
        )
        
        if response.choices and response.choices[0].message.content:
            print("✅ Success! GROQ API is working.")
            print(f"Response: {response.choices[0].message.content}")
            return True
        else:
            print("❌ Empty response from GROQ API")
            
    except Exception as e:
        print(f"❌ Error connecting to GROQ API: {str(e)}")
        
    return False

def main():
    """Main function to test GROQ API connection"""
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key or api_key == "your_actual_groq_api_key_here":
        print("❌ Please update the .env file with your actual GROQ API key")
        print("Get your API key from: https://console.groq.com/keys")
        return
    
    print(f"Testing GROQ API connection with key: {api_key[:8]}...{api_key[-4:]}")
    
    if not test_connection(api_key):
        print("\nTroubleshooting steps:")
        print("1. Verify your API key is correct")
        print("2. Check your internet connection")
        print("3. Visit https://status.groq.com/ to check if GROQ services are operational")
        print("4. Ensure your account has sufficient credits")

if __name__ == "__main__":
    main()
