"""Test GROQ API connection"""
import os
import pytest
from dotenv import load_dotenv
import groq

# Load environment variables from .env file
load_dotenv()

@pytest.fixture
def api_key():
    """Fixture to get the GROQ API key from environment variables"""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        pytest.skip("GROQ_API_KEY not found in environment variables")
    return key

@pytest.fixture
def model_name():
    """Fixture to get the GROQ model name from environment variables"""
    name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # Default to a common model if not set
    if not name:
        pytest.skip("GROQ_MODEL not found in environment variables")
    return name

def test_connection(api_key, model_name):
    """Test GROQ API connection with the provided API key and model"""
    # Initialize client
    client = groq.Client(api_key=api_key)
    
    # Simple test message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, world!"}
    ]
    
    # List available models first to see what's available
    print("Listing available models...")
    models = client.models.list()
    print("Available models:")
    for model in models.data:
        print(f"- {model.id}")
        
    print(f"\nAttempting to use model: {model_name}")
    
    # Make the API call
    response = client.chat.completions.create(
        messages=messages,
        model=model_name,
        max_tokens=20,
        temperature=0.7
    )
    
    # Assert the response is as expected
    assert response.choices is not None, "No choices in response"
    assert len(response.choices) > 0, "Empty choices list in response"
    assert response.choices[0].message is not None, "Message is None in response"
    assert response.choices[0].message.content is not None, "Message content is None"
    
    # Print success message
    print("✅ Success! GROQ API is working.")
    print(f"Response: {response.choices[0].message.content}")
    
    # No need to return anything - the test passes if no exceptions are raised

def main():
    """Main function to test GROQ API connection"""
    # Run pytest programmatically
    import sys
    sys.exit(pytest.main([__file__] + sys.argv[1:]))

if __name__ == "__main__":
    main()
