import google.generativeai as genai
import os

# Set your API key
api_key = "AIzaSyAt5jVVFdUJmCQSEA-U28oht6vCWXs2v_s"

try:
    # Configure the API
    genai.configure(api_key=api_key)
    
    # List available models to test connection
    models = genai.list_models()
    
    # Check if Gemini models are available
    gemini_models = [model.name for model in models if 'gemini' in model.name.lower()]
    
    if gemini_models:
        print("✅ API key is valid and has access to Gemini models")
        print("Available Gemini models:", gemini_models)
        
        # Test a simple prompt
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Hello, are you working?")
        print("Test response:", response.text)
    else:
        print("❌ API key is valid but no Gemini models found")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("This likely means your API key is invalid or doesn't have access to Gemini API")