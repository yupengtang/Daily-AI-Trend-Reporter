#!/usr/bin/env python3
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_api():
    token = os.getenv("HF_TOKEN")
    print(f"Token: {token[:10]}... (length: {len(token)})")
    
    # Test both endpoints
    endpoints = [
        "https://models.github.ai/inference/chat/completions",
        "https://models.inference.ai.azure.com/chat/completions"
    ]
    
    for endpoint in endpoints:
        print(f"\n🔄 Testing {endpoint}")
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "openai/gpt-4o-mini",
            "max_tokens": 50
        }
        
        try:
            response = requests.post(endpoint, headers=headers, json=data, timeout=30)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_api()