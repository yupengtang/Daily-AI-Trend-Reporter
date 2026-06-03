#!/usr/bin/env python3
"""
Test script to verify GitHub Models API connection and paper fetching
"""

import os
import requests


def test_paper_fetching():
    """Test fetching papers from Hugging Face"""
    print("🔍 Testing paper fetching from Hugging Face...")

    try:
        response = requests.get("https://huggingface.co/papers", timeout=10)
        if response.status_code == 200:
            print("✅ Successfully connected to Hugging Face Papers")
            print(f"📄 Page size: {len(response.text)} characters")
            return True
        else:
            print(f"❌ Failed to fetch papers: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing paper fetching: {e}")
        return False


def test_api():
    """Test the GitHub Models API connection using requests"""

    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN environment variable not set")
        return False

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    data = {
        "messages": [
            {"role": "user", "content": "Hello! Please respond with 'API test successful'"}
        ],
        "model": "openai/gpt-4o-mini",
        "max_tokens": 50,
    }

    try:
        response = requests.post(
            "https://models.github.ai/inference/chat/completions",
            headers=headers,
            json=data,
            timeout=30,
        )

        if response.status_code == 200:
            content = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            print(f"✅ API test successful!")
            print(f"Response: {content}")
            return True
        else:
            print(f"❌ API test failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ API test failed with exception: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting comprehensive tests...\n")

    paper_success = test_paper_fetching()
    print()

    api_success = test_api()
    print()

    if paper_success and api_success:
        print("🎉 All tests passed! The system is ready to generate content.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the configuration.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
