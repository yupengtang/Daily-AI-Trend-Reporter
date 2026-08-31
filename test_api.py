#!/usr/bin/env python3
"""
Preflight check: verify paper fetching and the configured LLM backend.

This runs before generation in CI so that a dead or misconfigured backend is
reported as such, rather than showing up later as a weekend post that silently
never appears.
"""

import requests

from batch_generate import API_KEY, ENDPOINT, LLM_PROVIDER, MODEL, PROVIDER


def test_paper_fetching():
    """Test fetching papers from Hugging Face"""
    print("Testing paper fetching from Hugging Face...")

    try:
        response = requests.get("https://huggingface.co/api/daily_papers", timeout=15)
        if response.status_code == 200:
            papers = response.json()
            print(f"OK: Hugging Face daily_papers reachable ({len(papers)} entries)")
            return True
        print(f"FAIL: could not fetch papers: HTTP {response.status_code}")
        return False
    except Exception as e:
        print(f"FAIL: error testing paper fetching: {e}")
        return False


def test_api():
    """Test the configured chat-completions backend"""
    print(f"Testing LLM backend: provider={LLM_PROVIDER} model={MODEL}")
    print(f"Endpoint: {ENDPOINT}")

    if not API_KEY:
        print(f"FAIL: {PROVIDER['key_env']} (or LLM_API_KEY) is not set")
        return False

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "messages": [
            {"role": "user", "content": "Reply with exactly: API test successful"}
        ],
        "model": MODEL,
        # Generous: reasoning models spend part of the budget before answering.
        "max_tokens": 2000,
    }

    try:
        response = requests.post(ENDPOINT, headers=headers, json=data, timeout=60)

        if response.status_code == 200:
            content = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            print(f"OK: API test successful. Response: {content!r}")
            return True

        print(f"FAIL: API test failed with status {response.status_code}")
        print(f"Response: {response.text[:500]}")
        return False

    except Exception as e:
        print(f"FAIL: API test failed with exception: {e}")
        return False


def main():
    """Run all tests"""
    print("Starting preflight checks...\n")

    paper_success = test_paper_fetching()
    print()

    api_success = test_api()
    print()

    if paper_success and api_success:
        print("All checks passed. The system is ready to generate content.")
        return True

    print("Some checks failed. Please check the configuration.")
    return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
