#!/usr/bin/env python3
"""
Test script to verify paper fetching logic
"""

import sys
import os
sys.path.append('.github/scripts')

# Import the function from generate_blog.py
from generate_blog import fetch_latest_papers

def test_paper_fetching():
    """Test the paper fetching function"""
    print("🔍 Testing paper fetching logic...")
    
    papers = fetch_latest_papers()
    
    if papers:
        print(f"✅ Successfully fetched {len(papers)} papers")
        print("\n📄 Sample papers:")
        for i, paper in enumerate(papers[:3], 1):
            print(f"{i}. {paper['title']}")
            print(f"   URL: {paper['url']}")
            print()
    else:
        print("❌ No papers fetched")
    
    # Check for fake URLs
    fake_url_patterns = ["paper-", "2507.2199", "paper-1", "paper-2"]
    has_fake_urls = False
    
    for paper in papers:
        url = paper.get('url', '')
        for pattern in fake_url_patterns:
            if pattern in url:
                print(f"⚠️ Found fake URL: {url}")
                has_fake_urls = True
    
    if not has_fake_urls:
        print("✅ No fake URLs detected")
    else:
        print("❌ Fake URLs detected - needs fixing")

if __name__ == "__main__":
    test_paper_fetching() 