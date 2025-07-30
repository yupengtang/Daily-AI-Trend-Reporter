#!/usr/bin/env python3
"""
Test script to verify paper fetching from Hugging Face for focused daily digest
"""

import requests
import re

def test_paper_fetching():
    """Test fetching papers from Hugging Face Papers"""
    print("🔍 Testing focused paper fetching from Hugging Face...")
    
    try:
        # Try multiple endpoints
        endpoints = [
            "https://huggingface.co/api/papers",
            "https://huggingface.co/api/papers?sort=date&direction=-1",
            "https://huggingface.co/api/papers?limit=100"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=15)
                if response.status_code == 200:
                    papers = response.json()
                    if papers and isinstance(papers, list):
                        print(f"✅ Successfully fetched {len(papers)} papers from {endpoint}")
                        print(f"📄 Latest 6 papers for focused digest:")
                        for i, paper in enumerate(papers[:6], 1):  # Show first 6 papers
                            title = paper.get('title', 'Unknown')
                            print(f"  {i}. {title}")
                        return True
            except Exception as e:
                print(f"⚠️ Failed to fetch from {endpoint}: {e}")
                continue
        
        # Fallback: try to scrape from the papers page
        try:
            response = requests.get("https://huggingface.co/papers", timeout=15)
            if response.status_code == 200:
                content = response.text
                title_pattern = r'<h[1-6][^>]*>([^<]+)</h[1-6]>'
                titles = re.findall(title_pattern, content)
                
                clean_titles = []
                for title in titles[:10]:
                    title = title.strip()
                    if len(title) > 10 and len(title) < 200:
                        clean_titles.append(title)
                
                if clean_titles:
                    print(f"✅ Successfully extracted {len(clean_titles)} papers from HTML")
                    print(f"📄 Latest 6 papers for focused digest:")
                    for i, title in enumerate(clean_titles[:6], 1):  # Show first 6 papers
                        print(f"  {i}. {title}")
                    return True
        except Exception as e:
            print(f"⚠️ Failed to scrape papers page: {e}")
        
        print("❌ Could not fetch papers from any source")
        return False
        
    except Exception as e:
        print(f"❌ Error testing paper fetching: {e}")
        return False

def main():
    """Run the test"""
    print("🚀 Testing focused paper fetching for daily digest...\n")
    
    success = test_paper_fetching()
    
    if success:
        print("\n🎉 Focused paper fetching test passed!")
        print("✅ The system will now generate focused daily digests")
        print("📊 Each digest will include:")
        print("   - Latest 6 research papers")
        print("   - Individual summaries for each paper")
        print("   - Direct links to original papers")
        print("   - Consistent format across all posts")
    else:
        print("\n⚠️ Paper fetching test failed")
        print("The system will fall back to general topic generation")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 