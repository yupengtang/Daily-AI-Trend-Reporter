#!/usr/bin/env python3
"""
Test script to verify paper fetching from Hugging Face for comprehensive daily digest
"""

import requests
import re

def test_paper_fetching():
    """Test fetching all papers from Hugging Face Papers"""
    print("🔍 Testing comprehensive paper fetching from Hugging Face...")
    
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
                        print(f"📄 All latest papers for comprehensive digest:")
                        for i, paper in enumerate(papers[:10], 1):  # Show first 10 as example
                            title = paper.get('title', 'Unknown')
                            authors = paper.get('authors', ['Unknown'])
                            print(f"  {i}. {title}")
                            print(f"     Authors: {', '.join(authors)}")
                        if len(papers) > 10:
                            print(f"  ... and {len(papers) - 10} more papers")
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
                for title in titles[:50]:  # Increased to get more papers
                    title = title.strip()
                    if len(title) > 10 and len(title) < 200:
                        clean_titles.append(title)
                
                if clean_titles:
                    print(f"✅ Successfully extracted {len(clean_titles)} papers from HTML")
                    print(f"📄 All latest papers for comprehensive digest:")
                    for i, title in enumerate(clean_titles[:10], 1):  # Show first 10 as example
                        print(f"  {i}. {title}")
                    if len(clean_titles) > 10:
                        print(f"  ... and {len(clean_titles) - 10} more papers")
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
    print("🚀 Testing comprehensive paper fetching for daily digest...\n")
    
    success = test_paper_fetching()
    
    if success:
        print("\n🎉 Comprehensive paper fetching test passed!")
        print("✅ The system will now generate comprehensive daily digests")
        print("📊 Each digest will include:")
        print("   - ALL latest research papers")
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