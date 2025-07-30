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
                        print(f"📄 Latest 10 papers for focused digest:")
                        for i, paper in enumerate(papers[:10], 1):  # Show first 10 papers
                            title = paper.get('title', 'Unknown')
                            url = paper.get('url', 'https://huggingface.co/papers')
                            print(f"  {i}. {title}")
                            print(f"     🔗 {url}")
                        return True
            except Exception as e:
                print(f"⚠️ Failed to fetch from {endpoint}: {e}")
                continue
        
        # Fallback: try to scrape from the papers page
        try:
            # Try to get papers from the specific date page first
            response = requests.get("https://huggingface.co/papers/date/2025-07-29", timeout=15)
            if response.status_code != 200:
                # Fallback to general papers page
                response = requests.get("https://huggingface.co/papers", timeout=15)
            
            if response.status_code == 200:
                content = response.text
                
                # Look for paper links with titles (most accurate method)
                paper_with_title_pattern = r'<a[^>]*href="/papers/([^"]+)"[^>]*>([^<]+)</a>'
                papers_with_titles = re.findall(paper_with_title_pattern, content)
                
                clean_papers = []
                if papers_with_titles:
                    for i, (paper_id, title) in enumerate(papers_with_titles[:10]):
                        title = title.strip()
                        if len(title) > 10 and len(title) < 200:
                            clean_papers.append({
                                'title': title,
                                'url': f"https://huggingface.co/papers/{paper_id}"
                            })
                
                if clean_papers:
                    print(f"✅ Successfully extracted {len(clean_papers)} papers from HTML")
                    print(f"📄 Latest 10 papers for focused digest:")
                    for i, paper in enumerate(clean_papers[:10], 1):  # Show first 10 papers
                        print(f"  {i}. {paper['title']}")
                        print(f"     🔗 {paper['url']}")
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