#!/usr/bin/env python3
"""
Test script to validate RSS feed format
"""

import xml.etree.ElementTree as ET
import requests
import sys

def validate_rss_feed(url):
    """Validate RSS feed format"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        
        # Check if it's a valid RSS feed
        if root.tag != 'rss':
            print("❌ Error: Root element is not 'rss'")
            return False
            
        # Check RSS version
        version = root.get('version')
        if version != '2.0':
            print(f"⚠️  Warning: RSS version is {version}, expected 2.0")
            
        # Check channel element
        channel = root.find('channel')
        if channel is None:
            print("❌ Error: No 'channel' element found")
            return False
            
        # Check required channel elements
        required_elements = ['title', 'link', 'description']
        for element in required_elements:
            if channel.find(element) is None:
                print(f"❌ Error: Missing required channel element: {element}")
                return False
                
        # Check items
        items = channel.findall('item')
        if not items:
            print("⚠️  Warning: No items found in RSS feed")
        else:
            print(f"✅ Found {len(items)} items in RSS feed")
            
        # Validate each item
        for i, item in enumerate(items):
            item_title = item.find('title')
            item_link = item.find('link')
            item_description = item.find('description')
            
            if item_title is None:
                print(f"⚠️  Warning: Item {i+1} missing title")
            if item_link is None:
                print(f"⚠️  Warning: Item {i+1} missing link")
            if item_description is None:
                print(f"⚠️  Warning: Item {i+1} missing description")
                
        print("✅ RSS feed validation completed")
        return True
        
    except requests.RequestException as e:
        print(f"❌ Error fetching RSS feed: {e}")
        return False
    except ET.ParseError as e:
        print(f"❌ Error parsing XML: {e}")
        return False

if __name__ == "__main__":
    rss_url = "https://yupengtang.github.io/Daily-AI-Trend-Reporter/feed.xml"
    print(f"Testing RSS feed: {rss_url}")
    print("-" * 50)
    
    success = validate_rss_feed(rss_url)
    
    if success:
        print("\n🎉 RSS feed is valid!")
    else:
        print("\n❌ RSS feed has issues that need to be fixed.")
        sys.exit(1) 