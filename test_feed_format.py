#!/usr/bin/env python3
"""
测试RSS feed格式是否正确
"""

import requests
import xml.etree.ElementTree as ET
import sys

def test_rss_format():
    """测试RSS feed格式"""
    url = "https://yupengtang.github.io/Daily-AI-Trend-Reporter/feed.xml"
    
    print("🔍 测试RSS feed格式...")
    print(f"URL: {url}")
    print("-" * 50)
    
    try:
        # 获取RSS feed内容
        response = requests.get(url)
        response.raise_for_status()
        content = response.text
        
        print("📄 获取到的内容前200字符:")
        print(content[:200])
        print("-" * 50)
        
        # 检查是否包含XML声明
        if not content.startswith('<?xml'):
            print("❌ 错误: 缺少XML声明")
            print("应该以 '<?xml version=\"1.0\" encoding=\"UTF-8\"?>' 开头")
            return False
        
        # 检查是否包含RSS根元素
        if '<rss' not in content:
            print("❌ 错误: 缺少RSS根元素")
            print("应该包含 '<rss version=\"2.0\">'")
            return False
        
        # 检查是否包含channel元素
        if '<channel>' not in content:
            print("❌ 错误: 缺少channel元素")
            return False
        
        # 检查必需的元素
        required_elements = ['<title>', '<link>', '<description>']
        for element in required_elements:
            if element not in content:
                print(f"❌ 错误: 缺少必需元素 {element}")
                return False
        
        # 检查是否有item元素
        if '<item>' not in content:
            print("⚠️  警告: 没有找到item元素（可能没有文章）")
        else:
            item_count = content.count('<item>')
            print(f"✅ 找到 {item_count} 个item元素")
        
        # 尝试解析XML
        try:
            root = ET.fromstring(content)
            print("✅ XML格式正确，可以成功解析")
            
            # 检查RSS版本
            if root.tag == 'rss':
                version = root.get('version')
                print(f"✅ RSS版本: {version}")
            else:
                print(f"⚠️  警告: 根元素是 {root.tag}，期望是 'rss'")
                
        except ET.ParseError as e:
            print(f"❌ XML解析错误: {e}")
            return False
        
        print("\n🎉 RSS feed格式测试通过！")
        return True
        
    except requests.RequestException as e:
        print(f"❌ 网络错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False

if __name__ == "__main__":
    success = test_rss_format()
    if not success:
        print("\n❌ RSS feed格式有问题，需要修复")
        sys.exit(1)
    else:
        print("\n✅ RSS feed格式正确！") 