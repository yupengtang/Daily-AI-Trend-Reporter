#!/usr/bin/env python3
"""
测试RSS feed格式修复
"""

import requests
import xml.etree.ElementTree as ET
import sys
import time

def test_rss_fix():
    """测试RSS feed格式修复"""
    url = "https://yupengtang.github.io/Daily-AI-Trend-Reporter/feed.xml"
    
    print("🔧 测试RSS feed格式修复...")
    print(f"URL: {url}")
    print("-" * 50)
    
    # 等待GitHub Pages更新
    print("⏳ 等待GitHub Pages更新（可能需要几分钟）...")
    
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            response = requests.get(url)
            response.raise_for_status()
            content = response.text
            
            print(f"📄 尝试 {attempt + 1}/{max_attempts} - 获取到的内容前200字符:")
            print(content[:200])
            print("-" * 50)
            
            # 检查是否包含XML声明
            if not content.startswith('<?xml'):
                print("❌ 错误: 缺少XML声明")
                if attempt < max_attempts - 1:
                    print("⏳ 等待30秒后重试...")
                    time.sleep(30)
                    continue
                else:
                    print("❌ 多次尝试后仍未修复")
                    return False
            
            # 检查是否包含RSS根元素
            if '<rss' not in content:
                print("❌ 错误: 缺少RSS根元素")
                if attempt < max_attempts - 1:
                    print("⏳ 等待30秒后重试...")
                    time.sleep(30)
                    continue
                else:
                    print("❌ 多次尝试后仍未修复")
                    return False
            
            # 检查是否包含channel元素
            if '<channel>' not in content:
                print("❌ 错误: 缺少channel元素")
                if attempt < max_attempts - 1:
                    print("⏳ 等待30秒后重试...")
                    time.sleep(30)
                    continue
                else:
                    print("❌ 多次尝试后仍未修复")
                    return False
            
            # 检查必需的元素
            required_elements = ['<title>', '<link>', '<description>']
            for element in required_elements:
                if element not in content:
                    print(f"❌ 错误: 缺少必需元素 {element}")
                    if attempt < max_attempts - 1:
                        print("⏳ 等待30秒后重试...")
                        time.sleep(30)
                        continue
                    else:
                        print("❌ 多次尝试后仍未修复")
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
                if attempt < max_attempts - 1:
                    print("⏳ 等待30秒后重试...")
                    time.sleep(30)
                    continue
                else:
                    return False
            
            print("\n🎉 RSS feed格式修复成功！")
            return True
            
        except requests.RequestException as e:
            print(f"❌ 网络错误: {e}")
            if attempt < max_attempts - 1:
                print("⏳ 等待30秒后重试...")
                time.sleep(30)
                continue
            else:
                return False
        except Exception as e:
            print(f"❌ 未知错误: {e}")
            if attempt < max_attempts - 1:
                print("⏳ 等待30秒后重试...")
                time.sleep(30)
                continue
            else:
                return False
    
    return False

if __name__ == "__main__":
    success = test_rss_fix()
    if not success:
        print("\n❌ RSS feed格式仍有问题")
        print("💡 建议：")
        print("1. 提交代码到GitHub")
        print("2. 等待GitHub Pages重新构建（通常需要5-10分钟）")
        print("3. 再次运行测试脚本")
        sys.exit(1)
    else:
        print("\n✅ RSS feed格式已修复！")
        print("🎉 现在用户可以正常订阅了！") 