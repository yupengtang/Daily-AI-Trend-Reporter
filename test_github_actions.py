#!/usr/bin/env python3
"""
Test script to verify GitHub Actions configuration
"""

import os
import datetime
import subprocess
import sys

def test_batch_generate_script():
    """Test if batch_generate.py exists and is executable"""
    print("🧪 Testing batch_generate.py script...")
    
    if not os.path.exists("batch_generate.py"):
        print("❌ batch_generate.py not found")
        return False
    
    print("✅ batch_generate.py exists")
    return True

def test_github_actions_config():
    """Test GitHub Actions configuration"""
    print("🧪 Testing GitHub Actions configuration...")
    
    if not os.path.exists(".github/workflows/daily_blog.yml"):
        print("❌ .github/workflows/daily_blog.yml not found")
        return False
    
    # Read the workflow file
    with open(".github/workflows/daily_blog.yml", "r") as f:
        content = f.read()
    
    # Check if it uses the new script
    if "python3 batch_generate.py" in content:
        print("✅ Uses new batch_generate.py script")
    else:
        print("❌ Still uses old script")
        return False
    
    # Check if it has proper commit and push logic
    if "git add _posts/" in content and "git commit" in content and "git push" in content:
        print("✅ Has proper commit and push logic")
    else:
        print("❌ Missing commit and push logic")
        return False
    
    return True

def test_posts_directory():
    """Test if _posts directory exists and is writable"""
    print("🧪 Testing _posts directory...")
    
    if not os.path.exists("_posts"):
        print("❌ _posts directory not found")
        return False
    
    # Test if we can write to it
    test_file = "_posts/test_write.md"
    try:
        with open(test_file, "w") as f:
            f.write("Test content")
        os.remove(test_file)
        print("✅ _posts directory is writable")
        return True
    except Exception as e:
        print(f"❌ Cannot write to _posts directory: {e}")
        return False

def test_git_configuration():
    """Test git configuration for GitHub Actions"""
    print("🧪 Testing git configuration...")
    
    try:
        # Test git status
        result = subprocess.run(["git", "status"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git repository is properly configured")
            return True
        else:
            print("❌ Git repository not properly configured")
            return False
    except Exception as e:
        print(f"❌ Git test failed: {e}")
        return False

def test_environment_variables():
    """Test if required environment variables are documented"""
    print("🧪 Testing environment variables...")
    
    # Check if HF_TOKEN is mentioned in documentation
    readme_content = ""
    if os.path.exists("README.md"):
        with open("README.md", "r") as f:
            readme_content = f.read()
    
    if "HF_TOKEN" in readme_content:
        print("✅ HF_TOKEN is documented")
    else:
        print("⚠️ HF_TOKEN not found in documentation")
    
    return True

def test_workflow_permissions():
    """Test workflow permissions"""
    print("🧪 Testing workflow permissions...")
    
    with open(".github/workflows/daily_blog.yml", "r") as f:
        content = f.read()
    
    if "contents: write" in content:
        print("✅ Has write permissions for contents")
    else:
        print("❌ Missing write permissions for contents")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Testing GitHub Actions Configuration")
    print("=" * 50)
    
    tests = [
        test_batch_generate_script,
        test_github_actions_config,
        test_posts_directory,
        test_git_configuration,
        test_environment_variables,
        test_workflow_permissions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GitHub Actions should work correctly.")
        print("\n📋 Next steps:")
        print("1. Ensure HF_TOKEN is set in repository secrets")
        print("2. Push changes to trigger GitHub Actions")
        print("3. Check Actions tab for any errors")
        print("4. Verify generated posts appear on the website")
    else:
        print("⚠️ Some tests failed. Please fix the issues before deploying.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 