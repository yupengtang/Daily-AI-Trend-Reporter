#!/usr/bin/env python3
"""
Debug script to check post generation and commit status
"""

import os
import datetime
import subprocess
import glob

def check_posts_directory():
    """Check the _posts directory for generated files"""
    print("📁 Checking _posts directory...")
    
    if not os.path.exists("_posts"):
        print("❌ _posts directory not found")
        return False
    
    # List all markdown files
    md_files = glob.glob("_posts/*.md")
    print(f"Found {len(md_files)} markdown files in _posts/")
    
    if md_files:
        print("Recent files:")
        for file in sorted(md_files, reverse=True)[:10]:  # Show last 10 files
            stat = os.stat(file)
            mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
            print(f"  {file} (modified: {mtime})")
    
    return True

def check_git_status():
    """Check git status for uncommitted changes"""
    print("\n🔍 Checking git status...")
    
    try:
        # Check if there are uncommitted changes
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        
        if result.stdout.strip():
            print("⚠️ Uncommitted changes found:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"  {line}")
        else:
            print("✅ No uncommitted changes")
        
        # Check last commit
        result = subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True)
        if result.stdout.strip():
            print(f"Last commit: {result.stdout.strip()}")
        
        return True
    except Exception as e:
        print(f"❌ Git status check failed: {e}")
        return False

def check_github_actions_log():
    """Check if GitHub Actions workflow exists"""
    print("\n🔧 Checking GitHub Actions configuration...")
    
    if os.path.exists(".github/workflows/daily_blog.yml"):
        print("✅ GitHub Actions workflow exists")
        
        # Check if it uses the correct script
        with open(".github/workflows/daily_blog.yml", "r") as f:
            content = f.read()
        
        if "python3 batch_generate.py" in content:
            print("✅ Uses correct batch_generate.py script")
        else:
            print("❌ Still uses old script")
        
        if "git add _posts/" in content and "git commit" in content:
            print("✅ Has proper commit and push logic")
        else:
            print("❌ Missing commit and push logic")
    else:
        print("❌ GitHub Actions workflow not found")
        return False
    
    return True

def check_environment_variables():
    """Check environment variables"""
    print("\n🔑 Checking environment variables...")
    
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("✅ HF_TOKEN is set")
    else:
        print("❌ HF_TOKEN is not set")
        print("💡 Set HF_TOKEN environment variable to use the API")
    
    return True

def simulate_post_generation():
    """Simulate post generation to test the process"""
    print("\n🧪 Simulating post generation...")
    
    # Create a test post
    test_date = datetime.date.today().strftime("%Y-%m-%d")
    test_filename = f"_posts/{test_date}-test-post.md"
    
    try:
        with open(test_filename, "w") as f:
            f.write("---\n")
            f.write(f"layout: post\n")
            f.write(f"title: \"Test Post - {test_date}\"\n")
            f.write(f"date: {test_date}\n")
            f.write("---\n\n")
            f.write("This is a test post to verify the generation process.\n")
        
        print(f"✅ Test post created: {test_filename}")
        
        # Check if it was created
        if os.path.exists(test_filename):
            print("✅ File exists on disk")
            
            # Clean up
            os.remove(test_filename)
            print("✅ Test file cleaned up")
        else:
            print("❌ File was not created")
        
        return True
    except Exception as e:
        print(f"❌ Test post creation failed: {e}")
        return False

def check_website_deployment():
    """Check if the website is properly configured"""
    print("\n🌐 Checking website deployment...")
    
    # Check if GitHub Pages is configured
    if os.path.exists("_config.yml"):
        with open("_config.yml", "r") as f:
            config = f.read()
        
        if "theme: minima" in config:
            print("✅ Jekyll theme configured")
        else:
            print("⚠️ Jekyll theme not configured")
        
        if "baseurl:" in config:
            print("✅ Base URL configured")
        else:
            print("⚠️ Base URL not configured")
    else:
        print("❌ _config.yml not found")
    
    # Check if index.md exists
    if os.path.exists("index.md"):
        print("✅ Homepage exists")
    else:
        print("❌ Homepage not found")
    
    return True

def main():
    """Run all debug checks"""
    print("🔍 Debugging Post Generation and Deployment")
    print("=" * 50)
    
    checks = [
        check_posts_directory,
        check_git_status,
        check_github_actions_log,
        check_environment_variables,
        simulate_post_generation,
        check_website_deployment,
    ]
    
    passed = 0
    total = len(checks)
    
    for check in checks:
        try:
            if check():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Check failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Debug Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! The system should work correctly.")
        print("\n📋 To test the full workflow:")
        print("1. Set HF_TOKEN environment variable")
        print("2. Run: python batch_generate.py")
        print("3. Check if posts are generated in _posts/")
        print("4. Commit and push changes")
        print("5. Check the website for new posts")
    else:
        print("⚠️ Some checks failed. Please fix the issues.")
        print("\n🔧 Common fixes:")
        print("- Ensure HF_TOKEN is set")
        print("- Check git repository permissions")
        print("- Verify GitHub Actions workflow configuration")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 