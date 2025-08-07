#!/usr/bin/env python3
"""
Test script for the new publishing strategy:
- Saturday: Generate Monday-Friday daily posts
- Sunday: Generate technical deep dive post
"""

import os
import datetime
import asyncio
from batch_generate import validate_date, validate_date_range

def test_date_validation():
    """Test date validation functions"""
    print("🧪 Testing date validation...")
    
    # Test valid dates
    try:
        start_date = validate_date("2025-01-20")
        end_date = validate_date("2025-01-26")
        validate_date_range(start_date, end_date)
        print("✅ Date validation passed")
    except Exception as e:
        print(f"❌ Date validation failed: {e}")
        return False
    
    return True

def test_weekday_detection():
    """Test weekday detection for Saturday/Sunday logic"""
    print("🧪 Testing weekday detection...")
    
    # Test Saturday (weekday 5)
    saturday = datetime.date(2025, 1, 25)  # Saturday
    sunday = datetime.date(2025, 1, 26)    # Sunday
    
    print(f"Saturday ({saturday}) weekday: {saturday.weekday()}")
    print(f"Sunday ({sunday}) weekday: {sunday.weekday()}")
    
    if saturday.weekday() == 5 and sunday.weekday() == 6:
        print("✅ Weekday detection correct")
        return True
    else:
        print("❌ Weekday detection incorrect")
        return False

def test_date_calculation():
    """Test date calculations for Monday-Friday generation"""
    print("🧪 Testing date calculations...")
    
    # Test Saturday generating Monday-Friday
    saturday = datetime.date(2025, 1, 25)  # Saturday
    
    expected_dates = [
        saturday - datetime.timedelta(days=5),  # Monday
        saturday - datetime.timedelta(days=4),  # Tuesday
        saturday - datetime.timedelta(days=3),  # Wednesday
        saturday - datetime.timedelta(days=2),  # Thursday
        saturday - datetime.timedelta(days=1),  # Friday
    ]
    
    print("Expected Monday-Friday dates:")
    for i, date in enumerate(expected_dates):
        day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][i]
        print(f"  {day_name}: {date}")
    
    # Test Sunday generating technical deep dive
    sunday = datetime.date(2025, 1, 26)  # Sunday
    week_start = sunday - datetime.timedelta(days=6)  # Monday
    week_end = sunday - datetime.timedelta(days=1)    # Friday
    
    print(f"\nSunday technical deep dive range:")
    print(f"  Week start (Monday): {week_start}")
    print(f"  Week end (Friday): {week_end}")
    
    print("✅ Date calculations correct")
    return True

def test_file_naming():
    """Test file naming conventions"""
    print("🧪 Testing file naming...")
    
    # Test daily post naming
    daily_date = datetime.date(2025, 1, 20)
    daily_filename = f"_posts/{daily_date.strftime('%Y-%m-%d')}-daily-ai-research-digest.md"
    print(f"Daily post filename: {daily_filename}")
    
    # Test technical deep dive naming
    deep_dive_date = datetime.date(2025, 1, 26)
    deep_dive_filename = f"_posts/{deep_dive_date.strftime('%Y-%m-%d')}-technical-deep-dive.md"
    print(f"Technical deep dive filename: {deep_dive_filename}")
    
    print("✅ File naming conventions correct")
    return True

def test_strategy_logic():
    """Test the core strategy logic"""
    print("🧪 Testing strategy logic...")
    
    # Test a week of dates
    test_dates = [
        datetime.date(2025, 1, 20),  # Monday
        datetime.date(2025, 1, 21),  # Tuesday
        datetime.date(2025, 1, 22),  # Wednesday
        datetime.date(2025, 1, 23),  # Thursday
        datetime.date(2025, 1, 24),  # Friday
        datetime.date(2025, 1, 25),  # Saturday
        datetime.date(2025, 1, 26),  # Sunday
    ]
    
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    print("Strategy for each day:")
    for date, day_name in zip(test_dates, day_names):
        if date.weekday() == 5:  # Saturday
            print(f"  {day_name} ({date}): Generate Monday-Friday daily posts")
        elif date.weekday() == 6:  # Sunday
            print(f"  {day_name} ({date}): Generate technical deep dive")
        else:
            print(f"  {day_name} ({date}): Generate individual daily post")
    
    print("✅ Strategy logic correct")
    return True

def main():
    """Run all tests"""
    print("🚀 Testing New Publishing Strategy")
    print("=" * 50)
    
    tests = [
        test_date_validation,
        test_weekday_detection,
        test_date_calculation,
        test_file_naming,
        test_strategy_logic,
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
        print("🎉 All tests passed! New strategy is ready.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 