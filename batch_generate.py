#!/usr/bin/env python3
"""
Batch generation script for Daily AI Trend Reporter
Generates posts for a specified date range including weekly reports and technical deep dives
"""

import os
import datetime
import asyncio
import requests
import argparse
from typing import List, Dict, Optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set API configuration
HF_TOKEN = os.getenv("HF_TOKEN")  # GitHub Models token
ENDPOINT = "https://models.github.ai/inference/chat/completions"
MODEL = "openai/gpt-5"

# Number of papers to summarize each day
PAPERS_PER_DAY = 10

def validate_date(date_str: str) -> datetime.date:
    """Validate and parse date string"""
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD format.")

def validate_date_range(start_date: datetime.date, end_date: datetime.date) -> None:
    """Validate date range"""
    today = datetime.date.today()
    
    if start_date > today:
        raise ValueError(f"Start date {start_date} is in the future!")
    
    if end_date > today:
        raise ValueError(f"End date {end_date} is in the future!")
    
    if start_date > end_date:
        raise ValueError("Start date must be before or equal to end date!")

def fetch_papers_for_date(target_date: datetime.date) -> List[Dict]:
    """Fetch papers for a specific date from the Hugging Face API"""
    print(f"🔍 Fetching papers for {target_date}...")
    date_str = target_date.strftime("%Y-%m-%d")

    try:
        url = f"https://huggingface.co/api/daily_papers?date={date_str}"
        response = requests.get(url, timeout=15)

        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list):
                papers = []
                for item in data[:PAPERS_PER_DAY]:
                    paper = item.get("paper", {})
                    paper_id = paper.get("id", "")
                    authors = [a.get("name", "Unknown") for a in paper.get("authors", [])]
                    abstract = paper.get("ai_summary") or paper.get("summary") or f"Latest research on {paper.get('title', paper_id)}"
                    papers.append({
                        'title': paper.get("title", f"Research Paper {paper_id}"),
                        'authors': authors or ['Research Team'],
                        'abstract': abstract,
                        'ai_keywords': paper.get("ai_keywords", []),
                        'url': f"https://huggingface.co/papers/{paper_id}"
                    })
                print(f"✅ Found {len(papers)} papers for {date_str} via API")
                return papers

        print(f"⚠️ API returned status {response.status_code}, trying fallback...")
    except Exception as e:
        print(f"❌ Error fetching papers for {target_date}: {e}")

    return fetch_papers_fallback(target_date)

def fetch_papers_fallback(target_date: datetime.date) -> List[Dict]:
    """Fallback: fetch today's general papers list from HF API"""
    print(f"🔄 Using fallback method for {target_date}...")

    try:
        response = requests.get("https://huggingface.co/api/papers", timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list):
                papers = []
                for item in data[:PAPERS_PER_DAY]:
                    paper_id = item.get("id", "")
                    abstract = item.get("ai_summary") or item.get("summary") or f"Latest research on {item.get('title', paper_id)}"
                    papers.append({
                        'title': item.get("title", f"Research Paper {paper_id}"),
                        'authors': [a.get("name", "Unknown") for a in item.get("authors", [])],
                        'abstract': abstract,
                        'url': f"https://huggingface.co/papers/{paper_id}"
                    })
                print(f"✅ Fallback: found {len(papers)} papers")
                return papers
    except Exception as e:
        print(f"❌ Fallback also failed: {e}")

    return []

async def call_ai_api(messages, max_tokens=1024, temperature=0.3):
    """Call GitHub Models API"""
    if not HF_TOKEN:
        print("❌ HF_TOKEN not available")
        return None
        
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": messages,
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        print("🤖 Calling GitHub Models API...")
        response = requests.post(ENDPOINT, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            print("✅ GitHub Models API successful")
            return response
        else:
            print(f"❌ GitHub Models API failed with status {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ GitHub Models API call error: {e}")
        return None

async def generate_daily_post(target_date: datetime.date) -> Optional[str]:
    """Generate a daily blog post for a specific date"""
    print(f"🚀 Generating daily post for {target_date}...")
    
    # Get papers for the specific date
    papers = fetch_papers_for_date(target_date)
    
    if not papers:
        print(f"❌ No papers available for {target_date}")
        return None
    
    print(f"📝 Generating summaries for {len(papers)} papers...")
    
    # Generate summaries for each paper
    paper_summaries = []
    
    for i, paper in enumerate(papers, 1):
        print(f"📄 Processing paper {i}/{len(papers)}: {paper['title'][:50]}...")
        
        # Use the real abstract directly as summary
        summary = paper.get('abstract', f"Latest research on {paper['title']} with potential applications in AI/ML.")
        
        paper_summaries.append({
            'title': paper['title'],
            'summary': summary,
            'url': paper['url']
        })
        print(f"✅ Using real abstract for paper {i}")
    
    if paper_summaries:
        # Extract keywords from HF ai_keywords, ranked by frequency
        from collections import Counter
        kw_counter = Counter()
        for paper in papers:
            for kw in paper.get('ai_keywords', []):
                kw = kw.strip()
                if len(kw) >= 3 and not kw.startswith('$'):
                    kw_counter[kw] += 1
        # Take top 8 by frequency (most representative of today's papers)
        keywords_list = [kw for kw, _ in kw_counter.most_common(8)]

        # If fewer than 5, supplement with AI-generated keywords
        if len(keywords_list) < 5:
            keywords_prompt = (
                f"Based on these {len(paper_summaries)} research papers, generate exactly 8 relevant keywords.\n"
                f"Return ONLY a comma-separated list, nothing else.\n\n"
            )
            for i, paper in enumerate(paper_summaries, 1):
                keywords_prompt += f"{i}. {paper['title']}\n"
            keywords_response = await call_ai_api([
                {"role": "system", "content": "Output only a comma-separated keyword list."},
                {"role": "user", "content": keywords_prompt}
            ], max_tokens=100, temperature=0.3)
            if keywords_response and keywords_response.status_code == 200:
                ai_kw = keywords_response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                keywords_list = [k.strip() for k in ai_kw.split(",")][:8]

        if not keywords_list:
            keywords_list = ["AI research", "machine learning", "deep learning"]
        keywords = ", ".join(keywords_list)

        # Create content
        date_str = target_date.strftime("%Y-%m-%d")
        content = f"Keywords: {keywords}\n\n---\n\n"

        for i, paper in enumerate(paper_summaries, 1):
            content += f"### {i}. {paper['title']}\n\n"
            content += f"[Read Paper]({paper['url']})\n\n"
            content += f"{paper['summary']}\n\n"
        
    # Generate safe filename with weekday
    weekday = target_date.strftime("%A")  # Get weekday name (Monday, Tuesday, etc.)
    safe_title = "daily-ai-research-digest"
    filename = f"_posts/{date_str}-{weekday.lower()}-{safe_title}.md"
    
    # Format date for title with weekday (e.g., "Monday, July 29, 2024")
    weekday = target_date.strftime("%A")  # Get weekday name (Monday, Tuesday, etc.)
    formatted_date = target_date.strftime("%B %d, %Y")
    title_with_weekday = f"Daily AI Research Papers - {weekday}, {formatted_date}"
    
    # Save content to markdown file
    with open(filename, "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write(f"layout: post\n")
        f.write(f"title: \"{title_with_weekday}\"\n")
        f.write(f"date: {date_str}\n")
        f.write("---\n\n")
        f.write(content.strip() + "\n")
    
    print(f"✅ Generated daily digest: {filename}")
    return filename

async def generate_technical_deep_dive(week_start: datetime.date, week_end: datetime.date, publish_date: Optional[datetime.date] = None) -> Optional[str]:
    """Generate technical deep dive post, published on publish_date (Sunday)"""
    if publish_date is None:
        publish_date = week_end + datetime.timedelta(days=2)
    print(f"🔬 Generating technical deep dive for {week_start} to {week_end}...")
    
    # Get weekly report from Saturday
    weekly_report_filename = f"_posts/{week_start.strftime('%Y-%m-%d')}-to-{week_end.strftime('%Y-%m-%d')}-weekly-report.md"
    
    week_content = ""
    if os.path.exists(weekly_report_filename):
        with open(weekly_report_filename, 'r', encoding='utf-8') as f:
            week_content = f.read()
            print(f"  📄 Found weekly report: {weekly_report_filename}")
    else:
        print(f"  ⚠️ Missing weekly report: {weekly_report_filename}")
        return None
    
    # Create technical deep dive prompt based on weekly report
    deep_dive_prompt = (
        f"You are a senior AI researcher and technical writer with deep expertise in machine learning, deep learning, and AI systems. "
        f"Based on the weekly research summary from {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}, "
        f"identify the MOST FRONTIER, MOST ATTRACTIVE, and MOST USEFUL research topic mentioned in the weekly report. "
        f"Then write a comprehensive technical deep dive that includes:\n\n"
        f"1. **Introduction**: Explain why this research is groundbreaking and exciting\n"
        f"2. **Technical Background**: Provide the necessary theoretical foundation\n"
        f"3. **Core Innovation**: Deep dive into the key technical contribution\n"
        f"4. **Implementation**: Provide detailed, well-commented code examples in Python\n"
        f"5. **Practical Applications**: Show real-world use cases\n"
        f"6. **Future Implications**: Discuss the broader impact\n\n"
        f"Requirements:\n"
        f"- Choose the most significant and practical research from the weekly report\n"
        f"- Include detailed Python code with comprehensive comments\n"
        f"- Make the code educational and implementable\n"
        f"- Focus on cutting-edge techniques (transformers, diffusion models, RL, etc.)\n"
        f"- Include mathematical formulations where relevant\n"
        f"- Do NOT use any emojis. Write in a professional, academic tone.\n"
        f"- Use clean markdown formatting with proper headings\n\n"
        f"Weekly Report Content:\n{week_content}\n"
    )

    # Call API for technical deep dive
    deep_dive_response = await call_ai_api([
        {"role": "system", "content": "You are a senior AI researcher and technical writer. You write precise, technically accurate content with code examples. Never use emojis. Write in a professional academic tone. For math formulas, use $ for inline math and $$ on its own line for display math. Do NOT use \\( \\) or \\[ \\] delimiters."},
        {"role": "user", "content": deep_dive_prompt}
    ], max_tokens=3000, temperature=0.4)
    
    if deep_dive_response and deep_dive_response.status_code == 200:
        deep_dive_data = deep_dive_response.json()
        deep_dive_content = deep_dive_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        # Generate technical deep dive filename — published on Sunday
        weekday = publish_date.strftime("%A")
        deep_dive_filename = f"_posts/{publish_date.strftime('%Y-%m-%d')}-{weekday.lower()}-technical-deep-dive.md"

        # Save technical deep dive
        with open(deep_dive_filename, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f"layout: post\n")
            f.write(f"title: \"Technical Deep Dive - {week_start.strftime('%B %d')} to {week_end.strftime('%B %d, %Y')}\"\n")
            f.write(f"date: {publish_date.strftime('%Y-%m-%d')}\n")
            f.write("category: technical-deep-dive\n")
            f.write("---\n\n")
            f.write(deep_dive_content.strip() + "\n")
        
        print(f"✅ Generated technical deep dive: {deep_dive_filename}")
        return deep_dive_filename
    else:
        print(f"❌ Technical deep dive generation failed: {deep_dive_response.status_code if deep_dive_response else 'No response'}")
        return None

async def generate_weekly_report(week_start: datetime.date, week_end: datetime.date, publish_date: Optional[datetime.date] = None) -> Optional[str]:
    """Generate weekly report, published on publish_date (Saturday)"""
    if publish_date is None:
        publish_date = week_end + datetime.timedelta(days=1)
    print(f"📊 Generating weekly report for {week_start} to {week_end}...")
    
    # Get posts from this week
    week_posts = []
    
    current_date = week_start
    while current_date <= week_end:
        weekday = current_date.strftime("%A").lower()
        filename = f"_posts/{current_date.strftime('%Y-%m-%d')}-{weekday}-daily-ai-research-digest.md"
        
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                week_posts.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'content': content
                })
                print(f"  📄 Found daily post: {current_date.strftime('%Y-%m-%d')} ({weekday})")
        else:
            print(f"  ⚠️ Missing daily post: {current_date.strftime('%Y-%m-%d')} ({weekday})")
        
        current_date += datetime.timedelta(days=1)
    
    if not week_posts:
        print("⚠️ No posts found for this week")
        return None
    
    # Check if we have enough posts for a meaningful weekly report
    if len(week_posts) < 3:
        print(f"⚠️ Only {len(week_posts)} posts found for the week, skipping weekly report")
        return None
    
    # Create weekly report prompt
    weekly_prompt = (
        f"You are a very senior research scientist with 20+ years of experience in AI/ML who has published extensively in top-tier conferences and journals. "
        f"Write a comprehensive weekly research report analyzing this week's AI research papers.\n\n"
        f"Week: {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}\n"
        f"Total posts: {len(week_posts)}\n\n"
        f"Write the report in a natural, conversational style that reflects your deep expertise and personal insights. "
        f"Use first-person perspective and share your genuine thoughts about the research. "
        f"Be enthusiastic about promising developments, concerned about challenges, and thoughtful about implications.\n\n"
        f"Structure your report with these sections (start directly with Executive Summary, no main title):\n"
        f"1. **Executive Summary**: Start with your overall impression of the week's research and what excites you most\n"
        f"2. **Technical Trends Analysis**: Discuss the main research directions you observe, using phrases like 'I've been particularly impressed by...' and 'What strikes me most is...'\n"
        f"3. **Key Innovations and Breakthroughs**: Highlight the most significant technical breakthroughs with your personal assessment\n"
        f"4. **Methodological Insights**: Share your deep technical insights about novel approaches\n"
        f"5. **Practical Implications**: Discuss real-world applications and impact from your perspective\n"
        f"6. **Future Directions**: Predict where these research areas are heading based on your experience\n"
        f"7. **Technical Recommendations**: Provide specific recommendations for researchers and practitioners\n"
        f"8. **Conclusion**: End with your overall assessment of the field's direction\n\n"
        f"Use natural language, personal insights, and expert judgment throughout. "
        f"Express genuine interest in promising work, concern about challenges, and thoughtful analysis of implications. "
        f"Write like a senior researcher sharing weekly thoughts with colleagues.\n"
        f"Do NOT use any emojis. Write in a professional, academic tone with clean markdown.\n\n"
        f"Weekly Posts:\n"
    )

    for i, post in enumerate(week_posts, 1):
        weekly_prompt += f"\n--- Day {i} ({post['date']}) ---\n{post['content']}\n"

    # Call API for weekly report
    weekly_response = await call_ai_api([
        {"role": "system", "content": "You are a senior research scientist with deep expertise in AI/ML. You write in a natural, conversational style reflecting personal insights and first-person perspective. Never use emojis. Write in a professional academic tone. Start directly with the Executive Summary section, no main title."},
        {"role": "user", "content": weekly_prompt}
    ], max_tokens=2000, temperature=0.4)
    
    if weekly_response and weekly_response.status_code == 200:
        weekly_data = weekly_response.json()
        weekly_content = weekly_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        # Generate weekly report filename using week range
        weekly_filename = f"_posts/{week_start.strftime('%Y-%m-%d')}-to-{week_end.strftime('%Y-%m-%d')}-weekly-report.md"
        
        # Save weekly report
        with open(weekly_filename, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f"layout: post\n")
            f.write(f"title: \"Weekly Report - {week_start.strftime('%B %d')} to {week_end.strftime('%B %d, %Y')}\"\n")
            f.write(f"date: {publish_date.strftime('%Y-%m-%d')}\n")
            f.write("category: weekly-report\n")
            f.write("---\n\n")
            f.write(weekly_content.strip() + "\n")
        
        print(f"✅ Generated weekly report: {weekly_filename}")
        return weekly_filename
    else:
        print(f"❌ Weekly report generation failed: {weekly_response.status_code if weekly_response else 'No response'}")
        return None

async def batch_generate(start_date: datetime.date, end_date: datetime.date) -> None:
    """Generate posts for the entire date range with new strategy"""
    os.makedirs("_posts", exist_ok=True)
    print(f"🚀 Starting batch generation from {start_date} to {end_date}...")
    print("📅 New Strategy:")
    print("  - Saturday: Generate weekly report (Monday-Friday summary)")
    print("  - Sunday: Generate technical deep dive post")
    
    generated_files = []
    current_date = start_date
    
    while current_date <= end_date:
        print(f"\n📅 Processing {current_date}...")
        
        # Check if it's Saturday (weekday 5)
        if current_date.weekday() == 5:  # Saturday
            print("📊 Saturday detected - generating weekly report...")
            
            # Generate weekly report for Monday to Friday
            week_start = current_date - datetime.timedelta(days=5)  # Monday
            week_end = current_date - datetime.timedelta(days=1)    # Friday
            weekly_file = await generate_weekly_report(week_start, week_end, current_date)
            if weekly_file:
                generated_files.append(weekly_file)
        
        # Check if it's Sunday (weekday 6)
        elif current_date.weekday() == 6:  # Sunday
            print("🔬 Sunday detected - generating technical deep dive...")

            # Generate technical deep dive for the week
            week_start = current_date - datetime.timedelta(days=6)  # Monday
            week_end = current_date - datetime.timedelta(days=2)    # Friday (matches weekly report range)
            deep_dive_file = await generate_technical_deep_dive(week_start, week_end, current_date)
            if deep_dive_file:
                generated_files.append(deep_dive_file)
        
        # For other days, generate normal daily posts
        else:
            daily_file = await generate_daily_post(current_date)
            if daily_file:
                generated_files.append(daily_file)
        
        current_date += datetime.timedelta(days=1)
    
    print(f"\n🎉 Batch generation completed!")
    print(f"📁 Generated {len(generated_files)} files:")
    for file in generated_files:
        print(f"  - {file}")

def main():
    """Main function - auto-generates for today by default (CI-friendly)"""
    parser = argparse.ArgumentParser(description="Daily AI Trend Reporter - Batch Generator")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode with prompts")
    args = parser.parse_args()

    print("🌟 Daily AI Trend Reporter - Batch Generator")
    print("=" * 50)

    if not HF_TOKEN:
        print("❌ HF_TOKEN environment variable is not set")
        print("💡 Set HF_TOKEN to use the GitHub Models API")
        exit(1)

    try:
        if args.interactive:
            start_str = input("📅 Enter start date (YYYY-MM-DD): ").strip()
            end_str = input("📅 Enter end date (YYYY-MM-DD): ").strip()
            start_date = validate_date(start_str)
            end_date = validate_date(end_str)
            validate_date_range(start_date, end_date)
            confirm = input("\n🤔 Proceed with generation? (y/N): ").strip().lower()
            if confirm != 'y':
                print("❌ Generation cancelled")
                return
        elif args.start and args.end:
            start_date = validate_date(args.start)
            end_date = validate_date(args.end)
            validate_date_range(start_date, end_date)
        else:
            start_date = datetime.date.today()
            end_date = datetime.date.today()
            print(f"📅 Auto-generating for today: {start_date}")

        print(f"✅ Date range: {start_date} to {end_date}")
        asyncio.run(batch_generate(start_date, end_date))

    except ValueError as e:
        print(f"❌ Error: {e}")
        exit(1)
    except KeyboardInterrupt:
        print("\n❌ Generation interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        exit(1)

if __name__ == "__main__":
    main()