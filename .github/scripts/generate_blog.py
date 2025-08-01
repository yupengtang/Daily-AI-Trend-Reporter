import os
import json
import datetime
import re
import asyncio
import requests

# Set API configuration
HF_TOKEN = os.getenv("HF_TOKEN")  # GitHub Models token
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI API key as backup
ENDPOINT = "https://models.github.ai/inference/chat/completions"
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
MODEL = "openai/gpt-4.1"  # Use GPT-4.1 as default
OPENAI_MODEL = "gpt-4o-mini"  # OpenAI model

# Number of papers to summarize each day (increased to 10)
PAPERS_PER_DAY = 10  # Changed from 6 to 10

def fetch_latest_papers():
    """Fetch the latest papers from Hugging Face Papers"""
    try:
        # Try multiple endpoints to get latest papers
        endpoints = [
            "https://huggingface.co/api/papers",
            "https://huggingface.co/api/papers?sort=date&direction=-1",
            "https://huggingface.co/api/papers?limit=100"  # Increased limit
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=15)
                if response.status_code == 200:
                    papers = response.json()
                    if papers and isinstance(papers, list):
                        print(f"✅ Successfully fetched {len(papers)} papers from {endpoint}")
                        return papers
            except Exception as e:
                print(f"⚠️ Failed to fetch from {endpoint}: {e}")
                continue
        
        # Fallback: try to scrape from the papers page
        try:
            # Try to get papers from today's date page first
            today = datetime.date.today()
            today_str = today.strftime("%Y-%m-%d")
            response = requests.get(f"https://huggingface.co/papers/date/{today_str}", timeout=15)
            if response.status_code != 200:
                # Fallback to yesterday's page
                yesterday = today - datetime.timedelta(days=1)
                yesterday_str = yesterday.strftime("%Y-%m-%d")
                response = requests.get(f"https://huggingface.co/papers/date/{yesterday_str}", timeout=15)
            
            if response.status_code == 200:
                # Extract paper information from HTML (basic parsing)
                content = response.text
                
                # Look for paper links with titles (most accurate method)
                paper_with_title_pattern = r'<a[^>]*href="/papers/([^"]+)"[^>]*>([^<]+)</a>'
                papers_with_titles = re.findall(paper_with_title_pattern, content)
                
                # Also look for paper links in the HTML
                paper_link_patterns = [
                    r'href="/papers/([^"]+)"',
                    r'href="https://huggingface\.co/papers/([^"]+)"',
                    r'data-paper-id="([^"]+)"'
                ]
                
                paper_links = []
                for pattern in paper_link_patterns:
                    links = re.findall(pattern, content)
                    if links:
                        paper_links.extend(links)
                        break
                
                # Look for paper titles in the HTML
                title_pattern = r'<h[1-6][^>]*>([^<]+)</h[1-6]>'
                titles = re.findall(title_pattern, content)
                
                # Also look for paper titles in other formats
                alt_title_pattern = r'<a[^>]*href="/papers/[^"]*"[^>]*>([^<]+)</a>'
                alt_titles = re.findall(alt_title_pattern, content)
                
                # Combine all titles
                all_titles = titles + alt_titles
                
                # Combine titles with links if available
                clean_papers = []
                
                # First try to use papers with titles (most accurate)
                if papers_with_titles:
                    for i, (paper_id, title) in enumerate(papers_with_titles[:PAPERS_PER_DAY]):
                        title = title.strip()
                        if len(title) > 10 and len(title) < 200:  # Reasonable title length
                            clean_papers.append({
                                'title': title,
                                'authors': ['Research Team'],
                                'abstract': f'Latest research on {title}',
                                'url': f"https://huggingface.co/papers/{paper_id}"
                            })
                
                # Fallback to other methods if needed
                if not clean_papers and paper_links:
                    for i, paper_id in enumerate(paper_links[:PAPERS_PER_DAY]):
                        if i < len(all_titles):
                            title = all_titles[i].strip()
                        else:
                            title = f"Research Paper {i+1}"
                        
                        if len(title) > 10 and len(title) < 200:  # Reasonable title length
                            clean_papers.append({
                                'title': title,
                                'authors': ['Research Team'],
                                'abstract': f'Latest research on {title}',
                                'url': f"https://huggingface.co/papers/{paper_id}"
                            })
                
                # Last fallback: use real paper IDs from recent papers
                if not clean_papers:
                    real_paper_ids = [
                        "2507.14111", "2507.20254", "2507.21183", "2507.20240", 
                        "2507.22061", "2507.21503", "2507.21364", "2507.14112",
                        "2507.20255", "2507.21184"
                    ]
                    
                    for i, title in enumerate(all_titles[:PAPERS_PER_DAY]):
                        title = title.strip()
                        if len(title) > 10 and len(title) < 200 and i < len(real_paper_ids):
                            clean_papers.append({
                                'title': title,
                                'authors': ['Research Team'],
                                'abstract': f'Latest research on {title}',
                                'url': f"https://huggingface.co/papers/{real_paper_ids[i]}"
                            })
                
                if clean_papers:
                    print(f"✅ Successfully extracted {len(clean_papers)} papers from HTML")
                    return clean_papers
                
        except Exception as e:
            print(f"❌ Failed to scrape papers page: {e}")
        
        return []
        
    except Exception as e:
        print(f"❌ Paper fetching error: {e}")
        return []

def get_latest_papers():
    """Get the latest papers and format them for processing"""
    papers = fetch_latest_papers()
    
    if not papers:
        print("❌ No papers fetched")
        return []
    
    # Take the latest papers
    latest_papers = papers[:PAPERS_PER_DAY]
    print(f"📄 Processing {len(latest_papers)} latest papers")
    
    return latest_papers

async def call_github_models_api(messages, max_tokens=1024, temperature=0.3):
    """Call GitHub Models API using requests"""
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
        response = requests.post(ENDPOINT, headers=headers, json=data, timeout=60)
        return response
    except Exception as e:
        print(f"❌ GitHub Models API call error: {e}")
        return None

async def call_openai_api(messages, max_tokens=1024, temperature=0.3):
    """Call OpenAI API as backup"""
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not available")
        return None
        
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": messages,
        "model": OPENAI_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(OPENAI_ENDPOINT, headers=headers, json=data, timeout=60)
        return response
    except Exception as e:
        print(f"❌ OpenAI API call error: {e}")
        return None

async def call_ai_api(messages, max_tokens=1024, temperature=0.3):
    """Try GitHub Models first, then OpenAI as backup"""
    print("🤖 Trying GitHub Models API...")
    response = await call_github_models_api(messages, max_tokens, temperature)
    
    if response and response.status_code == 200:
        print("✅ GitHub Models API successful")
        return response
    
    print("⚠️ GitHub Models API failed, trying OpenAI API...")
    response = await call_openai_api(messages, max_tokens, temperature)
    
    if response and response.status_code == 200:
        print("✅ OpenAI API successful")
        return response
    
    print("❌ Both APIs failed")
    return None

async def generate_blog_post():
    """Generate a daily blog post with the latest AI research papers"""
    print("🚀 Starting daily AI research digest generation...")
    
    # Get latest papers from Hugging Face
    papers = get_latest_papers()
    
    if not papers:
        print("❌ No papers available for today's digest")
        return None
    
    print(f"📝 Generating summaries for {len(papers)} papers...")
    
    # Generate summaries for each paper
    paper_summaries = []
    
    for i, paper in enumerate(papers, 1):
        print(f"📄 Processing paper {i}/{len(papers)}: {paper['title'][:50]}...")
        
        # Create prompt for paper summary
        summary_prompt = (
            f"Summarize this AI research paper in 2-3 sentences, focusing on key innovations and practical impact:\n\n"
            f"Title: {paper['title']}\n"
            f"Abstract: {paper.get('abstract', 'Latest research in AI/ML')}\n\n"
            f"Provide a concise technical summary that highlights the main contribution and potential applications."
        )
        
        # Call API for summary
        response = await call_ai_api([
            {"role": "system", "content": "You are a technical writer specializing in AI/ML research. Provide concise, accurate summaries."},
            {"role": "user", "content": summary_prompt}
        ], max_tokens=150, temperature=0.3)
        
        if response and response.status_code == 200:
            response_data = response.json()
            summary = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            paper_summaries.append({
                'title': paper['title'],
                'summary': summary,
                'url': paper['url']
            })
            print(f"✅ Summary generated for paper {i}")
        else:
            # Fallback summary
            summary = f"Latest research on {paper['title']} with potential applications in AI/ML."
            paper_summaries.append({
                'title': paper['title'],
                'summary': summary,
                'url': paper['url']
            })
            print(f"⚠️ Using fallback summary for paper {i}")
    
    if paper_summaries:
        # Generate keywords based on the papers
        print("🔑 Generating keywords based on today's papers...")
        keywords_prompt = (
            f"Based on these {len(paper_summaries)} research papers, generate 8-12 relevant keywords.\n"
            f"Focus on the main research areas and technologies mentioned.\n\n"
            f"Papers:\n"
        )
        for i, paper in enumerate(paper_summaries, 1):
            keywords_prompt += f"{i}. {paper['title']}\n"
        
        keywords_prompt += "\nGenerate keywords in this format: keyword1, keyword2, keyword3, keyword4, keyword5, keyword6, keyword7, keyword8"
        
        # Call API for keywords
        keywords_response = await call_ai_api([
            {"role": "system", "content": "You are a technical writer. Generate relevant keywords based on research paper titles and topics."},
            {"role": "user", "content": keywords_prompt}
        ], max_tokens=100, temperature=0.3)
        
        if keywords_response and keywords_response.status_code == 200:
            keywords_data = keywords_response.json()
            keywords = keywords_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        else:
            keywords = "AI research, machine learning, deep learning, computer vision, natural language processing, model optimization, edge computing, autonomous systems"
        
        # Create content
        today = datetime.date.today().strftime("%Y-%m-%d")
        content = f"**🔑 Keywords**: {keywords}\n\n"
        
        for i, paper in enumerate(paper_summaries, 1):
            content += f"**{i}. {paper['title']}**\n\n"
            content += f"🔗 [Read Paper]({paper['url']})\n\n"
            content += f"📋 Summary: {paper['summary']}\n\n"
        
    else:
        # Fallback to general topic generation
        prompt = (
            "Generate a concise technical digest (max 300 words) about a frontier topic in MLE or SDE.\n"
            "Focus on the latest cutting-edge research and practical applications.\n"
            "Format exactly as follows:\n"
            "🗓️ Date: <today's date>\n"
            "🎯 Topic: <Short Title>\n"
            "📌 Summary: <150 words max technical summary>\n"
            "🔑 Keywords: keyword1, keyword2, keyword3"
        )
        
        today = datetime.date.today().strftime("%Y-%m-%d")
        
        # Call API
        response = await call_ai_api([
            {"role": "system", "content": "You are a helpful technical writer specialized in AI/ML. Focus on the latest cutting-edge research and practical applications."},
            {"role": "user", "content": prompt}
        ], max_tokens=1024, temperature=0.3)
        
        if response and response.status_code == 200:
            response_data = response.json()
            content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            print("❌ API call failed")
            return None
    
    # Generate safe filename
    safe_title = "daily-ai-research-digest"
    filename = f"_posts/{today}-{safe_title}.md"
    
    # Format date for title (e.g., "July 29, 2024")
    date_obj = datetime.date.today()
    formatted_date = date_obj.strftime("%B %d, %Y")
    
    # Save content to markdown file
    with open(filename, "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write(f"layout: post\n")
        f.write(f"title: \"Daily AI Research Papers - {formatted_date}\"\n")
        f.write(f"date: {today}\n")
        f.write("---\n\n")
        f.write(content.strip() + "\n")
    
    print(f"✅ Generated daily digest: {filename}")
    return filename

async def generate_weekly_report():
    """Generate weekly report on Sundays"""
    today = datetime.datetime.now()
    
    # Only generate on Sundays
    if today.weekday() != 6:  # Sunday is 6
        print(f"📅 Today is {today.strftime('%A')}, skipping weekly report (only generated on Sundays)")
        return None
    
    print("📊 Generating weekly research report...")
    
    # Get posts from this week
    week_start = today - datetime.timedelta(days=6)
    week_posts = []
    
    for i in range(7):
        date = week_start + datetime.timedelta(days=i)
        filename = f"_posts/{date.strftime('%Y-%m-%d')}-daily-ai-research-digest.md"
        
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                week_posts.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'content': content
                })
    
    if not week_posts:
        print("⚠️ No posts found for this week")
        return None
    
    # Create weekly report prompt
    weekly_prompt = (
        f"You are a very senior research scientist with 20+ years of experience in AI/ML who has published extensively in top-tier conferences and journals. "
        f"Write a comprehensive weekly research report analyzing this week's AI research papers.\n\n"
        f"Week: {today.strftime('%Y-%m-%d')}\n"
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
        f"Express genuine excitement about promising work, concern about challenges, and thoughtful analysis of implications. "
        f"Make it sound like a senior researcher sharing weekly thoughts with colleagues.\n\n"
        f"Weekly Posts:\n"
    )
    
    for i, post in enumerate(week_posts, 1):
        weekly_prompt += f"\n--- Day {i} ({post['date']}) ---\n{post['content']}\n"
    
    # Call API for weekly report
    weekly_response = await call_ai_api([
        {"role": "system", "content": "You are a very senior research scientist with 20+ years of experience in AI/ML who has published extensively in top-tier conferences and journals. You write in a natural, conversational style that reflects deep expertise and personal insights. You use first-person perspective, share genuine thoughts about research, express enthusiasm for promising developments, show concern about challenges, and provide thoughtful analysis of implications. Your writing style is like a senior researcher sharing weekly thoughts with colleagues - natural, insightful, and personally engaged with the research. Do NOT include a main title at the beginning - start directly with the Executive Summary section."},
        {"role": "user", "content": weekly_prompt}
    ], max_tokens=2000, temperature=0.4)
    
    if weekly_response and weekly_response.status_code == 200:
        weekly_data = weekly_response.json()
        weekly_content = weekly_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        # Calculate week start (Monday) for filename and title
        week_start = today - datetime.timedelta(days=6)
        
        # Generate weekly report filename using week range
        weekly_filename = f"_posts/{week_start.strftime('%Y-%m-%d')}-to-{today.strftime('%Y-%m-%d')}-weekly-report.md"
        
        # Save weekly report
        with open(weekly_filename, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f"layout: post\n")
            f.write(f"title: \"Weekly Report - {week_start.strftime('%B %d')} to {today.strftime('%B %d, %Y')}\"\n")
            f.write(f"date: {today.strftime('%Y-%m-%d')}\n")
            f.write("category: weekly-report\n")
            f.write("---\n\n")
            f.write(weekly_content.strip() + "\n")
        
        print(f"✅ Generated weekly report: {weekly_filename}")
        return weekly_filename
    else:
        print(f"❌ Weekly report generation failed: {weekly_response.status_code if weekly_response else 'No response'}")
        return None

# Run the async function
if __name__ == "__main__":
    if not HF_TOKEN and not OPENAI_API_KEY:
        print("❌ Neither HF_TOKEN nor OPENAI_API_KEY environment variable is set")
        print("💡 Set at least one of them to use the API")
        exit(1)
    
    # Generate daily blog post
    daily_result = asyncio.run(generate_blog_post())
    
    # Generate weekly report on Sundays
    weekly_result = asyncio.run(generate_weekly_report())
    
    if daily_result:
        print(f"✅ Daily digest generated: {daily_result}")
    if weekly_result:
        print(f"✅ Weekly report generated: {weekly_result}")
