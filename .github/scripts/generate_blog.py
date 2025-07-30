import os
import json
import datetime
import re
import asyncio
import requests
from azure.rest.ai_inference import ModelClient
from azure.core.credentials import AzureKeyCredential

# Set API configuration
HF_TOKEN = os.getenv("HF_TOKEN")  # Changed from GITHUB_TOKEN
ENDPOINT = "https://models.github.ai/inference"
MODEL = "openai/gpt-4.1"

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
            # Try to get papers from the specific date page first
            response = requests.get("https://huggingface.co/papers/date/2025-07-29", timeout=15)
            if response.status_code != 200:
                # Fallback to general papers page
                response = requests.get("https://huggingface.co/papers", timeout=15)
            
            if response.status_code == 200:
                # Extract paper information from HTML (basic parsing)
                content = response.text
                
                # Look for paper links in the HTML - try to find specific paper URLs
                # Updated pattern to match various paper link formats
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
                
                # Look for paper links with titles (most accurate method)
                paper_with_title_pattern = r'<a[^>]*href="/papers/([^"]+)"[^>]*>([^<]+)</a>'
                papers_with_titles = re.findall(paper_with_title_pattern, content)
                
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
                if not clean_papers:
                    for i, title in enumerate(all_titles[:50]):  # Increased to get more papers
                        title = title.strip()
                        if len(title) > 10 and len(title) < 200:  # Reasonable title length
                            # Try to find corresponding paper link
                            paper_url = 'https://huggingface.co/papers'
                            if i < len(paper_links):
                                # Clean the paper ID and construct URL
                                paper_id = paper_links[i].strip()
                                if paper_id and not paper_id.startswith('http'):
                                    paper_url = f"https://huggingface.co/papers/{paper_id}"
                            
                            clean_papers.append({
                                'title': title,
                                'authors': ['Research Team'],
                                'abstract': f'Latest research on {title}',
                                'url': paper_url
                            })
                
                if clean_papers:
                    print(f"✅ Successfully extracted {len(clean_papers)} papers from HTML")
                    return clean_papers
        except Exception as e:
            print(f"⚠️ Failed to scrape papers page: {e}")
        
        print("❌ Could not fetch papers from any source")
        return []
        
    except Exception as e:
        print(f"❌ Error fetching papers: {e}")
        return []

def get_latest_papers():
    """Get the latest 10 papers from Hugging Face Papers"""
    papers = fetch_latest_papers()
    
    if not papers:
        print("⚠️ Could not fetch papers, using fallback topic generation")
        return []
    
    # Return the latest 10 papers
    selected_papers = papers[:PAPERS_PER_DAY]
    print(f"📄 Selected {len(selected_papers)} latest papers:")
    for i, paper in enumerate(selected_papers, 1):
        print(f"  {i}. {paper['title']}")
    return selected_papers

async def generate_blog_post():
    # Get latest papers from Hugging Face
    latest_papers = get_latest_papers()
    
    if latest_papers:
        # Create structured content with individual summaries
        today = datetime.date.today().strftime("%Y-%m-%d")
        
        # Generate individual summaries for each paper
        paper_summaries = []
        for i, paper in enumerate(latest_papers, 1):
            print(f"📝 Generating summary for paper {i}/{len(latest_papers)}: {paper['title']}")
            
            summary_prompt = (
                f"Generate a brief 2-3 sentence summary for this research paper:\n"
                f"Title: {paper['title']}\n"
                f"Abstract: {paper['abstract'][:200]}...\n\n"
                f"Focus on the key innovation and practical impact. Keep it concise and technical."
            )
            
            # Create client for this request
            client = ModelClient(
                ENDPOINT,
                AzureKeyCredential(HF_TOKEN),  # Changed from GITHUB_TOKEN
            )
            
            # Make API request for individual summary
            response = await client.path("/chat/completions").post({
                "body": {
                    "messages": [
                        {"role": "system", "content": "You are a technical writer. Provide concise, accurate summaries of AI/ML research papers."},
                        {"role": "user", "content": summary_prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 150,
                    "model": MODEL
                }
            })
            
            if response.status_code == 200:
                response_data = response.json()
                summary = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            else:
                summary = f"Latest research on {paper['title']} with practical applications in AI/ML."
            
            paper_summaries.append({
                'title': paper['title'],
                'summary': summary,
                'url': paper.get('url', 'https://huggingface.co/papers')
            })
        
        # Create the main content
        content = f"🗓️ Date: {today}\n"
        content += f"🎯 Topic: Daily AI Research Digest\n\n"
        
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
        
        # Create client for keywords generation
        client = ModelClient(
            ENDPOINT,
            AzureKeyCredential(HF_TOKEN),
        )
        
        # Make API request for keywords
        keywords_response = await client.path("/chat/completions").post({
            "body": {
                "messages": [
                    {"role": "system", "content": "You are a technical writer. Generate relevant keywords based on research paper titles and topics."},
                    {"role": "user", "content": keywords_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 100,
                "model": MODEL
            }
        })
        
        if keywords_response.status_code == 200:
            keywords_data = keywords_response.json()
            keywords = keywords_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        else:
            keywords = "AI research, machine learning, deep learning, computer vision, natural language processing, model optimization, edge computing, autonomous systems"
        
        content += f"🔑 Keywords: {keywords}\n\n"
        
        for i, paper in enumerate(paper_summaries, 1):
            content += f"**{i}. {paper['title']}**\n"
            content += f"🔗 [Read Paper]({paper['url']})\n"
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
        
        # Create client
        client = ModelClient(
            ENDPOINT,
            AzureKeyCredential(HF_TOKEN),  # Changed from GITHUB_TOKEN
        )
        
        # Make API request
        response = await client.path("/chat/completions").post({
            "body": {
                "messages": [
                    {"role": "system", "content": "You are a helpful technical writer specialized in AI/ML. Focus on the latest cutting-edge research and practical applications."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1024,
                "model": MODEL
            }
        })
        
        if response.status_code != 200:
            print("❌ GitHub Models API error:", response.status_code, response.text)
            return None
        
        # Get response content
        response_data = response.json()
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
    
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
    if latest_papers:
        print(f"📄 Based on {len(latest_papers)} latest papers with individual summaries")
    return filename

async def generate_weekly_report():
    """Generate a weekly research report on Sundays"""
    today = datetime.date.today()
    
    # Check if it's Sunday
    if today.weekday() != 6:  # 6 = Sunday
        print("📅 Not Sunday, skipping weekly report generation")
        return None
    
    print("📊 Generating weekly research report...")
    
    # Get all posts from this week (Monday to Sunday)
    week_posts = []
    week_start = today - datetime.timedelta(days=6)  # Monday
    
    for i in range(7):
        date = week_start + datetime.timedelta(days=i)  # Monday to Sunday
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
        f"Make it sound like a senior researcher sharing their weekly thoughts with colleagues.\n\n"
        f"Weekly Posts:\n"
    )
    
    for i, post in enumerate(week_posts, 1):
        weekly_prompt += f"\n--- Day {i} ({post['date']}) ---\n{post['content']}\n"
    
    # Create client for weekly report
    client = ModelClient(
        ENDPOINT,
        AzureKeyCredential(HF_TOKEN),
    )
    
    # Make API request for weekly report
    weekly_response = await client.path("/chat/completions").post({
        "body": {
            "messages": [
                {"role": "system", "content": "You are a very senior research scientist with 20+ years of experience in AI/ML who has published extensively in top-tier conferences and journals. You write in a natural, conversational style that reflects deep expertise and personal insights. You use first-person perspective, share genuine thoughts about research, express enthusiasm for promising developments, show concern about challenges, and provide thoughtful analysis of implications. Your writing style is like a senior researcher sharing weekly thoughts with colleagues - natural, insightful, and personally engaged with the research. Do NOT include a main title at the beginning - start directly with the Executive Summary section."},
                {"role": "user", "content": weekly_prompt}
            ],
            "temperature": 0.4,
            "max_tokens": 2000,
            "model": MODEL
        }
    })
    
    if weekly_response.status_code == 200:
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
        print(f"❌ Weekly report generation failed: {weekly_response.status_code}")
        return None

# Run the async function
if __name__ == "__main__":
    if not HF_TOKEN:  # Changed from GITHUB_TOKEN
        print("❌ HF_TOKEN environment variable not set")
        exit(1)
    
    # Generate daily blog post
    daily_result = asyncio.run(generate_blog_post())
    
    # Generate weekly report on Sundays
    weekly_result = asyncio.run(generate_weekly_report())
    
    if daily_result:
        print(f"✅ Daily digest generated: {daily_result}")
    if weekly_result:
        print(f"✅ Weekly report generated: {weekly_result}")
