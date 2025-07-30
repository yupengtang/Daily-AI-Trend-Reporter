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

# Number of papers to summarize each day (reduced to 6)
PAPERS_PER_DAY = 6  # Changed from 50 to 6

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
            response = requests.get("https://huggingface.co/papers", timeout=15)
            if response.status_code == 200:
                # Extract paper information from HTML (basic parsing)
                content = response.text
                # Look for paper titles in the HTML
                title_pattern = r'<h[1-6][^>]*>([^<]+)</h[1-6]>'
                titles = re.findall(title_pattern, content)
                
                # Filter and clean titles
                clean_titles = []
                for title in titles[:50]:  # Increased to get more papers
                    title = title.strip()
                    if len(title) > 10 and len(title) < 200:  # Reasonable title length
                        clean_titles.append({
                            'title': title,
                            'authors': ['Research Team'],
                            'abstract': f'Latest research on {title}',
                            'url': 'https://huggingface.co/papers'
                        })
                
                if clean_titles:
                    print(f"✅ Successfully extracted {len(clean_titles)} papers from HTML")
                    return clean_titles
        except Exception as e:
            print(f"⚠️ Failed to scrape papers page: {e}")
        
        print("❌ Could not fetch papers from any source")
        return []
        
    except Exception as e:
        print(f"❌ Error fetching papers: {e}")
        return []

def get_latest_papers():
    """Get the latest 6 papers from Hugging Face Papers"""
    papers = fetch_latest_papers()
    
    if not papers:
        print("⚠️ Could not fetch papers, using fallback topic generation")
        return []
    
    # Return the latest 6 papers
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
        content += f"📌 Today's Latest Research Papers ({len(paper_summaries)} papers):\n\n"
        
        for i, paper in enumerate(paper_summaries, 1):
            content += f"**{i}. {paper['title']}**\n"
            content += f"🔗 [Read Paper]({paper['url']})\n"
            content += f"📋 Summary: {paper['summary']}\n\n"
        
        content += f"🔑 Keywords: AI research, machine learning, deep learning, computer vision, natural language processing, model optimization, edge computing, autonomous systems"
        
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

# Run the async function
if __name__ == "__main__":
    if not HF_TOKEN:  # Changed from GITHUB_TOKEN
        print("❌ HF_TOKEN environment variable not set")
        exit(1)
    
    asyncio.run(generate_blog_post())
