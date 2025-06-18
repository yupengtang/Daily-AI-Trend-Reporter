import os
import requests
import json
import datetime
import re

# Set API URL and Key
GMI_API_URL = "https://api.gmi-serving.com/v1/chat/completions"
GMI_API_KEY = os.getenv("GMI_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {GMI_API_KEY}",
    "Content-Type": "application/json"
}

# Load used topics
USED_TOPICS_FILE = "memory/topics_seen.txt"
if not os.path.exists(USED_TOPICS_FILE):
    with open(USED_TOPICS_FILE, "w"): pass
with open(USED_TOPICS_FILE, "r", encoding="utf-8") as f:
    used_topics = set(line.strip() for line in f if line.strip())

# Build prompt and payload
prompt = (
    "Generate a concise technical digest (max 300 words) about a frontier topic in MLE or SDE.\n"
    "Format exactly as follows:\n"
    "🗓️ Date: <today's date>\n"
    "🎯 Topic: <Short Title>\n"
    "📌 Summary: <150 words max technical summary>\n"
    "🔑 Keywords: keyword1, keyword2, keyword3"
)
today = datetime.date.today().strftime("%Y-%m-%d")
payload = {
    "model": "deepseek-ai/DeepSeek-R1",
    "temperature": 0.3,
    "max_tokens": 1024,
    "messages": [
        {"role": "system", "content": "You are a helpful technical writer specialized in AI/ML."},
        {"role": "user", "content": prompt}
    ]
}

# Make API request
resp = requests.post(GMI_API_URL, headers=HEADERS, json=payload)
if resp.status_code != 200:
    print("❌ GMI API error:", resp.status_code, resp.text)
    exit(1)

content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
topic_match = re.search(r"🎯 Topic: (.+)", content)
if not topic_match:
    print("❌ Failed to extract topic.")
    exit(1)

topic = topic_match.group(1).strip()
if topic in used_topics:
    print(f"⚠️ Topic '{topic}' already used. Skipping.")
    exit(0)

# Generate safe filename
safe_title = re.sub(r'[^a-zA-Z0-9]+', '-', topic.lower()).strip("-")
filename = f"_posts/{today}-{safe_title}.md"

# Save content to markdown file
with open(filename, "w", encoding="utf-8") as f:
    f.write("---\n")
    f.write(f"layout: post\n")
    f.write(f"title: \"🌱 {topic}\"\n")
    f.write(f"date: {today}\n")
    f.write("---\n\n")
    f.write(content.strip() + "\n")

# Save topic to used list
with open(USED_TOPICS_FILE, "a", encoding="utf-8") as f:
    f.write(topic + "\n")

print(f"✅ Generated post: {filename}")
