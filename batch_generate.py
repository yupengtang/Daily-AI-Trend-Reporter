#!/usr/bin/env python3
"""
Batch generation script for Daily AI Trend Reporter
Generates posts for a specified date range including weekly reports and technical deep dives
"""

import os
import re
import time
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

# ---------------------------------------------------------------------------
# LLM backend configuration
#
# GitHub Models (models.github.ai) was retired and now returns HTTP 410 for
# every request, which is what silently killed the weekend reports. The backend
# is therefore pluggable: every provider below speaks the OpenAI-compatible
# /chat/completions wire format, so switching is a matter of setting
# LLM_PROVIDER (and that provider's key) - no code change required.
# ---------------------------------------------------------------------------
PROVIDERS = {
    "groq": {
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
    },
    "openai": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
    },
    "huggingface": {
        "endpoint": "https://router.huggingface.co/v1/chat/completions",
        "key_env": "HF_TOKEN",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct",
    },
    "together": {
        "endpoint": "https://api.together.xyz/v1/chat/completions",
        "key_env": "TOGETHER_API_KEY",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
if LLM_PROVIDER not in PROVIDERS:
    raise SystemExit(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
        f"Choose one of: {', '.join(sorted(PROVIDERS))}"
    )

PROVIDER = PROVIDERS[LLM_PROVIDER]
ENDPOINT = os.getenv("LLM_ENDPOINT") or PROVIDER["endpoint"]
MODEL = os.getenv("LLM_MODEL") or PROVIDER["default_model"]
# Provider-specific key first, then a generic escape hatch.
API_KEY = os.getenv(PROVIDER["key_env"]) or os.getenv("LLM_API_KEY")
# Only sent when explicitly set - reasoning models (e.g. gpt-oss) accept it,
# most others reject unknown parameters with a 400.
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "").strip() or None

# Providers bill "requested tokens" as prompt + max_tokens, and Groq's free
# tier allows only 8000 of those per minute. These ceilings leave room for the
# prompt (~700-1500 tokens) while still giving reasoning models enough budget to
# think and then answer at length.
WEEKLY_REPORT_MAX_TOKENS = int(os.getenv("WEEKLY_REPORT_MAX_TOKENS", "5000"))
DEEP_DIVE_MAX_TOKENS = int(os.getenv("DEEP_DIVE_MAX_TOKENS", "5000"))

# Minimum seconds between LLM calls. A backfill fires many requests back to
# back, which trips a tokens-per-minute limit almost immediately; spacing them
# out is what makes a multi-post backfill finish rather than fail halfway.
_DEFAULT_INTERVAL = "65" if LLM_PROVIDER == "groq" else "0"
LLM_MIN_REQUEST_INTERVAL = float(
    os.getenv("LLM_MIN_REQUEST_INTERVAL", _DEFAULT_INTERVAL)
)

# Never shrink a request below this many completion tokens - past this point the
# post would be too truncated to be worth publishing.
MIN_COMPLETION_TOKENS = 1500

DEEP_DIVE_SYSTEM_PROMPT = (
    "You are a senior AI researcher and technical writer. You write precise, "
    "technically accurate content with code examples. Never use emojis. Write "
    "in a professional academic tone. For math formulas, use $ for inline math "
    "and $$ on its own line for display math. Do NOT use \\( \\) or \\[ \\] "
    "delimiters."
)

WEEKLY_REPORT_SYSTEM_PROMPT = (
    "You are a senior research scientist with deep expertise in AI/ML. You "
    "write in a natural, conversational style reflecting personal insights and "
    "first-person perspective. Never use emojis. Write in a professional "
    "academic tone. Start directly with the Executive Summary section, no main "
    "title."
)

# Number of papers to summarize each day
PAPERS_PER_DAY = 10


class LLMError(RuntimeError):
    """Raised when the LLM backend cannot produce a completion."""


class InsufficientSourceData(RuntimeError):
    """Raised when a week simply lacks enough daily digests to summarise.

    Distinct from LLMError: this is an expected skip, not a broken pipeline, and
    must not turn the scheduled run red.
    """


# Transient conditions worth retrying; anything else fails immediately.
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

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

_last_request_ts = 0.0


async def _pace_requests():
    """Sleep as needed so consecutive calls respect LLM_MIN_REQUEST_INTERVAL."""
    global _last_request_ts
    if LLM_MIN_REQUEST_INTERVAL > 0 and _last_request_ts:
        wait = LLM_MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_ts)
        if wait > 0:
            print(f"Pacing: waiting {wait:.0f}s to stay under the rate limit...")
            await asyncio.sleep(wait)
    _last_request_ts = time.monotonic()


def _retry_after_seconds(response, default=30.0):
    """How long the provider wants us to wait, from header or error message."""
    header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    # Groq embeds the wait in the message, e.g. "Please try again in 30.98s"
    match = re.search(r"try again in ([\d.]+)s", response.text)
    if match:
        return float(match.group(1))
    return default


async def _call_ai_api_raw(messages, max_tokens=1024, temperature=0.3, max_attempts=6):
    """Call the endpoint and return (content, finish_reason).

    Raises LLMError instead of returning None so that a dead backend surfaces as
    a real failure rather than a post that silently never gets written.
    """
    if not API_KEY:
        raise LLMError(
            f"No API key for provider '{LLM_PROVIDER}'. Set {PROVIDER['key_env']} "
            f"(or LLM_API_KEY) in the environment."
        )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "messages": messages,
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if REASONING_EFFORT:
        data["reasoning_effort"] = REASONING_EFFORT

    last_error = "unknown error"

    for attempt in range(1, max_attempts + 1):
        await _pace_requests()

        try:
            print(f"Calling {LLM_PROVIDER} ({MODEL}), max_tokens={data['max_tokens']} "
                  f"- attempt {attempt}/{max_attempts}...")
            response = requests.post(ENDPOINT, headers=headers, json=data, timeout=120)

            if response.status_code == 200:
                choice = response.json().get("choices", [{}])[0]
                content = (choice.get("message", {}).get("content", "") or "").strip()
                finish_reason = choice.get("finish_reason", "")
                if content:
                    print(f"OK: {LLM_PROVIDER} responded ({len(content)} chars, "
                          f"finish_reason={finish_reason})")
                    return content, finish_reason
                last_error = "API returned an empty completion"
                print(f"WARNING: {last_error}")

            elif response.status_code == 413:
                # "Request too large": the provider counts prompt + max_tokens
                # against the per-minute budget, so asking for a shorter answer
                # is what actually gets the request through.
                last_error = f"HTTP 413: {response.text[:200]}"
                if data["max_tokens"] > MIN_COMPLETION_TOKENS:
                    data["max_tokens"] = max(MIN_COMPLETION_TOKENS,
                                             data["max_tokens"] // 2)
                    print(f"Request too large - retrying with "
                          f"max_tokens={data['max_tokens']}")
                    continue
                raise LLMError(
                    f"{LLM_PROVIDER} request too large even at "
                    f"max_tokens={MIN_COMPLETION_TOKENS} - {last_error}"
                )

            elif response.status_code == 429:
                wait = min(_retry_after_seconds(response), 120.0) + 3
                last_error = f"HTTP 429: {response.text[:200]}"
                print(f"Rate limited - provider asked to wait {wait:.0f}s")
                if attempt < max_attempts:
                    await asyncio.sleep(wait)
                continue

            else:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                print(f"ERROR: {last_error}")
                if response.status_code not in RETRY_STATUS:
                    raise LLMError(f"{LLM_PROVIDER} request failed - {last_error}")

        except requests.RequestException as e:
            last_error = f"{type(e).__name__}: {e}"
            print(f"ERROR: request error: {last_error}")

        if attempt < max_attempts:
            delay = min(2 ** attempt, 30)
            print(f"Retrying in {delay}s...")
            await asyncio.sleep(delay)

    raise LLMError(
        f"{LLM_PROVIDER} request failed after {max_attempts} attempts - {last_error}"
    )


async def call_ai_api(messages, max_tokens=1024, temperature=0.3, max_attempts=6):
    """Call the backend and return just the message text."""
    content, _ = await _call_ai_api_raw(messages, max_tokens, temperature, max_attempts)
    return content


# How much of the truncated text to resend so the model can pick up the thread.
# Only the tail goes back, so the continuation prompt stays inside the token
# budget instead of growing with the document.
CONTINUATION_TAIL_CHARS = 600


def _close_open_code_fence(body: str) -> str:
    """Append a closing fence if the text ends inside a code block."""
    if body.count("```") % 2 == 1:
        return body.rstrip() + "\n```"
    return body


def _trim_to_complete(body: str) -> str:
    """Drop a trailing half-written line so a post never ends mid-sentence."""
    lines = body.rstrip().split("\n")

    # If the text ends inside a code block, only the final line is partial.
    # Walking back line by line here would delete the whole code block, since
    # source lines almost never end in sentence punctuation.
    if body.count("```") % 2 == 1:
        if lines:
            lines.pop()
        return "\n".join(lines).rstrip() + "\n```"

    # Outside code, walk back over trailing prose that does not end a sentence,
    # stopping at a fence so a complete code block is never eaten.
    while lines:
        last = lines[-1].rstrip()
        # A complete line ends a sentence, a table row, a fence, or a heading.
        if (not last or last.startswith("```")
                or last.endswith((".", "!", "?", ":", "|", "`", ")", '"', "*"))):
            break
        lines.pop()
    return "\n".join(lines).rstrip()


def _sanitize_prose(text: str) -> str:
    """Rewrite LaTeX artifacts into the markdown flavour the site renders."""
    # Display math first, so the inline pass cannot chew the bracket forms.
    text = re.sub(r"\\\[\s*(.+?)\s*\\\]",
                  lambda m: "\n$$\n" + m.group(1).strip() + "\n$$\n",
                  text, flags=re.S)
    text = re.sub(r"\\\((.+?)\\\)",
                  lambda m: "$" + m.group(1).strip() + "$",
                  text, flags=re.S)
    # LaTeX list environments mean nothing to kramdown; turn them into lists.
    text = re.sub(r"[ \t]*\\(?:begin|end)\{(?:enumerate|itemize|description)\}[ \t]*\n?",
                  "", text)
    text = re.sub(r"^[ \t]*\\item[ \t]+", "- ", text, flags=re.M)
    return text


def _sanitize_markdown(body: str) -> str:
    """Normalise LaTeX the model emits despite being told not to.

    The site sets `kramdown: math_engine: null`, so \\( \\) and \\[ \\] render as
    literal backslashes on the page. The system prompt already asks for $ and $$;
    this enforces it instead of trusting the model to comply. Fenced code is left
    untouched, since a backslash-paren there is usually a regex, not math.
    """
    parts = re.split(r"(```.*?```)", body, flags=re.S)
    for i in range(0, len(parts), 2):  # even indices sit outside code fences
        parts[i] = _sanitize_prose(parts[i])
    return "".join(parts).strip()


def _join_continuations(parts: List[str]) -> str:
    """Stitch continuation chunks back together at a sensible seam.

    A chunk can stop anywhere - mid-sentence, or inside a code block - so the
    separator has to match where the break happened. Always joining with a blank
    line would split a single sentence across two paragraphs.
    """
    usable = [part.strip() for part in parts if part and part.strip()]
    if not usable:
        return ""

    body = usable[0]
    for part in usable[1:]:
        if body.count("```") % 2 == 1:
            separator = "\n"        # still inside a code block
        elif body.endswith((".", "!", "?", ":", "|", "`", ")", '"')):
            separator = "\n\n"      # the previous chunk ended cleanly
        else:
            separator = " "         # cut mid-sentence; keep it one sentence
        body = body.rstrip() + separator + part

    return body


async def generate_long_form(system_prompt: str, user_prompt: str, max_tokens: int,
                             temperature: float = 0.4,
                             max_continuations: int = 3) -> str:
    """Produce a long document, continuing past the model's output-token ceiling.

    Free-tier rate limits cap how much can be requested per call - less than a
    full technical deep dive with code. Rather than publish a post that stops
    mid-sentence (which is exactly what an unchecked finish_reason produces),
    ask the model to continue from the tail of what it already wrote.
    """
    content, finish_reason = await _call_ai_api_raw(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    parts = [content]

    for i in range(1, max_continuations + 1):
        if finish_reason != "length":
            break

        tail = "\n".join(parts)[-CONTINUATION_TAIL_CHARS:]
        print(f"Output hit the token ceiling - requesting continuation "
              f"{i}/{max_continuations}")

        content, finish_reason = await _call_ai_api_raw(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                    "You are continuing a technical article that was cut off "
                    "part-way through. These are its final lines:\n\n"
                    f"...{tail}\n\n"
                    "Continue from exactly that point. Do not repeat any of the "
                    "text above, do not restate the title, and do not add a "
                    "preamble such as 'continuing from'. Write the remaining "
                    "sections and finish the article."
                )},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        parts.append(content)

    body = _join_continuations(parts)

    if finish_reason == "length":
        print("WARNING: still truncated after continuations - trimming to the "
              "last complete line")
        return _sanitize_markdown(_trim_to_complete(body))

    return _sanitize_markdown(_close_open_code_fence(body))


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
    
    if not paper_summaries:
        print(f"No usable paper summaries for {target_date}")
        return None

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
        # Deliberately non-fatal: a daily digest is still worth publishing
        # without AI keywords, so a dead backend degrades rather than blocks.
        try:
            ai_kw = await call_ai_api([
                {"role": "system", "content": "Output only a comma-separated keyword list."},
                {"role": "user", "content": keywords_prompt}
            ], max_tokens=100, temperature=0.3)
            keywords_list = [k.strip() for k in ai_kw.split(",") if k.strip()][:8]
        except LLMError as e:
            print(f"WARNING: keyword generation unavailable ({e})")
            print("-> falling back to keywords derived from the papers themselves")

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
            week_content = _extract_post_summary(f.read(), max_chars=4000)
            print(f"  📄 Found weekly report: {weekly_report_filename}")
    else:
        # Previously a hard stop, which meant one failed Saturday automatically
        # took out Sunday too. Fall back to the same daily digests the weekly
        # report is built from, so the two failures stay independent.
        print(f"  Missing weekly report: {weekly_report_filename}")
        print("  -> falling back to this week's daily digests as source material")
        fallback_posts = _load_week_daily_posts(week_start, week_end)
        if not fallback_posts:
            print("  No daily digests for this week either - cannot write a deep dive")
            return None
        week_content = "\n\n".join(
            f"--- {post['date']} ---\n{post['content']}" for post in fallback_posts
        )[:4000]

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
        f"- Use clean markdown formatting with proper headings\n"
        # Without a length target this model writes past its output budget and
        # the article gets cut off mid-section. Keep it inside one response.
        f"- Keep the complete article between 1200 and 1800 words. Budget the "
        f"length so that all six sections fit and the article reaches a proper "
        f"conclusion; prefer one focused code example over several long ones.\n\n"
        f"Weekly Report Content:\n{week_content}\n"
    )

    # Call API for technical deep dive. Deep dives regularly exceed a single
    # response budget, so this goes through the continuation-aware path.
    deep_dive_content = await generate_long_form(
        DEEP_DIVE_SYSTEM_PROMPT,
        deep_dive_prompt,
        max_tokens=DEEP_DIVE_MAX_TOKENS,
        temperature=0.4,
    )

    # LLMError deliberately propagates to batch_generate, which records it as a
    # hard failure so the workflow exits non-zero instead of reporting success.

    # Generate technical deep dive filename - published on Sunday
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

    print(f"Generated technical deep dive: {deep_dive_filename}")
    return deep_dive_filename

def _load_week_daily_posts(week_start: datetime.date, week_end: datetime.date,
                           max_chars: int = 2000) -> List[Dict]:
    """Load the Monday-Friday daily digests backing a given week."""
    week_posts = []
    current_date = week_start
    while current_date <= week_end:
        weekday = current_date.strftime("%A").lower()
        filename = f"_posts/{current_date.strftime('%Y-%m-%d')}-{weekday}-daily-ai-research-digest.md"

        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                week_posts.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'content': _extract_post_summary(f.read(), max_chars=max_chars)
                })
                print(f"  Found daily post: {current_date.strftime('%Y-%m-%d')} ({weekday})")
        else:
            print(f"  Missing daily post: {current_date.strftime('%Y-%m-%d')} ({weekday})")

        current_date += datetime.timedelta(days=1)

    return week_posts


def _extract_post_summary(content: str, max_chars: int = 2000) -> str:
    """Extract key content from a daily post, stripping frontmatter and truncating."""
    lines = content.split('\n')
    in_frontmatter = False
    body_lines = []
    for line in lines:
        if line.strip() == '---':
            in_frontmatter = not in_frontmatter
            continue
        if not in_frontmatter:
            body_lines.append(line)
    body = '\n'.join(body_lines).strip()
    if len(body) > max_chars:
        body = body[:max_chars] + '\n[...truncated]'
    return body


async def generate_weekly_report(week_start: datetime.date, week_end: datetime.date, publish_date: Optional[datetime.date] = None) -> Optional[str]:
    """Generate weekly report, published on publish_date (Saturday)"""
    if publish_date is None:
        publish_date = week_end + datetime.timedelta(days=1)
    print(f"📊 Generating weekly report for {week_start} to {week_end}...")

    # Get posts from this week
    week_posts = _load_week_daily_posts(week_start, week_end)

    # Not enough source material is a legitimate skip, not a backend failure, so
    # it is signalled separately from the LLMError path.
    if not week_posts:
        print("No daily posts found for this week")
        raise InsufficientSourceData(f"no daily digests for {week_start}..{week_end}")

    if len(week_posts) < 3:
        print(f"Only {len(week_posts)} posts found for the week, skipping weekly report")
        raise InsufficientSourceData(
            f"only {len(week_posts)} daily digests for {week_start}..{week_end}"
        )

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

    # Call API for weekly report (LLMError propagates to batch_generate)
    weekly_content = await generate_long_form(
        WEEKLY_REPORT_SYSTEM_PROMPT,
        weekly_prompt,
        max_tokens=WEEKLY_REPORT_MAX_TOKENS,
        temperature=0.4,
    )

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

    print(f"Generated weekly report: {weekly_filename}")
    return weekly_filename

async def _generate_for_date(target_date, generated, failed, skipped):
    """Generate whatever artifact `target_date` calls for, recording the outcome.

    Failures are collected rather than swallowed: an LLM outage must make the
    scheduled run go red, which is precisely what did not happen when GitHub
    Models was retired.
    """
    try:
        if target_date.weekday() == 5:  # Saturday -> weekly report
            print("Saturday detected - generating weekly report...")
            week_start = target_date - datetime.timedelta(days=5)  # Monday
            week_end = target_date - datetime.timedelta(days=1)    # Friday
            path = await generate_weekly_report(week_start, week_end, target_date)

        elif target_date.weekday() == 6:  # Sunday -> technical deep dive
            print("Sunday detected - generating technical deep dive...")
            week_start = target_date - datetime.timedelta(days=6)  # Monday
            week_end = target_date - datetime.timedelta(days=2)    # Friday
            path = await generate_technical_deep_dive(week_start, week_end, target_date)

        else:
            path = await generate_daily_post(target_date)

        if path:
            generated.append(path)
        else:
            failed.append((target_date, "generator returned no file"))

    except InsufficientSourceData as e:
        # Expected and harmless: nothing to summarise for that week.
        print(f"SKIP {target_date}: {e}")
        skipped.append((target_date, str(e)))

    except LLMError as e:
        print(f"FAILED {target_date}: {e}")
        failed.append((target_date, str(e)))


def find_missing_weekend_posts(today: datetime.date, weeks_back: int) -> List[datetime.date]:
    """Return weekend dates in the recent past whose post was never produced.

    This is what makes the pipeline self-healing: a transient outage on one
    Saturday no longer leaves a permanent hole in the archive, because the next
    scheduled run notices the gap and fills it.
    """
    missing = []
    for offset in range(1, weeks_back * 7 + 1):
        day = today - datetime.timedelta(days=offset)

        if day.weekday() == 5:  # Saturday
            week_start = day - datetime.timedelta(days=5)
            week_end = day - datetime.timedelta(days=1)
            path = (f"_posts/{week_start.strftime('%Y-%m-%d')}"
                    f"-to-{week_end.strftime('%Y-%m-%d')}-weekly-report.md")
        elif day.weekday() == 6:  # Sunday
            path = f"_posts/{day.strftime('%Y-%m-%d')}-sunday-technical-deep-dive.md"
        else:
            continue

        if not os.path.exists(path):
            missing.append(day)

    # Oldest first, so a weekend's Saturday report exists before its Sunday
    # deep dive is written from it.
    return sorted(missing)


def _report(generated, failed, skipped) -> int:
    """Print a run summary and return the process exit code."""
    print("\n" + "=" * 50)
    print(f"Generated {len(generated)} file(s):")
    for path in generated:
        print(f"  + {path}")

    if skipped:
        print(f"\nSkipped {len(skipped)} date(s) for lack of source material:")
        for day, reason in skipped:
            print(f"  ~ {day}: {reason}")

    if failed:
        print(f"\nFAILED on {len(failed)} date(s):")
        for day, reason in failed:
            print(f"  ! {day}: {reason}")
        print("\nRun failed - see errors above.")
        return 1

    print("\nAll requested posts generated successfully.")
    return 0


async def batch_generate(start_date: datetime.date, end_date: datetime.date,
                         heal_weeks: int = 0) -> int:
    """Generate posts for the date range. Returns a process exit code."""
    os.makedirs("_posts", exist_ok=True)
    print(f"Starting batch generation from {start_date} to {end_date}...")
    print("Strategy:")
    print("  - Saturday: weekly report (Monday-Friday summary)")
    print("  - Sunday:   technical deep dive")
    print("  - Weekdays: daily research digest")

    generated, failed, skipped = [], [], []

    current_date = start_date
    while current_date <= end_date:
        print(f"\n--- Processing {current_date} ({current_date.strftime('%A')}) ---")
        await _generate_for_date(current_date, generated, failed, skipped)
        current_date += datetime.timedelta(days=1)

    if heal_weeks > 0:
        print(f"\n=== Checking the last {heal_weeks} week(s) for missing weekend posts ===")
        missing = [d for d in find_missing_weekend_posts(end_date, heal_weeks)
                   if not (start_date <= d <= end_date)]
        if not missing:
            print("No gaps found - the weekend archive is complete.")
        else:
            print(f"Backfilling {len(missing)} missing weekend post(s): "
                  f"{', '.join(str(d) for d in missing)}")
            for day in missing:
                print(f"\n--- Backfilling {day} ({day.strftime('%A')}) ---")
                await _generate_for_date(day, generated, failed, skipped)

    return _report(generated, failed, skipped)


def main():
    """Main function - auto-generates for today by default (CI-friendly)"""
    parser = argparse.ArgumentParser(description="Daily AI Trend Reporter - Batch Generator")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode with prompts")
    parser.add_argument("--heal-weeks", type=int, default=0, metavar="N",
                        help="After generating, backfill any weekend post missing "
                             "from the last N weeks (0 disables)")
    parser.add_argument("--heal-only", action="store_true",
                        help="Only backfill missing weekend posts; generate nothing else")
    args = parser.parse_args()

    print("Daily AI Trend Reporter - Batch Generator")
    print("=" * 50)
    print(f"Provider: {LLM_PROVIDER}  |  Model: {MODEL}")
    print(f"Endpoint: {ENDPOINT}")

    if not API_KEY:
        print(f"ERROR: no API key found for provider '{LLM_PROVIDER}'")
        print(f"Set {PROVIDER['key_env']} (or LLM_API_KEY) in the environment.")
        exit(1)

    try:
        if args.heal_only:
            today = datetime.date.today()
            weeks = args.heal_weeks or 8
            missing = find_missing_weekend_posts(today + datetime.timedelta(days=1), weeks)
            if not missing:
                print(f"No missing weekend posts in the last {weeks} weeks.")
                return
            print(f"Backfilling {len(missing)} missing weekend post(s): "
                  f"{', '.join(str(d) for d in missing)}")
            generated, failed, skipped = [], [], []

            async def run_heal():
                for day in missing:
                    print(f"\n--- Backfilling {day} ({day.strftime('%A')}) ---")
                    await _generate_for_date(day, generated, failed, skipped)

            asyncio.run(run_heal())
            exit(_report(generated, failed, skipped))

        if args.interactive:
            start_str = input("Enter start date (YYYY-MM-DD): ").strip()
            end_str = input("Enter end date (YYYY-MM-DD): ").strip()
            start_date = validate_date(start_str)
            end_date = validate_date(end_str)
            validate_date_range(start_date, end_date)
            confirm = input("\nProceed with generation? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Generation cancelled")
                return
        elif args.start and args.end:
            start_date = validate_date(args.start)
            end_date = validate_date(args.end)
            validate_date_range(start_date, end_date)
        else:
            start_date = datetime.date.today()
            end_date = datetime.date.today()
            print(f"Auto-generating for today: {start_date}")

        print(f"Date range: {start_date} to {end_date}")
        exit(asyncio.run(batch_generate(start_date, end_date, heal_weeks=args.heal_weeks)))

    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        exit(130)


if __name__ == "__main__":
    main()
