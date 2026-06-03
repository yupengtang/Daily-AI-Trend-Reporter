# Daily AI Trend Reporter

A fully automated pipeline that publishes a daily digest of frontier AI research. Every weekday, it pulls the top papers from Hugging Face, summarizes each one, and deploys the result as a static Jekyll site on GitHub Pages — no manual intervention required.

**Live site**: [yupengtang.github.io/Daily-AI-Trend-Reporter](https://yupengtang.github.io/Daily-AI-Trend-Reporter/)

## How it works

A GitHub Actions cron job runs daily at 14:00 UTC. The script hits the [Hugging Face Daily Papers API](https://huggingface.co/papers), picks the top 10 papers, pulls their AI-generated summaries and keywords, formats them as a Jekyll post, commits, and pushes. A second workflow builds the Jekyll site and deploys to Pages.

**Weekly schedule:**

- **Mon–Fri** — Daily digest: 10 papers with titles, links, and one-paragraph summaries.
- **Saturday** — Weekly report: an LLM-generated synthesis of the week's research trends.
- **Sunday** — Technical deep dive: in-depth analysis of the week's most interesting topic, with code.

## Setup

1. **Add your token.** Go to [GitHub Models](https://github.com/marketplace/models), generate an API key (Fine-grained PAT with **Models: Read** permission). Then add it as `HF_TOKEN` in Repository Settings → Secrets.
2. **Enable Pages.** Repository Settings → Pages → Source: "GitHub Actions".
3. **Test it.** Actions tab → "Daily AI Frontier - Daily Generation" → Run workflow.

That's it. The cron handles everything after that.

## Local development

```bash
pip install -r requirements.txt
export HF_TOKEN="your-token"

# Run today's generation
python3 batch_generate.py

# Or specify a date range
python3 batch_generate.py --start 2026-06-01 --end 2026-06-03

# Test API connectivity
python3 test_api.py
```

Jekyll preview:

```bash
bundle install
bundle exec jekyll serve
```

## Configuration

**Schedule** — edit the cron in `.github/workflows/daily_blog.yml`:

```yaml
schedule:
  - cron: '0 14 * * *'
```

**Model** — change the `MODEL` variable in `batch_generate.py`. Currently `openai/gpt-4o-mini` via GitHub Models.

**Papers per day** — `PAPERS_PER_DAY` in `batch_generate.py`, default 10.

## License

MIT
