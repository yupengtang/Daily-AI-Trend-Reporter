# Daily AI Trend Reporter

A fully automated pipeline that publishes a daily digest of frontier AI research. Every weekday, it pulls the top papers from Hugging Face, summarizes each one, and deploys the result as a static Jekyll site on GitHub Pages — no manual intervention required.

**Live site**: [yupengtang.github.io/Daily-AI-Trend-Reporter](https://yupengtang.github.io/Daily-AI-Trend-Reporter/)

## How it works

A GitHub Actions cron job runs daily at 14:00 UTC. The script hits the [Hugging Face Daily Papers API](https://huggingface.co/papers), picks the top 10 papers, pulls their AI-generated summaries and keywords, formats them as a Jekyll post, commits, and pushes. A second workflow builds the Jekyll site and deploys to Pages.

**Weekly schedule:**

- **Mon–Fri** — Daily digest: 10 papers with titles, links, and one-paragraph summaries.
- **Saturday** — Weekly report: an LLM-generated synthesis of the week's research trends.
- **Sunday** — Technical deep dive: in-depth analysis of the week's most interesting topic, with code.

Weekday digests are built straight from the Hugging Face abstracts and only use the LLM to top up keywords, so they survive a backend outage. The Saturday and Sunday posts are fully LLM-generated and cannot.

## Setup

1. **Add your API key.** Create a [Groq API key](https://console.groq.com/keys) and add it as `GROQ_API_KEY` in Repository Settings → Secrets.
2. **Enable Pages.** Repository Settings → Pages → Source: "GitHub Actions".
3. **Test it.** Actions tab → "Daily AI Frontier - Daily Generation" → Run workflow.

That's it. The cron handles everything after that.

## Local development

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your-key"

# Run today's generation
python3 batch_generate.py

# Or specify a date range
python3 batch_generate.py --start 2026-06-01 --end 2026-06-03

# Fill in any weekend post missing from the last 8 weeks
python3 batch_generate.py --heal-only --heal-weeks 8

# Check the papers feed and the LLM backend
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

**LLM backend** — set via environment variables, no code change needed. `LLM_PROVIDER` selects one of `groq` (default), `openai`, `huggingface`, or `together`; `LLM_MODEL` overrides that provider's default model. Both are set in the workflow `env:` block. Each provider reads its own key (`GROQ_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`, `TOGETHER_API_KEY`), or `LLM_API_KEY` as a generic fallback. New OpenAI-compatible providers can be added to the `PROVIDERS` dict in `batch_generate.py`.

> The pipeline originally ran on GitHub Models, which was retired and now returns HTTP 410 for every request. The provider table exists so the next such retirement is a one-line config change.

**Rate limits** — `LLM_MIN_REQUEST_INTERVAL` (seconds between calls, default 65 on Groq) keeps backfills under the free tier's 8000 tokens/minute cap. `WEEKLY_REPORT_MAX_TOKENS` and `DEEP_DIVE_MAX_TOKENS` bound each response; the client automatically halves them and retries if the provider rejects a request as too large, and honors `Retry-After` on rate limits.

**Papers per day** — `PAPERS_PER_DAY` in `batch_generate.py`, default 10.

## Recovering from gaps

The daily workflow scans the previous 4 weeks on every run and regenerates any weekend post that is missing, so a transient outage heals itself the next day. For a larger hole, run the **Backfill Weekend Reports** workflow (Actions tab) with a wider `weeks_back` window, or `--heal-only --heal-weeks N` locally.

If a run fails, it now exits non-zero and the Actions run goes red. Silent success with zero generated files is no longer possible.

## License

MIT
