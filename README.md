# 🌟 Daily AI Trend Reporter

Automatically generated daily technical digest on MLE/SDE frontiers using GitHub Models API and Jekyll static site.

## 🚀 Features

- **Daily Auto-Update**: Scheduled generation via GitHub Actions (once per day at 4:00 UTC)
- **Focused Daily Digest**: Summarizes the latest 10 research papers from [Hugging Face Papers](https://huggingface.co/papers)
- **Individual Paper Summaries**: Each paper gets its own 2-3 sentence summary
- **Direct Paper Links**: Includes specific links to original papers (e.g., https://huggingface.co/papers/2507.14111) for further reading
- **Auto-Generated Keywords**: Keywords are automatically generated based on the day's research papers
- **Weekly Research Reports**: Every Sunday, generates a comprehensive technical analysis by a senior research scientist
- **Consistent Format**: Uniform structure across all daily posts
- **Static Website**: Beautiful GitHub Pages site using Jekyll

## 📋 Setup Instructions

### 1. Configure GitHub Token

Add the following secret in your GitHub repository Settings > Secrets and variables > Actions:

- **Name**: `HF_TOKEN`
- **Value**: Your GitHub Personal Access Token (requires `models:read` permission)

### 2. Enable GitHub Pages

1. Go to repository Settings > Pages
2. Source select "GitHub Actions"
3. Ensure the repository is public (required for GitHub Pages)

### 3. Manual Test Trigger

1. Go to Actions tab
2. Select "🌱 Daily AI Frontier - Daily Generation" workflow
3. Click "Run workflow" to test

## ⚙️ Configuration

### Schedule Settings

Currently set to run daily at UTC 4:00 (12:00 AM Beijing Time), generating 1 article per day.

To modify frequency, edit the cron expression in `.github/workflows/daily_blog.yml`:

```yaml
schedule:
  - cron: '0 4 * * *'  # Daily at 4:00 UTC
```

### Model Configuration

Currently using `openai/gpt-4.1` model. To change models, edit the `MODEL` variable in `.github/scripts/generate_blog.py`.

### Paper Source Configuration

The system automatically fetches the latest 10 papers from [Hugging Face Papers](https://huggingface.co/papers) and generates individual summaries for each paper. Each summary includes the paper title, specific direct link to the original paper (e.g., https://huggingface.co/papers/2507.14111), and a concise 2-3 sentence summary focusing on key innovations and practical impact. Keywords are automatically generated based on the day's research papers to reflect current trends. Additionally, every Sunday, the system generates a comprehensive weekly research report written by a senior research scientist, providing deep technical analysis and insights.

## 📁 Project Structure

```
Daily-AI-Trend-Reporter/
├── .github/
│   ├── workflows/
│   │   └── daily_blog.yml          # GitHub Actions configuration
│   └── scripts/
│       └── generate_blog.py        # Content generation script
├── _posts/                         # Generated blog posts
├── _layouts/                       # Jekyll layout templates
├── _includes/                      # Jekyll include templates
├── _config.yml                     # Jekyll configuration
├── index.md                        # Homepage
├── test_api.py                     # API connection test script
├── test_paper_fetch.py             # Paper fetching test script
└── requirements.txt                # Python dependencies
```

## 🔧 Local Development

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Local Testing

```bash
export HF_TOKEN="your-token-here"
python3 test_api.py  # Test API connection and paper fetching
python3 test_paper_fetch.py  # Test paper fetching specifically
python3 .github/scripts/generate_blog.py  # Test content generation with latest papers
```

## 📝 Article Format

Each daily digest follows a consistent structure:

- **🗓️ Date**: Publication date
- **🎯 Topic**: Daily AI Research Digest
- **📌 Today's Latest Research Papers**: List of 10 latest papers, each with:
  - **Paper Title**: Full research paper title
  - **🔗 Read Paper**: Direct link to original paper
  - **📋 Summary**: 2-3 sentence technical summary
- **🔑 Keywords**: Relevant AI/ML keywords

## 🛠️ Troubleshooting

### Common Issues

1. **API Errors**: Check if `HF_TOKEN` is correctly set with `models:read` permission
2. **Paper Fetching Issues**: Check network connection to Hugging Face
3. **Generation Failures**: Check API limits and network connectivity
4. **Fake Paper URLs**: The system now uses real paper IDs from Hugging Face (fixed in July 2025)

### View Logs

View detailed logs in GitHub Actions:
1. Go to Actions tab
2. Click on the latest workflow run
3. Check the "Daily Generation" step logs

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📋 Version History

See [VERSION_HISTORY.md](VERSION_HISTORY.md) for detailed changelog and recent fixes.

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to improve this project!

---

**Note**: Make sure your GitHub Token has sufficient permissions and the repository is set to public to support GitHub Pages.

## 📅 Start Date

This project is configured to start generating content from July 30th, 2025. 