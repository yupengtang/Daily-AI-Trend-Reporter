# 🌟 Daily AI Trend Reporter

Automatically generated daily technical digest on MLE/SDE frontiers using GitHub Models API and Jekyll static site.

## 🚀 Features

- **Smart Weekly Schedule**: 
  - **Saturday**: Generate Monday-Friday daily posts
  - **Sunday**: Generate technical deep dive post with detailed code examples
- **Focused Daily Digest**: Summarizes the latest 10 research papers from [Hugging Face Papers](https://huggingface.co/papers)
- **Individual Paper Summaries**: Each paper gets its own 2-3 sentence summary
- **Direct Paper Links**: Includes specific links to original papers (e.g., https://huggingface.co/papers/2507.14111) for further reading
- **Auto-Generated Keywords**: Keywords are automatically generated based on the day's research papers
- **Technical Deep Dives**: Every Sunday, generates comprehensive technical analysis with detailed code implementations
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

Currently set to run daily at UTC 14:00 (7:00 AM PDT, 10:00 PM CST), with smart weekly scheduling:

- **Monday-Friday**: Generate individual daily posts
- **Saturday**: Generate Monday-Friday daily posts (batch mode)
- **Sunday**: Generate technical deep dive post

To modify frequency, edit the cron expression in `.github/workflows/daily_blog.yml`:

```yaml
schedule:
  - cron: '0 14 * * *'  # Daily at 14:00 UTC
```

### Model Configuration

Currently using `openai/gpt-4o-mini` model. To change models, edit the `MODEL` variable in `batch_generate.py`.

### Paper Source Configuration

The system automatically fetches the latest 10 papers from [Hugging Face Papers](https://huggingface.co/papers) and generates individual summaries for each paper. Each summary includes the paper title, specific direct link to the original paper (e.g., https://huggingface.co/papers/2507.14111), and a concise 2-3 sentence summary focusing on key innovations and practical impact. Keywords are automatically generated based on the day's research papers to reflect current trends.

## 📁 Project Structure

```
Daily-AI-Trend-Reporter/
├── .github/
│   ├── workflows/
│   │   └── daily_blog.yml          # GitHub Actions configuration
│   └── scripts/
│       └── generate_blog.py        # Content generation script
├── _posts/                         # Generated blog posts
│   ├── YYYY-MM-DD-weekday-daily-ai-research-digest.md    # Daily posts
│   └── YYYY-MM-DD-weekday-technical-deep-dive.md         # Sunday technical deep dives
├── _layouts/                       # Jekyll layout templates
├── _includes/                      # Jekyll include templates
├── _config.yml                     # Jekyll configuration
├── index.md                        # Homepage
├── batch_generate.py               # Batch generation script with new strategy
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
python3 batch_generate.py  # Test batch generation with new strategy
```

## 📝 Article Format

### Daily Digest Format

Each daily digest follows a consistent structure:

- **📅 Title**: "Daily AI Research Papers - [Weekday], [Month Day, Year]" (e.g., "Daily AI Research Papers - Thursday, August 07, 2025")
- **🔑 Keywords**: Relevant AI/ML keywords
- **📋 Today's Latest Research Papers**: List of 10 latest papers, each with:
  - **Paper Title**: Full research paper title
  - **🔗 Read Paper**: Direct link to original paper
  - **📋 Summary**: 2-3 sentence technical summary

### Technical Deep Dive Format (Sundays)

Every Sunday generates a comprehensive technical deep dive:

- **📅 Title**: "Technical Deep Dive - [Start Date] to [End Date] ([Weekday])" (e.g., "Technical Deep Dive - August 04 to August 09, 2025 (Sunday)")
- **Introduction**: Why the research is groundbreaking and exciting
- **Technical Background**: Theoretical foundation
- **Core Innovation**: Deep dive into key technical contribution
- **Implementation**: Detailed, well-commented Python code examples
- **Practical Applications**: Real-world use cases
- **Future Implications**: Broader impact and research directions

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

## 🔄 New Publishing Strategy

### Weekly Schedule

- **Monday-Friday**: Individual daily posts with latest research papers
- **Saturday**: Batch generation of Monday-Friday daily posts
- **Sunday**: Technical deep dive post with detailed code implementations

### Technical Deep Dives

Every Sunday, the system analyzes the week's research papers and selects the most frontier, attractive, and useful topic for a comprehensive technical deep dive. These posts include:

- Detailed technical explanations
- Complete Python code implementations with comprehensive comments
- Real-world application examples
- Mathematical formulations where relevant
- Future research directions

This new strategy ensures consistent content generation while providing valuable technical depth for readers interested in implementation details.

## 🕐 Time Schedule

### Current Schedule

| Time Zone | Time | Description |
|-----------|------|-------------|
| **UTC** | 2:00 PM | Standard time |
| **Pacific Time (PDT)** | 7:00 AM | Pacific Daylight Time |
| **China Time (CST)** | 10:00 PM | China Standard Time |
| **Eastern Time (EDT)** | 10:00 AM | Eastern Daylight Time |

### Time Logic

- **Trigger Time**: Daily at 2:00 PM UTC
- **Content Generation**: Uses `datetime.date.today()` for current date
- **File Naming**: `YYYY-MM-DD-weekday-daily-ai-research-digest.md`
- **User Experience**: Users see content for the current date at appropriate local times 