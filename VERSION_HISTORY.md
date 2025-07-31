# Version History

## [1.1.0] - 2025-07-31

### Fixed
- Fixed fake paper URLs generation issue
- Improved paper fetching reliability with better error handling
- Enhanced API fallback mechanism for better uptime

### Changed
- Updated workflow frequency from 3 times per day to once per day
- Changed default model to `openai/gpt-4.1`
- Reduced GitHub Actions timeout from 150 minutes to 30 minutes
- Improved paper fetching logic to use real Hugging Face paper IDs

### Technical Improvements
- Added OpenAI API as fallback when GitHub Models API fails
- Enhanced regex patterns for extracting paper information
- Better error handling and fallback mechanisms
- Simplified execution logic in GitHub Actions

## [1.0.0] - 2025-07-30

### Added
- Initial release of Daily AI Trend Reporter
- Daily AI research digest generation
- GitHub Actions automation for scheduled content generation
- Jekyll static site generation
- Weekly research reports on Sundays
- Individual paper summaries with direct links
- Auto-generated keywords based on research papers
- Support for GitHub Models API and OpenAI API 