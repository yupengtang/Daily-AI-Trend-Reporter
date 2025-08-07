# 📦 Batch Generator for Daily AI Trend Reporter

批量生成脚本，可以生成指定时间段内的所有posts（包括周报）。

## 🚀 功能特性

- **日期范围生成**：支持指定起始和截止日期
- **真实论文链接**：从Hugging Face爬取真实的论文信息
- **自动周报生成**：在周日自动生成周报
- **GitHub Models API**：使用GitHub Models API生成内容
- **错误处理完善**：未来日期验证、网络错误处理等
- **软换行格式**：使用行末两个空格的Markdown格式

## 📋 使用前准备

### 1. 设置环境变量

```bash
# 设置GitHub Models API Token
export HF_TOKEN="your-github-models-token"
```

### 2. 安装依赖

```bash
pip install requests asyncio
```

## 🎯 使用方法

### 基本使用

```bash
python3 batch_generate.py
```

### 交互式输入

脚本会提示你输入：

1. **起始日期**：格式为 `YYYY-MM-DD`
2. **截止日期**：格式为 `YYYY-MM-DD`
3. **确认生成**：输入 `y` 确认开始生成

### 示例运行

```
🌟 Daily AI Trend Reporter - Batch Generator
==================================================
📅 Enter start date (YYYY-MM-DD): 2025-07-25
📅 Enter end date (YYYY-MM-DD): 2025-07-31

✅ Date range validated: 2025-07-25 to 2025-07-31

🤔 Proceed with generation? (y/N): y

🚀 Starting batch generation from 2025-07-25 to 2025-07-31...

📅 Processing 2025-07-25...
🔍 Fetching papers for 2025-07-25...
✅ Successfully fetched page for 2025-07-25
✅ Found 10 papers for 2025-07-25
📝 Generating summaries for 10 papers...
...
```

## 🔧 技术特性

### 论文爬取逻辑

1. **直接爬取**：访问 `https://huggingface.co/papers/date/YYYY-MM-DD`
2. **HTML解析**：使用正则表达式提取论文标题和链接
3. **多重备选**：多个正则模式确保最大成功率
4. **真实链接**：确保所有论文链接都是真实的Hugging Face链接

### API调用策略

1. **GitHub Models API**：使用 `openai/gpt-4o-mini`
2. **错误处理**：API失败时使用fallback内容

### 格式保证

- **软换行**：使用 `  \n` 实现Markdown换行
- **统一格式**：与现有文件格式完全一致
- **文件名规范**：`YYYY-MM-DD-daily-ai-research-digest.md`

## ⚠️ 错误处理

### 日期验证

- ✅ 检查日期格式是否正确
- ✅ 验证起始日期不能大于截止日期
- ✅ 禁止未来日期（防止无效爬取）

### 网络错误

- ✅ 超时处理（15秒）
- ✅ 多重重试机制
- ✅ 备用数据源

### API错误

- ✅ GitHub Models API调用
- ✅ 自动fallback内容
- ✅ 详细错误日志

## 📁 生成文件

### 每日报告

```
_posts/2025-07-25-daily-ai-research-digest.md
_posts/2025-07-26-daily-ai-research-digest.md
_posts/2025-07-27-daily-ai-research-digest.md
...
```

### 周报（周日生成）

```
_posts/2025-07-21-to-2025-07-27-weekly-report.md
_posts/2025-07-28-to-2025-08-03-weekly-report.md
...
```

## 🎨 内容格式

### 每日报告格式

```markdown
---
layout: post
title: "Daily AI Research Papers - July 25, 2025"
date: 2025-07-25
---

**🔑 Keywords**: keyword1, keyword2, keyword3, keyword4, keyword5, keyword6, keyword7, keyword8

**1. Paper Title**  
🔗 [Read Paper](https://huggingface.co/papers/real-paper-id)  
📋 Summary: Technical summary of the paper...

**2. Another Paper Title**  
🔗 [Read Paper](https://huggingface.co/papers/another-real-id)  
📋 Summary: Another technical summary...
```

### 周报格式

```markdown
---
layout: post
title: "Weekly Report - July 21 to July 27, 2025"
date: 2025-07-27
category: weekly-report
---

**Executive Summary**

This week's research has been particularly exciting...

**Technical Trends Analysis**

I've been particularly impressed by...
```

## 🔍 调试信息

脚本提供详细的调试信息：

- 🔍 论文爬取状态
- 📄 论文处理进度
- 🤖 API调用状态
- ✅ 生成成功确认
- ⚠️ 警告信息
- ❌ 错误信息

## 🚨 注意事项

1. **API限制**：注意API调用次数限制
2. **网络稳定性**：确保网络连接稳定
3. **磁盘空间**：确保有足够空间存储生成的文件
4. **时间消耗**：大量日期范围可能需要较长时间

## 🐛 故障排除

### 常见问题

1. **API错误**：检查HF_TOKEN环境变量设置
2. **网络超时**：检查网络连接
3. **日期格式错误**：使用 `YYYY-MM-DD` 格式
4. **未来日期错误**：只能生成过去日期的内容

### 日志查看

脚本会输出详细的处理日志，包括：
- 每个步骤的状态
- API调用结果
- 文件生成路径
- 错误信息

## 📞 支持

如有问题，请检查：
1. HF_TOKEN环境变量是否正确设置
2. 网络连接是否正常
3. 日期格式是否正确
4. 是否有足够的磁盘空间 