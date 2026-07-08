---
title: OpenCodeReview
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
sdk_version: ""
python_version: "3.12"
app_file: app.py
pinned: false
---

# OpenCodeReview

AI-powered PR review with human-in-the-loop approval.

Reviews pull requests for **correctness**, **security**, and **test coverage** using LLMs (Gemini/Groq), then presents findings for human approval before posting comments.

## Stack

- **Frontend**: Gradio web UI
- **Backend**: LangGraph pipeline with structured output
- **LLMs**: Google Gemini (primary) → Groq (fallback)
- **Tracing**: LangSmith & Langfuse

## Usage

### Web UI (default)

Open the Space and enter a GitHub repository + PR number.

### CLI

```bash
# Smoke test with demo data
python main.py review --smoke

# Review a real PR
python main.py review --repo org/repo --pr 42
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes (primary) | Google Gemini API key |
| `GROQ_API_KEY` | Yes (fallback) | Groq API key |
| `GITHUB_TOKEN` | Yes | GitHub token for fetching PRs |
| `LANGCHAIN_API_KEY` | No | LangSmith tracing |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | No | Langfuse tracing |
