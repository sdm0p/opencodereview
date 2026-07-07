---
title: OpenCodeReview
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# OpenCodeReview

AI-powered PR review with human-in-the-loop approval.

## How to use

1. Go to the **Review a PR** tab
2. Enter a GitHub repo (e.g. `psf/requests`) and PR number
3. Click **Run Review**
4. Review the findings and Approve/Reject

## Environment Variables

| Secret | Required | Description |
|--------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for AI reviewers |
| `GITHUB_TOKEN` | No | GitHub token for fetching PRs and posting comments |

Built with LangGraph · Groq · ChromaDB · Gradio

