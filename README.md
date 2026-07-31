---
title: OpenCodeReview
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

<h1 align="center">🔍 OpenCodeReview</h1>

<p align="center">
  <b>AI-powered pull request review with human-in-the-loop approval.</b><br>
  Three specialist LLM reviewers (correctness · security · test coverage) analyze every PR,
  a vector store grounds them in your codebase's history, and nothing is posted to GitHub
  until a human approves it.
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/sdm0/opencodereview"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue" alt="Hugging Face Space"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+"></a>
  <a href="#"><img src="https://img.shields.io/badge/LangGraph-1.x-1C3C3C?logo=langchain" alt="LangGraph"></a>
  <a href="#"><img src="https://img.shields.io/badge/LLMs-Gemini%20%7C%20Groq-blueviolet" alt="LLMs"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-Unlicensed-lightgrey" alt="License"></a>
</p>

---

## 📑 Table of Contents

- [✨ Features](#features)
- [🔄 How It Works](#how-it-works)
- [🏗️ Architecture](#architecture)
- [🧱 Tech Stack](#tech-stack)
- [🚀 Quick Start](#quick-start)
- [💻 CLI Usage](#cli-usage)
- [🖥️ Web UI (Gradio)](#web-ui-gradio)
- [🔐 Authentication](#authentication)
- [⚙️ Configuration](#configuration)
- [🔑 Environment Variables](#environment-variables)
- [🐳 Docker](#docker)
- [📊 Evaluation](#evaluation)
- [🔭 Observability](#observability)
- [🔔 Alerting](#alerting)
- [📁 Project Structure](#project-structure)
- [🧪 Testing](#testing)
- [📜 License](#license)

---

## ✨ Features

- **Three parallel specialist reviewers** — correctness, security, and test-coverage reviewers
  run concurrently on the same diff, each with its own versioned system prompt.
- **Retrieval-augmented review** — a persistent [ChromaDB](https://www.trychroma.com/) vector
  store indexes your repository's source code, past merged-PR discussions, and docs, so
  reviewers get relevant context (e.g. *"this function is called from X"*) — not just the diff.
- **Human-in-the-loop approval** — the pipeline **pauses** before posting anything. A human
  reviews the verdict and findings, then approves, rejects, or quits.
- **Inline comments** — approved findings are posted as **PR review comments at the exact
  file/line position**, with automatic fallback to issue-level comments when a line has gone stale (422).
- **Smart aggregation** — heuristic deduplication (file/line proximity + confidence) plus a
  rule-based verdict (with an optional LLM critic pass) that filters noise before humans see it.
- **LLM fallback chain** — Google Gemini (primary, generous free tier) → Groq (fallback),
  zero code changes required.
- **Full observability** — LangSmith & Langfuse tracing, per-run token cost & TTFT tracking,
  RAGAS retrieval-quality scoring, and a health-check CLI (`doctor`).
- **Evaluation harness** — measure precision/recall/F1 against ground-truth human review
  comments, plus RAGAS retrieval metrics.
- **Web UI & CLI** — a polished Gradio interface and a full-featured Click CLI.
- **GitHub OAuth (Device Flow)** — authenticate interactively; tokens stored in your OS keyring.

---

## 🔄 How It Works

```
                         ┌────────────────────────────────────────────────┐
                         │           GitHub Pull Request                │
                         └──────────────────────┬───────────────────────┘
                                                ▼
                                   ┌────────────────────────┐
        START ────────────────────▶│       ingestion        │  Fetch PR metadata, unified
                                   │      (no-op if diff    │  diff, and full file contents
                                   │      already present)  │  from the GitHub API
                                   └───────────┬────────────┘
                                               ▼
                                   ┌────────────────────────┐
                                   │       retrieval        │  Build/load a ChromaDB vector
                                   │                        │  store of the repo's source,
                                   │                        │  past merged PRs & docs; query
                                   │                        │  it with the diff → context
                                   └───────────┬────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    ▼                          ▼                          ▼
        ┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
        │  correctness_review │     │   security_review  │     │ test_coverage_review│
        │  (logic, edge cases,│     │  (injection, auth, │     │  (are the changes  │
        │   null checks, ...) │     │   secrets, ...)    │     │   properly tested?)│
        └─────────┬──────────┘     └─────────┬──────────┘     └─────────┬──────────┘
                  │                          │                          │
                  └──────────────────────────┼──────────────────────────┘
                                             ▼
                                   ┌────────────────────────┐
                                   │       aggregator       │  Deduplicate findings, filter
                                   │                        │  noise, produce final verdict
                                   │                        │  (rule-based or LLM critic)
                                   └───────────┬────────────┘
                                               ▼
                                   ┌────────────────────────┐
                                   │    human_approval      │  ⏸ PAUSES here — a human
                                   │   (interrupt() pause)  │  must approve before posting
                                   └───────────┬────────────┘
                                               ▼
                                   ┌────────────────────────┐
                                   │        executor        │  Post inline review comments
                                   │                        │  (422 → issue-comment fallback)
                                   └───────────┬────────────┘
                                               ▼
                                              END
```

### Pipeline steps in detail

| # | Node | What happens |
|---|------|--------------|
| 1 | **ingestion** | Fetches PR metadata, the full unified diff, and per-file content (handles fork PRs, pagination, rate limits). Uses the head commit SHA. |
| 2 | **retrieval** | Builds (once per base-branch SHA, cached) a ChromaDB collection embedding the repo's source files, up to 20 past merged PRs, and docs (README, CONTRIBUTING…). Queries it with the diff, returns the top 15 context chunks. |
| 3 | **correctness_review** | LLM scans for logic errors, off-by-one bugs, missing null checks, resource leaks, concurrency hazards. |
| 4 | **security_review** | LLM scans for injection, secrets, unsafe deserialization, auth flaws, crypto misuse, information disclosure. |
| 5 | **test_coverage_review** | LLM checks whether the changed logic has adequate test coverage. |
| 6 | **aggregator** | Merges findings that point at the same file/line (±5 lines), keeps the highest-confidence one, drops noise below `0.25` confidence, then produces a verdict — either rule-based (default, ~10 ms, zero LLM cost) or via an optional LLM critic (`AGGREGATOR_USE_LLM_CRITIC=true`). |
| 7 | **human_approval** | Calls LangGraph's `interrupt()` — the graph **pauses** and returns the verdict + findings to the caller. Nothing is posted until the human resumes with `approve` / `reject`. |
| 8 | **executor** | Posts each finding as an **inline PR review comment** (right side of the diff, multi-line ranges supported). If GitHub returns `422` (stale line), falls back to a general issue comment. Finally posts the overall verdict as a summary comment. |

### Verdict logic (rule-based default)

| Condition | Recommendation | Approved |
|---|---|---|
| ≥ 1 critical finding | `block` | ❌ |
| ≥ 1 high finding | `request_changes` | ❌ |
| any medium/low/info finding | `comment` | ✅ |
| no findings | `approve` | ✅ |

Score: `10 − 4·critical − 2·high − 0.5·medium` (clamped to 0–10).

---

## 🏗️ Architecture

The system is a **LangGraph state machine** with a typed Pydantic state schema and SQLite
checkpointing (needed for the interrupt/resume round-trip):

- **`OpenCodeReviewState`** carries PR metadata (`repo`, `pr_number`, `base_sha`, `diff`,
  `changed_files`), retrieved context (`context_chunks`, accumulated via an `add` reducer),
  findings (`findings` → `final_findings`), the `verdict`, and the `human_approved` flag.
- **Parallelism** — retrieval fans out to the three reviewers in parallel; LangGraph merges
  their accumulated findings at the aggregator.
- **Checkpointing** — every run is persisted to SQLite (`SqliteSaver`), so a paused graph can
  be resumed after a page reload or a process restart via the same thread ID.

### Tech stack

| Layer | Technology |
|---|---|
| Orchestration | [LangGraph](https://www.langchain.com/langgraph) (StateGraph + `interrupt()`) |
| LLMs | Google Gemini `gemini-3.1-flash-lite` (primary) → Groq `llama-3.3-70b-versatile` (fallback) |
| Structured output | Pydantic v2 schemas via `with_structured_output` |
| Vector store | ChromaDB (PersistentClient) + ONNX MiniLM-L6-v2 embeddings |
| GitHub integration | `requests`-based client with token resolution, 401 resilience, 15-min HTTP cache |
| Web UI | Gradio (dark/light theme, live pipeline status) |
| CLI | Click (with `OCR_` env-var prefix support) |
| Tracing | LangSmith (auto) + Langfuse (explicit callback) |
| Quality | RAGAS retrieval metrics + ground-truth eval harness |
| Auth | GitHub OAuth Device Flow, tokens stored in OS keyring |
| Deployment | Docker (multi-stage), Docker Compose, Hugging Face Spaces, GitHub Actions |

---

## 🚀 Quick Start

### Prerequisites

- Python **3.12+**
- A **GitHub token** (`GITHUB_TOKEN` or `opencodereview auth login`) — required to fetch PRs and post comments
- **One LLM key**: `GEMINI_API_KEY` (preferred) or `GROQ_API_KEY` (fallback)

### Installation

```bash
# 1. Clone & enter
git clone https://github.com/sdm0p/opencodereview.git
cd opencodereview

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API keys
export GITHUB_TOKEN="ghp_..."
export GEMINI_API_KEY="AIza..."      # primary LLM
# export GROQ_API_KEY="gsk_..."      # fallback LLM
```

### Run your first review (smoke test)

```bash
# Runs the full pipeline on a synthetic demo PR — no API keys or network needed
python main.py review --smoke
```

### Review a real PR

```bash
python main.py review --repo psf/requests --pr 42
```

The pipeline fetches the PR, reviews it, and **pauses** for your decision:

```text
  HUMAN-IN-THE-LOOP: Approve this review for posting?
  [a] Approve — post findings as GitHub comments
  [r] Reject  — discard results, post nothing
  [q] Quit    — leave graph paused
```

---

## 💻 CLI Usage

```
python main.py [COMMAND] [SUBCOMMAND] [OPTIONS]
```

`review` and `doctor` are standalone commands; `auth` and `config` have their own subcommands
(e.g. `python main.py auth login`, `python main.py config show`).

Every option also works as an environment variable with the `OCR_` prefix
(e.g. `--repo` → `OCR_REPO`, `--pr` → `OCR_PR_NUMBER`).

### `review` — run the review pipeline

| Option | Default | Description |
|---|---|---|
| `--repo OWNER/REPO` | `demo-org/demo-repo` | GitHub repository to review |
| `--pr N` | `1` | Pull request number |
| `--smoke` | — | Non-interactive smoke test on synthetic data (skips the HITL prompt) |

> **Note:** with the default repo/PR the CLI uses the built-in synthetic payload so you can
> try the tool instantly. Pass real `--repo`/`--pr` values to review an actual PR.

```bash
python main.py review --smoke                            # offline pipeline check
python main.py review --repo psf/requests --pr 42        # review a real PR
python main.py review --repo private/org --pr 7          # private repos need a token
```

### `doctor` — health checks

```bash
python main.py doctor
```

Checks Python version, API keys, observability backends, and live connectivity to
Groq / Langfuse.

### `auth` — GitHub authentication

```bash
python main.py auth login      # OAuth Device Flow (prints a code to enter on GitHub)
python main.py auth status     # show the authenticated user
python main.py auth logout     # remove the stored token
```

### `config` — secrets & settings management

```bash
python main.py config set-key --provider anthropic      # store an API key in the keyring
python main.py config show                              # masked summary of configured keys
python main.py config set-observability \
  --langsmith-api-key ls_... \
  --langfuse-public-key pk-lf-... \
  --langfuse-secret-key sk-lf-...
```

---

## 🖥️ Web UI (Gradio)

A full web interface ships with the project — this is also what runs on
[Hugging Face Spaces](https://huggingface.co/spaces/sdm0/opencodereview).

```bash
python app.py            # serves at http://localhost:7860
```

Features:

- **Review a PR** tab — enter `owner/repo` + PR number, watch the pipeline run, then
  **Approve & Post** or **Reject** the results.
- **Smoke Test** tab — run the pipeline on synthetic demo data (no keys needed).
- Severity-colored findings table, verdict card, RAGAS retrieval-quality bars,
  per-run cost summary.
- Dark/light theme toggle (persisted in `localStorage`).
- **Configured Keys & Health** accordion showing which keys are set.

---

## 🔐 Authentication

OpenCodeReview needs GitHub credentials to fetch PR data and (on approval) post comments.
Tokens are resolved in this priority order: **OS keyring → `GITHUB_TOKEN` env var**.

### Option A — OAuth Device Flow (recommended)

```bash
export GITHUB_OAUTH_CLIENT_ID="<your OAuth app client id>"
python main.py auth login
```

1. Register an OAuth app at `https://github.com/settings/developers` and **enable Device Flow**.
2. Run `auth login` — you'll get a one-time code and a verification URL.
3. Open the URL, enter the code, authorize. The token is stored securely in your keyring
   (scope: `repo`).

### Option B — Personal access token

```bash
export GITHUB_TOKEN="ghp_..."
```

> 💡 The client automatically detects invalid/revoked tokens (HTTP 401) and raises a
> helpful error telling you to re-authenticate.

---

## ⚙️ Configuration

Configuration is resolved with the priority **environment variable → system keyring → error**:

- Provider API keys (e.g. Anthropic via `config set-key`) follow
  `OPENCODEREVIEW_<FIELD>` env naming.
- GitHub token and observability keys can be stored in the keyring via
  `auth login` / `config set-observability`; env vars always take precedence.
- The HTTP cache (`requests-cache`, 15-min TTL) and the ChromaDB vector store live under
  `.opencodereview/` in the working directory unless overridden (see env vars below).

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Primary LLM | Google Gemini key — used for all reviewer nodes |
| `GROQ_API_KEY` | Fallback LLM | Groq key — used when Gemini is unavailable |
| `GITHUB_TOKEN` | Yes¹ | GitHub token for fetching PRs & posting comments |
| `GITHUB_OAUTH_CLIENT_ID` | For `auth login` | GitHub OAuth App client ID (Device Flow) |
| `LANGSMITH_API_KEY` | No | LangSmith tracing (legacy alias: `LANGCHAIN_API_KEY`) |
| `LANGSMITH_PROJECT` | No | LangSmith project name (default `opencodereview`) |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | No | Langfuse secret key |
| `LANGFUSE_HOST` | No | Langfuse host (default `https://cloud.langfuse.com`) |
| `OPENCODEREVIEW_CACHE_DIR` | No | HTTP cache directory (default `.opencodereview/`) |
| `OPENCODEREVIEW_VECTOR_DIR` | No | ChromaDB vector-store directory (default `.opencodereview/vectors`) |
| `AGGREGATOR_USE_LLM_CRITIC` | No | `true` to enable the optional LLM aggregator pass (default: rule-based) |
| `PORT` | No | Gradio server port (default `7860`) |
| `DISCORD_WEBHOOK` / `SLACK_WEBHOOK` | For alerting | Alerting webhooks (see [Alerting](#alerting)) |

¹ No token is required when reviewing the built-in demo payload — a token is only strictly
required to review real PRs.

---

## 🐳 Docker

### Build & run

```bash
# Build
docker build -t opencodereview .

# Smoke test
docker run --rm opencodereview python -m main review --smoke

# Review a real PR
docker run --rm \
  -e GROQ_API_KEY=gsk_... \
  -e GITHUB_TOKEN=ghp_... \
  opencodereview python -m main review --repo org/repo --pr 42

# Web UI
docker run --rm -p 7860:7860 \
  -e GROQ_API_KEY=gsk_... \
  -e GITHUB_TOKEN=ghp_... \
  opencodereview
```

### Docker Compose

```bash
# 1. Put your keys in a .env file
#      GROQ_API_KEY=gsk_...
#      GITHUB_TOKEN=ghp_...

# 2. Smoke test
docker compose run --rm opencodereview review --smoke

# 3. Review a real PR
docker compose run --rm opencodereview review --repo org/repo --pr 42

# 4. Run evaluation
docker compose run --rm opencodereview python eval_data/evaluate.py --max-prs 2
```

The vector store is persisted in a named volume (`opencodereview_vectors`) so embeddings are
not rebuilt on every run.

### Image variants

- **`runtime`** (default) — full app; vector-store retrieval uses the lightweight **ONNX**
  MiniLM embedder, so no PyTorch is needed.
- **`runtime-full`** — adds `sentence-transformers` (PyTorch, ~800 MB) if you need it:
  `docker build --target runtime-full -t opencodereview:full .`

### Deployment

- **Hugging Face Spaces**: push to `main` triggers
  `.github/workflows/deploy-hf.yml`, which syncs the repo to
  `sdm0/opencodereview` and restarts the Space (Docker SDK, port 7860). Set
  `GROQ_API_KEY` / `GITHUB_TOKEN` as Space secrets.

---

## 📊 Evaluation

Evaluate the agent against **ground-truth human review comments**.

### 1. Build the dataset

```bash
python eval_data/fetcher.py        # requires GITHUB_TOKEN
```

Produces `eval_data/prs.jsonl` — one JSON object per PR with its diff, changed files, and the
human review comments used as ground truth.

### 2. Run evaluation

```bash
python eval_data/evaluate.py                    # full run
python eval_data/evaluate.py --max-prs 2        # quick smoke run
python eval_data/evaluate.py --verbose          # per-finding match details
python eval_data/evaluate.py --match-keywords   # keyword-overlap matching bonus
python eval_data/evaluate.py --ragas            # + RAGAS retrieval metrics
python eval_data/evaluate.py --log-to-observability   # log scores to LangSmith/Langfuse
```

Findings are matched to human comments by **file path + line proximity (±3 lines)** (with an
optional keyword-overlap bonus), then **precision / recall / F1** are computed per PR plus
micro- and macro-averages. Results are written to `eval_data/eval_results.json`.

> ⚠️ The evaluator refuses to run while `GITHUB_TOKEN` is set (the executor would post to real
> PRs). Unset it, or pass `--force` to acknowledge.

### RAGAS retrieval metrics

`eval_data/ragas_eval.py` scores retrieval & generation quality:

| Metric | Question it answers |
|---|---|
| `context_precision` | Are the most relevant code chunks ranked first? |
| `context_recall` | Did we find *all* the relevant code? (needs ground truth) |
| `faithfulness` | Does the review stick to the retrieved code? |
| `answer_relevancy` | Does the review address the PR diff? |
| `mmr` | How early does the first useful chunk appear? |

Standalone CLI: `python eval_data/ragas_eval.py --data eval_data/prs.jsonl`

---

## 🔭 Observability

Two optional backends — the app runs fine with neither:

- **LangSmith** — enabled automatically when `LANGSMITH_API_KEY` (or legacy
  `LANGCHAIN_API_KEY`) is set. Full graph traces out of the box.
- **Langfuse** — enabled when `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` are set.
  The pipeline creates a trace per review, and logs **scores** linked to that trace:
  - `verdict_score`, `findings_count`
  - `ragas_context_precision`, `ragas_faithfulness`, `ragas_answer_relevancy`, `ragas_mmr`, …

Every run also tracks **token cost** (USD, via Groq pricing) and **TTFT**, printed at the end
of each CLI run and shown in the web UI.

---

## 🔔 Alerting

`scripts/alert.py` is a continuous health monitor: it runs the smoke-test review, compares the
resulting score against a stored baseline, and pings a **Discord or Slack webhook** when the
score deviates by more than `2.5` points or the run fails.

```bash
python scripts/alert.py \
  --baseline-file scripts/alert_baseline.json \
  --discord-webhook "$DISCORD_WEBHOOK"

python scripts/alert.py --slack-webhook "$SLACK_WEBHOOK" --update-baseline
```

A GitHub Actions workflow (`.github/workflows/alert.yml`) runs this **every 6 hours**
(set `GROQ_API_KEY`, `GITHUB_TOKEN`, and the webhook URLs as repository secrets).

---

## 📁 Project Structure

```
opencodereview/
├── app.py                       # Gradio web UI (entry point for HF Spaces)
├── main.py                      # Click CLI — review / doctor / auth / config
├── auth.py                      # GitHub OAuth Device Flow
├── config.py                    # Settings resolution (env → keyring), config CLI
├── graph.py                     # LangGraph pipeline + SqliteSaver checkpointing
├── state.py                     # Typed Pydantic state schema (Findings, Verdict, …)
├── llm_factory.py               # Gemini → Groq fallback model factory
├── github_client.py             # GitHub API session, token resolution, HTTP caching
├── observability.py             # LangSmith/Langfuse, cost & TTFT tracking, health checks
├── requirements.txt
├── Dockerfile                   # Multi-stage build (builder → runtime → runtime-full)
├── docker-compose.yml
├── docker-entrypoint.sh         # Fixes vector-store volume perms, drops to appuser
├── nodes/                       # One module per graph node
│   ├── ingestion.py             #   Fetch PR data from GitHub
│   ├── retrieval.py             #   ChromaDB vector store + diff query
│   ├── correctness_reviewer.py  #   LLM: logic & correctness issues
│   ├── security_reviewer.py     #   LLM: security vulnerabilities
│   ├── test_coverage_reviewer.py#   LLM: test-coverage gaps
│   ├── aggregator.py            #   Dedupe, filter, verdict
│   ├── human_approval.py        #   interrupt() — human-in-the-loop gate
│   └── executor.py              #   Post inline comments to GitHub
├── prompts/                     # Versioned system prompts (correctness/security/…_v1.txt)
├── eval_data/                   # Evaluation harness
│   ├── fetcher.py               #   Build ground-truth dataset (prs.jsonl)
│   ├── evaluate.py              #   Precision/recall/F1 vs human comments
│   ├── ragas_eval.py            #   RAGAS retrieval & generation metrics
│   ├── prs.jsonl                #   Ground-truth dataset
│   └── eval_results.json        #   Latest evaluation results
├── scripts/
│   └── alert.py                 # Smoke-test alert monitor (Discord/Slack)
└── .github/workflows/
    ├── alert.yml                # Every-6h smoke-test alert
    └── deploy-hf.yml            # Deploy to Hugging Face Spaces on push to main
```

---

## 🧪 Testing

There are no unit-test files yet — the built-in verification paths are:

```bash
# Smoke test: compiles the graph, runs it on synthetic data, and verifies the
# interrupt/resume round-trip (assertions included)
python main.py review --smoke

# Doctor: checks env, keys, and connectivity
python main.py doctor

# Quick evaluation run on the first 2 PRs
python eval_data/evaluate.py --max-prs 2
```

---

## 📜 License

This project is currently **unlicensed** (all rights reserved) — no license file is
distributed with the repository. Please contact the maintainers before reusing code.
