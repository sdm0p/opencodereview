# OpenCodeReview — Project Description

> **Spec file** · Audience: future maintainers & contributors · Goal: balanced
> overview (features + architecture + how to run) · Accuracy: **as-is**, i.e. the
> working tree is the source of truth, including uncommitted changes. No
> recommendations section — this document describes only.
>
> Written: 2026-08-12 · Based on code reading (no commands executed).

---

## 1. What the project is

**OpenCodeReview** is an AI-powered pull-request reviewer with a
**human-in-the-loop** approval gate. It runs a LangGraph pipeline in which three
specialist LLM reviewers — **correctness**, **security**, and **test coverage** —
analyze a PR in parallel, grounded by a ChromaDB vector store over the repo's
codebase history. Nothing is posted to GitHub until a human reviews the verdict
and explicitly approves it. Findings are posted as inline PR review comments with
an automatic fallback to issue-level comments when lines go stale.

It ships as both a **Click CLI** (`main.py`) and a **Gradio web UI** (`app.py`,
the entry point for Hugging Face Spaces), and includes an evaluation harness
(ground-truth precision/recall/F1 + RAGAS retrieval metrics), optional
observability (LangSmith + Langfuse), cost/TTFT tracking, a health-check CLI
(`doctor`), and a continuous alerting script.

**One-paragraph elevator summary:** "Enter `owner/repo` + PR number (or run the
CLI), the pipeline fetches the PR, indexes/retrieves relevant code context,
runs three parallel LLM reviews, dedupes and scores the findings, pauses for a
human decision, and only then posts comments back to GitHub."

---

## 2. Feature summary

- **Three parallel specialist reviewers** — correctness, security, and
  test-coverage nodes run concurrently on the same diff, each with its own
  versioned system prompt under `prompts/`.
- **Retrieval-augmented review** — a persistent ChromaDB store embeds the repo's
  source, up to 20 past merged PRs, and docs (README, CONTRIBUTING…); the diff is
  used as a query and the top ~15 chunks are injected as reviewer context.
- **Human-in-the-loop approval** — the graph calls LangGraph's `interrupt()` and
  pauses; the caller (CLI prompt or UI buttons) resumes with `approve` / `reject`.
- **Inline comments** — approved findings are posted at the exact file/line
  position; a GitHub `422` (stale line) triggers an issue-comment fallback.
- **Smart aggregation** — heuristic deduplication (file/line proximity ±5 lines +
  confidence) and noise filtering (< 0.25 confidence dropped), plus a rule-based
  verdict with an optional LLM-critic pass.
- **LLM flexibility** — Gemini primary → Groq fallback, plus **custom endpoints**
  (BYO keys) for any OpenAI-compatible API, Anthropic, or Google, configured via
  `OCR_ENDPOINT_*` env vars or added at runtime in the UI.
- **Evaluation & observability** — ground-truth eval harness, RAGAS retrieval
  metrics, LangSmith/Langfuse tracing, per-run token cost & TTFT.
- **CLI + Web UI** — identical pipeline driven from either interface.
- **GitHub OAuth Device Flow** — interactive login, tokens stored in the OS
  keyring (priority: keyring → `GITHUB_TOKEN` env).

---

## 3. System architecture (pipeline-focused)

The system is a **LangGraph state machine** (`graph.py`) with a typed Pydantic
state schema and SQLite checkpointing. Topology:

```
START → ingestion → retrieval ──→ correctness_review ──┐
                              ├──→ security_review    ├──→ aggregator → human_approval (interrupt ⏸) → executor → END
                              └──→ test_coverage_review ──┘
```

### 3.1 Pipeline nodes

| # | Node | Module | What it does |
|---|------|--------|--------------|
| 1 | `ingestion` | `nodes/ingestion.py` | Fetches PR metadata, the full unified diff, and per-file contents from the GitHub API (handles fork PRs, pagination, rate limits). No-op when a diff is already present in state (e.g. synthetic payloads). |
| 2 | `retrieval` | `nodes/retrieval.py` | Builds/loads a ChromaDB collection embedding the repo source, past merged PRs, and docs; queries it with the diff; returns top-15 context chunks. Cached per base-branch SHA so embeddings aren't rebuilt per run. |
| 3–5 | `correctness_review`, `security_review`, `test_coverage_review` | `nodes/{correctness,security,test_coverage}_reviewer.py` | Three parallel LLM reviewers. Each emits Pydantic-validated findings (category, severity, confidence, line range, suggested fix) via `with_structured_output`. Prompts live in `prompts/*_v1.txt`. |
| 6 | `aggregator` | `nodes/aggregator.py` | Merges findings pointing at the same file/line (±5 lines, highest confidence wins), drops noise below 0.25, then produces a verdict. Default is a rule-based verdict (~10 ms, zero LLM cost); `AGGREGATOR_USE_LLM_CRITIC=true` enables an optional LLM critic pass. |
| 7 | `human_approval` | `nodes/human_approval.py` | Calls `interrupt()` — the graph **pauses** and returns verdict + findings to the caller. Nothing is posted until resumed with `approve` / `reject`. |
| 8 | `executor` | `nodes/executor.py` | Posts each finding as an inline PR review comment (multi-line ranges supported). On GitHub `422` (stale line), falls back to a general issue comment. Posts the overall verdict as a summary comment. |

**Verdict rules (rule-based default):**

| Condition | Recommendation | Approved |
|---|---|---|
| ≥ 1 critical finding | `block` | ❌ |
| ≥ 1 high finding | `request_changes` | ❌ |
| any medium/low/info finding | `comment` | ✅ |
| no findings | `approve` | ✅ |

Score = `10 − 4·critical − 2·high − 0.5·medium`, clamped to 0–10.

### 3.2 State schema (minimal)

`OpenCodeReviewState` (`state.py`) is a Pydantic `BaseModel`. Key fields:

- PR metadata: `repo`, `pr_number`, `base_sha`, `diff`, `changed_files`
- `endpoint` — chosen custom LLM endpoint name (empty = built-in default)
- Accumulated lists (`Annotated[list, add]` reducers so parallel branches merge):
  `context_chunks`, `findings`
- Aggregated output: `final_findings`, `verdict` (`Verdict` model), `human_approved`

Sub-models: `ChangedFile`, `ContextChunk`, `Finding`, `Verdict`, and a `Severity`
enum (`critical`/`high`/`medium`/`low`/`info`). See `state.py` for field-level
detail.

### 3.3 Checkpointing & resume

- Every run is persisted via **`SqliteSaver`** on a SQLite DB
  (`graph.py`, `checkpoints.db` by default; `:memory:` for ephemeral runs).
- A `JsonPlusSerializer` with an allowlist of the Pydantic models handles state
  (de)serialization.
- The compiled graph exposes `_opencodereview_conn` (informal handle) so callers
  close the SQLite connection when a run finishes — prevents file-handle leaks in
  the long-running UI. The DB file is cleaned up after runs (`_cleanup_db()`).
- Because checkpoints persist to disk, a paused graph can be resumed after a page
  reload or process restart via the same `thread_id`.

---

## 4. Supporting layers

### 4.1 LLM layer (`llm_factory.py`)

- **Default chain:** Google Gemini (`gemini-3.1-flash-lite`, primary) → Groq
  (`llama-3.3-70b-versatile`, fallback); `temperature=0`. `ValueError` if no key.
- **Custom endpoints** (`endpoints.py` + `llm_factory.create_endpoint_llm`):
  - OpenAI-compatible (`langchain-openai`) — DeepSeek, OpenRouter, vLLM, Ollama,
    Together, local proxies…
  - Anthropic (`langchain-anthropic`), Google (`langchain-google-genai`).
- **Endpoint registry** (`endpoints.py`):
  - Env convention: `OCR_ENDPOINT_<n>_NAME/TYPE/API_KEY/BASE_URL/MODEL`
    (`TYPE`: `openai` | `anthropic` | `google`; default `openai`).
  - Built-in entries: `Gemini` and `Grok` are always selectable as endpoints.
  - UI-added endpoints live in an in-memory, thread-locked session registry
    (never written to disk; vanish on process restart).
  - Name precedence on collision: **session (UI) > env-var > built-in**.
  - Keys are only read from the environment and masked for display.

### 4.2 GitHub integration (`github_client.py`, `auth.py`)

- `requests`-based client; token resolution priority **keyring → `GITHUB_TOKEN`
  env**; 401 detection with a helpful re-auth error.
- HTTP caching via `requests-cache` (15-min TTL) under `.opencodereview/`.
- `auth.py`: GitHub OAuth **Device Flow** — `auth login` prints a one-time code;
  token stored in the OS keyring (scope `repo`).

### 4.3 Aggregation & verdict (see §3.1 node 6)

### 4.4 Configuration (`config.py`)

- `Settings` (pydantic-settings) resolves env var → system keyring → error.
- Provider registry currently covers `anthropic` (`OPENCODEREVIEW_ANTHROPIC_KEY`).
- CLI: `config set-key`, `config show` (masked), `config set-observability`.

---

## 5. Interfaces & configuration

### 5.1 CLI (`main.py`)

| Command | Purpose |
|---|---|
| `review [--smoke] [--repo OWNER/REPO] [--pr N] [--endpoint NAME]` | Run the pipeline. Default repo/pr uses a built-in synthetic payload (no keys/network needed). Interactive HITL prompt (`a`/`r`/`q`). |
| `doctor` | Health checks: Python version, API keys, LLM endpoints, Groq/Langfuse connectivity. |
| `auth login/status/logout` | GitHub OAuth Device Flow management. |
| `config …` | `set-key`, `show`, `set-observability` (see §4.4). |

Every option also works as an env var with the `OCR_` prefix
(e.g. `--repo` → `OCR_REPO`, `--pr` → `OCR_PR_NUMBER`).

### 5.2 Gradio web UI (`app.py`)

- **Review a PR** tab (owner/repo + PR number, live pipeline status, then
  **Approve & Post** / **Reject**).
- **Smoke Test** tab (synthetic demo, no keys).
- Severity-colored findings table, verdict card, RAGAS retrieval-quality bars,
  per-run cost summary.
- **Configured Keys & Health** accordion; dark/light theme toggle (persisted).
- **LLM Endpoint** dropdown with a "Built-in" optgroup (Gemini, Grok) and flat
  custom-endpoint entries; an add-any-custom-endpoint form (BYO name/type/base
  URL/key/model); endpoint test buttons that probe connectivity
  (`_probe_endpoint` — classifies 401/403 as "server rejected the key", falls
  back to a `chat/completions` probe when `GET /models` is blocked/missing).

### 5.3 Environment variables (key ones; full list in README)

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` / `GROQ_API_KEY` | Primary / fallback LLM keys |
| `GITHUB_TOKEN` | GitHub token (keyring preferred) |
| `OCR_ENDPOINT_<n>_*` | Custom endpoint configs (name/type/key/base_url/model) |
| `AGGREGATOR_USE_LLM_CRITIC` | `true` enables LLM-critic aggregation pass |
| `LANGSMITH_API_KEY`, `LANGFUSE_*` | Observability backends |
| `PORT` | Gradio port (default 7860) |
| `OCR_RAGAS_INTERVAL` | RAGAS throttle interval in seconds (default 3.0; 0 disables) |

---

## 6. Evaluation & observability

### 6.1 Ground-truth evaluation (`eval_data/`)

- `fetcher.py` builds `prs.jsonl` — one JSON per PR with diff, changed files, and
  human review comments as ground truth (requires `GITHUB_TOKEN`).
- `evaluate.py` matches findings to human comments by **file path + line
  proximity (±3 lines)** (optional keyword-overlap bonus), computes
  **precision / recall / F1** per PR plus micro/macro averages →
  `eval_data/eval_results.json`. Refuses to run while `GITHUB_TOKEN` is set
  (executor would post to real PRs) unless `--force`.
- `ragas_eval.py` computes RAGAS retrieval metrics: `context_precision`,
  `context_recall`, `faithfulness`, `answer_relevancy`, `mmr`.

### 6.2 Observability (`observability.py`)

- **LangSmith** — enabled automatically when `LANGSMITH_API_KEY` (or legacy
  `LANGCHAIN_API_KEY`) is set.
- **Langfuse** — enabled when public+secret keys are set; a trace per review;
  scores logged: `verdict_score`, `findings_count`, `ragas_*` metrics.
- **Cost & TTFT** — `TokenCostCallback` tracks per-run USD cost (via a
  `MODEL_PRICING` table covering Groq models and Gemini) and time-to-first-token;
  printed by CLI and shown in the UI.
- **Health checks** — `HealthStatus`, `check_groq_connectivity`,
  `check_langfuse_connectivity` (used by `doctor`).

### 6.3 RAGAS details worth knowing

- RAGAS scoring is routed **Groq-first** (avoids Gemini's ~15 req/min quota),
  then Gemini. Requires `GROQ_API_KEY` or `GEMINI_API_KEY`.
- Uncommitted work in the working tree throttles RAGAS LLM calls through a
  `_ThrottledChatModel` wrapper (`OCR_RAGAS_INTERVAL`, default 3 s between calls)
  to stay under free-tier per-minute rate limits, and failed metrics are logged
  as `None` (not fake `0.0`), with consumers guarded against `None`.

### 6.4 Alerting (`scripts/alert.py`)

Runs the smoke-test review, compares the score to a stored baseline, and pings a
**Discord or Slack webhook** when the score deviates by > 2.5 points or the run
fails. `alert.yml` runs it every 6 hours.

---

## 7. Deployment & CI

- **Dockerfile** — multi-stage: `builder` → `runtime` (default; ONNX MiniLM
  embedder, no PyTorch) → `runtime-full` (adds sentence-transformers/PyTorch).
- **docker-compose.yml** — one service, named volume for the vector store
  (`opencodereview_vectors`) so embeddings persist across runs;
  `docker-entrypoint.sh` fixes volume permissions and drops to a non-root user.
- **Hugging Face Spaces** — `.github/workflows/deploy-hf.yml` syncs the repo to
  the Space and restarts it (Docker SDK, port 7860) on push to `main`.
- **GitHub Actions** — `alert.yml` (every-6h smoke-test alert).

---

## 8. Project structure (file map)

```
├── app.py                 # Gradio web UI (HF Spaces entry point)
├── main.py                # Click CLI — review / doctor / auth / config
├── graph.py               # LangGraph pipeline + SqliteSaver checkpointing
├── state.py               # Typed Pydantic state schema
├── llm_factory.py         # Gemini → Groq fallback + custom-endpoint factory
├── endpoints.py           # Custom LLM endpoint registry (env + UI session)
├── github_client.py       # GitHub API client, token resolution, HTTP cache
├── auth.py                # GitHub OAuth Device Flow
├── config.py              # Settings (env → keyring → error) + config CLI
├── observability.py       # LangSmith/Langfuse, cost & TTFT, health checks
├── nodes/                 # One module per graph node (8 nodes)
├── prompts/               # Versioned system prompts (…_v1.txt)
├── eval_data/             # fetcher.py · evaluate.py · ragas_eval.py · prs.jsonl · eval_results.json
├── scripts/alert.py       # Continuous health monitor (Discord/Slack)
├── .github/workflows/     # alert.yml · deploy-hf.yml
└── Dockerfile · docker-compose.yml · docker-entrypoint.sh
```

---

## 9. Current state — known gaps & observations

These are the honest as-is observations a maintainer should know (no
recommendations here — just description):

- **No automated tests.** There is no `tests/` directory; verification relies on
  `python main.py review --smoke`, `python main.py doctor`, and the eval harness.
  The README acknowledges this.
- **Breadth-first exception swallowing.** Many layers catch `Exception` and only
  warn or silently degrade: Langfuse score logging, reviewer nodes (LLM failure →
  empty review), the aggregator's optional LLM critic, and RAGAS metric failures.
  This historically hid real failures (documented in `FIXES.md`).
- **FIXES.md backlog** — 14 documented fixes, most already applied to the code
  (verified: `import requests` in `nodes/executor.py`, the `_opencodereview_conn`
  connection-close pattern in `graph.py`/`main.py`/`app.py`, deleted dead
  `passthrough`/`post_results` nodes, and a `MODEL_PRICING` table incl. Gemini).
  The master checklist at the end of `FIXES.md` is unchecked — treat it as a
  record of intent rather than current status.
- **Uncommitted working-tree changes** (described as current state above):
  - `app.py` — improved endpoint probing (401/403 classified as key-rejected,
    chat/completions fallback probe for servers without `GET /models`, clearer
    status text in cards and the test form).
  - `eval_data/ragas_eval.py` — `_ThrottledChatModel` rate-limit throttle,
    Groq-first with `max_retries=5`, failed metrics → `None`.
- **Minor artifacts:** a stray `nodes__init__.py` file sits at the repo root
  (an accidental duplicate of `nodes/__init__.py`); `.opencodereview/vectors/`
  contains a large checked-in ChromaDB sqlite file.
- **Naming inconsistencies:** the UI/CLI refer to the built-in Groq provider as
  "Grok" (`builtin_endpoints()`), which differs from the actual provider name.
- **`nodes/aggregator.py`** (unlike the reviewers) imports a shared
  `GROQ_MODEL`-style constant rather than a centralised `constants.py` — there is
  no `constants.py`; constants are still duplicated across modules.

---

## 10. How to run it (quick reference)

```bash
pip install -r requirements.txt
export GITHUB_TOKEN=ghp_...        # real PRs only; keyring preferred
export GEMINI_API_KEY=AIza...      # or GROQ_API_KEY

python main.py review --smoke                       # offline pipeline check
python main.py review --repo psf/requests --pr 42   # real PR
python main.py review --endpoint DeepSeek ...       # custom endpoint
python main.py doctor                               # health checks
python app.py                                       # web UI at :7860

python eval_data/evaluate.py --max-prs 2            # quick eval run
python eval_data/ragas_eval.py --data eval_data/prs.jsonl   # RAGAS metrics
```

---

*End of project description. Source of truth: the working tree at commit
`bd08fdb` + uncommitted changes, verified by code reading on 2026-08-12.*
