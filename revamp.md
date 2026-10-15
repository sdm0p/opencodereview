# OpenCodeReview Revamp Plan

## Vision

Turn OpenCodeReview into a reliable, fast, human-approved code review platform:

- React workspace on Vercel
- Python review API and worker on Hugging Face or another container host
- GitHub OAuth/App authentication
- Durable job and approval state
- Hugging Face Bucket for large artifacts and caches
- Fast, low-cost model routing
- Observable, testable, repeatable review results

The current Gradio Space remains useful as a reference implementation, but React should become the primary product surface.

## Current baseline

Working today:

- React frontend deployed to Vercel
- FastAPI boundary deployed with the Hugging Face Space
- LangGraph review pipeline
- GitHub PR ingestion and comment posting
- Human approval checkpoint
- Correctness, security, and test-coverage reviewers
- Background RAGAS task creation
- Endpoint fallback support
- Basic CORS and token protection
- Live review polling, findings, verdict, approval, rejection, and delivery receipt in React

Known gaps:

- Job registry is process-local
- SQLite checkpoints are not durable on a restarted Space
- React does not yet receive reliable final RAGAS metric values
- Authentication is not implemented for browser users
- API rate limiting and per-user authorization are missing
- GitHub posting status needs durable receipts and idempotency
- Automated tests are incomplete
- Vercel and Hugging Face deployments are not fully release-gated

## Target architecture

```
Browser
  |
  | GitHub OAuth session
  v
React/Vercel
  |
  | authenticated API requests
  v
FastAPI API
  |
  +--> Supabase/Postgres
  |      users, installations, reviews, approvals, statuses
  |
  +--> Durable queue/workflow
  |      queued, retrying, running, awaiting approval, posting
  |
  +--> Review worker
  |      LangGraph, ingestion, retrieval, reviewers, aggregator
  |
  +--> Hugging Face Bucket
  |      diffs, vectors, reports, RAGAS, cache, traces
  |
  +--> GitHub App API
  |      read PR, create review, post approved comments
  |
  +--> LLM router
         free/local/paid providers with fallback and budgets
```

Use Postgres for transactional state. Use the HF Bucket for large mutable artifacts. Buckets provide S3-like object storage and `hf://buckets/` access, but they are not a transactional job database and do not provide Git-style history. See the [Hugging Face Buckets documentation](https://huggingface.co/docs/hub/main/en/storage-buckets).

## Product features

### Review workspace

- Repository and pull request input
- GitHub repository picker after OAuth
- Open PR picker with title, author, labels, and changed-file count
- Review mode:
  - full review
  - security only
  - correctness only
  - tests only
  - quick review
- Model/provider selector with availability and latency
- Review policy selector
- Path ignore rules
- Start, cancel, retry, and duplicate-review detection
- Live progress timeline:
  - fetching PR
  - reading changed files
  - retrieving context
  - running reviewers
  - aggregating findings
  - computing RAGAS
  - awaiting approval
  - posting
- Findings grouped by severity and reviewer
- Finding filters:
  - critical/high/medium/low/info
  - security/correctness/tests/performance/style
  - confidence threshold
- Diff viewer with source lines
- Suggested fix viewer
- Edit finding before approval
- Approve all, approve selected, reject, or request rerun
- GitHub posting receipt
- Review history
- Compare reviews across commits
- Download Markdown, JSON, and SARIF reports
- Shareable private review links
- Keyboard navigation and screen-reader support
- Mobile responsive layout
- Dark and light themes

### GitHub integration

Use a GitHub App rather than a long-lived personal access token.

Required capabilities:

- Sign in with GitHub OAuth
- List repositories available to the user
- List open pull requests
- Read pull request metadata and diffs
- Read repository files
- Post approved review comments
- Submit one atomic pull request review
- Add GitHub Checks results
- Receive pull request webhooks
- Re-review only when the head SHA changes
- Avoid duplicate comments using a review marker
- Handle renamed/deleted files and stale lines
- Handle GitHub rate limits and retry-after headers
- Support private repositories only after explicit installation permission

Use least privilege:

- Contents: read
- Pull requests: read/write only for posting
- Checks: write if Checks integration is enabled
- Metadata: read

Never expose GitHub access tokens to the browser.

### Repository policies

Per-repository configuration should support:

- Minimum confidence threshold
- Severity threshold for blocking
- Required reviewers
- Files or directories to ignore
- Generated-file patterns
- Test-file requirements
- Security-sensitive paths
- Maximum review cost
- Maximum changed-file count
- Required model or provider
- Auto-review on pull request event
- Whether approval is required for every run

Store policies in Postgres and export them as versioned JSON.

### Team features

- Organizations and teams
- Repository ownership
- Reviewer roles
- Audit log
- Review activity dashboard
- Accepted/rejected finding feedback
- Team-level cost budgets
- Model quality dashboard
- Slack or email notifications
- Shared policies and prompt versions

## Authentication and authorization

### Browser login

Implement GitHub OAuth or GitHub App user authorization:

1. React sends the user to the backend authorization route.
2. Backend redirects to GitHub.
3. GitHub redirects back with a short-lived code.
4. Backend exchanges the code server-side.
5. Backend creates a secure, HTTP-only session cookie.
6. React calls `/api/me` to load the session.

Security requirements:

- HTTP-only, Secure, SameSite cookies
- State and nonce validation
- Short-lived sessions
- Refresh or reauthorization flow
- Logout endpoint
- CSRF protection for cookie-authenticated mutations
- Never store GitHub client secrets in Vite environment variables
- Never put access tokens in localStorage
- Per-user ownership checks on every review job
- Separate GitHub App installation access from user identity

### API protection

- Require authentication outside local development
- Add per-user and per-IP rate limits
- Add idempotency keys to review creation
- Limit maximum repository/PR request size
- Validate repository access before queueing
- Return structured error codes
- Redact tokens, prompts, and repository secrets from logs

## Model strategy

### Routing principles

Use a model router rather than one fixed provider:

1. Select the fastest eligible model for the requested review mode.
2. Use a free or local model for quick triage.
3. Use a stronger model only for high-risk or ambiguous findings.
4. Retry only transient provider failures.
5. Enforce per-review token and cost budgets.
6. Record provider, model, prompt version, latency, tokens, and outcome.

### Free and low-cost options

Availability and free quotas change, so configure them through environment variables and health checks instead of hard-coding promises.

Recommended tiers:

- Local development:
  - Ollama with a small code model
  - llama.cpp-compatible model
  - Transformers.js for lightweight browser experiments
- Free/low-cost hosted fallback:
  - Hugging Face Inference Providers when a suitable free quota is available
  - Groq free tier when available
  - Gemini free tier when available
  - OpenAI-compatible community endpoints only when trusted and explicitly configured
- Stronger optional providers:
  - paid Gemini, Anthropic, OpenAI-compatible providers, or Groq tiers

### Fastest inference profile

Create a `fast` profile:

- Small code-capable model
- Low temperature
- Short context
- Diff-only input first
- Retrieval only for changed symbols
- Parallel reviewer calls
- Strict output schema
- Maximum latency budget
- Early exit when no actionable change is detected

Create a `deep` profile:

- Larger context
- Repository retrieval
- Three specialist reviewers
- More validation
- Higher token budget
- Required human approval

A useful default route:

```
quick triage -> fast model
security/high-risk change -> stronger model
provider failure -> next configured provider
all providers unavailable -> clear failed state
```

### Model quality controls

- JSON schema validation for every model result
- Automatic retry for malformed output
- Finding deduplication
- Confidence calibration
- Prompt versioning
- Golden PR dataset
- Precision, recall, false-positive, and latency tracking
- Human feedback labels:
  - useful
  - duplicate
  - false positive
  - missed issue
  - accepted fix

## Hugging Face Bucket plan

Create a private bucket such as:

```text
Sdm0/opencodereview-data
```

Suggested layout:

```text
review-artifacts/<owner>__<repo>/pr-<number>/<head-sha>/
  request.json
  metadata.json
  diff.patch
  changed-files.json
  findings.json
  verdict.json
  report.md
  report.sarif
ragas/<review-id>/
  status.json
  scores.json
vectors/<repository-key>/
  chroma/
cache/github/<owner>__<repo>/pr-<number>/<head-sha>/
  pull.json
  files.json
  diff.patch
logs/<date>/<review-id>.jsonl
```

Use the Bucket for:

- Chroma/vector persistence
- PR diff cache
- Raw GitHub responses
- Review reports
- RAGAS results
- Evaluation runs
- Model comparison artifacts
- Debug traces
- Rolling backups

Do not use it as the only source of truth for:

- Job status
- User sessions
- Approvals
- Authorization
- Idempotency locks

Use Postgres/Supabase for those.

## Job and persistence redesign

Create these tables:

### users

- id
- github_user_id
- login
- avatar_url
- created_at
- last_seen_at

### github_installations

- id
- owner
- installation_id
- encrypted credential reference
- permissions
- created_at

### reviews

- id
- user_id
- repo
- pr_number
- head_sha
- status
- requested_mode
- requested_model
- created_at
- started_at
- completed_at
- error_code
- error_message

### review_findings

- id
- review_id
- file_path
- line_start
- line_end
- severity
- category
- comment
- confidence
- suggested_fix
- decision

### review_events

- id
- review_id
- event_type
- payload
- created_at

### approvals

- id
- review_id
- user_id
- action
- created_at

### model_runs

- id
- review_id
- provider
- model
- prompt_version
- latency_ms
- input_tokens
- output_tokens
- cost
- status

## RAGAS implementation

- Persist a RAGAS task ID with the review.
- Write task state to Postgres and scores to the HF Bucket.
- Expose `GET /api/reviews/{id}/ragas`.
- Return:
  - pending
  - completed
  - failed
  - timed_out
- Persist individual metrics:
  - context precision
  - context recall
  - faithfulness
  - answer relevancy
- Show timestamps and model/provider used.
- Do not label a task “completed” when the score object is empty.
- Add a timeout and a clear failure reason.
- Add a rerun-RAGAS action without rerunning the full review.

## API design

Recommended routes:

```text
GET    /api/health
GET    /api/me
GET    /api/auth/github/start
GET    /api/auth/github/callback
POST   /api/auth/logout

GET    /api/repos
GET    /api/repos/{owner}/{repo}/pulls

POST   /api/reviews
GET    /api/reviews/{id}
POST   /api/reviews/{id}/cancel
POST   /api/reviews/{id}/retry
POST   /api/reviews/{id}/decision
GET    /api/reviews/{id}/events
GET    /api/reviews/{id}/ragas
GET    /api/reviews/{id}/artifacts/{type}

POST   /api/webhooks/github
GET    /api/policies/{owner}/{repo}
PUT    /api/policies/{owner}/{repo}
```

Use Server-Sent Events for live progress. Keep polling as a fallback.

## GitHub posting reliability

- Re-fetch the PR head SHA immediately before posting.
- Refuse to post if the SHA changed.
- Submit comments atomically where possible.
- Record each GitHub comment ID.
- Use idempotency keys to prevent duplicate comments.
- Return:
  - inline posted count
  - fallback posted count
  - failed count
  - verdict posted
  - GitHub URLs
- Mark results as:
  - posted
  - partially_posted
  - post_failed
  - dry_run
  - rejected

## Testing strategy

### Unit tests

- Repository and PR validation
- Endpoint validation
- RAGAS state transitions
- Finding deduplication
- Severity filtering
- GitHub line fallback
- Idempotency
- Authorization checks

### Integration tests

- Fake GitHub API
- Fake LLM provider
- LangGraph interrupt/resume
- Review restart recovery
- Queue retry
- Bucket upload/download
- Postgres migrations
- API contract tests

### Browser tests

- Login
- Start review
- Progress states
- Findings rendering
- RAGAS pending/completed/failed states
- Approve flow
- Reject flow
- Posting receipt
- Network failure recovery
- Mobile layout
- Keyboard navigation

### CI gates

- Ruff/format/lint
- Mypy or pyright
- Pytest
- Frontend build
- ESLint
- Playwright smoke tests
- Secret scan
- Dependency audit
- Container build
- Container vulnerability scan
- Deployment health check
- Automatic rollback if health checks fail

## Observability

Track:

- review duration
- queue delay
- ingestion duration
- retrieval duration
- reviewer duration
- aggregation duration
- RAGAS duration
- model latency
- tokens and cost
- provider failures
- GitHub API latency
- comment success/failure
- false-positive feedback
- restart recovery count

Use structured JSON logs with a review ID and user ID. Keep sensitive code and credentials out of logs.

## Deployment plan

### Vercel

- React frontend only
- Production environment points to the authenticated API
- Preview deployments use a staging API
- Add CSP and security headers
- Add runtime configuration endpoint for non-secret feature flags

### Hugging Face

- API and worker container
- Private Bucket mounted at a stable path
- Secrets stored in Space secrets
- Health endpoint and readiness endpoint
- Separate worker process if supported
- Deployment smoke test after every build

### Supabase

- Jobs, users, approvals, policies, model runs, and audit events
- Row-level security
- Database migrations
- Backups and retention
- Indexes on user ID, status, repo, PR number, and head SHA

## Phased delivery

### Phase 1: reliability

- Fix RAGAS empty-score handling
- Persist jobs and approvals
- Add restart recovery
- Add structured status transitions
- Add executor delivery receipts
- Add API tests and browser smoke tests

### Phase 2: authentication

- GitHub OAuth/App login
- User-owned reviews
- Repository picker
- Private repository support
- Logout and session expiry

### Phase 3: speed and cost

- Fast model profile
- Provider health checks
- Token budgets
- Cache GitHub responses
- Cache retrieval vectors in HF Bucket
- Parallel reviewer limits
- Early-exit triage

### Phase 4: product depth

- Review history
- Diff viewer
- Editable findings
- Policy editor
- SARIF export
- GitHub Checks
- Webhooks
- Notifications
- Team dashboards

### Phase 5: quality moat

- Golden benchmark dataset
- Human feedback loop
- Model comparison dashboard
- Prompt version management
- Confidence calibration
- Repository architecture memory
- Regression alerts

## Definition of done

The revamp is complete when:

- A user can sign in with GitHub.
- The user can select an accessible repository and PR.
- A review survives an API/worker restart.
- The UI shows progress and findings.
- RAGAS reports real metric values or a clear failure state.
- Approval is authenticated and authorized.
- GitHub posting is idempotent and accurately reported.
- Artifacts are recoverable from the HF Bucket.
- Tests pass in CI.
- Deployments run health checks and can roll back.
- Free-tier operation has documented limits and graceful degradation.

