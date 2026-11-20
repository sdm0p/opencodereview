# OpenCodeReview Production Revamp and Offline Evaluation Plan

## Scope

This plan includes the earlier audit recommendations and adds a repeatable offline correctness benchmark using `code-review-bench/code-review-bench`.

The goal is to improve review correctness before allowing automatic use:

```
offline benchmark -> inspect failures -> change prompts/retrieval/router -> rerun benchmark -> live PR cross-check
```

## Earlier recommendations included

- Durable job storage instead of the process-local API registry
- Durable LangGraph checkpoints instead of ephemeral SQLite
- Correct posting receipts with posted/partial/failed states
- React polling, findings, verdict, approval, rejection, and delivery status
- GitHub OAuth/GitHub App authentication
- Per-user authorization and rate limits
- Idempotent GitHub posting
- HF Bucket storage for diffs, vectors, reports, caches, and RAGAS artifacts
- Supabase/Postgres for jobs, users, approvals, and audit events
- RAGAS status and score persistence
- Fast/free model routing with fallback and budgets
- Repository policies and review modes
- Automated unit, integration, and browser tests
- CI deployment gates and health checks
- Model, prompt, latency, cost, and quality observability
- Live validation against a real pull request before release

## Offline benchmark

### Primary dataset

Use [code-review-bench/code-review-bench](https://huggingface.co/datasets/code-review-bench/code-review-bench).

Use the `offline` split first because it contains expert-curated golden issues. Use `online` later for comparison against other review tools.

The benchmark must run without:

- GitHub comments
- GitHub write permissions
- human approval
- repository mutation
- production credentials
- external posting

### Prediction format

The review engine must export one JSON object per row:

```json
{
  "task_id": "stable-row-id",
  "findings": [
    {
      "file_path": "src/example.py",
      "line_start": 42,
      "line_end": 44,
      "severity": "high",
      "category": "correctness",
      "comment": "Explanation of the issue",
      "confidence": 0.87
    }
  ],
  "verdict": {
    "recommendation": "request_changes",
    "overall_score": 6.5
  },
  "model": "provider/model",
  "prompt_version": "review-v1",
  "latency_ms": 12000,
  "cost": 0.0
}
```

### Offline scoring

Score every generated finding against the expert issue using:

- file path match
- line overlap with tolerance
- semantic keyword overlap
- severity agreement
- category agreement

Report:

- precision
- recall
- F1
- finding-level accuracy
- file-location accuracy
- severity accuracy
- false-positive rate
- no-issue accuracy
- average confidence
- confidence calibration
- latency
- cost

Do not use exact patch matching as the primary metric. Several valid fixes can solve the same issue.

### Error buckets

Every failure should be assigned one bucket:

- missed issue
- wrong file
- wrong line
- wrong severity
- wrong category
- vague explanation
- false positive
- duplicate finding
- unsupported claim
- malformed output
- timeout/provider failure
- retrieval failure

### Improvement loop

After each run:

1. Save predictions and scores.
2. Export false positives and missed issues.
3. Inspect the worst examples.
4. Classify root cause:
   - ingestion
   - retrieval
   - prompt
   - model
   - aggregation
   - line mapping
   - schema validation
5. Apply the smallest targeted change.
6. Rerun the same fixed validation split.
7. Compare against the previous result.
8. Keep the change only if quality improves without unacceptable latency/cost regression.

Keep a regression set of every previously fixed failure.

## Implementation phases

### Phase 1: offline correctness harness

- Add `eval_data/code_review_bench_eval.py`.
- Download the dataset through `datasets`.
- Normalize offline rows into a stable input format.
- Accept agent predictions as JSONL.
- Compute matching metrics and failure buckets.
- Write `eval_data/code_review_bench_results.json`.
- Add a smoke fixture with three rows for CI.

### Phase 2: dry-run review adapter

- Add a review-engine mode that accepts a prepared diff and changed files.
- Disable GitHub writes.
- Disable human interrupt.
- Return findings, verdict, model metadata, and RAGAS task metadata.
- Add `--dataset code-review-bench` and `--max-rows N`.
- Store large inputs/outputs in the HF Bucket.

### Phase 3: RAGAS correctness

- Persist RAGAS task state with the review ID.
- Do not mark RAGAS complete when scores are empty.
- Return completed metric values or a clear failure reason.
- Compare retrieval metrics against issue-detection metrics.
- Add a no-LLM deterministic retrieval baseline.

### Phase 4: durable production workflow

- Move jobs and approvals to Supabase/Postgres.
- Store LangGraph thread IDs and head SHAs.
- Use a durable queue/worker.
- Add retries, timeouts, cancellation, and dead-letter states.
- Store artifacts and reports in the HF Bucket.

### Phase 5: authentication and GitHub

- Implement GitHub OAuth or GitHub App login.
- Use secure HTTP-only sessions.
- Associate jobs with users.
- Enforce repository access and job ownership.
- Add idempotency keys.
- Recheck PR head SHA before posting.
- Return exact GitHub comment URLs and failure counts.

### Phase 6: model routing

- Add `fast`, `balanced`, and `deep` profiles.
- Prefer free/local providers when configured.
- Use provider health checks.
- Retry only transient errors.
- Enforce token, latency, and cost budgets.
- Record provider/model/prompt version for every run.

### Phase 7: release verification

- Unit tests
- API contract tests
- Graph interrupt/resume tests
- Fake GitHub integration tests
- Browser tests
- Container build
- Vercel build
- Secret scan
- Dependency audit
- Deployment health check
- Offline benchmark gate
- Live PR cross-check

## Release gates

Do not call a release production-ready unless:

- offline F1 meets the agreed threshold
- false-positive rate is measured
- no malformed predictions occur in the smoke set
- RAGAS returns real metrics or an explicit failure state
- review state survives a worker restart
- approval is authenticated
- posting is idempotent
- a live PR review matches an independent cross-check
- Vercel and backend health checks pass

## Live cross-check procedure

1. Select a public PR not used in development.
2. Run it through the React/Vercel flow.
3. Save the generated findings.
4. Run the Hugging Face/legacy flow against the same repository, PR, and head SHA.
5. Compare:
   - finding count
   - file and line locations
   - severity
   - explanation
   - verdict
   - RAGAS status and scores
6. Do not approve or post until the comparison is complete.
7. Record the result in the evaluation report.

## Storage design

HF Bucket:

```text
eval/<dataset>/<run-id>/
  manifest.json
  predictions.jsonl
  scores.json
  failures.jsonl
  ragas.json
  reports/
```

Postgres:

```text
reviews
review_findings
review_events
model_runs
approvals
evaluation_runs
```

## Correctness principles

- A finding is correct only when its claim is supported by the diff/context.
- A high confidence score must correspond to a high likelihood of correctness.
- A valid review may contain no findings.
- A patch that passes tests is not automatically a good review.
- The system must distinguish “no issue found” from “review failed.”
- Every result must include model, prompt version, dataset revision, and timestamp.

