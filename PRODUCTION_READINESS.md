# Production readiness checklist

## Verified

- Supabase Postgres stores review jobs and survives a Space restart.
- Review findings and verdicts are recoverable after restart.
- RAGAS inputs and terminal failures are persisted in job payloads.
- Groq provider routing avoids an exhausted Gemini quota when Groq is configured.
- RAGAS can produce real context precision, faithfulness, and MMR values.
- RAGAS provider calls have bounded retries and timeouts.
- GitHub posting receipts distinguish inline, fallback, finding, and verdict failures.
- React production build and Vercel deployment succeed.
- A reproducible diff-backed benchmark fixture exists for five public PRs.

## Required before production release

- Generate predictions from the five diff-backed fixtures and record precision,
  recall, F1, severity, category, and false-positive results.
- Run an approve decision against a safe test PR and verify the GitHub posting
  receipt and resulting comments.
- Add per-user GitHub OAuth or GitHub App authentication and job ownership.
- Enable required API authentication and rate limits after OAuth is wired into
  the frontend.
- Add browser, API lifecycle, restart-recovery, and posting integration tests.
- Verify a second repository and PR independently of the project repository.

RAGAS scores must remain explicitly marked unavailable or failed when the
provider cannot return. The system must never substitute invented metrics.
