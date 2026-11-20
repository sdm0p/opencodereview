# Production deployment runbook

## Required before enabling public access

1. Set `GITHUB_TOKEN` or configure the GitHub App/OAuth flow.
2. Set one remote inference credential (`GEMINI_API_KEY`, `GROQ_API_KEY`, or a configured endpoint).
3. Set `OCR_REQUIRE_AUTH=true` and a strong `OCR_API_KEY` until web OAuth is enabled.
4. Set `OCR_ALLOWED_ORIGINS` to the exact Vercel origin.
5. Set `OPENCODEREVIEW_DB_PATH` to a mounted persistent path, or migrate the
   job/checkpoint store to Postgres before running more than one replica.
6. Keep `OPENCODEREVIEW_ALLOW_LOCAL_EMBEDDINGS=false` on small hosts.

## Verification gates

- `GET /api/health` reports `remote_provider_configured` and `configured_path`.
- A real PR reaches `awaiting_approval` without posting comments automatically.
- The decision endpoint returns a structured posting receipt.
- RAGAS reaches `completed` with non-empty metric values, or reports a bounded
  terminal failure with an actionable reason.
- Restart the worker and confirm an awaiting-approval job remains readable.
- Run the offline Code Review Bench scorer and retain its JSON result artifact.

The current Hugging Face Space passes the health and live-review gates, but its
default filesystem is ephemeral. Configure persistent storage before claiming
restart-safe production durability.
