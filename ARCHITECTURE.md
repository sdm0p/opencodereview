# OpenCodeReview architecture

The deployed path is intentionally remote-inference only. The API and Gradio
process orchestrate GitHub retrieval and LangGraph; Gemini, Groq, or another
configured OpenAI-compatible provider performs model inference. No model is
downloaded or executed by the hosted API by default.

## Runtime boundaries

- `api.py`: authenticated HTTP job lifecycle, persistence, approval, and
  posting receipts.
- `app.py`: Gradio entry point and review orchestration adapter.
- `graph.py`, `nodes/`, `state.py`: LangGraph workflow and typed state.
- `github_client.py`: GitHub reads, caching, and write operations.
- `llm_factory.py`, `endpoints.py`: remote provider selection and fallback.
- `frontend/`: Vercel React client; it talks only to the API boundary.
- `eval_data/`: offline benchmark scorer and RAGAS evaluation utilities.
- `prompts/`: review prompts kept outside orchestration code.

## Local model policy

Local embeddings are disabled unless `OPENCODEREVIEW_ALLOW_LOCAL_EMBEDDINGS=true`
is explicitly set. This keeps laptops and hosted containers from pulling
PyTorch or embedding models. Retrieval gracefully returns no extra context and
the review continues with the PR diff when the flag is disabled.

## Persistence boundary

The API persists jobs in SQLite so a process restart can recover queued and
awaiting-approval jobs. SQLite on an ephemeral Space filesystem is not a
replacement for a managed database or mounted volume; production deployment
should point `DB_PATH` at durable storage before horizontal scaling.
