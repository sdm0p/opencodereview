# OpenCodeReview architecture

The deployed path is intentionally remote-inference only. The API and Gradio
process orchestrate GitHub retrieval and LangGraph; Gemini, Groq, or another
configured OpenAI-compatible provider performs model inference. No model is
downloaded or executed by the hosted API by default.

## Runtime boundaries

- `backend/api.py`: authenticated HTTP job lifecycle, persistence, approval, and
  posting receipts.
- `backend/app.py`: Gradio UI and review orchestration adapter.
- `backend/graph.py`, `nodes/`, `backend/state.py`: LangGraph workflow and typed
  state.
- `backend/integrations/github_client.py`: GitHub reads, caching, and writes.
- `backend/integrations/llm_factory.py` and
  `backend/integrations/endpoints.py`: remote provider selection and fallback.
- `backend/integrations/auth.py` and `config.py`: authentication and settings.
- Root modules such as `main.py`, `app.py`, and `combined_server.py` are
  compatibility entry points. The application implementation lives in
  `backend/`.
- `frontend/`: Vercel React client; review requests use FastAPI, while provider
  connectivity tests use the Vercel function.
- `eval_data/`: offline benchmark scorer and RAGAS evaluation utilities.
- `prompts/`: review prompts kept outside orchestration code.

## Local model policy

Local embeddings are disabled unless `OPENCODEREVIEW_ALLOW_LOCAL_EMBEDDINGS=true`
is explicitly set. This keeps laptops and hosted containers from pulling
PyTorch or embedding models. Retrieval gracefully returns no extra context and
the review continues with the PR diff when the flag is disabled.

## Persistence boundary

The API stores job records in Supabase Postgres when `SUPABASE_DB_URL` is
configured and otherwise uses SQLite. LangGraph checkpoints use a separate
SQLite saver. Process-local worker and session state is not shared between
replicas, so a durable queue and shared checkpoint/session storage are needed
before horizontal scaling.
