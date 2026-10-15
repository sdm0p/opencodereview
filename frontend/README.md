# OpenCodeReview React frontend

This Vite app is the first migration boundary for the Gradio UI. Deploy it as a Vercel static project with the root directory set to `frontend` and `VITE_REVIEW_API_URL` pointing at the persistent Python review API.

The Python API boundary is now available locally with `uvicorn api:app --reload --port 8000`. Set `VITE_REVIEW_API_URL=http://localhost:8000` for local browser integration. The process-local job registry is suitable for development only; production needs a durable job store and queue.
