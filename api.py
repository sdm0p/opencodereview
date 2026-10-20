"""Small HTTP boundary for the React client.

The review engine remains in the existing LangGraph modules. This service only
owns job lifecycle and approval state; production deployments should replace
the process-local registry with a durable store and queue.
"""
from __future__ import annotations

import os
import json
import secrets
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel, Field

from app import DB_PATH, _build_and_stream, _normalize_endpoint_name, _ragas_tasks, _ragas_lock, _spawn_ragas_background
from graph import build_graph
from observability import TokenCostCallback, flush_langfuse, get_langfuse_handler
from state import Finding, Verdict

app = FastAPI(title="OpenCodeReview API", version="0.1.0")
allowed_origins = [item.strip() for item in os.getenv("OCR_ALLOWED_ORIGINS", "http://localhost:5173").split(",") if item.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["GET", "POST"], allow_headers=["*"], allow_credentials=False)

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=int(os.getenv("OCR_MAX_WORKERS", "2")))


def _postgres_connect():
    """Return a Supabase Postgres connection when configured, else None."""
    url = os.getenv("SUPABASE_DB_URL", "").strip()
    if not url:
        return None
    try:
        import psycopg
        return psycopg.connect(url)
    except Exception:
        return None

def _init_jobs() -> None:
    pg = _postgres_connect()
    if pg is not None:
        with pg:
            with pg.cursor() as cur:
                cur.execute("""create table if not exists public.review_jobs (
                    id uuid primary key, repo text not null, pr_number integer not null,
                    endpoint text, status text not null, payload jsonb not null,
                    created_at timestamptz not null default now(), updated_at timestamptz not null default now())""")
                cur.execute("create index if not exists review_jobs_status_idx on public.review_jobs (status)")
        pg.close()
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS review_jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        conn.commit()

_init_jobs()

def _save_job(job: dict[str, Any]) -> None:
    payload = {k: v for k, v in job.items() if k not in {"future"}}
    if "config" in payload:
        payload["config"] = {"configurable": payload["config"].get("configurable", {})}
    pg = _postgres_connect()
    if pg is not None:
        with pg:
            with pg.cursor() as cur:
                cur.execute("""insert into public.review_jobs (id, repo, pr_number, endpoint, status, payload, updated_at)
                    values (%s, %s, %s, %s, %s, %s::jsonb, now())
                    on conflict (id) do update set repo=excluded.repo, pr_number=excluded.pr_number,
                    endpoint=excluded.endpoint, status=excluded.status, payload=excluded.payload, updated_at=now()""",
                    (job["id"], job["repo"], job["pr_number"], job.get("endpoint"), job["status"], json.dumps(payload, default=str)))
        pg.close()
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR REPLACE INTO review_jobs (id, payload) VALUES (?, ?)", (job["id"], json.dumps(payload, default=str)))
        conn.commit()

def _load_job(job_id: str) -> dict[str, Any] | None:
    pg = _postgres_connect()
    if pg is not None:
        with pg.cursor() as cur:
            cur.execute("select payload from public.review_jobs where id = %s", (job_id,))
            row = cur.fetchone()
        pg.close()
        return json.loads(row[0]) if row else None
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT payload FROM review_jobs WHERE id = ?", (job_id,)).fetchone()
    return json.loads(row[0]) if row else None

def _load_jobs() -> list[dict[str, Any]]:
    pg = _postgres_connect()
    if pg is not None:
        with pg.cursor() as cur:
            cur.execute("select payload from public.review_jobs")
            rows = cur.fetchall()
        pg.close()
        return [json.loads(row[0]) if isinstance(row[0], str) else row[0] for row in rows]
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT payload FROM review_jobs").fetchall()
    return [json.loads(row[0]) for row in rows]


class ReviewRequest(BaseModel):
    repo: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    pr_number: int = Field(gt=0)
    endpoint: str | None = None


class DecisionRequest(BaseModel):
    action: Literal["approve", "reject"]


def _authorize(token: str | None) -> None:
    expected = os.getenv("OCR_API_KEY", "").strip()
    # Authentication is opt-in for local development, but deployments can
    # require it explicitly.  Never silently treat a missing production key
    # as authenticated.
    require_auth = os.getenv("OCR_REQUIRE_AUTH", "false").strip().lower() in {"1", "true", "yes", "on"}
    if require_auth and not expected:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    if expected and (not token or not secrets.compare_digest(token, expected)):
        raise HTTPException(status_code=401, detail="Invalid API key")


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in job.items() if key not in {"config", "future"}}
    return result

def _with_ragas(job: dict[str, Any]) -> dict[str, Any]:
    result = _public_job(job)
    task_id = result.get("ragas_task_id")
    if not task_id:
        result["ragas"] = {
            "status": "skipped",
            "scores": result.get("ragas_scores") or {},
            "reason": "No retrieval context was available for RAGAS evaluation",
        }
    else:
        with _ragas_lock:
            task = _ragas_tasks.get(task_id)
        if not task:
            # Task registries are process-local. A restart invalidates an
            # in-flight task instead of making the client poll forever.
            if result.get("ragas_scores"):
                result["ragas"] = {"status": "completed", "scores": result["ragas_scores"]}
            else:
                result["ragas"] = {"status": "unavailable", "error": "RAGAS task was lost during process restart"}
        elif not task.get("done"):
            timeout = float(os.getenv("OCR_RAGAS_TIMEOUT_SECONDS", "90"))
            started = float(task.get("started", 0) or 0)
            if started and datetime.now(timezone.utc).timestamp() - started > timeout:
                result["ragas"] = {"status": "failed", "error": f"RAGAS exceeded {int(timeout)} second timeout"}
            else:
                result["ragas"] = {"status": "pending"}
        elif task.get("error"):
            result["ragas"] = {"status": "failed", "error": task.get("error")}
        else:
            scores = task.get("scores") or {}
            if scores:
                result["ragas"] = {"status": "completed", "scores": scores}
                if not job.get("ragas_scores"):
                    with _lock:
                        live = _jobs.get(job.get("id"))
                        if live is not None:
                            live["ragas_scores"] = scores
                            live["ragas_status"] = "completed"
                            _save_job(live)
            else:
                result["ragas"] = {"status": "completed_no_scores", "scores": {}, "error": "RAGAS completed without metric values"}
    return result


def _run_job(job_id: str, request: ReviewRequest) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "processing"
            _save_job(_jobs[job_id])
    try:
        state, config, cost, ragas, ragas_inputs = _build_and_stream(request.repo, request.pr_number, _normalize_endpoint_name(request.endpoint), ragas_background=True)
        verdict: Verdict | None = state.values.get("verdict")
        findings: list[Finding] = state.values.get("final_findings", [])
        with _lock:
            _jobs[job_id].update(status="awaiting_approval", config=config, verdict=verdict.model_dump(mode="json") if verdict else None, findings=[item.model_dump(mode="json") for item in findings], cost=cost, ragas_task_id=ragas if isinstance(ragas, str) else None, ragas_scores=ragas if isinstance(ragas, dict) else None, ragas_inputs=ragas_inputs or None)
            _save_job(_jobs[job_id])
    except Exception as exc:
        with _lock:
            _jobs[job_id].update(status="failed", error=str(exc)[:500])
            _save_job(_jobs[job_id])


def _hydrate_jobs() -> None:
    """Restore persisted jobs after a process restart and resume interrupted work."""
    for persisted in _load_jobs():
        job_id = persisted.get("id")
        if not job_id:
            continue
        status = persisted.get("status")
        _jobs[job_id] = persisted
        if persisted.get("ragas_inputs") and persisted.get("ragas_task_id") and not persisted.get("ragas_scores"):
            new_task = _spawn_ragas_background(
                persisted["ragas_inputs"],
                persisted.get("repo", ""),
                int(persisted.get("pr_number", 0)),
                None,
                None,
            )
            persisted["ragas_task_id"] = new_task
            _save_job(persisted)
        if status in {"queued", "processing", "processing_decision"}:
            # Decision processing cannot be safely replayed without an explicit
            # user action; review execution can be resumed idempotently.
            if status == "processing_decision":
                _jobs[job_id]["status"] = "awaiting_approval"
                _save_job(_jobs[job_id])
                continue
            try:
                request = ReviewRequest(repo=persisted["repo"], pr_number=int(persisted["pr_number"]), endpoint=persisted.get("endpoint"))
            except Exception:
                _jobs[job_id].update(status="failed", error="Persisted job request is invalid")
                _save_job(_jobs[job_id])
                continue
            _jobs[job_id]["status"] = "queued"
            _save_job(_jobs[job_id])
            _jobs[job_id]["future"] = _executor.submit(_run_job, job_id, request)


_hydrate_jobs()


@app.get("/api/health")
def health() -> dict[str, Any]:
    provider_configured = any(
        os.getenv(name, "").strip()
        for name in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )
    require_auth = os.getenv("OCR_REQUIRE_AUTH", "false").strip().lower() in {"1", "true", "yes", "on"}
    persistence = "local_default_path"
    if os.getenv("SUPABASE_DB_URL"):
        probe = _postgres_connect()
        if probe is not None:
            probe.close()
            persistence = "supabase_postgres"
        else:
            persistence = "supabase_configured_unreachable"
    elif os.getenv("OPENCODEREVIEW_DB_PATH"):
        persistence = "configured_path"
    return {
        "status": "ok",
        "inference": "remote_provider_configured" if provider_configured else "fallback_or_unconfigured",
        "authentication": "required" if require_auth else "optional",
        "persistence": persistence,
        "local_embeddings": os.getenv("OPENCODEREVIEW_ALLOW_LOCAL_EMBEDDINGS", "false").lower()
        in {"1", "true", "yes", "on"},
    }


@app.post("/api/reviews", status_code=202)
def create_review(request: ReviewRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(x_api_key)
    job_id = str(uuid.uuid4())
    job = {"id": job_id, "repo": request.repo, "pr_number": request.pr_number, "endpoint": request.endpoint, "status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    with _lock:
        _jobs[job_id] = job
        _save_job(job)
    job["future"] = _executor.submit(_run_job, job_id, request)
    return _with_ragas(job)


@app.get("/api/reviews/{job_id}")
def get_review(job_id: str, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(x_api_key)
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Review not found")
    return _with_ragas(job)


@app.post("/api/reviews/{job_id}/decision")
def decide_review(job_id: str, request: DecisionRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(x_api_key)
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            job = _load_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Review not found")
        if job.get("status") != "awaiting_approval":
            raise HTTPException(status_code=409, detail="Review is not awaiting approval")
        config = job["config"]
        job["status"] = "processing_decision"
        _save_job(job)
    if request.action == "reject":
        with _lock:
            job.update(status="rejected", decision="reject")
            _save_job(job)
        return _public_job(job)
    graph = build_graph(DB_PATH)
    try:
        callbacks = [TokenCostCallback()]
        handler = get_langfuse_handler()
        if handler:
            callbacks.append(handler)
        config = dict(config)
        config["callbacks"] = callbacks
        result = graph.invoke(Command(resume={"action": "approve"}), config)
        posting_result = result.get("posting_result") if isinstance(result, dict) else None
        final_status = (posting_result or {}).get("status", "posted")
        with _lock:
            job.update(status=final_status, decision="approve", posting_result=posting_result)
            _save_job(job)
        return _public_job(job)
    except Exception as exc:
        with _lock:
            job.update(status="failed", error=str(exc)[:500])
            _save_job(job)
        raise HTTPException(status_code=500, detail="Decision processing failed") from exc
    finally:
        conn = getattr(graph, "_opencodereview_conn", None)
        if conn:
            conn.close()
        flush_langfuse()
