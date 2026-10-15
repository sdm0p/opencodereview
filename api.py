"""Small HTTP boundary for the React client.

The review engine remains in the existing LangGraph modules. This service only
owns job lifecycle and approval state; production deployments should replace
the process-local registry with a durable store and queue.
"""
from __future__ import annotations

import os
import json
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

from app import DB_PATH, _build_and_stream, _normalize_endpoint_name, _ragas_tasks, _ragas_lock
from graph import build_graph
from observability import TokenCostCallback, flush_langfuse, get_langfuse_handler
from state import Finding, Verdict

app = FastAPI(title="OpenCodeReview API", version="0.1.0")
allowed_origins = [item.strip() for item in os.getenv("OCR_ALLOWED_ORIGINS", "http://localhost:5173").split(",") if item.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["GET", "POST"], allow_headers=["*"], allow_credentials=False)

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=int(os.getenv("OCR_MAX_WORKERS", "2")))

def _init_jobs() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS review_jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        conn.commit()

_init_jobs()

def _save_job(job: dict[str, Any]) -> None:
    payload = {k: v for k, v in job.items() if k not in {"future"}}
    if "config" in payload:
        payload["config"] = {"configurable": payload["config"].get("configurable", {})}
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR REPLACE INTO review_jobs (id, payload) VALUES (?, ?)", (job["id"], json.dumps(payload, default=str)))
        conn.commit()

def _load_job(job_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT payload FROM review_jobs WHERE id = ?", (job_id,)).fetchone()
    return json.loads(row[0]) if row else None


class ReviewRequest(BaseModel):
    repo: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    pr_number: int = Field(gt=0)
    endpoint: str | None = None


class DecisionRequest(BaseModel):
    action: Literal["approve", "reject"]


def _authorize(token: str | None) -> None:
    expected = os.getenv("OCR_API_KEY", "").strip()
    if expected and token != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in job.items() if key not in {"config", "future"}}
    return result

def _with_ragas(job: dict[str, Any]) -> dict[str, Any]:
    result = _public_job(job)
    task_id = result.get("ragas_task_id")
    if task_id:
        with _ragas_lock:
            task = _ragas_tasks.get(task_id)
        if not task or not task.get("done"):
            result["ragas"] = {"status": "pending"}
        elif task.get("error"):
            result["ragas"] = {"status": "failed", "error": task.get("error")}
        else:
            result["ragas"] = {"status": "completed", "scores": task.get("scores")}
    return result


def _run_job(job_id: str, request: ReviewRequest) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "processing"
            _save_job(_jobs[job_id])
    try:
        state, config, cost, ragas = _build_and_stream(request.repo, request.pr_number, _normalize_endpoint_name(request.endpoint), ragas_background=True)
        verdict: Verdict | None = state.values.get("verdict")
        findings: list[Finding] = state.values.get("final_findings", [])
        with _lock:
            _jobs[job_id].update(status="awaiting_approval", config=config, verdict=verdict.model_dump(mode="json") if verdict else None, findings=[item.model_dump(mode="json") for item in findings], cost=cost, ragas_task_id=ragas if isinstance(ragas, str) else None, ragas_scores=ragas if isinstance(ragas, dict) else None)
            _save_job(_jobs[job_id])
    except Exception as exc:
        with _lock:
            _jobs[job_id].update(status="failed", error=str(exc)[:500])
            _save_job(_jobs[job_id])


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/reviews", status_code=202)
def create_review(request: ReviewRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(x_api_key)
    job_id = str(uuid.uuid4())
    job = {"id": job_id, "repo": request.repo, "pr_number": request.pr_number, "status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
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
