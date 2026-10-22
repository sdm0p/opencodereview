"""Small HTTP boundary for the React client.

The review engine remains in the existing LangGraph modules. This service only
owns job lifecycle and approval state; production deployments should replace
the process-local registry with a durable store and queue.
"""
from __future__ import annotations

import os
import hashlib
import time
from pathlib import Path
import json
import secrets
import sqlite3
import threading
import uuid
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel, Field

from app import DB_PATH, _build_and_stream, _normalize_endpoint_name, _ragas_tasks, _ragas_lock, _spawn_ragas_background
from graph import build_graph
from observability import TokenCostCallback, flush_langfuse, get_langfuse_handler
from state import Finding, Verdict
from endpoints import register_endpoint, unregister_endpoint

app = FastAPI(title="OpenCodeReview API", version="0.1.0")
allowed_origins = [item.strip() for item in os.getenv("OCR_ALLOWED_ORIGINS", "http://localhost:5173").split(",") if item.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["GET", "POST"], allow_headers=["*"], allow_credentials=True)

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=int(os.getenv("OCR_MAX_WORKERS", "2")))
_rate_lock = threading.Lock()
_rate_windows: dict[str, list[float]] = {}
_oauth_states: set[str] = set()
_sessions: dict[str, dict[str, Any]] = {}

def _rate_limit(request: Request) -> None:
    limit = int(os.getenv("OCR_RATE_LIMIT_PER_MINUTE", "0") or 0)
    if limit <= 0:
        return
    now = datetime.now(timezone.utc).timestamp()
    forwarded = request.headers.get("x-forwarded-for") or request.headers.get("x-real-ip")
    key = (forwarded.split(",", 1)[0].strip() if forwarded else None) or "global"
    with _rate_lock:
        window = [stamp for stamp in _rate_windows.get(key, []) if now - stamp < 60]
        if len(window) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        window.append(now)
        _rate_windows[key] = window


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
    payload = {k: v for k, v in job.items() if k not in {"future", "ragas_inputs"}}
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
        if not row:
            return None
        return json.loads(row[0]) if isinstance(row[0], str) else row[0]
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
    return [json.loads(row[0]) if isinstance(row[0], str) else row[0] for row in rows]


class ReviewRequest(BaseModel):
    repo: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    pr_number: int = Field(gt=0)
    endpoint: str | None = None
    provider: str | None = Field(default=None, pattern=r"^(openai|anthropic|google)$")
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    review_mode: Literal["fast", "thorough"] = "thorough"
    baseline_sha: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{40}$")


class DecisionRequest(BaseModel):
    action: Literal["approve", "reject"]


class FindingFeedbackRequest(BaseModel):
    finding_index: int = Field(ge=0)
    label: Literal['correct', 'incorrect', 'useful']


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

def _authorize_request(token: str | None, request: Request) -> None:
    if _session(request):
        return
    _authorize(token)

def _session(request: Request) -> dict[str, Any] | None:
    sid = request.cookies.get("ocr_session")
    return _sessions.get(sid) if sid else None

@app.get("/api/auth/github/start")
def github_login() -> RedirectResponse:
    client_id = os.getenv("GITHUB_OAUTH_CLIENT_ID", "").strip()
    if not client_id:
        raise HTTPException(status_code=503, detail="GitHub OAuth is not configured")
    state = secrets.token_urlsafe(32)
    _oauth_states.add(state)
    callback = os.getenv("GITHUB_OAUTH_CALLBACK_URL", "").strip() or "http://localhost:8000/api/auth/github/callback"
    url = "https://github.com/login/oauth/authorize?" + requests.compat.urlencode({"client_id": client_id, "redirect_uri": callback, "scope": "read:user user:email repo", "state": state})
    return RedirectResponse(url)

@app.get("/api/auth/github/callback")
def github_callback(code: str, state: str, response: Request) -> RedirectResponse:
    if state not in _oauth_states:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")
    _oauth_states.discard(state)
    client_id = os.getenv("GITHUB_OAUTH_CLIENT_ID", "").strip()
    client_secret = os.getenv("GITHUB_OAUTH_CLIENT_SECRET", "").strip()
    callback = os.getenv("GITHUB_OAUTH_CALLBACK_URL", "").strip() or "http://localhost:8000/api/auth/github/callback"
    token_resp = requests.post("https://github.com/login/oauth/access_token", data={"client_id": client_id, "client_secret": client_secret, "code": code, "redirect_uri": callback}, headers={"Accept": "application/json"}, timeout=20)
    token_resp.raise_for_status()
    access_token = token_resp.json().get("access_token")
    if not access_token:
        raise HTTPException(status_code=502, detail="GitHub did not return an access token")
    user_resp = requests.get("https://api.github.com/user", headers={"Authorization": f"Bearer {access_token}", "Accept": "application/vnd.github+json"}, timeout=20)
    user_resp.raise_for_status()
    user = user_resp.json()
    sid = secrets.token_urlsafe(32)
    _sessions[sid] = {"token": access_token, "github_id": user.get("id"), "login": user.get("login")}
    frontend = os.getenv("OCR_FRONTEND_URL", allowed_origins[0] if allowed_origins else "http://localhost:5173")
    out = RedirectResponse(frontend)
    out.set_cookie("ocr_session", sid, httponly=True, secure=frontend.startswith("https://"), samesite="lax", max_age=86400)
    return out

@app.get("/api/auth/me")
def github_me(request: Request) -> dict[str, Any]:
    session = _session(request)
    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"github_id": session["github_id"], "login": session["login"]}

@app.post("/api/auth/logout")
def github_logout(request: Request) -> dict[str, str]:
    sid = request.cookies.get("ocr_session")
    if sid:
        _sessions.pop(sid, None)
    return {"status": "logged_out"}


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in job.items() if key not in {"config", "future", "request_fingerprint", "owner_id", "ragas_inputs"}}
    now = time.time()
    result["elapsed_seconds"] = round(max(0, now - job.get("started_timestamp", now)), 1) if job.get("status") in {"processing", "queued"} else job.get("elapsed_seconds", 0)
    if job.get("status") == "queued":
        result["queue_position"] = sum(1 for item in _jobs.values() if item.get("status") == "queued" and item.get("created_at", "") <= job.get("created_at", ""))
    return result

def _with_ragas(job: dict[str, Any]) -> dict[str, Any]:
    result = _public_job(job)
    task_id = result.get("ragas_task_id")
    with _ragas_lock:
        task = _ragas_tasks.get(task_id) if task_id else None
    def finish():
        metrics = (task or {}).get('metrics') or job.get('ragas_metrics') or {}
        evaluation = (task or {}).get('evaluation') or job.get('ragas_evaluation') or {}
        result.setdefault('ragas', {}).update(metrics=metrics, evaluation=evaluation)
        if metrics and result['ragas'].get('scores') and any(m.get('status') != 'completed' for m in metrics.values()):
            result['ragas']['status'] = 'partial'
        if metrics != job.get('ragas_metrics', {}) or evaluation != job.get('ragas_evaluation', {}):
            with _lock:
                live = _jobs.get(job.get('id'))
                if live is not None:
                    live.update(ragas_metrics=metrics, ragas_evaluation=evaluation)
                    _save_job(live)
        return result
    if not task_id:
        result["ragas"] = {
            "status": "skipped",
            "scores": result.get("ragas_scores") or {},
            "reason": "Review incomplete; scoring was not started" if result.get("status") == "incomplete" else "No retrieval context was available for RAGAS evaluation",
        }
    else:
        if job.get("ragas_status") in {"failed", "unavailable", "completed_no_scores"}:
            result["ragas"] = {
                "status": job["ragas_status"],
                "scores": job.get("ragas_scores") or {},
                "error": job.get("ragas_error"),
            }
            return finish()
        with _ragas_lock:
            task = _ragas_tasks.get(task_id)
        if not task:
            # Task registries are process-local. A restart invalidates an
            # in-flight task instead of making the client poll forever.
            if result.get("ragas_scores"):
                result["ragas"] = {"status": "completed", "scores": result["ragas_scores"]}
            else:
                result["ragas"] = {"status": "unavailable", "error": "RAGAS task was lost during process restart"}
                with _lock:
                    live = _jobs.get(job.get("id"))
                    if live is not None:
                        live["ragas_status"] = "unavailable"
                        live["ragas_error"] = result["ragas"]["error"]
                        _save_job(live)
        elif not task.get("done"):
            timeout = float(task.get("timeout_seconds") or os.getenv("OCR_RAGAS_TIMEOUT_SECONDS", "90"))
            started = float(task.get("started", 0) or 0)
            if started and datetime.now(timezone.utc).timestamp() - started > timeout:
                result["ragas"] = {"status": "failed", "error": f"RAGAS exceeded {int(timeout)} second timeout"}
                with _lock:
                    live = _jobs.get(job.get("id"))
                    if live is not None:
                        live["ragas_status"] = "failed"
                        live["ragas_error"] = result["ragas"]["error"]
                        _save_job(live)
            else:
                result["ragas"] = {"status": "pending"}
        elif task.get("error"):
            result["ragas"] = {"status": "failed", "error": task.get("error")}
            with _lock:
                live = _jobs.get(job.get("id"))
                if live is not None:
                    live["ragas_status"] = "failed"
                    live["ragas_error"] = task.get("error")
                    _save_job(live)
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
                with _lock:
                    live = _jobs.get(job.get("id"))
                    if live is not None:
                        live["ragas_status"] = "completed_no_scores"
                        live["ragas_error"] = result["ragas"]["error"]
                        _save_job(live)
    return finish()


def _run_job(job_id: str, request: ReviewRequest) -> None:
    with _lock:
        if _jobs.get(job_id, {}).get("status") == "cancelled":
            return
        if job_id in _jobs:
            _jobs[job_id].update(status="processing", started_timestamp=time.time(), progress={"stage": "ingestion", "completed_stages": []})
            _save_job(_jobs[job_id])
    try:
        ephemeral_endpoint = None
        ragas_started = False
        endpoint_name = _normalize_endpoint_name(request.endpoint)
        if request.api_key:
            if not request.provider or not request.model:
                raise ValueError("provider and model are required when supplying a user API key")
            ephemeral_endpoint = f"user-{job_id}"
            register_endpoint(ephemeral_endpoint, request.provider, request.api_key, request.base_url, request.model)
            endpoint_name = ephemeral_endpoint
        def save_ragas(scores: dict, error: str | None) -> None:
            try:
                with _lock:
                    live = _jobs.get(job_id)
                    if live is None:
                        return
                    live["ragas_scores"] = scores or None
                    live["ragas_status"] = "failed" if error else ("completed" if scores else "completed_no_scores")
                    live["ragas_error"] = error
                    _save_job(live)
            finally:
                # Custom providers are registered only for this job. Keep the
                # endpoint alive until the background RAGAS task has finished;
                # otherwise evaluation silently falls back to built-ins.
                if ephemeral_endpoint:
                    unregister_endpoint(ephemeral_endpoint)

        def progress(stage: str, elapsed: float, update: dict | None = None) -> None:
            with _lock:
                live = _jobs[job_id]
                if live.get("status") == "cancelled":
                    raise RuntimeError("Review cancelled")
                completed = live.setdefault("progress", {}).setdefault("completed_stages", [])
                if stage not in completed:
                    completed.append(stage)
                live["progress"]["stage"] = stage
                live.setdefault("stage_timings", {})[stage] = round(elapsed, 2)
                if isinstance(update, dict):
                    stages = update.get("review_stages", [])
                    if isinstance(stages, list):
                        live.setdefault("review_stages", []).extend(stages)
                    for key in ("findings",):
                        for finding in update.get(key, []) or []:
                            if hasattr(finding, "model_dump"):
                                finding = finding.model_dump(mode="json")
                            if isinstance(finding, dict):
                                live.setdefault("findings", []).append(finding)
                _save_job(live)

        state, config, cost, ragas, ragas_inputs = _build_and_stream(
            request.repo,
            request.pr_number,
            endpoint_name,
            ragas_background=True,
            ragas_result_callback=save_ragas,
            progress_callback=progress,
            review_mode=request.review_mode,
            baseline_sha=request.baseline_sha,
        )
        ragas_started = isinstance(ragas, str)
        actual_head = state.values.get("head_sha")
        verdict: Verdict | None = state.values.get("verdict")
        findings: list[Finding] = state.values.get("final_findings", [])
        with _lock:
            if _jobs[job_id].get("status") == "cancelled":
                return
            if actual_head and _jobs[job_id].get("head_sha") != actual_head:
                # The PR moved between metadata lookup and ingestion. The result
                # can still be viewed, but must not satisfy the old cache key.
                _jobs[job_id]["request_fingerprint"] = None
                _jobs[job_id]["head_sha"] = actual_head
            stages = state.values.get("review_stages", [])
            incomplete = bool(verdict and verdict.recommendation == "incomplete")
            _jobs[job_id].update(review_coverage=state.values.get("review_coverage", {}), review_stages=stages, elapsed_seconds=round(time.time() - _jobs[job_id]["started_timestamp"], 2), status="incomplete" if incomplete else "awaiting_approval", config=config, verdict=verdict.model_dump(mode="json") if verdict else None, findings=[item.model_dump(mode="json") for item in findings], cost=cost, cost_details=config.get("metadata", {}).get("cost_details"), ragas_task_id=ragas if isinstance(ragas, str) else None, ragas_scores=ragas if isinstance(ragas, dict) else None, ragas_inputs=None)
            _save_job(_jobs[job_id])
    except Exception as exc:
        with _lock:
            if _jobs[job_id].get("status") != "cancelled":
                _jobs[job_id].update(status="failed", error=str(exc)[:500])
            _save_job(_jobs[job_id])
    finally:
        if 'ephemeral_endpoint' in locals() and ephemeral_endpoint and not ragas_started:
            unregister_endpoint(ephemeral_endpoint)


def _hydrate_jobs() -> None:
    """Restore persisted jobs after a process restart and resume interrupted work."""
    for persisted in _load_jobs():
        # Evaluation inputs may include an in-memory endpoint object. Never
        # retain or expose that snapshot: it can contain a user's provider key.
        if "ragas_inputs" in persisted:
            persisted.pop("ragas_inputs", None)
            _save_job(persisted)
        job_id = persisted.get("id")
        if not job_id:
            continue
        status = persisted.get("status")
        _jobs[job_id] = persisted
        if persisted.get("ragas_task_id") and not persisted.get("ragas_status"):
            persisted.update(ragas_status="unavailable", ragas_error="Evaluation interrupted by restart; it was not automatically replayed.")
            _save_job(persisted)
        if status in {"queued", "processing"} and persisted.get("custom_provider"):
            persisted.update(status="failed", error="Provider credentials were not persisted; resubmit this review after restart.")
            _save_job(persisted)
            continue
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
        "rate_limit_per_minute": int(os.getenv("OCR_RATE_LIMIT_PER_MINUTE", "0") or 0),
        "persistence": persistence,
        "local_embeddings": os.getenv("OPENCODEREVIEW_ALLOW_LOCAL_EMBEDDINGS", "false").lower()
        in {"1", "true", "yes", "on"},
    }


@app.post("/api/reviews", status_code=202)
def create_review(request: ReviewRequest, http_request: Request, x_api_key: str | None = Header(default=None), idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> dict[str, Any]:
    _authorize_request(x_api_key, http_request)
    _rate_limit(http_request)
    owner = str((_session(http_request) or {}).get("github_id") or "api-key")
    material = request.model_dump()
    head_sha = base_sha = None
    try:
        token = (_session(http_request) or {}).get("token") or os.getenv("GITHUB_TOKEN", "")
        headers = {"Accept": "application/vnd.github+json", "Cache-Control": "no-cache"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = requests.get(f"https://api.github.com/repos/{request.repo}/pulls/{request.pr_number}", headers=headers, timeout=6)
        response.raise_for_status()
        pull = response.json()
        head_sha, base_sha = pull["head"]["sha"], pull["base"]["sha"]
    except (requests.RequestException, KeyError, ValueError):
        pass  # Active deduplication still works if GitHub metadata is unavailable.
    prompt_hash = hashlib.sha256()
    for source in sorted(Path(__file__).parent.joinpath("prompts").glob("*.txt")):
        prompt_hash.update(source.read_bytes())
    for source in sorted(Path(__file__).parent.joinpath("nodes").glob("*.py")):
        prompt_hash.update(source.read_bytes())
    material.update(head_sha=head_sha, base_sha=base_sha, pipeline_version=prompt_hash.hexdigest())
    fingerprint = hashlib.sha256(json.dumps({"owner": owner, "request": material}, sort_keys=True).encode()).hexdigest()
    # Reuse completed previews only for an identical immutable diff and configuration.
    with _lock:
        existing = next((item for item in _jobs.values() if item.get("owner_id") == owner and item.get("request_fingerprint") == fingerprint and (item.get("status") in {"queued", "processing"} or (head_sha and base_sha and item.get("status") == "awaiting_approval" and time.time() - item.get("started_timestamp", 0) < 900))), None)
        if existing is None:
            job_id = str(uuid.uuid4())
            job = {"id": job_id, "repo": request.repo, "pr_number": request.pr_number, "endpoint": request.endpoint, "review_mode": request.review_mode, "baseline_sha": request.baseline_sha, "custom_provider": bool(request.api_key), "head_sha": head_sha, "base_sha": base_sha, "owner_id": owner, "request_fingerprint": fingerprint, "status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
            _jobs[job_id] = job
            _save_job(job)
        else:
            job = existing
    if existing is not None:
        return {**_with_ragas(job), "reused": True}
    job["future"] = _executor.submit(_run_job, job_id, request)
    return _with_ragas(job)


@app.post("/api/reviews/{job_id}/cancel")
def cancel_review(job_id: str, request: Request, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize_request(x_api_key, request)
    owner = str((_session(request) or {}).get("github_id") or "api-key")
    with _lock:
        job = _jobs.get(job_id)
        if not job or job.get("owner_id", "api-key") != owner:
            raise HTTPException(status_code=404, detail="Review not found")
        if job.get("status") not in {"queued", "processing"}:
            raise HTTPException(status_code=409, detail="Only queued or processing reviews can be cancelled")
        future = job.get("future")
        if future:
            future.cancel()
        job.update(status="cancelled", cancellation_note="Running calls finish within their timeout; no results will be posted.")
        _save_job(job)
    return _public_job(job)


@app.get("/api/reviews/{job_id}")
def get_review(job_id: str, request: Request, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    # API-key authorization remains supported for polling; OAuth callers use the cookie path.
    _authorize_request(x_api_key, request)
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            job = _load_job(job_id)
    owner = str((_session(request) or {}).get("github_id") or "api-key")
    if not job or job.get("owner_id", "api-key") != owner:
        raise HTTPException(status_code=404, detail="Review not found")
    return _with_ragas(job)


@app.post('/api/reviews/{job_id}/feedback')
def finding_feedback(job_id: str, payload: FindingFeedbackRequest, request: Request,
                     x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize_request(x_api_key, request)
    _rate_limit(request)
    owner = str((_session(request) or {}).get('github_id') or 'api-key')
    with _lock:
        job = _jobs.get(job_id) or _load_job(job_id)
        if not job or job.get('owner_id', 'api-key') != owner:
            raise HTTPException(status_code=404, detail='Review not found')
        if job.get('status') not in {'awaiting_approval', 'incomplete', 'posted', 'partially_posted', 'rejected', 'post_failed'}:
            raise HTTPException(status_code=409, detail='Wait until the review finishes before giving feedback')
        if payload.finding_index >= len(job.get('findings') or []):
            raise HTTPException(status_code=422, detail='Finding does not exist')
        feedback = dict(job.get('feedback') or {})
        feedback[str(payload.finding_index)] = payload.label
        job['feedback'] = feedback
        job['feedback_updated_at'] = datetime.now(timezone.utc).isoformat()
        _jobs[job_id] = job
        _save_job(job)
    return {'feedback': feedback}


@app.post("/api/reviews/{job_id}/decision")
def decide_review(job_id: str, request: DecisionRequest, http_request: Request, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize_request(x_api_key, http_request)
    _rate_limit(http_request)
    with _lock:
        preview = _jobs.get(job_id) or _load_job(job_id)
    owner = str((_session(http_request) or {}).get("github_id") or "api-key")
    if not preview or preview.get("owner_id", "api-key") != owner:
        raise HTTPException(status_code=404, detail="Review not found")
    if request.action == "approve" and preview.get("baseline_sha"):
        raise HTTPException(status_code=409, detail="Incremental reviews are preview-only. Run a full review before posting.")
    if preview and request.action == "approve" and preview.get("head_sha"):
        headers = {"Accept": "application/vnd.github+json", "Cache-Control": "no-cache"}
        token = (_session(http_request) or {}).get("token") or os.getenv("GITHUB_TOKEN", "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = requests.get(f"https://api.github.com/repos/{preview['repo']}/pulls/{preview['pr_number']}", headers=headers, timeout=6)
            response.raise_for_status()
            current = response.json()
            if current["head"]["sha"] != preview["head_sha"] or current["base"]["sha"] != preview.get("base_sha"):
                raise HTTPException(status_code=409, detail="PR changed since this review; start a new review before posting.")
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Cannot verify the current PR commit; retry before posting.") from exc
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
