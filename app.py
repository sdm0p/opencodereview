#!/usr/bin/env python3
"""OpenCodeReview — Gradio Web UI for Hugging Face Spaces.

Run locally:
    python app.py

On Hugging Face Spaces:
    The Dockerfile now defaults to running this module.
    Set GROQ_API_KEY (and optionally GITHUB_TOKEN) as Space secrets.
"""

from __future__ import annotations

import html
import json
import logging
import os
import time
import uuid

import gradio as gr
from langgraph.types import Command

from graph import build_graph
from main import NODE_LABELS, _cleanup_db, SYNTHETIC_STATE
from observability import (
    build_run_metadata,
    enable_langsmith,
    get_langfuse_handler,
    is_langfuse_enabled,
    is_langsmith_enabled,
    estimate_groq_cost,
    flush_langfuse,
    format_cost,
    langfuse_trace,
    log_error_to_backends,
    log_langfuse_score,
    _resolve_langfuse_trace_id,
    HealthStatus,
    TokenCostCallback,
    check_groq_connectivity,
    check_langfuse_connectivity,
)
from endpoints import (
    clear_session_endpoints,
    discover_endpoints,
    endpoint_choices,
    register_endpoint,
    session_endpoints,
)
from state import Verdict

import subprocess
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = "checkpoints.db"

SEVERITY_COLORS = {
    "critical": "#dc2626",
    "high": "#ea580c",
    "medium": "#ca8a04",
    "low": "#16a34a",
    "info": "#2563eb",
}

RECOMMENDATION_ICONS = {
    "approve": "✅",
    "request_changes": "🔄",
    "block": "🚫",
    "comment": "💬",
}


# ─── Orchestration ───────────────────────────────────────────────────────────


def _init_observability_ui() -> None:
    """Initialise observability backends and HTTP cache for the Gradio UI."""
    from github_client import _init_cache as _init_http_cache
    _init_http_cache()

    # LangSmith: enable if key is in environment (check both modern and legacy names)
    ls_key = (
        os.environ.get("LANGSMITH_API_KEY", "").strip()
        or os.environ.get("LANGCHAIN_API_KEY", "").strip()
    )
    if ls_key:
        project = os.environ.get("LANGSMITH_PROJECT",
                    os.environ.get("LANGCHAIN_PROJECT", "opencodereview"))
        enable_langsmith(ls_key, project)
        logger.info("Observability: LangSmith enabled for Gradio UI")
    if get_langfuse_handler():
        logger.info("Observability: Langfuse enabled for Gradio UI")


def _build_and_stream(repo: str, pr_number: int, endpoint_name: str = "") -> tuple:
    """Build the graph, stream up to the interrupt, and return state + config.

    Parameters
    ----------
    endpoint_name : str
        Optional name of a configured custom LLM endpoint
        (``OCR_ENDPOINT_*``).  Empty string uses the built-in provider.
    """
    graph = build_graph(DB_PATH)
    thread_id = str(uuid.uuid4())
    trace_name = f"opencodereview/review/{repo}#{pr_number}"
    tags = ["gradio_ui", repo]
    prompt_versions = {
        "correctness": "v1",
        "security": "v1",
        "test_coverage": "v1",
        "aggregator": "v1",
    }
    cost_tracker = TokenCostCallback()
    metadata = build_run_metadata(
        source="gradio_ui", repo=repo, pr_number=pr_number,
        session_id=thread_id,
        prompt_versions=prompt_versions,
    )
    config: dict = {
        "configurable": {"thread_id": f"web-{thread_id}"},
        "metadata": metadata,
    }

    # Use langfuse_trace context manager which sets trace metadata via
    # propagate_attributes() (langfuse v4+ API) and yields the handler.
    with langfuse_trace(
        trace_name=trace_name,
        tags=tags,
        session_id=thread_id,
        metadata={
            "source": "gradio_ui",
            "repo": repo,
            "pr_number": pr_number,
            "prompt_versions": prompt_versions,
        },
    ) as handler:
        callbacks = [cost_tracker]
        if handler:
            callbacks.append(handler)
        config["callbacks"] = callbacks

        try:
            t0 = time.time()
            initial = {"repo": repo, "pr_number": pr_number}
            if endpoint_name:
                initial["endpoint"] = endpoint_name
            for event in graph.stream(initial, config):
                for node_name in event:
                    label = NODE_LABELS.get(node_name, node_name)
                    logger.info("  ✔ %s  (+%.1fs)", label, time.time() - t0)
            logger.info("  ─── Pipeline complete in %.1fs ───", time.time() - t0)
        except Exception as exc:
            log_error_to_backends(exc, context={"source": "gradio_ui", "phase": "stream", "repo": repo, "pr_number": pr_number})
            raise

        state = graph.get_state(config)

        # ── Log verdict/findings scores and capture trace_id ───────
        # Capture trace_id NOW (immediately after graph completes)
        # so it can be used after the long-running RAGAS computation.
        # _resolve_langfuse_trace_id() can return None after ~60s delays
        # because contextvars may be lost.
        run_trace_id = _resolve_langfuse_trace_id(handler)

        verdict = state.values.get("verdict")
        findings = state.values.get("final_findings", [])
        if verdict:
            log_langfuse_score(
                name="verdict_score",
                value=verdict.overall_score,
                comment=f"{repo}#{pr_number} — {verdict.recommendation}: {verdict.summary[:100]}",
                trace_id=run_trace_id,
                handler=handler,
            )
            log_langfuse_score(
                name="findings_count",
                value=len(findings),
                comment=f"{repo}#{pr_number}",
                trace_id=run_trace_id,
                handler=handler,
            )

        # ── Compute and log RAGAS retrieval scores ─────────────────
        ragas_scores: dict[str, float | None] = {}
        context_chunks = state.values.get("context_chunks", [])
        if context_chunks:
            try:
                from eval_data.ragas_eval import compute_ragas_retrieval_scores

                query = state.values.get("diff", "") or ""
                contexts = [c.content for c in context_chunks if c.content]

                # Build answer text from findings + verdict for
                # faithfulness & answer_relevancy metrics
                answer_lines = []
                for f in findings:
                    answer_lines.append(
                        f"[{f.severity.value}] {f.file_path}:{f.line_start}-{f.line_end}: {f.comment}"
                    )
                if state.values.get("verdict"):
                    v = state.values["verdict"]
                    answer_lines.append(
                        f"Verdict: {v.recommendation} (score={v.overall_score}/10) — {v.summary}"
                    )
                answer_text = "\n".join(answer_lines) if answer_lines else None

                ragas_scores = compute_ragas_retrieval_scores(
                    query=query,
                    retrieved_contexts=contexts,
                    answer=answer_text,
                    endpoint=endpoint_name or None,
                )

                # Log RAGAS scores with the explicitly captured trace_id
                # to avoid contextvar loss during the long computation.
                # handler=handler is also passed as fallback resolution.
                for metric_name, value in ragas_scores.items():
                    if value is None:
                        continue
                    log_langfuse_score(
                        name=f"ragas_{metric_name}",
                        value=value,
                        comment=f"{repo}#{pr_number} — {metric_name}",
                        trace_id=run_trace_id,
                        handler=handler,
                    )

                logger.info(
                    "RAGAS scores logged to Langfuse: %s",
                    " | ".join(
                        f"{k}={v:.4f}" for k, v in ragas_scores.items()
                        if v is not None
                    ),
                )
            except ImportError:
                logger.info("RAGAS not installed — skipping RAGAS scoring (pip install ragas)")
            except Exception as exc:
                logger.warning("RAGAS scoring failed: %s", exc)

    # Close the checkpointer connection (checkpoints persist to disk so the
    # resume path opens a fresh one) and ship any queued Langfuse scores.
    conn = getattr(graph, "_opencodereview_conn", None)
    if conn:
        conn.close()
    flush_langfuse()

    return state, config, cost_tracker.summary(), ragas_scores


def _format_findings_table(findings: list) -> str:
    """Build an HTML table of findings with severity-coloured badges."""
    if not findings:
        return '<p style="color:var(--text-muted)">No issues found.</p>'

    rows: list[str] = []
    for f in findings:
        color = SEVERITY_COLORS.get(f.severity.value, "#666")
        rows.append(
            f"<tr>"
            f'<td><span style="display:inline-block;padding:2px 10px;'
            f'border-radius:12px;background:{color}22;'
            f'color:{color};font-weight:700;font-size:0.85em">'
            f"{f.severity.value.upper()}</span></td>"
            f"<td>{f.category}</td>"
            f'<td><code>{f.file_path}:{f.line_start}–{f.line_end}</code></td>'
            f"<td>{f.comment}</td>"
            f'<td style="text-align:center">{f.confidence:.0%}</td>'
            f"</tr>"
        )

    return (
        '<div style="overflow-x:auto">'
        '<table style="width:100%;border-collapse:collapse;font-size:0.9em">'
        "<thead>"
        f'<tr style="background:var(--bg-table-header);border-bottom:2px solid var(--border-color)">'
        '<th style="padding:10px 12px;text-align:left;color:var(--text-primary)">Severity</th>'
        '<th style="padding:10px 12px;text-align:left;color:var(--text-primary)">Category</th>'
        '<th style="padding:10px 12px;text-align:left;color:var(--text-primary)">Location</th>'
        '<th style="padding:10px 12px;text-align:left;color:var(--text-primary)">Comment</th>'
        '<th style="padding:10px 12px;text-align:left;color:var(--text-primary)">Confidence</th>'
        "</tr>"
        "</thead>"
        f'<tbody style="color:var(--text-secondary)">' + "".join(rows) + "</tbody>"
        "</table>"
        "</div>"
    )


def _format_verdict(verdict: Verdict | None) -> str:
    """Build a verdict summary card with a score bar and severity chips."""
    if verdict is None:
        return '<p class="muted">No verdict produced.</p>'

    icon = RECOMMENDATION_ICONS.get(verdict.recommendation, "📋")
    color = (
        "#16a34a" if verdict.recommendation == "approve"
        else "#dc2626" if verdict.recommendation == "block"
        else "#ea580c"
    )
    pct = max(0, min(100, int(verdict.overall_score * 10)))

    chips = []
    if verdict.critical_count:
        chips.append(
            f'<span class="sev-chip" style="color:#dc2626;border-color:#dc262644;background:#dc262611">'
            f"critical × {verdict.critical_count}</span>"
        )
    if verdict.high_count:
        chips.append(
            f'<span class="sev-chip" style="color:#ea580c;border-color:#ea580c44;background:#ea580c11">'
            f"high × {verdict.high_count}</span>"
        )
    chips_html = (
        f'<div class="chips-row" style="margin-top:10px">' + "".join(chips) + "</div>"
        if chips else ""
    )

    return (
        '<div class="ocr-card verdict-card" '
        f'style="border-left-color:{color}">'
        '<div class="verdict-head">'
        f'<span class="verdict-icon">{icon}</span>'
        f'<span class="verdict-reco" style="color:{color}">{verdict.recommendation.upper()}</span>'
        f'<span class="verdict-score">Score <b>{verdict.overall_score:.1f}</b>/10</span>'
        "</div>"
        f'<div class="score-track"><div class="score-fill" style="width:{pct}%;background:{color}"></div></div>'
        f'<p class="verdict-summary">{verdict.summary}</p>'
        + chips_html
        + "</div>"
    )


RAGAS_METRIC_LABELS = {
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "mmr": "Mean Reciprocal Rank",
}

RAGAS_METRIC_TOOLTIPS = {
    "context_precision": "Are the most relevant code chunks ranked first?",
    "context_recall": "Did we find ALL the relevant code? (requires ground truth)",
    "faithfulness": "Does the review answer stick to the retrieved code?",
    "answer_relevancy": "Does the review address the PR diff?",
    "mmr": "How early in the list does the first useful chunk appear?",
}


def _format_ragas_scores(scores: dict[str, float | None]) -> str:
    """Build an HTML card showing RAGAS retrieval quality metrics."""
    if not scores:
        return ""

    def _score_color(val: float) -> str:
        if val >= 0.8:
            return "#16a34a"
        elif val >= 0.5:
            return "#ca8a04"
        else:
            return "#dc2626"

    rows: list[str] = []
    failed = 0
    for key, value in scores.items():
        label = RAGAS_METRIC_LABELS.get(key, key.replace("_", " ").title())
        tooltip = RAGAS_METRIC_TOOLTIPS.get(key, "")
        tip_html = (
            f'<span class="tip" title="{tooltip}">ℹ️</span>' if tooltip else ""
        )
        if value is None:
            failed += 1
            rows.append(
                f'<div class="metric-row">'
                f'<span class="metric-label">{label} {tip_html}</span>'
                f'<span class="metric-value muted">N/A — failed</span>'
                f"</div>"
            )
            continue
        color = _score_color(value)
        pct = int(value * 100)
        rows.append(
            f'<div class="metric-row">'
            f'<span class="metric-label">{label} {tip_html}</span>'
            f'<span class="metric-value" style="color:{color}">{value:.3f}</span>'
            f"</div>"
            f'<div class="metric-track"><div class="metric-fill" style="width:{pct}%;background:{color}"></div></div>'
        )

    note = ""
    if failed:
        note = (
            '<p class="muted small" style="margin:10px 0 0">'
            f"{failed} metric(s) failed (likely LLM rate-limited) and are shown as N/A."
            "</p>"
        )

    return (
        '<div class="ocr-card">'
        '<div class="card-title">📊 Retrieval Quality (RAGAS)</div>'
        + "".join(rows)
        + note
        + "</div>"
    )


def _format_findings_count(findings: list) -> str:
    """Build a compact summary count per severity as pill chips."""
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.severity.value] = counts.get(f.severity.value, 0) + 1

    parts = []
    for sev in ("critical", "high", "medium", "low", "info"):
        if counts.get(sev, 0) > 0:
            color = SEVERITY_COLORS.get(sev, "#666")
            parts.append(
                f'<span class="sev-chip" style="color:{color};border-color:{color}44;background:{color}11">'
                f"{sev} × {counts[sev]}</span>"
            )
    return (
        '<div class="chips-row">'
        + ("".join(parts) if parts else '<span class="muted">No issues</span>')
        + "</div>"
    )


def _format_run_summary(
    repo: str, pr_number: int, endpoint_name: str, duration: float, cost_summary: str
) -> str:
    """Build a compact run-summary card shown above the results."""
    endpoint_display = html.escape(endpoint_name or "built-in default")
    return (
        '<div class="ocr-card" style="margin-top:12px">'
        '<div class="run-summary">'
        f'<span>📦 <b>{html.escape(repo)}</b>#{pr_number}</span>'
        f'<span>🤖 <b>{endpoint_display}</b></span>'
        f'<span>⏱️ <b>{duration:.1f}s</b></span>'
        f'<span>{cost_summary}</span>'
        "</div>"
        "</div>"
    )


def _format_session_endpoints() -> str:
    """Render cards for endpoints added at runtime via the UI form."""
    eps = session_endpoints()
    if not eps:
        return (
            '<p class="muted small">No endpoints added in this session yet — '
            "fill the form above and click <b>Save & use endpoint</b>.</p>"
        )
    cards = "".join(
        f'<div class="endpoint-card">'
        f'<div class="endpoint-name">{html.escape(ep.name)}'
        f'<span class="endpoint-badge">{ep.provider}</span></div>'
        f'<div class="endpoint-meta"><code>{html.escape(ep.model)}</code>'
        f' · {html.escape(ep.base_url or "default URL")} · key {ep.masked_key}</div>'
        f'</div>'
        for ep in eps
    )
    return (
        f'<div class="card-title" style="margin:6px 0 6px">'
        f'Added in this session ({len(eps)})</div><div>{cards}</div>'
    )


def _probe_endpoint(ep) -> tuple[bool, str]:
    """Quick connectivity probe for a custom endpoint (no LLM generation)."""
    import urllib.request as _ur

    if ep.provider == "google":
        try:
            from google import genai
            client = genai.Client(api_key=ep.api_key)
            list(client.models.list())
            return True, "models.list OK"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"[:140]

    if ep.provider == "anthropic":
        base = (ep.base_url or "https://api.anthropic.com/v1").rstrip("/")
        candidates = [f"{base}/models"]
        if not base.endswith("/v1"):
            candidates.append(f"{base}/v1/models")
    else:  # openai-compatible
        base = (ep.base_url or "https://api.openai.com/v1").rstrip("/")
        candidates = [f"{base}/models"]
        if not base.endswith("/v1"):
            candidates.append(f"{base}/v1/models")

    last_err = "no response"
    for url in candidates:
        try:
            if ep.provider == "anthropic":
                req = _ur.Request(
                    url,
                    headers={
                        "x-api-key": ep.api_key,
                        "anthropic-version": "2023-06-01",
                    },
                )
            else:
                req = _ur.Request(
                    url, headers={"Authorization": f"Bearer {ep.api_key}"}
                )
            with _ur.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return True, f"HTTP 200 · {url}"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
    return False, last_err[:140]


def test_endpoints() -> str:
    """Probe every configured endpoint and render status cards."""
    eps = discover_endpoints()
    if not eps:
        return (
            '<p class="muted">No custom endpoints configured — add '
            'OCR_ENDPOINT_* secrets first.</p>'
        )

    cards = []
    for ep in eps:
        ok, detail = _probe_endpoint(ep)
        status_color = "#16a34a" if ok else "#dc2626"
        cards.append(
            f'<div class="endpoint-card">'
            f'<div class="endpoint-name">{ep.name}'
            f'<span class="endpoint-badge">{ep.provider}</span></div>'
            f'<div class="endpoint-meta"><code>{ep.model}</code>'
            f' · {ep.base_url or "default URL"} · key {ep.masked_key}</div>'
            f'<div class="endpoint-meta" style="color:{status_color};font-weight:700">'
            f'{"✅ reachable" if ok else "❌ unreachable"} — {detail}</div>'
            f'</div>'
        )
    return '<div>' + "".join(cards) + '</div>'


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _get_deploy_info() -> tuple[str, str]:
    """Return Git commit hash and deploy timestamp for the footer."""
    commit = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
    except Exception:
        pass
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return commit, ts


# ─── Gradio callbacks ───────────────────────────────────────────────────────


# _prune_db replaced with direct _cleanup_db() calls below


def run_review(repo: str, pr_number: int, endpoint_name: str = "", progress=gr.Progress()):
    """Run the review pipeline up to human-in-the-loop interrupt."""
    if not repo or "/" not in repo:
        yield [None, None, None, None, None, None], "❌ Enter a repo in `owner/name` format."
        return

    _cleanup_db()

    progress(0.1, desc="Building graph…")

    t0 = time.time()
    try:
        state, config, cost_summary, ragas_scores = _build_and_stream(
            repo.strip(), pr_number, endpoint_name,
        )
    except Exception as exc:
        log_error_to_backends(exc, context={"source": "gradio_ui", "phase": "run_review", "repo": repo, "pr_number": pr_number})
        yield ([None, None, None, None, None, None], f"❌ Review failed: {exc}")
        return

    duration = time.time() - t0

    tasks = state.tasks
    values = state.values

    if not tasks:
        yield [None, None, None, None, None, None], "ℹ️ Review completed without findings."
        return

    verdict: Verdict | None = values.get("verdict")
    findings: list = values.get("final_findings", [])

    progress(0.8, desc="Formatting results…")

    verdict_html = _format_verdict(verdict)
    ragas_html = _format_ragas_scores(ragas_scores)
    findings_html = _format_findings_table(findings)
    count_html = _format_findings_count(findings)
    # Strip non-serializable callbacks before saving config to state
    config_json = json.dumps(
        {k: v for k, v in config.items() if k != "callbacks"}
    )

    run_summary_html = _format_run_summary(
        repo.strip(), pr_number, endpoint_name, duration, cost_summary,
    )

    yield (
        [run_summary_html, verdict_html, findings_html, count_html, ragas_html, config_json],
        None,
    )


def resume_review(config_json: str, action: str, progress=gr.Progress()):
    """Resume the paused graph with approve/reject."""
    if not config_json:
        return "ℹ️ No pending review to resume. Run a review first."

    config = json.loads(config_json)
    graph = build_graph(DB_PATH)
    handler = get_langfuse_handler()
    cost_tracker = TokenCostCallback()
    callbacks = [cost_tracker]
    if handler:
        callbacks.append(handler)
    config.setdefault("callbacks", []).extend(callbacks)

    progress(0.3, desc=f"Processing {action}…")

    try:
        graph.invoke(Command(resume={"action": action}), config)
        final_state = graph.get_state(config)
        approved = final_state.values.get("human_approved")
        msg = (
            f"✅ Review **approved**! Findings posted to GitHub."
            if approved
            else "❌ Review **rejected**. No comments posted."
        )
        if cost_tracker.usage:
            msg += f" _{cost_tracker.summary()}_"
    except Exception as exc:
        log_error_to_backends(exc, context={"source": "gradio_ui", "phase": "resume_review", "action": action})
        msg = f"❌ Error: {exc}"
    finally:
        # Close the checkpointer connection and ship any queued scores.
        conn = getattr(graph, "_opencodereview_conn", None)
        if conn:
            conn.close()
        flush_langfuse()
    return msg


def run_smoke(progress=gr.Progress()):
    """Run the synthetic smoke test."""
    _cleanup_db()

    progress(0.1, desc="Building graph…")
    graph = build_graph(DB_PATH)
    thread_id = str(uuid.uuid4())
    prompt_versions = {
        "correctness": "v1",
        "security": "v1",
        "test_coverage": "v1",
        "aggregator": "v1",
    }
    cost_tracker = TokenCostCallback()
    metadata = build_run_metadata(
        source="gradio_ui", repo="demo-org/demo-repo", pr_number=1,
        session_id=thread_id,
        prompt_versions=prompt_versions,
    )
    config: dict = {
        "configurable": {"thread_id": f"smoke-web-{thread_id}"},
        "metadata": metadata,
    }

    progress(0.3, desc="Running on synthetic data…")
    with langfuse_trace(
        trace_name="opencodereview/smoke-test",
        tags=["gradio_ui", "smoke", "demo-org/demo-repo"],
        session_id=thread_id,
        metadata={"prompt_versions": prompt_versions},
    ) as handler:
        callbacks = [cost_tracker]
        if handler:
            callbacks.append(handler)
        config["callbacks"] = callbacks

        try:
            list(graph.stream(SYNTHETIC_STATE, config))
        except Exception as exc:
            log_error_to_backends(exc, context={"source": "gradio_ui", "phase": "smoke_stream"})
            logger.exception("Smoke test failed: %s", exc)
            raise

    state = graph.get_state(config)
    tasks = state.tasks

    # Close the checkpointer connection and ship any queued Langfuse scores.
    conn = getattr(graph, "_opencodereview_conn", None)
    if conn:
        conn.close()
    flush_langfuse()

    if not tasks:
        yield [None, None, None, None, None], "ℹ️ Smoke test completed without findings."
        return

    verdict: Verdict | None = state.values.get("verdict")
    findings: list = state.values.get("final_findings", [])

    progress(0.8, desc="Formatting…")
    verdict_html = _format_verdict(verdict)
    findings_html = _format_findings_table(findings)
    # Strip non-serializable callbacks before saving config to state
    config_json = json.dumps(
        {k: v for k, v in config.items() if k != "callbacks"}
    )
    cost_html = f'<p style="margin-top:8px;font-size:0.85em;color:var(--text-muted)">{cost_tracker.summary()}</p>'

    yield [verdict_html, findings_html + cost_html, config_json, verdict_html, findings_html], None





# ─── UI ──────────────────────────────────────────────────────────────────────

CSS = """
/* ── CSS Custom Properties (light mode) ───────────────────────────── */
:root {
    --bg-page: #f1f5f9;
    --bg-card: #ffffff;
    --bg-card-alt: #f8fafc;
    --bg-table-header: #f1f5f9;
    --border-color: #e2e8f0;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --text-footer: #94a3b8;
    --spinner-track: #e2e8f0;
    --accent: #6366f1;
    --accent-soft: rgba(99, 102, 241, 0.12);
    --radius: 14px;
    --shadow: 0 1px 2px rgba(15, 23, 42, 0.05), 0 10px 30px rgba(15, 23, 42, 0.06);
}

/* ── Dark Mode Overrides ─────────────────────────────────────────── */
body.dark-mode {
    --bg-page: #0b1220;
    --bg-card: #131c31;
    --bg-card-alt: #0f172a;
    --bg-table-header: #1e293b;
    --border-color: #2b3a55;
    --text-primary: #f1f5f9;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --text-footer: #64748b;
    --spinner-track: #334155;
    --accent: #818cf8;
    --accent-soft: rgba(129, 140, 248, 0.15);
    --shadow: 0 1px 2px rgba(0, 0, 0, 0.4), 0 10px 30px rgba(0, 0, 0, 0.3);
}
body.dark-mode .gradio-container {
    background: radial-gradient(1200px 500px at 20% -10%, #1e1b4b33, transparent 60%),
                radial-gradient(1000px 400px at 90% 10%, #17255433, transparent 55%),
                var(--bg-page) !important;
}
body.dark-mode .gr-box,
body.dark-mode .tabs,
body.dark-mode .tab-nav {
    background-color: var(--bg-card) !important;
    border-color: var(--border-color) !important;
}
body.dark-mode input, body.dark-mode textarea, body.dark-mode select {
    background-color: var(--bg-card-alt) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-color) !important;
}
body.dark-mode label { color: var(--text-secondary) !important; }
body.dark-mode button:not(.lg) { color: var(--text-primary) !important; }
body.dark-mode .footer { color: var(--text-footer) !important; }
body.dark-mode details summary { color: var(--text-secondary) !important; }
body.dark-mode [data-testid="block-info"] { color: var(--text-muted) !important; }

/* ── Global ──────────────────────────────────────────────────────── */
.gradio-container {
    background: radial-gradient(1200px 500px at 20% -10%, #eef2ff66, transparent 60%),
                radial-gradient(1000px 400px at 90% 10%, #e0f2fe55, transparent 55%),
                var(--bg-page) !important;
}
.gr-container { max-width: 980px; margin: 0 auto; }
.footer { text-align: center; color: var(--text-footer); font-size: 0.85em; padding: 22px 0; }
details { margin-top: 8px; }
details summary { cursor: pointer; color: var(--text-secondary); font-weight: 500; }
code { background: var(--bg-card-alt); padding: 1px 6px; border-radius: 6px; font-size: 0.92em; }
.muted { color: var(--text-muted) !important; }
.small { font-size: 0.85em; }

/* ── Hero ────────────────────────────────────────────────────────── */
.hero { display: flex; align-items: center; gap: 16px; padding: 8px 0 4px; }
.hero-icon {
    font-size: 2.2em; width: 58px; height: 58px; display: flex; align-items: center;
    justify-content: center; border-radius: 16px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35);
}
.hero-title { margin: 0; font-size: 1.6em; font-weight: 800; letter-spacing: -0.02em; color: var(--text-primary); }
.hero-sub { margin: 2px 0 0; color: var(--text-muted); font-size: 0.95em; }

/* ── Cards ───────────────────────────────────────────────────────── */
.ocr-card {
    background: var(--bg-card); border: 1px solid var(--border-color);
    border-radius: var(--radius); padding: 16px 20px; margin-top: 12px;
    box-shadow: var(--shadow);
}
.card-title { font-weight: 700; color: var(--text-primary); margin-bottom: 12px; font-size: 1.02em; }
.chips-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }

/* ── Chips / badges ──────────────────────────────────────────────── */
.key-chip {
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    font-size: 0.85em; font-weight: 600; border: 1px solid;
}
.key-chip.ok { color: #16a34a; background: #16a34a14; border-color: #16a34a44; }
.key-chip.no { color: var(--text-muted); background: var(--bg-card-alt); border-color: var(--border-color); }
.sev-chip {
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    font-size: 0.85em; font-weight: 700; border: 1px solid;
}

/* ── Endpoint cards ──────────────────────────────────────────────── */
.endpoint-card {
    border: 1px solid var(--border-color); border-radius: 10px;
    padding: 10px 14px; margin-bottom: 8px; background: var(--bg-card-alt);
}
.endpoint-name { font-weight: 700; color: var(--text-primary); display: flex; gap: 8px; align-items: center; }
.endpoint-badge {
    font-size: 0.72em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;
    padding: 2px 8px; border-radius: 999px; color: var(--accent);
    background: var(--accent-soft); border: 1px solid #6366f144;
}
.endpoint-meta { margin-top: 4px; font-size: 0.88em; color: var(--text-secondary); }

/* ── Run summary ─────────────────────────────────────────────────── */
.run-summary { display: flex; gap: 20px; flex-wrap: wrap; align-items: center; font-size: 0.92em; color: var(--text-secondary); }
.run-summary b { color: var(--text-primary); }

/* ── Verdict ─────────────────────────────────────────────────────── */
.verdict-card { border-left: 5px solid var(--accent); }
.verdict-head { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; flex-wrap: wrap; }
.verdict-icon { font-size: 1.8em; }
.verdict-reco { font-size: 1.35em; font-weight: 800; letter-spacing: 0.02em; }
.verdict-score { margin-left: auto; font-size: 1.05em; color: var(--text-secondary); }
.verdict-score b { color: var(--text-primary); font-size: 1.25em; }
.verdict-summary { margin: 10px 0 0; color: var(--text-secondary); line-height: 1.55; }
.score-track { height: 10px; background: var(--spinner-track); border-radius: 999px; overflow: hidden; margin-top: 4px; }
.score-fill { height: 100%; border-radius: 999px; transition: width 0.7s ease; }

/* ── RAGAS metrics ───────────────────────────────────────────────── */
.metric-row { display: flex; justify-content: space-between; align-items: center; margin: 8px 0 4px; }
.metric-label { font-weight: 600; color: var(--text-secondary); font-size: 0.93em; }
.metric-value { font-weight: 800; font-size: 0.98em; }
.metric-track { height: 8px; background: var(--spinner-track); border-radius: 999px; overflow: hidden; }
.metric-fill { height: 100%; border-radius: 999px; transition: width 0.6s ease; }
.tip { cursor: help; font-size: 0.85em; }

/* ── Findings table ──────────────────────────────────────────────── */
table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
thead tr { background: var(--bg-table-header); border-bottom: 2px solid var(--border-color); }
th { padding: 10px 12px; text-align: left; color: var(--text-primary); }
td { padding: 10px 12px; color: var(--text-secondary); border-bottom: 1px solid var(--border-color); vertical-align: top; }
tbody tr:hover { background: var(--bg-card-alt); }

/* ── Theme toggle ────────────────────────────────────────────────── */
#theme-toggle-btn {
    float: right; margin-top: 18px !important; background: transparent !important;
    border: 1px solid var(--border-color) !important; border-radius: 10px !important;
    padding: 4px 10px !important; font-size: 1.1em !important; min-width: 44px !important;
    transition: all 0.2s ease !important; box-shadow: none !important;
}
#theme-toggle-btn:hover { background: var(--bg-card-alt) !important; border-color: var(--text-muted) !important; }

/* ── Loading spinner ─────────────────────────────────────────────── */
@keyframes spin { to { transform: rotate(360deg); } }
.loading-spinner {
    display: flex; align-items: center; gap: 12px; padding: 24px 0; justify-content: center;
}
.loading-spinner .spinner {
    width: 24px; height: 24px; border: 3px solid var(--spinner-track);
    border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite;
}
.loading-spinner .label { color: var(--text-muted); font-size: 0.95em; }
"""

JS_RESTORE_THEME = """
() => {
    const isDark = localStorage.getItem("ocr-theme") === "dark";
    if (isDark) document.body.classList.add("dark-mode");
    const btn = document.querySelector("#theme-toggle-btn");
    if (btn) btn.textContent = isDark ? "☀️" : "🌙";
}
"""

JS_TOGGLE_THEME = """
() => {
    const isDark = document.body.classList.toggle("dark-mode");
    localStorage.setItem("ocr-theme", isDark ? "dark" : "light");
    const btn = document.querySelector("#theme-toggle-btn");
    if (btn) btn.textContent = isDark ? "☀️" : "🌙";
    return [];
}
"""

# ── Initialise observability backends ────────────────────────────────
# Wrap in try/except so a failure here doesn't prevent the UI from loading.
# This is important on Hugging Face Spaces where dependencies may have
# version incompatibilities.
try:
    _init_observability_ui()
except Exception as exc:
    logger.warning("Observability initialisation failed (non-fatal): %s", exc)

with gr.Blocks(title="OpenCodeReview") as demo:
    demo.load(js=JS_RESTORE_THEME)

    # Discover configured endpoints once — reused by the hero chip, the
    # dropdown, and the health accordion.
    _configured_endpoints = discover_endpoints()
    _endpoint_choices = [ep.name for ep in _configured_endpoints]

    with gr.Row():
        with gr.Column(scale=4):
            gr.HTML(
                '<div class="hero">'
                '<div class="hero-icon">🔍</div>'
                '<div>'
                '<h1 class="hero-title">OpenCodeReview</h1>'
                '<p class="hero-sub">AI-powered PR review with human-in-the-loop approval</p>'
                f'<div class="chips-row" style="margin-top:8px">'
                f'<span class="key-chip ok">⚡ {len(_configured_endpoints)} custom endpoint'
                f'{"s" if len(_configured_endpoints) != 1 else ""}</span>'
                '<span class="key-chip" style="color:var(--text-muted);background:var(--bg-card-alt);border-color:var(--border-color)">🔐 BYO keys</span>'
                '</div>'
                '</div>'
                '</div>'
            )
        with gr.Column(scale=1, min_width=60):
            theme_toggle = gr.Button("🌙", elem_id="theme-toggle-btn", size="sm", min_width=50, visible=True)
            theme_toggle.click(js=JS_TOGGLE_THEME)

    # ── State ────────────────────────────────────────────────────────────
    pr_state = gr.State()
    smoke_state = gr.State()

    # ── Tab: Review a PR ─────────────────────────────────────────────────
    with gr.Tab("Review a PR"):
        with gr.Row():
            repo_input = gr.Textbox(
                label="Repository",
                placeholder="owner/repo  (e.g. psf/requests)",
                scale=3,
            )
            pr_input = gr.Number(
                label="PR #",
                value=1,
                minimum=1,
                precision=0,
                scale=1,
            )

        with gr.Row():
            endpoint_input = gr.Dropdown(
                choices=[""] + _endpoint_choices,
                value="",
                label="LLM Endpoint",
                info=(
                    "Pick a saved endpoint, or add your own below. "
                    "Empty = built-in default."
                    if _endpoint_choices
                    else "No endpoints yet — add one below or set OCR_ENDPOINT_* secrets."
                ),
                scale=3,
            )

        # ── Add your own endpoint (BYO key) ─────────────────────────
        with gr.Accordion("➕ Add custom endpoint (BYO key)", open=False):
            with gr.Row():
                ep_name_input = gr.Textbox(
                    label="Endpoint name",
                    placeholder="e.g. My DeepSeek",
                    scale=1,
                )
                ep_type_input = gr.Dropdown(
                    choices=["openai", "anthropic", "google"],
                    value="openai",
                    label="Type",
                    info="openai = any OpenAI-compatible API (DeepSeek, OpenRouter, Ollama…)",
                    scale=1,
                )
            with gr.Row():
                ep_base_url_input = gr.Textbox(
                    label="API endpoint (base URL)",
                    placeholder="https://api.deepseek.com/v1   (optional for google)",
                    scale=2,
                )
                ep_model_input = gr.Textbox(
                    label="Model",
                    placeholder="deepseek-chat / claude-sonnet-4-20250514 / gemini-3.1-flash",
                    scale=2,
                )
            ep_api_key_input = gr.Textbox(
                label="API key",
                type="password",
                placeholder="sk-...  (stored in memory only, never written to disk)",
            )
            with gr.Row():
                add_endpoint_btn = gr.Button("💾 Save & use endpoint", variant="primary", size="sm")
                clear_endpoints_btn = gr.Button("🗑 Clear added endpoints", size="sm")
            endpoint_save_msg = gr.Markdown()
            session_endpoints_display = gr.HTML(value=_format_session_endpoints())

        with gr.Row():
            run_btn = gr.Button("▶ Run Review", variant="primary", size="lg", scale=2)
            cancel_btn = gr.Button("⏹ Cancel", variant="stop", size="lg", visible=False)

        loading_box = gr.HTML(
            '<div class="loading-spinner"><div class="spinner"></div>'
            '<span class="label">Running review pipeline...</span></div>',
            visible=False,
        )
        status_msg = gr.Markdown(visible=False)

        with gr.Column(visible=False) as results_panel:
            run_summary_display = gr.HTML()
            verdict_display = gr.HTML()
            count_display = gr.HTML()
            findings_display = gr.HTML()
            ragas_display = gr.HTML()

            with gr.Row():
                approve_btn = gr.Button("✅ Approve & Post", variant="primary")
                reject_btn = gr.Button("❌ Reject", variant="secondary")

            resume_msg = gr.Markdown()

        # ── Event wiring ──────────────────────────────────────────────
        def on_run_click(*args):
            """Generator that clears old state, runs review, shows results."""
            # Yield initial loading state
            yield [
                gr.update(visible=True),
                gr.update(visible=True, variant="stop"),
                gr.update(visible=False),
                gr.update(visible=False, value=""),
                gr.update(value=""),
                *[None, None, None, None],
                None,
                gr.update(visible=False),
            ]

            for result, err in run_review(*args):
                if err:
                    yield [
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=True, value=err),
                        gr.update(value=""),
                        *[None, None, None, None],
                        None,
                        gr.update(visible=False),
                    ]
                else:
                    run_summary_html, verdict_html, findings_html, count_html, ragas_html, config_json = result
                    yield [
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=False),
                        gr.update(value=run_summary_html),
                        gr.update(value=verdict_html),
                        gr.update(value=count_html),
                        gr.update(value=findings_html),
                        gr.update(value=ragas_html),
                        config_json,
                        gr.update(visible=True),
                    ]

        run_event = run_btn.click(
            fn=on_run_click,
            inputs=[repo_input, pr_input, endpoint_input],
            outputs=[
                loading_box,
                cancel_btn,
                run_btn,
                status_msg,
                run_summary_display,
                verdict_display,
                count_display,
                findings_display,
                ragas_display,
                pr_state,
                results_panel,
            ],
        )
        cancel_btn.click(fn=None, cancels=[run_event])

        def on_approve(config_json):
            msg = resume_review(config_json, "approve")
            return gr.update(value=msg)

        def on_reject(config_json):
            msg = resume_review(config_json, "reject")
            return gr.update(value=msg)

        approve_btn.click(
            fn=on_approve,
            inputs=[pr_state],
            outputs=[resume_msg],
        )
        reject_btn.click(
            fn=on_reject,
            inputs=[pr_state],
            outputs=[resume_msg],
        )

        # ── Add / clear custom endpoints ─────────────────────────────
        def on_add_endpoint(name, ptype, base_url, api_key, model):
            """Register the form values, refresh the dropdown, auto-select."""
            try:
                cfg = register_endpoint(name, ptype, api_key, base_url, model)
                choices = [""] + endpoint_choices()
                return (
                    gr.update(choices=choices, value=cfg.name),
                    f"✅ Saved **{cfg.name}** ({cfg.provider} / `{cfg.model}`) — "
                    "it's selected above. Hit **▶ Run Review** to use it. "
                    "Tip: use 🔌 Test endpoint connectivity (health tab) to check the key first.",
                    _format_session_endpoints(),
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(value=""),
                )
            except ValueError as exc:
                return (
                    gr.update(),
                    f"❌ {exc}",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

        def on_clear_endpoints():
            """Remove all UI-added endpoints and reset the dropdown."""
            clear_session_endpoints()
            return (
                gr.update(choices=[""] + endpoint_choices(), value=""),
                "🗑 Cleared — dropdown reset to the built-in default.",
                _format_session_endpoints(),
            )

        add_endpoint_btn.click(
            fn=on_add_endpoint,
            inputs=[
                ep_name_input,
                ep_type_input,
                ep_base_url_input,
                ep_api_key_input,
                ep_model_input,
            ],
            outputs=[
                endpoint_input,
                endpoint_save_msg,
                session_endpoints_display,
                ep_name_input,
                ep_base_url_input,
                ep_api_key_input,
                ep_model_input,
            ],
        )
        clear_endpoints_btn.click(
            fn=on_clear_endpoints,
            outputs=[endpoint_input, endpoint_save_msg, session_endpoints_display],
        )

    # ── Tab: Smoke Test ──────────────────────────────────────────────────
    with gr.Tab("Smoke Test"):
        gr.Markdown(
            "Run the review pipeline on **synthetic demo data** "
            "(no API keys needed, no interrupt required)."
        )
        with gr.Row():
            smoke_btn = gr.Button("▶ Run Smoke Test", variant="primary", size="lg", scale=2)
            smoke_cancel_btn = gr.Button("⏹ Cancel", variant="stop", size="lg", visible=False)

        smoke_loading = gr.HTML(
            '<div class="loading-spinner"><div class="spinner"></div>'
            '<span class="label">Running smoke test...</span></div>',
            visible=False,
        )

        with gr.Column(visible=False) as smoke_results:
            smoke_verdict = gr.HTML()
            smoke_findings = gr.HTML()
            with gr.Row():
                smoke_approve = gr.Button("✅ Approve (demo)", variant="primary")
                smoke_reject = gr.Button("❌ Reject (demo)", variant="secondary")
            smoke_resume_msg = gr.Markdown()

        smoke_status = gr.Markdown(visible=False)

        def on_smoke_click():
            # Show loading state
            yield [
                gr.update(visible=True),
                gr.update(visible=True, variant="stop"),
                gr.update(visible=False),
                gr.update(visible=False),
                None,
                gr.update(value=""),
                gr.update(value=""),
                gr.update(visible=False),
            ]
            for result, err in run_smoke():
                if err:
                    yield [
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=False),
                        None,
                        gr.update(value=""),
                        gr.update(value=""),
                        gr.update(visible=True, value=err),
                    ]
                else:
                    verdict_html, findings_html, config_json, s_verdict, s_findings = result
                    yield [
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=True),
                        config_json,
                        gr.update(value=s_verdict),
                        gr.update(value=s_findings),
                        gr.update(visible=False),
                    ]

        smoke_event = smoke_btn.click(
            fn=on_smoke_click,
            outputs=[smoke_loading, smoke_cancel_btn, smoke_btn, smoke_results, smoke_state,
                     smoke_verdict, smoke_findings, smoke_status],
        )
        smoke_cancel_btn.click(fn=None, cancels=[smoke_event])

        smoke_approve.click(
            fn=lambda c: resume_review(c, "approve"),
            inputs=[smoke_state],
            outputs=[smoke_resume_msg],
        )
        smoke_reject.click(
            fn=lambda c: resume_review(c, "reject"),
            inputs=[smoke_state],
            outputs=[smoke_resume_msg],
        )

    # ── Environment status footer ──────────────────────────────────────
    with gr.Accordion("🔑 Configured Keys & Health", open=False):
        h = HealthStatus()
        health_data = h.summary()
        gemini_ok = bool(os.environ.get("GEMINI_API_KEY", "").strip())
        groq_ok = bool(os.environ.get("GROQ_API_KEY", "").strip())
        gh_ok = bool(os.environ.get("GITHUB_TOKEN", "").strip())
        lines = [
            f"<li>GEMINI_API_KEY: {'✅ Set' if gemini_ok else '❌ Not set'}</li>",
            f"<li>GROQ_API_KEY: {'✅ Set' if groq_ok else '❌ Not set'}</li>",
            f"<li>GITHUB_TOKEN: {'✅ Set' if gh_ok else '❌ Not set'}</li>",
            f"<li>LangSmith: {health_data['LangSmith']}</li>",
            f"<li>Langfuse: {health_data['Langfuse']}</li>",
        ]
        gr.HTML("<ul>" + "".join(lines) + "</ul>")

        gr.HTML(
            '<div class="card-title" style="margin:14px 0 8px">⚡ Custom LLM Endpoints</div>'
        )
        if _configured_endpoints:
            endpoint_cards = "".join(
                f'<div class="endpoint-card">'
                f'<div class="endpoint-name">{html.escape(ep.name)}'
                f'<span class="endpoint-badge">{ep.provider}</span>'
                + (
                    '<span class="endpoint-badge" style="color:#dc2626;background:#dc262611;border-color:#dc262644">missing key</span>'
                    if not ep.api_key else ""
                )
                + '</div>'
                f'<div class="endpoint-meta"><code>{html.escape(ep.model)}</code>'
                f' · {html.escape(ep.base_url or "default URL")} · key {ep.masked_key}</div>'
                f'</div>'
                for ep in _configured_endpoints
            )
            gr.HTML(f'<div>{endpoint_cards}</div>')
        else:
            gr.HTML(
                '<p class="muted small">None configured — set '
                '<code>OCR_ENDPOINT_1_NAME</code> / <code>_TYPE</code> / '
                '<code>_API_KEY</code> / <code>_BASE_URL</code> / <code>_MODEL</code> '
                'as Space secrets to add one.</p>'
            )
        with gr.Row():
            test_endpoints_btn = gr.Button("🔌 Test endpoint connectivity", size="sm", variant="secondary")
        endpoint_test_out = gr.HTML()
        test_endpoints_btn.click(fn=test_endpoints, outputs=[endpoint_test_out])

    # ── Footer ─────────────────────────────────────────────────────────
    _commit, _ts = _get_deploy_info()
    gr.HTML(
        f'<p class="footer">'
        f'Built with LangGraph · Groq · ChromaDB · Gradio'
        f'<br><span style="opacity:0.5;font-size:0.85em;color:var(--text-footer)">'
        f'deploy: <code>{_commit}</code> · {_ts}'
        f'</span></p>'
    )


def main() -> None:
    port = int(os.environ.get("PORT", 7860))
    # Gradio 6: theme/css moved from the Blocks constructor to launch().
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        theme=gr.themes.Soft(),
        css=CSS,
    )


if __name__ == "__main__":
    main()
