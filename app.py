#!/usr/bin/env python3
"""OpenCodeReview — Gradio Web UI for Hugging Face Spaces.

Run locally:
    python app.py

On Hugging Face Spaces:
    The Dockerfile now defaults to running this module.
    Set GROQ_API_KEY (and optionally GITHUB_TOKEN) as Space secrets.
"""

from __future__ import annotations

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


def _build_and_stream(repo: str, pr_number: int) -> tuple:
    """Build the graph, stream up to the interrupt, and return state + config."""
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
        ragas_scores: dict[str, float] = {}
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
                )

                # Log RAGAS scores with the explicitly captured trace_id
                # to avoid contextvar loss during the long computation.
                # handler=handler is also passed as fallback resolution.
                for metric_name, value in ragas_scores.items():
                    log_langfuse_score(
                        name=f"ragas_{metric_name}",
                        value=value,
                        comment=f"{repo}#{pr_number} — {metric_name}",
                        trace_id=run_trace_id,
                        handler=handler,
                    )

                logger.info(
                    "RAGAS scores logged to Langfuse: %s",
                    " | ".join(f"{k}={v:.4f}" for k, v in ragas_scores.items()),
                )
            except ImportError:
                logger.info("RAGAS not installed — skipping RAGAS scoring (pip install ragas)")
            except Exception as exc:
                logger.warning("RAGAS scoring failed: %s", exc)

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
    """Build a verdict summary card."""
    if verdict is None:
        return '<p style="color:var(--text-muted)">No verdict produced.</p>'

    icon = RECOMMENDATION_ICONS.get(verdict.recommendation, "📋")
    color = (
        "#16a34a" if verdict.recommendation == "approve"
        else "#dc2626" if verdict.recommendation == "block"
        else "#ea580c"
    )

    return (
        f'<div style="border:1px solid var(--border-color);border-radius:12px;'
        f'padding:16px 20px;background:var(--bg-card)">'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">'
        f'<span style="font-size:2em">{icon}</span>'
        f'<span style="font-size:1.4em;font-weight:700;color:{color}">'
        f"{verdict.recommendation.upper()}</span>"
        f'<span style="margin-left:auto;font-size:1.2em;font-weight:600;'
        f'color:var(--text-primary)">'
        f"Score: {verdict.overall_score}/10</span>"
        f"</div>"
        f"<p style='margin:0;color:var(--text-secondary)'>{verdict.summary}</p>"
        f"</div>"
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


def _format_ragas_scores(scores: dict[str, float]) -> str:
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

    bars: list[str] = []
    for key, value in scores.items():
        label = RAGAS_METRIC_LABELS.get(key, key.replace("_", " ").title())
        tooltip = RAGAS_METRIC_TOOLTIPS.get(key, "")
        color = _score_color(value)
        pct = int(value * 100)
        bars.append(
            f'<div style="margin-bottom:12px">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
            f'<span style="font-weight:500;color:var(--text-secondary)">'
            f'{label}'
            + (f'<span style="margin-left:4px;cursor:help;font-size:0.85em" '
               f'title="{tooltip}">ℹ️</span>' if tooltip else '')
            + f'</span>'
            f'<span style="font-weight:700;color:{color}">{value:.3f}</span>'
            f'</div>'
            f'<div style="height:8px;background:var(--spinner-track);border-radius:4px;overflow:hidden">'
            f'<div style="height:100%;width:{pct}%;background:{color};'
            f'border-radius:4px;transition:width 0.6s ease"></div>'
            f'</div>'
            f'</div>'
        )

    return (
        f'<div style="border:1px solid var(--border-color);border-radius:12px;'
        f'padding:16px 20px;margin-top:12px;background:var(--bg-card)">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:16px">'
        f'<span style="font-size:1.1em">📊</span>'
        f'<span style="font-weight:600;color:var(--text-primary)">'
        f'Retrieval Quality (RAGAS)</span>'
        f'</div>'
        + "".join(bars)
        + f'</div>'
    )


def _format_findings_count(findings: list) -> str:
    """Build a compact summary count per severity."""
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.severity.value] = counts.get(f.severity.value, 0) + 1

    parts = []
    for sev in ("critical", "high", "medium", "low", "info"):
        if counts.get(sev, 0) > 0:
            color = SEVERITY_COLORS.get(sev, "#666")
            parts.append(
                f'<span style="color:{color};font-weight:600">'
                f"{sev}: {counts[sev]}</span>"
            )
    return "  |  ".join(parts) if parts else '<span style="color:var(--text-muted)">No issues</span>'


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


def run_review(repo: str, pr_number: int, progress=gr.Progress()):
    """Run the review pipeline up to human-in-the-loop interrupt."""
    if not repo or "/" not in repo:
        yield [None, None, None, None, None], "❌ Enter a repo in `owner/name` format."
        return

    _cleanup_db()

    progress(0.1, desc="Building graph…")

    try:
        state, config, cost_summary, ragas_scores = _build_and_stream(repo.strip(), pr_number)
    except Exception as exc:
        log_error_to_backends(exc, context={"source": "gradio_ui", "phase": "run_review", "repo": repo, "pr_number": pr_number})
        yield [
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True, value=f"❌ Review failed: {exc}"),
            None, None, None, None, None,
            gr.update(visible=False),
        ]
        return

    tasks = state.tasks
    values = state.values

    if not tasks:
        yield [None, None, None, None, None], "ℹ️ Review completed without findings."
        return

    verdict: Verdict | None = values.get("verdict")
    findings: list = values.get("final_findings", [])

    progress(0.8, desc="Formatting results…")

    verdict_html = _format_verdict(verdict)
    ragas_html = _format_ragas_scores(ragas_scores)
    findings_html = _format_findings_table(findings)
    count_html = _format_findings_count(findings)
    # Append cost info to count display
    count_html += f'<p style="margin-top:8px;font-size:0.85em;color:var(--text-muted)">{cost_summary}</p>'
    # Strip non-serializable callbacks before saving config to state
    config_json = json.dumps(
        {k: v for k, v in config.items() if k != "callbacks"}
    )

    yield (
        [verdict_html, findings_html, count_html, ragas_html, config_json],
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
        return msg
    except Exception as exc:
        log_error_to_backends(exc, context={"source": "gradio_ui", "phase": "resume_review", "action": action})
        return f"❌ Error: {exc}"


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
    --bg-card: #f9fafb;
    --bg-table-header: #f3f4f6;
    --border-color: #e5e7eb;
    --text-primary: #111827;
    --text-secondary: #4b5563;
    --text-muted: #6b7280;
    --text-footer: #9ca3af;
    --spinner-track: #e5e7eb;
}

/* ── Dark Mode Overrides ─────────────────────────────────────────── */
body.dark-mode {
    --bg-card: #1f2937;
    --bg-table-header: #374151;
    --border-color: #4b5563;
    --text-primary: #f3f4f6;
    --text-secondary: #d1d5db;
    --text-muted: #9ca3af;
    --text-footer: #6b7280;
    --spinner-track: #4b5563;
}
body.dark-mode .gradio-container {
    background-color: #111827 !important;
}
body.dark-mode .gr-box,
body.dark-mode .tabs,
body.dark-mode .tab-nav {
    background-color: #1f2937 !important;
    border-color: #4b5563 !important;
}
body.dark-mode input, body.dark-mode textarea {
    background-color: #374151 !important;
    color: #f3f4f6 !important;
    border-color: #4b5563 !important;
}
body.dark-mode label {
    color: #d1d5db !important;
}
body.dark-mode button:not(.lg) {
    color: #f3f4f6 !important;
}
body.dark-mode .footer {
    color: var(--text-footer) !important;
}
body.dark-mode details summary {
    color: #d1d5db !important;
}
body.dark-mode [data-testid="block-info"] {
    color: #d1d5db !important;
}

/* ── Global Styles ────────────────────────────────────────────────── */
.gr-container { max-width: 960px; margin: 0 auto; }
h1 { display: flex; align-items: center; gap: 10px; }
.footer { text-align: center; color: var(--text-footer); font-size: 0.85em; padding: 20px 0; }
details { margin-top: 8px; }
details summary { cursor: pointer; color: #4b5563; font-weight: 500; }

/* ── Theme toggle ─────────────────────────────────────────────────── */
#theme-toggle-btn {
    float: right;
    margin-top: 24px !important;
    background: transparent !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    padding: 4px 10px !important;
    font-size: 1.1em !important;
    min-width: 44px !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}
#theme-toggle-btn:hover {
    background: #f3f4f6 !important;
    border-color: #9ca3af !important;
}
body.dark-mode #theme-toggle-btn {
    border-color: #4b5563 !important;
    color: #f3f4f6 !important;
}
body.dark-mode #theme-toggle-btn:hover {
    background: #374151 !important;
}

/* ── Loading spinner ──────────────────────────────────────────────── */
@keyframes spin { to { transform: rotate(360deg); } }
.loading-spinner {
    display: flex; align-items: center; gap: 12px;
    padding: 24px 0; justify-content: center;
}
.loading-spinner .spinner {
    width: 24px; height: 24px; border: 3px solid var(--spinner-track);
    border-top-color: #6366f1; border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
.loading-spinner .label {
    color: var(--text-muted); font-size: 0.95em;
}
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

with gr.Blocks(css=CSS, title="OpenCodeReview", theme=gr.themes.Soft()) as demo:
    demo.load(js=JS_RESTORE_THEME)

    with gr.Row():
        with gr.Column(scale=4):
            gr.HTML(
                '<h1>🔍 OpenCodeReview</h1>'
                '<p style="color:var(--text-muted);margin-top:-8px">'
                'AI-powered PR review with human-in-the-loop approval</p>'
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
            run_btn = gr.Button("▶ Run Review", variant="primary", size="lg", scale=2)
            cancel_btn = gr.Button("⏹ Cancel", variant="stop", size="lg", visible=False)

        loading_box = gr.HTML(
            '<div class="loading-spinner"><div class="spinner"></div>'
            '<span class="label">Running review pipeline...</span></div>',
            visible=False,
        )
        status_msg = gr.Markdown(visible=False)

        with gr.Column(visible=False) as results_panel:
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
                *[None, None, None, None, None],
                gr.update(visible=False),
            ]

            for result, err in run_review(*args):
                if err:
                    yield [
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=True, value=err),
                        *[None, None, None, None, None],
                        gr.update(visible=False),
                    ]
                else:
                    verdict_html, findings_html, count_html, ragas_html, config_json = result
                    yield [
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=False),
                        gr.update(value=verdict_html),
                        gr.update(value=count_html),
                        gr.update(value=findings_html),
                        gr.update(value=ragas_html),
                        config_json,
                        gr.update(visible=True),
                    ]

        run_event = run_btn.click(
            fn=on_run_click,
            inputs=[repo_input, pr_input],
            outputs=[
                loading_box,
                cancel_btn,
                run_btn,
                status_msg,
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
        ls_ok = h.langsmith
        lf_ok = h.langfuse
        lines = [
            f"<li>GEMINI_API_KEY: {'✅ Set' if gemini_ok else '❌ Not set'}</li>",
            f"<li>GROQ_API_KEY: {'✅ Set' if groq_ok else '❌ Not set'}</li>",
            f"<li>GITHUB_TOKEN: {'✅ Set' if gh_ok else '❌ Not set'}</li>",
            f"<li>LangSmith: {health_data['LangSmith']}</li>",
            f"<li>Langfuse: {health_data['Langfuse']}</li>",
        ]
        gr.HTML(f"<ul>" + "".join(lines) + "</ul>")

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
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )


if __name__ == "__main__":
    main()
