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
import uuid

import gradio as gr
from langgraph.types import Command

from graph import build_graph
from main import _cleanup_db, SYNTHETIC_STATE
from state import Verdict

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


def _build_and_stream(repo: str, pr_number: int) -> tuple:
    """Build the graph, stream up to the interrupt, and return state + config."""
    graph = build_graph(DB_PATH)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": f"web-{thread_id}"}}

    initial = {"repo": repo, "pr_number": pr_number}
    list(graph.stream(initial, config))
    state = graph.get_state(config)
    return state, config


def _format_findings_table(findings: list) -> str:
    """Build an HTML table of findings with severity-coloured badges."""
    if not findings:
        return '<p style="color:#666">No issues found.</p>'

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
        '<tr style="background:#f3f4f6;border-bottom:2px solid #e5e7eb">'
        '<th style="padding:10px 12px;text-align:left">Severity</th>'
        '<th style="padding:10px 12px;text-align:left">Category</th>'
        '<th style="padding:10px 12px;text-align:left">Location</th>'
        '<th style="padding:10px 12px;text-align:left">Comment</th>'
        '<th style="padding:10px 12px;text-align:left">Confidence</th>'
        "</tr>"
        "</thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
        "</div>"
    )


def _format_verdict(verdict: Verdict | None) -> str:
    """Build a verdict summary card."""
    if verdict is None:
        return '<p style="color:#666">No verdict produced.</p>'

    icon = RECOMMENDATION_ICONS.get(verdict.recommendation, "📋")
    color = (
        "#16a34a" if verdict.recommendation == "approve"
        else "#dc2626" if verdict.recommendation == "block"
        else "#ea580c"
    )

    return (
        f'<div style="border:1px solid #e5e7eb;border-radius:12px;'
        f'padding:16px 20px;background:#f9fafb">'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">'
        f'<span style="font-size:2em">{icon}</span>'
        f'<span style="font-size:1.4em;font-weight:700;color:{color}">'
        f"{verdict.recommendation.upper()}</span>"
        f'<span style="margin-left:auto;font-size:1.2em;font-weight:600">'
        f"Score: {verdict.overall_score}/10</span>"
        f"</div>"
        f"<p style='margin:0;color:#4b5563'>{verdict.summary}</p>"
        f"</div>"
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
    return "  |  ".join(parts) if parts else "No issues"


# ─── Gradio callbacks ───────────────────────────────────────────────────────


# _prune_db replaced with direct _cleanup_db() calls below


def run_review(repo: str, pr_number: int, progress=gr.Progress()):
    """Run the review pipeline up to human-in-the-loop interrupt."""
    if not repo or "/" not in repo:
        yield [None, None, None, None], "❌ Enter a repo in `owner/name` format."
        return

    _cleanup_db()

    progress(0.1, desc="Building graph…")

    try:
        state, config = _build_and_stream(repo.strip(), pr_number)
    except Exception as exc:
        logger.exception("Review failed")
        yield [None, None, None, None], f"❌ Review failed: {exc}"
        return

    tasks = state.tasks
    values = state.values

    if not tasks:
        yield [None, None, None, None], "ℹ️ Review completed without findings."
        return

    verdict: Verdict | None = values.get("verdict")
    findings: list = values.get("final_findings", [])

    progress(0.8, desc="Formatting results…")

    verdict_html = _format_verdict(verdict)
    findings_html = _format_findings_table(findings)
    count_html = _format_findings_count(findings)
    config_json = json.dumps(config)

    yield (
        [verdict_html, findings_html, count_html, config_json],
        None,
    )


def resume_review(config_json: str, action: str, progress=gr.Progress()):
    """Resume the paused graph with approve/reject."""
    if not config_json:
        return "ℹ️ No pending review to resume. Run a review first."

    config = json.loads(config_json)
    graph = build_graph(DB_PATH)

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
        return msg
    except Exception as exc:
        logger.exception("Resume failed")
        return f"❌ Error: {exc}"


def run_smoke(progress=gr.Progress()):
    """Run the synthetic smoke test."""
    _cleanup_db()

    progress(0.1, desc="Building graph…")
    graph = build_graph(DB_PATH)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": f"smoke-web-{thread_id}"}}

    progress(0.3, desc="Running on synthetic data…")
    list(graph.stream(SYNTHETIC_STATE, config))
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

    yield [verdict_html, findings_html, json.dumps(config), verdict_html, findings_html], None





# ─── UI ──────────────────────────────────────────────────────────────────────

CSS = """
.gr-container { max-width: 960px; margin: 0 auto; }
h1 { display: flex; align-items: center; gap: 10px; }
.footer { text-align: center; color: #9ca3af; font-size: 0.85em; padding: 20px 0; }
details { margin-top: 8px; }
details summary { cursor: pointer; color: #4b5563; font-weight: 500; }
"""

with gr.Blocks(css=CSS, title="OpenCodeReview", theme=gr.themes.Soft()) as demo:
    gr.HTML(
        '<h1>🔍 OpenCodeReview</h1>'
        '<p style="color:#6b7280;margin-top:-8px">'
        "AI-powered PR review with human-in-the-loop approval</p>"
    )

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
        run_btn = gr.Button("▶ Run Review", variant="primary", size="lg")

        status_msg = gr.Markdown(visible=False)

        with gr.Column(visible=False) as results_panel:
            verdict_display = gr.HTML()
            count_display = gr.HTML()
            findings_display = gr.HTML()

            with gr.Row():
                approve_btn = gr.Button("✅ Approve & Post", variant="primary")
                reject_btn = gr.Button("❌ Reject", variant="secondary")

            resume_msg = gr.Markdown()

        # ── Event wiring ──────────────────────────────────────────────
        def on_run_click(*args):
            """Generator that clears old state, runs review, shows results."""
            for result, err in run_review(*args):
                if err:
                    yield [
                        *[None, None, None, None],
                        gr.update(visible=True, value=err),
                        gr.update(visible=False),
                    ]
                else:
                    verdict_html, findings_html, count_html, config_json = result
                    yield [
                        gr.update(value=verdict_html),
                        gr.update(value=count_html),
                        gr.update(value=findings_html),
                        config_json,
                        gr.update(visible=False),
                        gr.update(visible=True),
                    ]

        run_event = run_btn.click(
            fn=on_run_click,
            inputs=[repo_input, pr_input],
            outputs=[
                verdict_display,
                count_display,
                findings_display,
                pr_state,
                status_msg,
                results_panel,
            ],
        )

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
        smoke_btn = gr.Button("▶ Run Smoke Test", variant="primary", size="lg")

        with gr.Column(visible=False) as smoke_results:
            smoke_verdict = gr.HTML()
            smoke_findings = gr.HTML()
            with gr.Row():
                smoke_approve = gr.Button("✅ Approve (demo)", variant="primary")
                smoke_reject = gr.Button("❌ Reject (demo)", variant="secondary")
            smoke_resume_msg = gr.Markdown()

        def on_smoke_click():
            for result, err in run_smoke():
                if err:
                    yield [
                        gr.update(visible=True, value=err),
                        gr.update(visible=False),
                        None,
                        gr.update(value=""),
                        gr.update(value=""),
                    ]
                else:
                    verdict_html, findings_html, config_json, s_verdict, s_findings = result
                    yield [
                        gr.update(visible=False),
                        gr.update(visible=True),
                        config_json,
                        gr.update(value=s_verdict),
                        gr.update(value=s_findings),
                    ]

        smoke_btn.click(
            fn=on_smoke_click,
            outputs=[status_msg, smoke_results, smoke_state,
                     smoke_verdict, smoke_findings],
        )

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
    with gr.Accordion("🔑 Configured Keys", open=False):
        groq_ok = bool(os.environ.get("GROQ_API_KEY", "").strip())
        gh_ok = bool(os.environ.get("GITHUB_TOKEN", "").strip())
        groq_status = "✅ Set" if groq_ok else "❌ Not set — reviewers will be skipped"
        gh_status = "✅ Set" if gh_ok else "❌ Not set — cannot fetch real PRs"
        gr.HTML(
            f"<ul>"
            f"<li>GROQ_API_KEY: {groq_status}</li>"
            f"<li>GITHUB_TOKEN: {gh_status}</li>"
            f"</ul>"
        )

    # ── Footer ─────────────────────────────────────────────────────────
    gr.HTML(
        '<p class="footer">Built with LangGraph · Groq · ChromaDB · Gradio</p>'
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
