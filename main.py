#!/usr/bin/env python3
"""OpenCodeReview — AI-powered PR review with human-in-the-loop approval.

Usage
-----
    opencodereview auth login
    opencodereview auth status
    opencodereview auth logout
    opencodereview review [--smoke]
    opencodereview --help
"""

from __future__ import annotations

import logging
import os
import sys
import uuid

import click
from langgraph.types import Command

from auth import auth as auth_group
from config import config as config_group
from graph import build_graph
from observability import (
    build_run_metadata,
    enable_langsmith,
    get_langfuse_handler,
    is_langsmith_enabled,
    estimate_groq_cost,
    format_cost,
    log_error_to_backends,
    TokenCostCallback,
)
from state import ChangedFile, ContextChunk, Finding, Severity, Verdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = "checkpoints.db"


# ─── Observability initialisation ──────────────────────────────────────────


def _init_observability() -> None:
    """Initialise tracing backends from env vars / keyring.

    Called once at the start of each ``review`` command.
    """
    import keyring as _kr
    from config import SERVICE_NAME as _svc

    # LangSmith: favour env var (check modern LANGSMITH_API_KEY first, then legacy)
    ls_key = (
        os.environ.get("LANGSMITH_API_KEY")
        or os.environ.get("LANGCHAIN_API_KEY")
        or _kr.get_password(_svc, "langsmith_api_key")
    )
    if ls_key and not is_langsmith_enabled():
        ls_project = (
            os.environ.get("LANGSMITH_PROJECT")
            or os.environ.get("LANGCHAIN_PROJECT")
            or _kr.get_password(_svc, "langsmith_project")
            or "opencodereview"
        )
        enable_langsmith(ls_key, ls_project)

    # Langfuse: env var overrides keyring
    for env_key, ring_key in [
        ("LANGFUSE_PUBLIC_KEY", "langfuse_public_key"),
        ("LANGFUSE_SECRET_KEY", "langfuse_secret_key"),
        ("LANGFUSE_HOST", "langfuse_host"),
    ]:
        if not os.environ.get(env_key):
            val = _kr.get_password(_svc, ring_key)
            if val:
                os.environ[env_key] = val

    if is_langsmith_enabled():
        logger.info("Observability: LangSmith enabled")
    if get_langfuse_handler():
        logger.info("Observability: Langfuse enabled")


# ─── Synthetic payload for offline testing ──────────────────────────────────

SYNTHETIC_STATE = {
    "repo": "demo-org/demo-repo",
    "pr_number": 1,
    "diff": (
        "diff --git a/src/auth.py b/src/auth.py\n"
        "index abc123..def456 100644\n"
        "--- a/src/auth.py\n"
        "+++ b/src/auth.py\n"
        "@@ -42,7 +42,9 @@ def login(username, password):\n"
        "     user = db.get_user(username)\n"
        "-    if user.password == password:\n"
        "+    if constant_time_compare(password, user.password):\n"
        "         return create_session(user)\n"
        "     return None\n"
    ),
    "changed_files": [
        ChangedFile(
            path="src/auth.py",
            status="modified",
            content="def login(username, password):\n"
                    "    user = db.get_user(username)\n"
                    "    if constant_time_compare(password, user.password):\n"
                    "        return create_session(user)\n"
                    "    return None\n",
            diff_hunk="@@ -42,7 +42,9 @@ def login(username, password): ...",
        ),
    ],
    "context_chunks": [
        ContextChunk(
            source="codebase", file_path="src/db.py",
            content="def get_user(username): ...",
            relevance_score=0.92,
        ),
    ],
    "findings": [
        Finding(
            file_path="src/auth.py", line_start=42, line_end=50,
            severity=Severity.CRITICAL, category="security",
            comment="Use constant-time comparison for passwords.",
            confidence=0.95,
        ),
    ],
    "verdict": Verdict(
        summary="Critical security vulnerability found.",
        approved=False, overall_score=3.5, critical_count=1,
        high_count=0, recommendation="block",
    ),
}


# ─── Review Command ─────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--smoke", is_flag=True,
    help="Non-interactive smoke test (skips HITL prompt).",
)
@click.option(
    "--repo",
    default="demo-org/demo-repo",
    show_default=True,
    help="GitHub repository (owner/name).",
)
@click.option(
    "--pr", "pr_number",
    default=1,
    show_default=True,
    help="Pull request number.",
)
def review(repo: str, pr_number: int, smoke: bool) -> None:
    """Run a PR review with human-in-the-loop approval.

    By default this runs with a synthetic PR payload for demonstration.
    """
    # ── Enable LangSmith if key is available ─────────────────────────
    _init_observability()

    graph = build_graph(DB_PATH)

    if smoke:
        _test_offline(graph)
    else:
        _run_with_hitl(graph, repo, pr_number)

    _cleanup_db()


def _run_with_hitl(graph, repo: str, pr_number: int) -> None:
    """Run the graph on a real PR — fetches data from GitHub, pauses for
    human approval, then posts findings back as PR comments."""
    thread_id = str(uuid.uuid4())
    handler = get_langfuse_handler()
    cost_tracker = TokenCostCallback()
    metadata = build_run_metadata(
        source="cli", repo=repo, pr_number=pr_number, session_id=thread_id,
    )
    config: dict = {
        "configurable": {"thread_id": f"hitl-{thread_id}"},
        "metadata": metadata,
    }
    # Build callbacks list with cost tracker + optional Langfuse
    callbacks = [cost_tracker]
    if handler:
        callbacks.append(handler)
    config["callbacks"] = callbacks

    # Use synthetic demo data by default; only fetch real PRs when the
    # user explicitly provides a custom repo or PR number.
    is_default = (repo == "demo-org/demo-repo" and pr_number == 1)
    if is_default:
        logger.info(
            "Using synthetic demo data — pass --repo and --pr to "
            "review a real PR from GitHub.\n"
        )
        initial_state = SYNTHETIC_STATE
    else:
        logger.info(
            "Starting review of %s#%d … fetching PR data from GitHub.\n",
            repo, pr_number,
        )
        initial_state = {
            "repo": repo,
            "pr_number": pr_number,
        }

    events = list(graph.stream(initial_state, config))

    state = graph.get_state(config)
    tasks = state.tasks

    if tasks:
        logger.info("\n" + "=" * 60)
        logger.info("  REVIEW RESULTS — HUMAN APPROVAL REQUIRED")
        logger.info("=" * 60)

        if state.values.get("verdict"):
            v = state.values["verdict"]
            logger.info("  Verdict: %s", v.recommendation.upper())
            logger.info("  Score:   %.1f/10", v.overall_score)
            logger.info("  Summary: %s", v.summary)

        final_findings = state.values.get("final_findings", [])
        logger.info("  Findings: %d", len(final_findings))
        for i, f in enumerate(final_findings, 1):
            logger.info(
                "    %d. [%.0f%%] %s:%d–%d (%s)",
                i, f.confidence * 100, f.file_path,
                f.line_start, f.line_end, f.severity.value,
            )
            logger.info("       %s", f.comment[:120])

        # Show cost
        logger.info("  %s", cost_tracker.summary())

        logger.info("=" * 60)

        click.echo()
        click.echo("-" * 50)
        click.echo("  HUMAN-IN-THE-LOOP: Approve this review for posting?")
        click.echo("-" * 50)
        click.echo("  [a] Approve — post findings as GitHub comments")
        click.echo("  [r] Reject  — discard results, post nothing")
        click.echo("  [q] Quit    — leave graph paused")
        click.echo()

        choice = click.prompt("  Your choice", type=click.Choice(["a", "r", "q"], case_sensitive=False))

        if choice == "a":
            logger.info("  → Approved! Resuming graph to post results …")
            graph.invoke(Command(resume={"action": "approve"}), config)
        elif choice == "r":
            logger.info("  → Rejected. Resuming graph without posting …")
            graph.invoke(Command(resume={"action": "reject"}), config)
        else:
            logger.info("  → Exiting. Graph remains paused — resume later")
            logger.info(
                "    Resume with: graph.invoke(Command(resume=...), %s)",
                config,
            )
            return

        final_state = graph.get_state(config)
        final_values = final_state.values

        logger.info("\n" + "=" * 60)
        logger.info("  FINAL STATE")
        logger.info("=" * 60)
        logger.info("  Human approved: %s", final_values.get("human_approved"))
        if final_values.get("verdict"):
            logger.info("  Verdict: %s", final_values["verdict"].recommendation)
        logger.info("  Final findings: %d", len(final_values.get("final_findings", [])))
        logger.info("  %s", cost_tracker.summary())
        logger.info("=" * 60)
    else:
        logger.info("Graph completed without interrupt (no findings to review?)")
        logger.info("  %s", cost_tracker.summary())


def _test_offline(graph) -> None:
    """Non-interactive test that the graph compiles and runs up to interrupt."""
    thread_id = str(uuid.uuid4())
    handler = get_langfuse_handler()
    cost_tracker = TokenCostCallback()
    metadata = build_run_metadata(
        source="cli", repo="demo-org/demo-repo", pr_number=1, session_id=thread_id,
    )
    config: dict = {
        "configurable": {"thread_id": f"smoke-{thread_id}"},
        "metadata": metadata,
    }
    callbacks = [cost_tracker]
    if handler:
        callbacks.append(handler)
    config["callbacks"] = callbacks

    list(graph.stream(SYNTHETIC_STATE, config))
    state = graph.get_state(config)
    tasks = state.tasks
    values = state.values

    verdict = values.get("verdict")
    final_findings = values.get("final_findings", [])

    print(f"Smoke test: paused=True, tasks={len(tasks)}, "
          f"findings={len(final_findings)}, "
          f"verdict={verdict.recommendation if verdict else 'None'}")
    print(cost_tracker.summary())

    assert len(tasks) > 0, "Graph did not pause at interrupt"
    assert verdict is not None, "No verdict produced"
    assert len(final_findings) >= 0, "Missing final_findings"

    graph.invoke(Command(resume={"action": "reject"}), config)
    print("Smoke test: resume OK, graph completed")


def _cleanup_db() -> None:
    """Remove the checkpoint database after a run."""
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            logger.info("Cleaned up %s", DB_PATH)
        except PermissionError:
            pass


# ─── Root CLI ───────────────────────────────────────────────────────────────


# ─── Doctor Command ─────────────────────────────────────────────────────────


@click.command()
def doctor() -> None:
    """Run system health checks and report status."""
    from config import _mask_key
    from observability import (
        HealthStatus,
        check_groq_connectivity,
        check_langfuse_connectivity,
    )

    click.echo()
    click.echo("=" * 50)
    click.echo("  OpenCodeReview — Health Check")
    click.echo("=" * 50)
    click.echo()

    # ── Environment ────────────────────────────────────────────────────
    click.echo("📦 Environment")
    click.echo(f"  Python:  {sys.version.split()[0]}")
    click.echo()

    # ── API keys ───────────────────────────────────────────────────────
    click.echo("🔑 API Keys")
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    click.echo(f"  GROQ_API_KEY:  {'✅ Set' if groq_key else '❌ Not set'}")
    gh_token = os.environ.get("GITHUB_TOKEN", "").strip()
    click.echo(f"  GITHUB_TOKEN:  {'✅ Set' if gh_token else '❌ Not set'}")
    click.echo()

    # ── Observability ──────────────────────────────────────────────────
    click.echo("🔭 Observability")
    h = HealthStatus()
    for comp, status in h.summary().items():
        click.echo(f"  {comp}: {status}")
    click.echo()

    # ── Connectivity tests ─────────────────────────────────────────────
    click.echo("🌐 Connectivity")

    # Groq
    groq_ok, groq_msg = check_groq_connectivity()
    click.echo(f"  Groq API:    {'✅' if groq_ok else '❌'} {groq_msg}")

    # Langfuse (only if configured)
    lf_ok, lf_msg = check_langfuse_connectivity()
    click.echo(f"  Langfuse:    {'✅' if lf_ok else '⏭️'} {lf_msg}")

    click.echo()
    click.echo("=" * 50)


# ─── Root CLI ───────────────────────────────────────────────────────────────


@click.group()
def cli() -> None:
    """OpenCodeReview — AI-powered PR review with human-in-the-loop approval."""


cli.add_command(review)
cli.add_command(doctor)
cli.add_command(auth_group)
cli.add_command(config_group)


def main() -> None:
    cli(auto_envvar_prefix="OCR")


if __name__ == "__main__":
    main()
