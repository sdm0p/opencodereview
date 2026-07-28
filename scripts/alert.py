#!/usr/bin/env python3
"""Alerting monitor for OpenCodeReview.

Runs a smoke-test review against a known reference PR and compares the
verdict/score against a last-known-good baseline.  Sends a webhook
notification (Discord or Slack) if the score deviates significantly or
the run errors out.

Usage
-----
    python scripts/alert.py \\
        --baseline-file alert_baseline.json \\
        --discord-webhook "$DISCORD_WEBHOOK_URL"

    # Or with Slack
    python scripts/alert.py \\
        --baseline-file alert_baseline.json \\
        --slack-webhook "$SLACK_WEBHOOK_URL"

Cron (GitHub Actions)
---------------------
    Runs via .github/workflows/alert.yml every 6 hours.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import urllib.request
import urllib.error

from graph import build_graph
from langgraph.types import Command
from main import SYNTHETIC_STATE
from observability import TokenCostCallback, build_run_metadata, langfuse_trace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BASELINE = Path(__file__).resolve().parent / "alert_baseline.json"

# A score deviation beyond this threshold (points) triggers alert
SCORE_TOLERANCE = 2.5


# ─── Baseline management ────────────────────────────────────────────────────


def _load_baseline(path: Path) -> dict[str, Any]:
    """Load the last-known-good baseline from disk."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_baseline(path: Path, data: dict[str, Any]) -> None:
    """Save the current run result as the new baseline."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Baseline saved to %s", path)


# ─── Run smoke test ─────────────────────────────────────────────────────────


def _run_smoke_test() -> dict[str, Any]:
    """Execute one smoke-test review and return the results.

    Returns
    -------
    dict with keys: success, score, verdict, findings_count, cost, error
    """
    db_path = str(_project_root / "alert_checkpoints.db")
    graph = build_graph(db_path)
    thread_id = str(uuid.uuid4())
    cost_tracker = TokenCostCallback()
    metadata = build_run_metadata(
        source="cli", repo="demo-org/demo-repo",
        pr_number=1, session_id=thread_id,
    )
    config: dict = {
        "configurable": {"thread_id": f"alert-{thread_id}"},
        "metadata": metadata,
    }

    with langfuse_trace(
        trace_name="opencodereview/alert-check",
        tags=["cli", "alert", "demo-org/demo-repo"],
        session_id=thread_id,
    ) as handler:
        callbacks = [cost_tracker]
        if handler:
            callbacks.append(handler)
        config["callbacks"] = callbacks

        try:
            list(graph.stream(SYNTHETIC_STATE, config))
            state = graph.get_state(config)
            tasks = state.tasks
            values = state.values

            if not tasks:
                # Graph completed without findings
                verdict = values.get("verdict")
                score = verdict.overall_score if verdict else 0.0
                result = {
                    "success": True,
                    "score": score,
                    "verdict": verdict.recommendation if verdict else "none",
                    "findings_count": len(values.get("final_findings", [])),
                    "cost": cost_tracker.summary(),
                    "error": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                # Resume with reject to complete
                graph.invoke(Command(resume={"action": "reject"}), config)
                final_state = graph.get_state(config)
                verdict = final_state.values.get("verdict")
                findings = final_state.values.get("final_findings", [])
                score = verdict.overall_score if verdict else 0.0
                result = {
                    "success": True,
                    "score": score,
                    "verdict": verdict.recommendation if verdict else "none",
                    "findings_count": len(findings),
                    "cost": cost_tracker.summary(),
                    "error": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            # Cleanup
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except PermissionError:
                    pass

            return result

        except Exception as exc:
            # Cleanup on error
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except PermissionError:
                    pass
            return {
                "success": False,
                "score": 0.0,
                "verdict": "error",
                "findings_count": 0,
                "cost": "N/A",
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


# ─── Webhook notification ────────────────────────────────────────────────────


def _send_discord(webhook: str, message: str) -> None:
    """Send a message to a Discord channel via webhook."""
    payload = json.dumps({"content": message}).encode("utf-8")
    req = urllib.request.Request(
        webhook,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=15)
        logger.info("Discord notification sent")
    except urllib.error.URLError as exc:
        logger.warning("Failed to send Discord webhook: %s", exc)


def _send_slack(webhook: str, message: str) -> None:
    """Send a message to a Slack channel via webhook."""
    payload = json.dumps({"text": message}).encode("utf-8")
    req = urllib.request.Request(
        webhook,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=15)
        logger.info("Slack notification sent")
    except urllib.error.URLError as exc:
        logger.warning("Failed to send Slack webhook: %s", exc)


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run smoke-test alert check for OpenCodeReview",
    )
    parser.add_argument(
        "--baseline-file", type=Path, default=DEFAULT_BASELINE,
        help="Path to baseline JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--discord-webhook", default=os.environ.get("DISCORD_WEBHOOK", ""),
        help="Discord webhook URL (or DISCORD_WEBHOOK env var)",
    )
    parser.add_argument(
        "--slack-webhook", default=os.environ.get("SLACK_WEBHOOK", ""),
        help="Slack webhook URL (or SLACK_WEBHOOK env var)",
    )
    parser.add_argument(
        "--update-baseline", action="store_true",
        help="Update the baseline file with this run's result (no alert on deviation)",
    )
    args = parser.parse_args()

    if not args.discord_webhook and not args.slack_webhook:
        logger.info("No webhook configured — running in dry-run mode (no notification)")

    # --- Run smoke test ---
    logger.info("Running smoke test alert check…")
    result = _run_smoke_test()

    if not result["success"]:
        msg = (
            f"🚨 **OpenCodeReview Alert — Smoke test FAILED**\n"
            f"Error: {result['error']}\n"
            f"Time: {result['timestamp']}"
        )
        logger.error(msg)
        if args.discord_webhook:
            _send_discord(args.discord_webhook, msg)
        if args.slack_webhook:
            _send_slack(args.slack_webhook, msg)
        sys.exit(1)

    logger.info(
        "Smoke test OK — score=%.1f, findings=%d, verdict=%s, %s",
        result["score"], result["findings_count"],
        result["verdict"], result["cost"],
    )

    # --- Compare with baseline ---
    baseline = _load_baseline(args.baseline_file)

    if args.update_baseline or not baseline:
        _save_baseline(args.baseline_file, result)
        logger.info("Baseline set — no comparison performed")
        return

    baseline_score = baseline.get("score", 0.0)
    deviation = abs(result["score"] - baseline_score)

    if deviation > SCORE_TOLERANCE:
        msg = (
            f"⚠️ **OpenCodeReview Alert — Score deviation detected**\n"
            f"Current score: {result['score']:.1f}/10 (baseline: {baseline_score:.1f}/10)\n"
            f"Deviation: {deviation:.1f} points (tolerance: {SCORE_TOLERANCE})\n"
            f"Verdict: {result['verdict']}\n"
            f"Findings: {result['findings_count']}\n"
            f"Cost: {result['cost']}\n"
            f"Time: {result['timestamp']}"
        )
        logger.warning(msg)
        if args.discord_webhook:
            _send_discord(args.discord_webhook, msg)
        if args.slack_webhook:
            _send_slack(args.slack_webhook, msg)
    else:
        logger.info(
            "Score within tolerance (deviation=%.1f) — no alert needed",
            deviation,
        )


if __name__ == "__main__":
    main()
