from __future__ import annotations

import logging
from typing import Literal

from langgraph.types import interrupt

from state import OpenCodeReviewState

logger = logging.getLogger(__name__)


# ─── Approval result that the human returns when resuming ────────────────────

ApprovalAction = Literal["approve", "reject", "edit"]


# ─── Node ────────────────────────────────────────────────────────────────────


def human_approval_node(state: OpenCodeReviewState) -> dict:
    """Pause graph execution and wait for a human to approve or reject the
    review findings before anything is posted to GitHub.

    The value passed to ``interrupt()`` is returned to the caller (the CLI
    or frontend).  The caller must resume with ``Command(resume=...)`` whose
    value becomes the return value of ``interrupt()``.
    """
    verdict = state.verdict
    findings = state.final_findings
    repo = state.repo
    pr_number = state.pr_number

    logger.info("=== HUMAN APPROVAL REQUIRED ===")
    logger.info("Repo: %s  |  PR #%d", repo, pr_number)

    if verdict:
        logger.info(
            "Verdict: %s (score=%.1f, approved=%s)",
            verdict.recommendation, verdict.overall_score, verdict.approved,
        )
        logger.info("Summary: %s", verdict.summary)

    logger.info("Filtered findings: %d", len(findings))
    for f in findings:
        logger.info(
            "  [%.2f] %s %s:%d–%d — %s",
            f.confidence, f.severity.value, f.file_path,
            f.line_start, f.line_end, f.comment[:80],
        )

    # ── Interrupt: pause here until the human resumes ─────────────────────
    interrupt_payload = {
        "type": "human_approval",
        "repo": repo,
        "pr_number": pr_number,
        "verdict": verdict.model_dump() if verdict else None,
        "findings_count": len(findings),
        "findings": [
            {
                "file_path": f.file_path,
                "line_start": f.line_start,
                "line_end": f.line_end,
                "severity": f.severity.value,
                "category": f.category,
                "comment": f.comment,
                "confidence": f.confidence,
                "suggested_fix": f.suggested_fix,
            }
            for f in findings
        ],
    }

    # This call blocks the graph until the human provides input via resume.
    # The return value is whatever was passed to `Command(resume=...)`.
    human_input: dict = interrupt(interrupt_payload)

    action: ApprovalAction = human_input.get("action", "reject")
    logger.info("Human decision: %s", action)

    if action == "reject":
        logger.info("Review rejected by human — no comments will be posted.")

    if action == "edit":
        edited_findings_raw = human_input.get("findings")
        if edited_findings_raw:
            logger.info("Human edited %d findings before approval.", len(edited_findings_raw))

    return {
        "human_approved": action == "approve" or action == "edit",
    }
