from __future__ import annotations

import logging
import os

import requests

from state import OpenCodeReviewState

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"


def _github_headers() -> dict[str, str]:
    headers: dict[str, str] = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "OpenCodeReview/1.0",
    }
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _post_comment(owner: str, repo: str, pr_number: int, body: str) -> bool:
    """Post a single PR review comment.  Returns True on success."""
    url = (
        f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues/{pr_number}/comments"
    )
    try:
        resp = requests.post(
            url,
            headers=_github_headers(),
            json={"body": body},
            timeout=30,
        )
        if resp.status_code in (201, 200):
            logger.info("Comment posted to %s/%s#%d", owner, repo, pr_number)
            return True
        else:
            logger.warning(
                "Failed to post comment: HTTP %s — %s",
                resp.status_code, resp.text[:200],
            )
            return False
    except requests.RequestException as exc:
        logger.warning("Failed to post comment: %s", exc)
        return False


def _build_comment_body(finding: dict) -> str:
    """Format a single finding as a Markdown PR comment."""
    sev_emoji = {
        "critical": "🚨",
        "high": "⚠️",
        "medium": "📌",
        "low": "💡",
        "info": "ℹ️",
    }
    emoji = sev_emoji.get(finding["severity"], "🔍")
    fix_section = ""
    if finding.get("suggested_fix"):
        fix_section = f"\n\n**Suggested fix:**\n```\n{finding['suggested_fix']}\n```"

    return (
        f"{emoji} **{finding['severity'].upper()}** — "
        f"*{finding['category']}*  \n"
        f"**File:** `{finding['file_path']}`  \n"
        f"**Lines:** {finding['line_start']}–{finding['line_end']}  \n"
        f"**Confidence:** {finding['confidence']:.0%}  \n\n"
        f"{finding['comment']}{fix_section}"
    )


def _build_verdict_body(verdict_dict: dict) -> str:
    """Format the overall verdict as a Markdown PR comment."""
    icon = {"approve": "✅", "request_changes": "🔄", "block": "🚫", "comment": "💬"}
    emoji = icon.get(verdict_dict.get("recommendation", ""), "📋")
    summary = verdict_dict.get("summary", "")
    score = verdict_dict.get("overall_score", 0)
    return (
        f"## OpenCodeReview — {emoji} {verdict_dict['recommendation'].upper()}\n\n"
        f"**Score:** {score}/10  \n"
        f"**Summary:** {summary}\n"
    )


def post_results_node(state: OpenCodeReviewState) -> dict:
    """Post the approved findings and verdict as GitHub PR comments.

    Only runs if the human approved (``state.human_approved is True``).
    Requires ``GITHUB_TOKEN`` to post.
    """
    if not state.human_approved:
        logger.info("Not approved — skipping PR comment posting")
        return {}

    # Parse owner/repo
    parts = state.repo.split("/")
    if len(parts) != 2:
        logger.warning("Cannot post results: invalid repo '%s'", state.repo)
        return {}
    owner, repo_name = parts
    pr_number = state.pr_number

    gh_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not gh_token:
        logger.warning(
            "GITHUB_TOKEN not set — cannot post to %s/%s#%d. "
            "Results logged instead.",
            owner, repo_name, pr_number,
        )
        # Log what we would have posted
        if state.verdict:
            logger.info("[WOULD POST] Verdict: %s", state.verdict.summary)
        for f in state.final_findings:
            logger.info(
                "[WOULD POST] %s:%d–%d %s — %s",
                f.file_path, f.line_start, f.line_end,
                f.severity.value, f.comment[:60],
            )
        return {}

    # Post verdict as the first comment
    if state.verdict:
        verdict_body = _build_verdict_body(state.verdict.model_dump())
        _post_comment(owner, repo_name, pr_number, verdict_body)

    # Post each finding as a separate comment
    findings_data = [
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
        for f in state.final_findings
    ]

    for fd in findings_data:
        comment_body = _build_comment_body(fd)
        _post_comment(owner, repo_name, pr_number, comment_body)

    logger.info(
        "Posted %d finding(s) + verdict to %s/%s#%d",
        len(findings_data), owner, repo_name, pr_number,
    )
    return {}
