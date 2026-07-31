from __future__ import annotations

import logging
import os

import requests
from typing import Any

from github_client import GitHubSession, resolve_token
from state import Finding, OpenCodeReviewState

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"

# ─── Helpers ─────────────────────────────────────────────────────────────────


_GITHUB_SESSION: GitHubSession | None = None


def _gh_session() -> GitHubSession:
    """Return a lazily-initialized :class:`GitHubSession`."""
    global _GITHUB_SESSION
    if _GITHUB_SESSION is None:
        _GITHUB_SESSION = GitHubSession()
    return _GITHUB_SESSION


def _fetch_pr_head_sha(owner: str, repo: str, pr_number: int) -> str | None:
    """Fetch the latest head SHA of the PR so inline comments target the correct
    commit even if the PR was updated since ingestion."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}"
    try:
        resp = _gh_session().get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()["head"]["sha"]
        logger.warning(
            "Could not fetch PR metadata: HTTP %s — %s",
            resp.status_code, resp.text[:120],
        )
        return None
    except requests.RequestException as exc:
        logger.warning("Request failed while fetching PR metadata: %s", exc)
        return None


def _post_review_comment(
    owner: str,
    repo: str,
    pr_number: int,
    commit_id: str,
    path: str,
    line: int,
    side: str,
    body: str,
    start_line: int | None = None,
) -> bool:
    """Post an inline PR review comment at the given file/line position.

    Uses ``POST /repos/{owner}/{repo}/pulls/{pull_number}/comments``.

    Parameters
    ----------
    line : int
        The line number in the diff. For multi-line comments this is the
        *last* line of the range.
    side : str
        ``"LEFT"`` or ``"RIGHT"`` — which side of the diff the comment
        targets. We always use ``"RIGHT"`` (the new/post-PR version).
    start_line : int | None
        When set, creates a multi-line comment spanning *start_line* →
        *line*.
    """
    payload: dict[str, Any] = {
        "body": body,
        "commit_id": commit_id,
        "path": path,
        "line": line,
        "side": side,
    }
    if start_line is not None:
        payload["start_line"] = start_line
        payload["start_side"] = side

    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}/comments"
    try:
        resp = _gh_session().post(url, json=payload, timeout=30)
        if resp.status_code in (201, 200):
            logger.debug(
                "Inline comment posted to %s:%d (%s)", path, line, side,
            )
            return True
        if resp.status_code == 422:
            # Stale line: the line no longer exists in the latest diff.
            logger.info(
                "Line %d no longer exists in %s (422) — will fall back "
                "to issue comment", line, path,
            )
            return False
        logger.warning(
            "Failed to post inline comment: HTTP %s — %s",
            resp.status_code, resp.text[:200],
        )
        return False
    except requests.RequestException as exc:
        logger.warning("Request failed for inline comment: %s", exc)
        return False


def _post_issue_comment(owner: str, repo: str, pr_number: int, body: str) -> bool:
    """Post a general (non-inline) issue/PR comment as a fallback.

    Uses ``POST /repos/{owner}/{repo}/issues/{issue_number}/comments``.
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues/{pr_number}/comments"
    try:
        resp = _gh_session().post(
            url, json={"body": body}, timeout=30,
        )
        if resp.status_code in (201, 200):
            logger.debug("Fallback issue comment posted")
            return True
        logger.warning(
            "Failed to post fallback comment: HTTP %s — %s",
            resp.status_code, resp.text[:200],
        )
        return False
    except requests.RequestException as exc:
        logger.warning("Request failed for fallback comment: %s", exc)
        return False


# ─── Formatting ──────────────────────────────────────────────────────────────

SEVERITY_EMOJI = {
    "critical": "\U0001f6a8",  # 🚨
    "high": "\u26a0\ufe0f",    # ⚠️
    "medium": "\U0001f4cc",    # 📌
    "low": "\U0001f4a1",       # 💡
    "info": "\u2139\ufe0f",    # ℹ️
}
VERDICT_ICON = {
    "approve": "\u2705",
    "request_changes": "\U0001f504",
    "block": "\U0001f6ab",
    "comment": "\U0001f4ac",
}


def _finding_body(finding: Finding) -> str:
    """Format a single finding as Markdown for the comment body."""
    emoji = SEVERITY_EMOJI.get(finding.severity.value, "\U0001f50d")
    fix = ""
    if finding.suggested_fix:
        fix = f"\n\n**Suggested fix:**\n```\n{finding.suggested_fix}\n```"
    return (
        f"{emoji} **{finding.severity.value.upper()}** — "
        f"*{finding.category}*  \n"
        f"**Confidence:** {finding.confidence:.0%}  \n\n"
        f"{finding.comment}{fix}"
    )


def _finding_fallback_body(finding: Finding) -> str:
    """Format a finding for a fallback issue comment when the line is stale."""
    return (
        f"*(This comment refers to a previous version of the code — "
        f"the line may no longer exist in the latest diff.)*\n\n"
        f"{_finding_body(finding)}\n\n"
        f"**Original location:** `{finding.file_path}` "
        f"(lines {finding.line_start}\u2013{finding.line_end})"
    )


def _verdict_body(verdict_dict: dict) -> str:
    """Format the overall verdict as a Markdown PR comment."""
    emoji = VERDICT_ICON.get(verdict_dict.get("recommendation", ""), "\U0001f4cb")
    summary = verdict_dict.get("summary", "")
    score = verdict_dict.get("overall_score", 0)
    return (
        f"## OpenCodeReview \u2014 {emoji} "
        f"{verdict_dict['recommendation'].upper()}\n\n"
        f"**Score:** {score}/10  \n"
        f"**Summary:** {summary}\n"
    )


# ─── Node ────────────────────────────────────────────────────────────────────


def executor_node(state: OpenCodeReviewState) -> dict:
    """Post approved findings as GitHub **PR review comments** at the correct
    file/line positions.

    Workflow
    --------
    1. Only runs if ``state.human_approved is True``.
    2. Fetches the latest PR head SHA every time (handles PR updates).
    3. For each finding:
       a. Try ``POST /pulls/{pr}/comments`` (inline review comment).
       b. On **422** (stale line): fall back to
          ``POST /issues/{pr}/comments`` (general comment).
    4. Posts the verdict as a general issue comment.

    Requires ``GITHUB_TOKEN`` to post.  Without it, logs what it *would*
    have posted.
    """
    if not state.human_approved:
        logger.info("Not approved \u2014 skipping executor")
        return {}

    parts = state.repo.split("/")
    if len(parts) != 2:
        logger.warning("Cannot post results: invalid repo '%s'", state.repo)
        return {}
    owner, repo_name = parts
    pr_number = state.pr_number

    gh_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not gh_token:
        gh_token = resolve_token() or ""
    if not gh_token:
        logger.warning(
            "No GitHub token available \u2014 cannot post to %s/%s#%d. "
            "Run 'opencodereview auth login' or set GITHUB_TOKEN.",
            owner, repo_name, pr_number,
        )
        if state.verdict:
            logger.info("[WOULD POST] Verdict: %s", state.verdict.summary)
        for f in state.final_findings:
            logger.info(
                "[WOULD POST] %s:%d\u2013%d %s \u2014 %s",
                f.file_path, f.line_start, f.line_end,
                f.severity.value, f.comment[:60],
            )
        return {}

    # ── 1. Fetch latest head SHA ─────────────────────────────────────────
    commit_id = _fetch_pr_head_sha(owner, repo_name, pr_number)
    if commit_id is None:
        logger.warning(
            "Could not fetch PR head SHA \u2014 cannot post inline "
            "comments. Falling back to issue-level comments."
        )

    # ── 2. Post each finding ─────────────────────────────────────────────
    inline_ok = 0
    inline_failed = 0
    fallback_ok = 0

    for f in state.final_findings:
        body = _finding_body(f)

        if commit_id:
            # Try inline review comment
            if f.line_start == f.line_end:
                ok = _post_review_comment(
                    owner, repo_name, pr_number,
                    commit_id, f.file_path, f.line_start, "RIGHT", body,
                )
            else:
                # Multi-line: end_line is the "line" param, start_line
                # marks the range start.
                ok = _post_review_comment(
                    owner, repo_name, pr_number,
                    commit_id, f.file_path, f.line_end, "RIGHT", body,
                    start_line=f.line_start,
                )

            if ok:
                inline_ok += 1
                continue
            inline_failed += 1

        # Fallback: issue-level comment with stale-line context
        fallback_body = (
            _finding_fallback_body(f) if commit_id else body
        )
        if _post_issue_comment(owner, repo_name, pr_number, fallback_body):
            fallback_ok += 1

    # ── 3. Post the verdict ──────────────────────────────────────────────
    if state.verdict:
        verdict_body = _verdict_body(state.verdict.model_dump())
        _post_issue_comment(owner, repo_name, pr_number, verdict_body)

    total = len(state.final_findings)
    logger.info(
        "Executor: %d/%d inline, %d/%d fallback, verdict %s \u2014 "
        "%s/%s#%d",
        inline_ok, total, fallback_ok, total - inline_ok,
        "posted" if state.verdict else "skipped",
        owner, repo_name, pr_number,
    )
    return {}
