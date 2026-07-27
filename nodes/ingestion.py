from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from github_client import GitHubSession, resolve_token
from state import ChangedFile, OpenCodeReviewState


logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
RAW_CONTENT_BASE = "https://raw.githubusercontent.com"


# ─── Custom Exception ────────────────────────────────────────────────────────


class IngestionError(Exception):
    """Raised when PR data ingestion fails and the graph cannot continue."""


def _check_rate_limit(response: requests.Response) -> None:
    """Raise ``IngestionError`` if the API quota is exhausted."""
    remaining = response.headers.get("X-RateLimit-Remaining")
    if remaining is not None and int(remaining) == 0:
        reset_ts = int(response.headers.get("X-RateLimit-Reset", "0"))
        reset_at = time.strftime("%H:%M:%S UTC", time.gmtime(reset_ts))
        raise IngestionError(
            f"GitHub API rate limit exhausted — resets at {reset_at}. "
            "Set the GITHUB_TOKEN environment variable for a much higher limit."
        )


def _parse_owner_repo(repo: str) -> tuple[str, str]:
    """Split ``owner/repo`` returning ``(owner, repo)``."""
    parts = repo.split("/")
    if len(parts) != 2 or not all(parts):
        raise IngestionError(
            f"Invalid repo format: '{repo}'. Expected 'owner/repo' "
            "(e.g. 'psf/requests')."
        )
    return parts[0], parts[1]


def _fetch_all_pages(url: str, session: requests.Session) -> list[dict]:
    """Follow ``Link``-header pagination and return every result item.

    Uses the provided ``session`` (should be a :class:`GitHubSession`) so
    that 401 handling and auth headers are applied automatically.
    """
    items: list[dict] = []
    while url:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        items.extend(resp.json())

        link_header = resp.headers.get("Link", "")
        next_url: Optional[str] = None
        for part in link_header.split(","):
            part = part.strip()
            if 'rel="next"' in part:
                start = part.index("<") + 1
                end = part.index(">")
                next_url = part[start:end]
                break
        url = next_url
    return items


def _fetch_file_content(
    owner: str, repo: str, ref: str, path: str, status: str
) -> str:
    """Return the full file content from the head branch of the PR.

    Deleted/removed files return an empty string.  Binary files that fail to
    fetch also return an empty string with a warning.
    """
    if status == "removed":
        return ""

    url = f"{RAW_CONTENT_BASE}/{owner}/{repo}/{ref}/{path}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            logger.warning("Content not found at %s (may have been deleted/renamed)", path)
            return ""
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.RequestException as exc:
        logger.warning("Could not fetch content for %s: %s", path, exc)
        return ""


# ─── Node ────────────────────────────────────────────────────────────────────


def ingestion_node(state: OpenCodeReviewState) -> dict:
    """Fetch PR metadata, diff, and changed-file contents from the GitHub API.

    Requires ``state.repo`` (``"owner/repo"``) and ``state.pr_number`` to be
    set.  Returns updates for ``diff`` and ``changed_files``.

    If ``state.diff`` is already populated (e.g. in tests with a synthetic
    payload) the node is a no-op.
    """
    # --- Early exit when data is already present (e.g. synthetic payloads) ----
    if state.diff:
        logger.info("PR data already populated — skipping ingestion")
        return {}

    # ── Token check before any API calls ─────────────────────────────────
    if not resolve_token():
        raise IngestionError(
            "No GitHub token found. OpenCodeReview needs a GitHub token to "
            "fetch PR data.\n\n"
            "  Authenticate via GitHub OAuth (recommended):\n"
            "    opencodereview auth login\n\n"
            "  Or set the GITHUB_TOKEN environment variable:\n"
            "    export GITHUB_TOKEN=\"ghp_...\"\n"
        )

    owner, repo_name = _parse_owner_repo(state.repo)
    pr_number = state.pr_number

    logger.info("Ingesting PR %s/%s#%d …", owner, repo_name, pr_number)

    # Use GitHubSession for automatic auth + 401 interception
    session = GitHubSession()

    # -- 1. PR metadata (to obtain the head SHA for content URLs) -------------
    pr_url = f"{GITHUB_API_BASE}/repos/{owner}/{repo_name}/pulls/{pr_number}"
    pr_resp = session.get(pr_url, timeout=30)

    if pr_resp.status_code == 404:
        raise IngestionError(
            f"PR {owner}/{repo_name}#{pr_number} was not found."
        )
    if pr_resp.status_code in (403, 401):
        _check_rate_limit(pr_resp)
        raise IngestionError(
            f"Access denied to {owner}/{repo_name}#{pr_number}. "
            "If the repo is private, set a valid GITHUB_TOKEN."
        )
    if pr_resp.status_code == 410:
        raise IngestionError(
            f"PR {owner}/{repo_name}#{pr_number} has been deleted."
        )
    pr_resp.raise_for_status()

    pr_data = pr_resp.json()
    head_sha = pr_data["head"]["sha"]
    head_ref = pr_data["head"]["ref"]
    base_sha = pr_data["base"]["sha"]
    base_ref = pr_data["base"]["ref"]

    logger.info(
        "PR #%d — %s — %s → %s (%s)",
        pr_number, pr_data.get("title", "(no title)"),
        base_ref, head_ref, head_sha[:7],
    )

    # -- 2. Determine where to fetch file content -----------------------------
    # For PRs from forks, the head SHA lives in the fork, not the base repo.
    head_repo_full: str = pr_data["head"]["repo"]["full_name"]  # "owner/repo"
    head_owner, head_repo_n = head_repo_full.split("/", 1)

    # -- 3. Raw unified diff of the whole PR ----------------------------------
    diff_resp = session.get(
        pr_url,
        headers={"Accept": "application/vnd.github.v3.diff"},
        timeout=30,
    )
    diff_resp.raise_for_status()
    diff_text = diff_resp.text
    logger.info("Fetched diff (%s)", _human_size(len(diff_text)))

    # -- 4. Changed files with per-file patches -------------------------------
    files_url = (
        f"{GITHUB_API_BASE}/repos/{owner}/{repo_name}/pulls/{pr_number}/files"
    )
    logger.info("Fetching changed-files list …")
    files_data = _fetch_all_pages(files_url, session)
    logger.info("Changed files: %d", len(files_data))

    # -- 5. Build ChangedFile objects (full content from the head branch) -----
    changed_files: list[ChangedFile] = []
    for file_info in files_data:
        file_path: str = file_info["filename"]
        status: str = file_info["status"]
        patch: str = file_info.get("patch") or ""
        additions: int = file_info.get("additions", 0)
        deletions: int = file_info.get("deletions", 0)

        # Fetch from the HEAD repo of the PR (handles fork PRs correctly)
        content = _fetch_file_content(
            head_owner, head_repo_n, head_sha, file_path, status
        )

        changed_files.append(ChangedFile(
            path=file_path,
            status=status,
            content=content,
            diff_hunk=patch,
        ))

        logger.debug(
            "  %s — %s (+%d/-%d, %s content)",
            file_path, status, additions, deletions,
            _human_size(len(content)),
        )

    return {
        "diff": diff_text,
        "changed_files": changed_files,
        "base_sha": base_sha,
    }


# ─── Utility ─────────────────────────────────────────────────────────────────


def _human_size(bytes_: int) -> str:
    """Return a short human-readable string for a byte count."""
    if bytes_ < 1024:
        return f"{bytes_} B"
    elif bytes_ < 1024 * 1024:
        return f"{bytes_ / 1024:.1f} KiB"
    else:
        return f"{bytes_ / (1024 * 1024):.1f} MiB"
