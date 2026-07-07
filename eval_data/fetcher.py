#!/usr/bin/env python3
"""Fetch 15 real past PRs with human-written review comments from popular
open-source Python repos for use as ground-truth evaluation data.

Requires a GITHUB_TOKEN environment variable for higher API rate limits.

Output: ``eval_data/prs.jsonl`` (one JSON line per PR with pre-fetched diff,
``changed_files``, and ``ground_truth`` review comments).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
RAW_CONTENT_BASE = "https://raw.githubusercontent.com"

# Popular Python repos with active PR reviews — we will fetch up to 3
# merged PRs per repo that bear review comments.
REPOS = [
    "psf/requests",
    "pydantic/pydantic",
    "fastapi/fastapi",
    "django/django",
    "pallets/flask",
]

PRS_PER_REPO = 3
TOTAL_TARGET = 15
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "prs.jsonl")


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "OpenCodeReviewEval/1.0",
    }
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_all_pages(url: str, headers: dict) -> list[dict]:
    """Follow GitHub Link-header pagination."""
    items: list[dict] = []
    while url:
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                logger.warning("HTTP %s for %s", resp.status_code, url)
                break
            items.extend(resp.json())
        except requests.RequestException as exc:
            logger.warning("Request failed: %s", exc)
            break

        link = resp.headers.get("Link", "")
        next_url: Optional[str] = None
        for part in link.split(","):
            part = part.strip()
            if 'rel="next"' in part:
                start = part.index("<") + 1
                end = part.index(">")
                next_url = part[start:end]
                break
        url = next_url
    return items


def _find_prs(repo: str, headers: dict, max_count: int) -> list[dict]:
    """Find merged PRs that have at least one review comment."""
    url = (
        f"{GITHUB_API_BASE}/repos/{repo}/pulls?"
        f"state=closed&sort=updated&direction=desc&per_page=100"
    )
    all_prs = _fetch_all_pages(url, headers)

    candidates = [
        pr for pr in all_prs
        if pr.get("merged_at") and pr.get("review_comments", 0) > 0
    ]

    logger.info(
        "  Found %d candidate PRs in %s (from %s total, %s merged)",
        len(candidates), repo, len(all_prs),
        sum(1 for p in all_prs if p.get("merged_at")),
    )
    return candidates[:max_count]


def _fetch_review_comments(
    owner: str, repo: str, pr_number: int, headers: dict,
) -> list[dict]:
    url = (
        f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls"
        f"/{pr_number}/comments"
    )
    return _fetch_all_pages(url, headers)


def _fetch_changed_files(
    owner: str, repo: str, pr_number: int, headers: dict,
) -> list[dict]:
    url = (
        f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls"
        f"/{pr_number}/files"
    )
    return _fetch_all_pages(url, headers)


def _fetch_file_content(owner: str, repo: str, ref: str, path: str) -> str:
    url = f"{RAW_CONTENT_BASE}/{owner}/{repo}/{ref}/{path}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.text
        return ""
    except requests.RequestException:
        return ""


def _build_ground_truth(review_comments: list[dict]) -> list[dict]:
    """Convert GitHub PR review comments to the ground-truth schema."""
    gt = []
    for rc in review_comments:
        # Use the line number from the diff; fall back to the original line.
        line = rc.get("line") or rc.get("original_line") or 0
        gt.append({
            "file_path": rc.get("path", ""),
            "line": line,
            "comment": rc.get("body", ""),
            "reviewer": rc.get("user", {}).get("login", "unknown"),
            "created_at": rc.get("created_at", ""),
        })
    return gt


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    headers = _headers()
    entries: list[dict] = []

    if "GITHUB_TOKEN" not in os.environ:
        logger.error(
            "GITHUB_TOKEN is not set! The GitHub API enforces very "
            "restrictive rate limits (60 req/hr) for unauthenticated "
            "requests — the fetcher WILL FAIL with HTTP 403 errors.\n"
            "  Set the environment variable and re-run:\n"
            "    export GITHUB_TOKEN=ghp_..."
        )
        logger.info("Exiting — no GITHUB_TOKEN set")
        sys.exit(1)

    for repo in REPOS:
        if len(entries) >= TOTAL_TARGET:
            break

        owner, repo_name = repo.split("/")
        prs = _find_prs(repo, headers, PRS_PER_REPO)

        for pr in prs:
            if len(entries) >= TOTAL_TARGET:
                break

            pr_number = pr["number"]
            title = pr.get("title", "") or "(no title)"
            head_sha = pr["head"]["sha"]

            logger.info(
                "Fetching %s/%s#%d — %s",
                owner, repo_name, pr_number, title[:80],
            )

            # --- Diff ---
            diff_headers = {**headers, "Accept": "application/vnd.github.v3.diff"}
            try:
                diff_resp = requests.get(
                    pr["url"], headers=diff_headers, timeout=30,
                )
                diff_text = diff_resp.text if diff_resp.status_code == 200 else ""
            except requests.RequestException as exc:
                logger.warning("  Diff fetch failed: %s", exc)
                diff_text = ""

            # --- Changed files with content ---
            files_data = _fetch_changed_files(owner, repo_name, pr_number, headers)
            changed_files: list[dict] = []
            for fd in files_data:
                file_path = fd["filename"]
                status = fd["status"]
                patch = fd.get("patch") or ""

                # Fetch file content from the head commit
                content = _fetch_file_content(owner, repo_name, head_sha, file_path)
                if status == "removed":
                    content = ""

                # For fork PRs, try the head repo (different from base)
                if not content:
                    head_full = pr["head"]["repo"].get("full_name", "")
                    if head_full and head_full != repo:
                        h_owner, h_repo = head_full.split("/", 1)
                        content = _fetch_file_content(
                            h_owner, h_repo, head_sha, file_path,
                        )

                changed_files.append({
                    "path": file_path,
                    "status": status,
                    "content": content,
                    "diff_hunk": patch,
                })

            # --- Review comments (ground truth) ---
            review_comments = _fetch_review_comments(
                owner, repo_name, pr_number, headers,
            )
            ground_truth = _build_ground_truth(review_comments)

            if not ground_truth:
                logger.info("  Skipping — no review comments found")
                continue

            entry: dict[str, Any] = {
                "id": f"{repo.replace('/', '_')}_{pr_number}",
                "repo": repo,
                "pr_number": pr_number,
                "title": title,
                "diff": diff_text,
                "changed_files": changed_files,
                "ground_truth": ground_truth,
            }
            entries.append(entry)

            logger.info(
                "  -> %d files, %d review comments, diff=%s",
                len(changed_files), len(ground_truth),
                f"{len(diff_text)} B" if diff_text else "empty",
            )

            # Polite rate-limiting
            time.sleep(0.5)

    # --- Write output ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total_gt = sum(len(e["ground_truth"]) for e in entries)
    logger.info(
        "Wrote %d PRs to %s — %d ground-truth comments (avg %.1f/PR)",
        len(entries), OUTPUT_PATH, total_gt,
        total_gt / len(entries) if entries else 0,
    )

    # Summary line for easy terminal inspection
    print(f"\n{'='*70}")
    print(f"  Dataset: {OUTPUT_PATH}")
    print(f"  PRs:     {len(entries)}")
    print(f"  Ground-truth comments: {total_gt}")
    for e in entries:
        print(f"    {e['id']:40s}  {len(e['ground_truth']):3d} comments  {e['title'][:50]}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
