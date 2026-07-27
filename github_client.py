#!/usr/bin/env python3
"""Shared GitHub API client with automatic token resolution and 401 resilience.

Caching
-------
All HTTP requests made through ``requests`` within this process are cached
to a local SQLite backend with a 15-minute TTL, so re-reviewing the same PR
does not re-fetch data from GitHub.

Usage
-----
    from github_client import GitHubSession, resolve_token, guard_401

    # Preferred: use the session for all API calls
    session = GitHubSession()
    resp = session.get("https://api.github.com/repos/psf/requests")

    # Decorator for standalone functions
    @guard_401
    def my_api_call():
        ...
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Optional

import keyring
import requests

logger = logging.getLogger(__name__)

SERVICE_NAME = "opencodereview"

# ─── Global HTTP cache ─────────────────────────────────────────────────────────
# Cache all requests (GitHub API + raw content) to a local SQLite store
# so re-reviewing the same PR avoids redundant network fetches.
_CACHE_BACKEND = None


def _init_cache() -> None:
    """Install ``requests-cache`` with a 15-minute TTL.

    Patches ``requests.Session`` globally, so :class:`GitHubSession`
    (which inherits from it) automatically gets caching.  Safe to call
    multiple times — subsequent calls are no-ops.
    """
    global _CACHE_BACKEND
    if _CACHE_BACKEND is not None:
        return
    try:
        import requests_cache

        cache_dir = os.environ.get(
            "OPENCODEREVIEW_CACHE_DIR",
            os.path.join(os.getcwd(), ".opencodereview"),
        )
        os.makedirs(cache_dir, exist_ok=True)
        db_path = os.path.join(cache_dir, "http_cache.sqlite")
        _CACHE_BACKEND = requests_cache.install_cache(
            cache_name=db_path,
            backend="sqlite",
            expire_after=900,  # 15 minutes
            allowable_codes=(200, 404),
        )
        logger.debug("HTTP cache initialised at %s", db_path)
    except ImportError:
        logger.debug("requests-cache not installed — skipping HTTP cache")
    except Exception as exc:
        logger.warning("Failed to initialise HTTP cache: %s", exc)


# ─── Exceptions ──────────────────────────────────────────────────────────────


class TokenRevokedError(Exception):
    """Raised when a GitHub API call returns 401 (token invalid or revoked)."""


# ─── Token resolution ───────────────────────────────────────────────────────


def resolve_token() -> Optional[str]:
    """Return a GitHub token from:

    1. The system keyring (set via ``opencodereview auth login``)
    2. The ``GITHUB_TOKEN`` environment variable

    Returns ``None`` if neither source has a token, so callers can fall back
    gracefully (e.g., dry-run mode in the executor).
    """
    # Try keyring first (auth login stores here)
    try:
        token = keyring.get_password(SERVICE_NAME, "github_token")
        if token:
            return token
    except Exception:
        pass  # keyring back-end unavailable (headless CI, etc.)

    # Fall back to env var
    token = os.environ.get("GITHUB_TOKEN")
    return token.strip() if token else None


# ─── 401 decorator ──────────────────────────────────────────────────────────


def guard_401(func):
    """Decorator: catch 401 responses from GitHub API calls and raise
    :class:`TokenRevokedError` with a helpful message.

    Use this on any standalone function that makes GitHub API calls and
    calls ``raise_for_status()``.

    For functions that use :class:`GitHubSession` the 401 handling happens
    automatically — no decorator needed.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 401:
                raise TokenRevokedError(
                    "Your GitHub token appears to be invalid or has been "
                    "revoked.\n\n"
                    "  Re-authenticate:\n"
                    "    opencodereview auth login\n\n"
                    "  Or update the GITHUB_TOKEN environment variable.\n"
                ) from exc
            raise
    return wrapper


# ─── Custom Session ─────────────────────────────────────────────────────────


class GitHubSession(requests.Session):
    """A :class:`requests.Session` subclass that:

    * Auto-resolves the GitHub token from keyring then env var
    * Sends ``Authorization`` and standard headers on every request
    * Raises :class:`TokenRevokedError` on ``401`` responses from
      ``api.github.com``

    Usage
    -----
        session = GitHubSession()
        resp = session.get("https://api.github.com/repos/psf/requests")
    """

    def __init__(self, token: Optional[str] = None) -> None:
        super().__init__()
        self.token = token or resolve_token()
        self.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "OpenCodeReview/1.0",
        })
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

    def request(self, method, url, *args, **kwargs):
        """Override :meth:`requests.Session.request` to intercept 401
        responses from ``api.github.com``."""
        resp = super().request(method, url, *args, **kwargs)
        # Only intercept 401 from the GitHub API, not raw content or other hosts
        if resp.status_code == 401 and "api.github.com" in url:
            raise TokenRevokedError(
                "Your GitHub token appears to be invalid or has been "
                "revoked.\n\n"
                "  Re-authenticate:\n"
                "    opencodereview auth login\n\n"
                "  Or update the GITHUB_TOKEN environment variable.\n"
            )
        return resp
