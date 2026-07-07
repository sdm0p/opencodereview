#!/usr/bin/env python3
"""Shared GitHub API client with automatic token resolution and 401 resilience.

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
