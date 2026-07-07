#!/usr/bin/env python3
"""GitHub OAuth Device Flow authentication for OpenCodeReview.

Usage
-----
    export GITHUB_OAUTH_CLIENT_ID="Iv23li..."
    opencodereview auth login
    opencodereview auth status
    opencodereview auth logout

The Device Flow is used instead of the Authorization Code flow because
there is no web server to handle a redirect URI.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional

import click
import keyring
import requests

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

SERVICE_NAME = "opencodereview"
GH_DEVICE_CODE_URL = "https://github.com/login/device/code"
GH_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
GH_API_BASE = "https://api.github.com"
DEFAULT_SCOPE = "repo"  # Full control of private repositories — needed for
                        # reading PRs and posting review comments on
                        # private repos.
POLL_SAFETY_MARGIN = 1  # Extra seconds to wait beyond the server's interval


# ─── Exceptions ─────────────────────────────────────────────────────────────


class AuthError(Exception):
    """Base exception for auth operations."""


class DeviceFlowExpired(AuthError):
    """The device code expired before the user approved."""


class DeviceFlowDenied(AuthError):
    """The user denied the authorization request."""


# ─── Auth Service ────────────────────────────────────────────────────────────


def _resolve_client_id(client_id: Optional[str] = None) -> str:
    """Return the GitHub OAuth client ID from the argument or environment."""
    cid = client_id or os.environ.get("GITHUB_OAUTH_CLIENT_ID")
    if not cid:
        raise AuthError(
            "GitHub OAuth client ID is required.\n\n"
            "  1. Register a new OAuth App at:\n"
            "     https://github.com/settings/developers\n\n"
            "  2. Enable Device Flow in the app settings.\n\n"
            "  3. Set the client ID:\n"
            f"       export GITHUB_OAUTH_CLIENT_ID=\"<your_client_id>\"\n"
            "     Or pass --client-id to the login command.",
        )
    return cid


class AuthService:
    """Manage GitHub OAuth tokens via the Device Flow."""

    def __init__(self, client_id: Optional[str] = None) -> None:
        self._client_id = client_id or os.environ.get("GITHUB_OAUTH_CLIENT_ID")
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "opencodereview/1.0",
        })

    # ── Device Flow Login ──────────────────────────────────────────────────

    def login(self) -> str:
        """Run the GitHub Device Flow and return the access token.

        Steps
        -----
        1. POST to ``/login/device/code`` to get a device + user code.
        2. Print the user code + verification URL for the user.
        3. Poll ``/login/oauth/access_token`` until the user approves.
        4. Validate that the granted scopes include what was requested.
        5. Save the token to the system keyring.
        6. Return the access token.
        """
        # Validate client_id before making any API calls
        self._client_id = _resolve_client_id(self._client_id)

        device_code, user_code, verification_uri, interval, expires_in = (
            self._request_device_code()
        )

        self._print_login_instructions(user_code, verification_uri, expires_in)

        access_token, granted_scope = self._poll_for_token(device_code, interval)

        # Validate that the granted scopes include what we asked for
        self._validate_scopes(granted_scope)

        # Save to keyring
        keyring.set_password(SERVICE_NAME, "github_token", access_token)
        logger.info("Token saved to system credential store (keyring).")

        # Fetch and display user info
        user = self._get_user(access_token)
        click.echo(f"  Logged in as: {user['login']}")

        return access_token

    def _request_device_code(
        self,
    ) -> tuple[str, str, str, int, int]:
        """POST to the device-code endpoint and return the response fields."""
        resp = self._session.post(
            GH_DEVICE_CODE_URL,
            data={
                "client_id": self._client_id,
                "scope": DEFAULT_SCOPE,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise AuthError(
                f"Device code request failed: {data.get('error_description', data['error'])}"
            )

        return (
            data["device_code"],
            data["user_code"],
            data["verification_uri"],
            data.get("interval", 5),
            data.get("expires_in", 900),
        )

    @staticmethod
    def _print_login_instructions(
        user_code: str, verification_uri: str, expires_in: int,
    ) -> None:
        """Print the user code and instructions to the terminal."""
        click.echo()
        click.echo("=" * 60)
        click.echo("  GitHub Device Authentication")
        click.echo("=" * 60)
        click.echo()
        click.echo(f"  1. Open this URL in your browser:")
        click.echo(f"     {verification_uri}")
        click.echo()
        click.echo(f"  2. Enter the following code:")
        click.echo(f"     {user_code}")
        click.echo()
        click.echo(f"  3. Authorize the OpenCodeReview application.")
        click.echo()
        click.echo(f"  This code expires in {expires_in // 60} minute(s).")
        click.echo(f"  Waiting for you to approve...")
        click.echo()

    def _poll_for_token(self, device_code: str, interval: int) -> tuple[str, str]:
        """Poll the access-token endpoint until the user approves or the code expires.

        Returns ``(access_token, granted_scope)`` where ``granted_scope`` is the
        space-separated scope string returned by GitHub (may be empty if the
        response omitted it).
        """
        poll_interval = interval
        start_time = time.monotonic()

        while True:
            time.sleep(poll_interval + POLL_SAFETY_MARGIN)

            resp = self._session.post(
                GH_ACCESS_TOKEN_URL,
                data={
                    "client_id": self._client_id,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            if "access_token" in data:
                return data["access_token"], data.get("scope", "")

            error = data.get("error", "")

            if error == "authorization_pending":
                # User hasn't acted yet — keep polling
                continue
            elif error == "slow_down":
                # Polling too fast — increase interval by 5 seconds
                poll_interval += 5
                logger.debug("Polling rate limited — increasing interval to %ds", poll_interval)
                continue
            elif error == "expired_token":
                raise DeviceFlowExpired(
                    "The device code has expired. Run 'opencodereview auth login' again."
                )
            elif error == "access_denied":
                raise DeviceFlowDenied("The authorization request was denied.")
            else:
                raise AuthError(
                    f"Unexpected polling response: {data.get('error_description', error)}"
                )

    # ── Status ──────────────────────────────────────────────────────────────

    def status(self) -> Optional[dict]:
        """Check whether a valid token is stored and return the authenticated user.

        Returns ``None`` if no token is stored.
        Raises ``AuthError`` if the token is invalid or expired.
        """
        token = keyring.get_password(SERVICE_NAME, "github_token")
        if not token:
            return None

        user = self._get_user(token)
        return user

    # ── Logout ──────────────────────────────────────────────────────────────

    def logout(self) -> bool:
        """Delete the stored token from the keyring.

        Returns ``True`` if a token was deleted, ``False`` if none existed.
        """
        existing = keyring.get_password(SERVICE_NAME, "github_token")
        if existing:
            try:
                keyring.delete_password(SERVICE_NAME, "github_token")
            except keyring.errors.PasswordDeleteError:
                logger.warning("Failed to delete token from keyring (back-end error).")
                return False
            return True
        return False

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _validate_scopes(self, granted_scope_str: str) -> None:
        """Compare the scopes GitHub actually granted against what was requested.

        The ``scope`` field in the access-token response is a space-separated
        string.  If the user denied some scopes, we warn them on stderr so
        they know certain features may not work.
        """
        if not granted_scope_str.strip():
            # Older GitHub Enterprise or unusual config — skip validation
            return

        requested = set(DEFAULT_SCOPE.split(","))
        granted = set(granted_scope_str.strip().split())
        missing = requested - granted

        if missing:
            click.echo(
                "\n"
                "Warning: Some requested permissions were not granted.\n"
                f"  Missing scopes: {', '.join(sorted(missing))}\n"
                f"  Granted scopes: {', '.join(sorted(granted))}\n"
                "\n"
                "  Features that need these scopes may not work correctly.\n"
                "  Re-run 'opencodereview auth login' and ensure you\n"
                "  authorize all requested permissions.\n",
                err=True,
            )

    def _get_user(self, token: str) -> dict:
        """Fetch the authenticated GitHub user via the API."""
        resp = self._session.get(
            f"{GH_API_BASE}/user",
            headers={"Authorization": f"token {token}"},
        )
        if resp.status_code == 401:
            raise AuthError(
                "The stored token is invalid or has been revoked.\n"
                "Run 'opencodereview auth login' to re-authenticate."
            )
        resp.raise_for_status()
        return resp.json()


# ─── CLI ─────────────────────────────────────────────────────────────────────


@click.group()
def auth() -> None:
    """Authenticate with GitHub via the OAuth Device Flow."""


@auth.command()
@click.option(
    "--client-id",
    envvar="GITHUB_OAUTH_CLIENT_ID",
    default=None,
    help="GitHub OAuth App client ID (or set GITHUB_OAUTH_CLIENT_ID).",
)
def login(client_id: Optional[str]) -> None:
    """Authenticate with GitHub and store the token securely."""
    try:
        svc = AuthService(client_id=client_id)
        svc.login()
    except (AuthError, requests.RequestException) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@auth.command()
def logout() -> None:
    """Remove the stored GitHub token."""
    svc = AuthService()
    if svc.logout():
        click.echo("Token removed from credential store.")
    else:
        click.echo("No token found. Nothing to remove.")


@auth.command()
def status() -> None:
    """Show whether a valid token is stored and which user it belongs to."""
    svc = AuthService()
    try:
        user = svc.status()
    except AuthError as exc:
        click.echo(f"Token status: invalid ({exc})", err=True)
        sys.exit(1)
    except requests.RequestException as exc:
        click.echo(f"Error checking token: {exc}", err=True)
        sys.exit(1)

    if user is None:
        click.echo("No token stored. Run 'opencodereview auth login'.", err=True)
        sys.exit(1)
    else:
        click.echo(f"Authenticated as: {user['login']}")
        click.echo(f"Token: valid")
        if user.get("name"):
            click.echo(f"Name:  {user['name']}")
