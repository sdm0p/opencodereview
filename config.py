#!/usr/bin/env python3
"""Configuration management for OpenCodeReview.

Resolves secrets in priority order: **environment variable > keyring > error**.

Providers
---------
- ``anthropic`` — ``ANTHROPIC_API_KEY`` → stored as ``opencodereview / anthropic_api_key``

Usage
-----
    opencodereview config set-key --provider anthropic
    opencodereview config show
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import click
import keyring
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

SERVICE_NAME = "opencodereview"

# ─── Provider registry ──────────────────────────────────────────────────────
#   slug → (keyring_key, env_var_name, display_name)

PROVIDERS: dict[str, tuple[str, str, str]] = {
    "anthropic": (
        "anthropic_api_key",         # keyring key
        "OPENCODEREVIEW_ANTHROPIC_KEY",  # env var
        "Anthropic API key",          # display name
    ),
}


# ─── Exceptions ─────────────────────────────────────────────────────────────


class ConfigError(Exception):
    """Configuration-related error with a user-facing message."""


# ─── Settings (pydantic-settings) ───────────────────────────────────────────


class Settings(BaseSettings):
    """Resolve configuration from env vars, falling back to the system keyring.

    Priority
    --------
    1. Environment variable (``OPENCODEREVIEW_<FIELD>``)
    2. System keyring (``opencodereview / <keyring_key>``)
    3. :class:`ConfigError` with a helpful message telling the user how to set it.
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENCODEREVIEW_",
        extra="ignore",
        validate_default=False,
    )

    anthropic_api_key: Optional[str] = Field(default=None)

    # ── Langfuse observability (optional) ──────────────────────────────────
    langfuse_public_key: Optional[str] = Field(default=None)
    langfuse_secret_key: Optional[str] = Field(default=None)
    langfuse_host: Optional[str] = Field(default=None)

    def model_post_init(self, __context) -> None:  # noqa: ANN101
        """After env-var resolution, fall back to keyring for any unset values."""
        for keyring_key, _, _ in PROVIDERS.values():
            if getattr(self, keyring_key) is None:
                stored = keyring.get_password(SERVICE_NAME, keyring_key)
                if stored:
                    setattr(self, keyring_key, stored)

    def get_anthropic_api_key(self) -> str:
        """Return the Anthropic API key or raise ``ConfigError``."""
        key = self.anthropic_api_key
        if not key:
            raise ConfigError(
                "Anthropic API key is not configured.\n\n"
                "  Set it interactively:\n"
                "    opencodereview config set-key --provider anthropic\n\n"
                "  Or as an environment variable:\n"
                "    export OPENCODEREVIEW_ANTHROPIC_KEY=\"sk-ant-...\"\n"
            )
        return key


# ─── Observability CLI ─────────────────────────────────────────────────────


@config.command(name="set-observability")
@click.option("--langsmith-api-key", help="LangSmith API key (for tracing)")
@click.option("--langsmith-project", default="opencodereview", show_default=True,
              help="LangSmith project name")
@click.option("--langfuse-public-key", help="Langfuse public key")
@click.option("--langfuse-secret-key", help="Langfuse secret key")
@click.option("--langfuse-host", default="https://cloud.langfuse.com",
              show_default=True, help="Langfuse host URL")
def set_observability(
    langsmith_api_key: Optional[str],
    langsmith_project: str,
    langfuse_public_key: Optional[str],
    langfuse_secret_key: Optional[str],
    langfuse_host: str,
) -> None:
    """Configure observability backends (LangSmith and/or Langfuse).

    Keys are stored in the system keyring.  Both backends can be configured
    simultaneously with no conflict.  If neither is set, the app runs
    exactly as before with zero tracing overhead.
    """
    if langsmith_api_key:
        keyring.set_password(SERVICE_NAME, "langsmith_api_key", langsmith_api_key)
        click.echo(f"LangSmith API key saved (project: {langsmith_project}).")

        # Warn if env var is already set
        if os.environ.get("LANGCHAIN_API_KEY"):
            click.echo(
                "  Note: LANGCHAIN_API_KEY env var is already set — "
                "it takes precedence over the keyring value.",
                err=True,
            )
    else:
        click.echo("  LangSmith: skipped (no --langsmith-api-key provided).")

    if langfuse_public_key and langfuse_secret_key:
        keyring.set_password(SERVICE_NAME, "langfuse_public_key", langfuse_public_key)
        keyring.set_password(SERVICE_NAME, "langfuse_secret_key", langfuse_secret_key)
        keyring.set_password(SERVICE_NAME, "langfuse_host", langfuse_host)
        click.echo(f"Langfuse keys saved (host: {langfuse_host}).")

        if os.environ.get("LANGFUSE_PUBLIC_KEY"):
            click.echo(
                "  Note: LANGFUSE_PUBLIC_KEY env var is already set — "
                "it takes precedence.",
                err=True,
            )
    else:
        click.echo("  Langfuse: skipped (need both --langfuse-public-key "
                    "and --langfuse-secret-key).")


# ─── Key masking ─────────────────────────────────────────────────────────────


def _mask_key(key: str) -> str:
    """Return a masked version showing only the prefix and last characters.

    Examples
    --------
    ``sk-ant-abcdefgh12345678`` → ``sk-ant-...5678``
    ``ghp_abc123def456``       → ``ghp_abc...f456``
    ``abcdefg``                 → ``abcd...fg``
    ``tiny``                    → ``t...``
    """
    n = len(key)
    if n <= 4:
        return key[:1] + "..."
    if n <= 8:
        return key[:4] + "..." + key[-2:]
    return key[:7] + "..." + key[-4:]


# ─── CLI ─────────────────────────────────────────────────────────────────────


@click.group()
def config() -> None:
    """Manage API keys and configuration."""


@config.command()
@click.option(
    "--provider",
    required=True,
    type=click.Choice(sorted(PROVIDERS), case_sensitive=True),
    help="Which provider's API key to store.",
)
def set_key(provider: str) -> None:
    """Prompt for an API key and store it securely in the system keyring.

    The key is entered via hidden input and never echoed to the terminal.
    """
    keyring_key, env_var, display_name = PROVIDERS[provider]

    # Warn if env var is already set (keyring value would be ignored)
    if os.environ.get(env_var):
        click.echo(
            f"Warning: {display_name} is already set via the"
            f" {env_var} environment variable.\n"
            f"  A keyring-stored key won't take effect until the env var is unset.\n",
            err=True,
        )

    api_key = click.prompt(
        f"Enter your {display_name}",
        hide_input=True,
    )

    if not api_key.strip():
        click.echo("Error: Key cannot be empty.", err=True)
        sys.exit(1)

    keyring.set_password(SERVICE_NAME, keyring_key, api_key.strip())
    click.echo(f"{display_name} saved to system credential store (keyring).")


@config.command()
def show() -> None:
    """Display which keys and tokens are configured (values masked).

    Full secrets are never printed — only the first 7 + last 4 characters.
    """
    click.echo()
    click.echo("=" * 50)
    click.echo("  Configuration Summary")
    click.echo("=" * 50)
    click.echo()
    has_any = False

    # ── GitHub token (managed by auth) ──────────────────────────────────────
    gh_token = keyring.get_password(SERVICE_NAME, "github_token")
    token_env = os.environ.get("GITHUB_TOKEN")
    if token_env:
        click.echo(f"  GitHub token:      {_mask_key(token_env)}  (from env)")
        has_any = True
    elif gh_token:
        click.echo(f"  GitHub token:      {_mask_key(gh_token)}  (from keyring)")
        has_any = True
    else:
        click.echo("  GitHub token:      not set")

    # ── Provider API keys ───────────────────────────────────────────────────
    for provider_slug, (keyring_key, env_var, display_name) in sorted(PROVIDERS.items()):
        env_value = os.environ.get(env_var)
        if env_value:
            click.echo(
                f"  {display_name}:  {_mask_key(env_value)}  (from env)"
            )
            has_any = True
        else:
            stored = keyring.get_password(SERVICE_NAME, keyring_key)
            if stored:
                click.echo(
                    f"  {display_name}:  {_mask_key(stored)}  (from keyring)"
                )
                has_any = True
            else:
                click.echo(f"  {display_name}:  not set")

    click.echo()
    if not has_any:
        click.echo("  No keys or tokens configured.")
        click.echo("  Run:  opencodereview config set-key --provider anthropic")
        click.echo()
