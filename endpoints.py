"""Custom LLM endpoint registry.

Lets users register any number of custom model endpoints via environment
variables (set as Space secrets or in the shell), so a review can run on a
user-chosen model instead of the single built-in key.

Env-var convention (index *n* is any number ≥ 1)::

    OCR_ENDPOINT_1_NAME=DeepSeek
    OCR_ENDPOINT_1_TYPE=openai            # openai | anthropic | google (default openai)
    OCR_ENDPOINT_1_API_KEY=sk-...
    OCR_ENDPOINT_1_BASE_URL=https://api.deepseek.com/v1
    OCR_ENDPOINT_1_MODEL=deepseek-chat

    OCR_ENDPOINT_2_NAME=Claude
    OCR_ENDPOINT_2_TYPE=anthropic
    OCR_ENDPOINT_2_API_KEY=sk-ant-...
    OCR_ENDPOINT_2_MODEL=claude-sonnet-4-20250514

Supported ``TYPE`` values:
    * ``openai``    — any OpenAI-compatible API (DeepSeek, OpenRouter, vLLM,
                      Ollama, Together, local proxies, ...)
    * ``anthropic`` — Anthropic Claude (``base_url`` optional, e.g. a proxy)
    * ``google``    — Google Gemini (``base_url`` not used)

Keys are only ever read from the environment — they are never written to
disk.  Use :func:`mask_key` anywhere a key is shown in the UI.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

ENDPOINT_ENV_PREFIX = "OCR_ENDPOINT_"

SUPPORTED_PROVIDERS = ("openai", "anthropic", "google")


@dataclass
class EndpointConfig:
    """A single custom LLM endpoint the user has configured."""

    name: str
    provider: str = "openai"  # openai | anthropic | google
    api_key: str = ""
    base_url: Optional[str] = None
    model: str = ""
    index: int = 0  # the env-var index the endpoint came from
    builtin: bool = False  # True for the always-available Gemini/Grok providers

    def __post_init__(self) -> None:
        self.provider = (self.provider or "openai").strip().lower()
        if self.provider not in SUPPORTED_PROVIDERS:
            logger.warning(
                "Endpoint %r has unsupported type %r — treating as openai",
                self.name, self.provider,
            )
            self.provider = "openai"

    @property
    def masked_key(self) -> str:
        """API key with only the first/last few chars visible."""
        return mask_key(self.api_key)

    def describe(self) -> str:
        """One-line human-readable summary for the UI / CLI."""
        base = f" ({self.base_url})" if self.base_url else ""
        return f"{self.name} [{self.provider}] {self.model}{base} — {self.masked_key}"


def mask_key(api_key: str) -> str:
    """Mask an API key for display, e.g. ``sk-abc…wxyz``."""
    if not api_key:
        return "(empty)"
    if len(api_key) <= 8:
        return api_key[:2] + "…" + api_key[-2:]
    return api_key[:6] + "…" + api_key[-4:]


# ── Runtime (UI-added) endpoints ────────────────────────────────────────────
# Endpoints entered in the web UI are kept in-memory only: they are never
# written to disk and disappear when the app process restarts.  They are
# layered on top of the env-var endpoints below, so both are available and
# a UI-added endpoint with the same name as an env one takes precedence.
_session_endpoints: dict[str, EndpointConfig] = {}
# Guards the registry above — Gradio serves events from a thread pool, so
# concurrent visitors may register/clear endpoints at the same time.
_session_lock = threading.Lock()


def register_endpoint(
    name: str,
    provider: str = "openai",
    api_key: str = "",
    base_url: Optional[str] = None,
    model: str = "",
) -> EndpointConfig:
    """Register (or replace) an endpoint added at runtime via the UI.

    Validation is strict so a review never runs on a half-configured entry:

    * ``name`` and ``model`` are required
    * ``api_key`` is required (use a placeholder such as ``dummy`` for
      keyless local servers like Ollama)
    * ``provider`` must be one of :data:`SUPPORTED_PROVIDERS`

    Returns the stored :class:`EndpointConfig`.  Re-registering the same
    name overwrites the previous entry.
    """
    name = (name or "").strip()
    provider = (provider or "openai").strip().lower()
    model = (model or "").strip()
    api_key = (api_key or "").strip()
    base_url = (base_url or "").strip() or None

    if not name:
        raise ValueError("Endpoint name is required.")
    if not model:
        raise ValueError("Model name is required.")
    if not api_key:
        raise ValueError("API key is required (use 'dummy' for keyless local servers).")
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported type {provider!r} — use one of {SUPPORTED_PROVIDERS}."
        )

    cfg = EndpointConfig(
        name=name,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    with _session_lock:
        if name in _scan_env_endpoints_by_name() or any(
            ep.name == name for ep in builtin_endpoints()
        ):
            logger.warning(
                "Session endpoint %r shadows an existing endpoint "
                "(OCR_ENDPOINT_* env or built-in) with the same name for "
                "the rest of this process.",
                name,
            )
        _session_endpoints[name] = cfg
    logger.info("Registered session endpoint %r (%s / %s)", name, provider, model)
    return cfg


def clear_session_endpoints() -> None:
    """Remove every endpoint that was added at runtime via the UI."""
    with _session_lock:
        _session_endpoints.clear()


def session_endpoints() -> list[EndpointConfig]:
    """Endpoints added at runtime via the UI (env-var ones excluded)."""
    with _session_lock:
        return list(_session_endpoints.values())


def _scan_env_endpoints() -> list[EndpointConfig]:
    """Scan the environment for ``OCR_ENDPOINT_<n>_*`` configs.

    Each index is read independently.  An endpoint is registered only when
    it has at least a name and a model (the API key is required to actually
    build a client, but a key-less entry is still listed so the UI can show
    \"missing key\").  Entries are returned sorted by index.
    """
    groups: dict[int, dict[str, str]] = {}
    for env_key, value in os.environ.items():
        if not env_key.startswith(ENDPOINT_ENV_PREFIX):
            continue
        rest = env_key[len(ENDPOINT_ENV_PREFIX):]
        if "_" not in rest:
            continue
        idx_str, attr = rest.split("_", 1)
        if not idx_str.isdigit():
            continue
        groups.setdefault(int(idx_str), {})[attr.lower()] = value.strip()

    endpoints: list[EndpointConfig] = []
    for idx in sorted(groups):
        g = groups[idx]
        name = g.get("name") or f"endpoint-{idx}"
        if not g.get("model"):
            # Incomplete config — skip (must at least have a model)
            continue
        endpoints.append(
            EndpointConfig(
                index=idx,
                name=name,
                provider=g.get("type") or g.get("provider") or "openai",
                api_key=g.get("api_key") or "",
                base_url=g.get("base_url") or None,
                model=g.get("model") or "",
            )
        )
    return endpoints


def _scan_env_endpoints_by_name() -> dict[str, EndpointConfig]:
    """Env endpoints keyed by name (helper for collision checks)."""
    return {ep.name: ep for ep in _scan_env_endpoints()}


def builtin_endpoints() -> list[EndpointConfig]:
    """The always-available built-in providers (Gemini, Grok).

    These are surfaced as selectable endpoints so the UI/CLI can choose
    explicitly between them.  They read their keys from the standard
    ``GEMINI_API_KEY`` / ``GROQ_API_KEY`` env vars and run on the same
    models as the default (no-endpoint) path.
    """
    from llm_factory import GEMINI_MODEL, GROQ_MODEL

    return [
        EndpointConfig(
            name="Gemini",
            provider="google",
            api_key=os.environ.get("GEMINI_API_KEY", "").strip(),
            model=GEMINI_MODEL,
            builtin=True,
        ),
        EndpointConfig(
            name="Grok",
            provider="openai",
            api_key=os.environ.get("GROQ_API_KEY", "").strip(),
            base_url="https://api.groq.com/openai/v1",
            model=GROQ_MODEL,
            builtin=True,
        ),
    ]


def discover_endpoints() -> list[EndpointConfig]:
    """All available endpoints: built-ins + env-var configs + UI-added ones.

    Precedence on name collision: UI-added (session) > env-var > built-in.
    Entries are returned sorted by name.
    """
    merged: dict[str, EndpointConfig] = {
        ep.name: ep for ep in builtin_endpoints()
    }
    merged.update(_scan_env_endpoints_by_name())
    with _session_lock:
        for ep in _session_endpoints.values():
            merged[ep.name] = ep
    return [merged[k] for k in sorted(merged)]


def get_endpoint(name: str | None) -> Optional[EndpointConfig]:
    """Return the endpoint whose name matches, or ``None``.

    Resolution order: UI-added (session) → env-var → built-in.
    """
    if not name:
        return None
    with _session_lock:
        session_ep = _session_endpoints.get(name)
    if session_ep is not None:
        return session_ep
    env_ep = _scan_env_endpoints_by_name().get(name)
    if env_ep is not None:
        return env_ep
    return next((b for b in builtin_endpoints() if b.name == name), None)


def dropdown_choices() -> list[str | tuple[str, list[str]]]:
    """Grouped choices for the UI dropdown.

    Returns a Gradio-compatible choice list with two optgroups:
    ``("Built-in", ["Gemini", "Grok"])`` and ``("Custom", [...])`` —
    so the user can pick a *section* for the built-in providers vs the
    custom endpoints they configured.  An empty string entry keeps the
    "auto" (no-endpoint) option first.

    A built-in is omitted from its group when a custom endpoint shadows it
    by name (session > env > built-in precedence), so each name appears
    exactly once.
    """
    custom_names = [
        ep.name for ep in discover_endpoints() if not ep.builtin
    ]
    builtin_names = [
        ep.name for ep in builtin_endpoints()
        if ep.name not in custom_names
    ]
    choices: list = [""]
    if builtin_names:
        choices.append(("Built-in", builtin_names))
    if custom_names:
        choices.append(("Custom", custom_names))
    return choices
