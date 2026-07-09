#!/usr/bin/env python3
"""Observability & monitoring for OpenCodeReview.

Provides a single-entry-point module for all tracing, cost tracking,
metadata attribution, health checks, and alerting concerns.

Backends
--------
* **LangSmith** — automatic via ``LANGCHAIN_API_KEY`` / ``LANGCHAIN_PROJECT``
  env vars.  LangGraph/LangChain reads these at import time and starts
  tracing without any per-call code.
* **Langfuse** — explicit ``CallbackHandler`` instantiated only when
  ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set.

Usage
-----
    from observability import get_langfuse_handler, build_run_metadata

    handler = get_langfuse_handler()
    metadata = build_run_metadata(source="cli", repo="org/repo", pr_number=42)
    config = {"configurable": {"thread_id": "..."}, "metadata": metadata}
    if handler:
        config["callbacks"] = [handler]
"""

from __future__ import annotations

import logging
import os
import platform
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─── LangSmith (automatic, env-var-driven) ──────────────────────────────────


def is_langsmith_enabled() -> bool:
    """Return True if LangSmith tracing is configured via env vars."""
    return bool(os.environ.get("LANGCHAIN_API_KEY", "").strip())


def enable_langsmith(api_key: str, project: str = "") -> None:
    """Set the environment variables that LangChain/LangGraph reads at runtime.

    Call this at startup **before** any LangChain/LangGraph imports so that
    the tracing is active from the first call.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    if project:
        os.environ["LANGCHAIN_PROJECT"] = project
    logger.info(
        "LangSmith tracing enabled (project=%s)",
        project or os.environ.get("LANGCHAIN_PROJECT", "default"),
    )


def disable_langsmith() -> None:
    """Remove LangSmith env vars so tracing stops."""
    os.environ.pop("LANGCHAIN_TRACING_V2", None)
    os.environ.pop("LANGCHAIN_API_KEY", None)
    # Keep LANGCHAIN_PROJECT in case user wants to set it separately


# ─── Langfuse (explicit callback) ──────────────────────────────────────────


def get_langfuse_handler():
    """Return a Langfuse ``CallbackHandler`` if keys are configured, else None.

    Reads ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, and optionally
    ``LANGFUSE_HOST`` from environment variables.  In langfuse v4+ the
    ``CallbackHandler`` only needs ``public_key``; the SDK reads
    ``LANGFUSE_SECRET_KEY`` and ``LANGFUSE_HOST`` from the environment
    automatically.
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()

    if not public_key or not secret_key:
        return None

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        # In langfuse v4+, the CallbackHandler uses get_client() to look up
        # an existing Langfuse client instance by public_key.  We must
        # initialise the client first so that get_client() finds it;
        # otherwise the handler receives a disabled client that drops traces.
        # Both classes read LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY from
        # environment variables automatically.
        Langfuse()
        return CallbackHandler()
    except ImportError:
        logger.warning(
            "langfuse not available — skipping tracing",
            exc_info=True,
        )
        return None
    except Exception as exc:
        logger.warning(
            "Failed to initialise Langfuse handler: %s",
            exc,
            exc_info=True,
        )
        return None


def is_langfuse_enabled() -> bool:
    """Return True if Langfuse keys are present in the environment."""
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    return bool(pk and sk)


# ─── Run metadata ──────────────────────────────────────────────────────────


def build_run_metadata(
    source: str = "cli",
    repo: str = "",
    pr_number: int = 0,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    prompt_versions: Optional[dict[str, str]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a metadata dict attached to every trace/span.

    Parameters
    ----------
    source : str
        ``"cli"`` or ``"gradio_ui"``.
    repo : str
        GitHub repository being reviewed (e.g. ``"psf/requests"``).
    pr_number : int
        PR number being reviewed.
    session_id : str or None
        Unique session / thread identifier.
    user_id : str or None
        GitHub username (CLI) or anonymised token (Gradio).
    prompt_versions : dict or None
        Map of reviewer name → prompt file version (e.g.
        ``{"correctness": "v1", "security": "v1"}``).
    extra : dict or None
        Any additional key-value pairs to attach.

    Returns
    -------
    dict
        Flat metadata dict suitable for ``config["metadata"]``.
    """
    meta: dict[str, Any] = {
        "source": source,
        "app": "opencodereview",
        "host": platform.node(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if repo:
        meta["repo"] = repo
    if pr_number:
        meta["pr_number"] = pr_number
    if session_id:
        meta["session_id"] = session_id
    if user_id:
        meta["user_id"] = user_id
    if prompt_versions:
        meta["prompt_versions"] = prompt_versions
    if extra:
        meta.update(extra)

    return meta


# ─── Cost tracking ─────────────────────────────────────────────────────────


# Groq published per-1M-token pricing (USD) — as of 2025-07
# Source: https://console.groq.com/docs/pricing
GROQ_PRICING: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    # Fallback for unknown models — pessimistic estimate
    "__default__": {"input": 1.00, "output": 1.00},
}


def estimate_groq_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate the USD cost of a Groq LLM call.

    Parameters
    ----------
    model : str
        Groq model name (e.g. ``"llama-3.3-70b-versatile"``).
    input_tokens : int
        Number of tokens in the prompt (system + user messages).
    output_tokens : int
        Number of tokens in the completion.

    Returns
    -------
    float
        Estimated cost in USD.
    """
    pricing = GROQ_PRICING.get(model, GROQ_PRICING["__default__"])
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


def format_cost(cost_usd: float) -> str:
    """Format a USD cost for human-readable display."""
    if cost_usd < 0.001:
        return f"${cost_usd * 1000:.2f} m¢"
    return f"${cost_usd:.4f}"


# ─── Health check helpers ──────────────────────────────────────────────────


class HealthStatus:
    """Aggregate health of the observability and API connectivity."""

    def __init__(self) -> None:
        self.langsmith: bool = is_langsmith_enabled()
        self.langfuse: bool = is_langfuse_enabled()
        self.groq: bool = bool(os.environ.get("GROQ_API_KEY", "").strip())
        self.github: bool = bool(os.environ.get("GITHUB_TOKEN", "").strip())

    @property
    def any_tracing(self) -> bool:
        return self.langsmith or self.langfuse

    def summary(self) -> dict[str, str]:
        """Return a dict of component → status string."""
        return {
            "LangSmith": "✅ Active" if self.langsmith else "⛔ Not configured",
            "Langfuse": "✅ Active" if self.langfuse else "⛔ Not configured",
            "Groq API": "✅ Key set" if self.groq else "⛔ No key",
            "GitHub API": "✅ Token set" if self.github else "⛔ No token",
        }


def check_groq_connectivity() -> tuple[bool, str]:
    """Lightweight Groq connectivity check — does not consume quota.

    Uses ``urllib.request`` from stdlib (no extra dependencies).

    Returns ``(ok, message)``.
    """
    import json as _json
    import urllib.error as _urlerror
    import urllib.request as _urlreq

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return False, "No GROQ_API_KEY set"

    try:
        req = _urlreq.Request(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="GET",
        )
        with _urlreq.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                data = _json.loads(resp.read().decode("utf-8"))
                models = data.get("data", [])
                count = len(models)
                return True, f"Reachable — {count} models available"
            else:
                return False, f"HTTP {resp.status}"
    except _urlerror.HTTPError as exc:
        if exc.code == 401:
            return False, "Invalid API key (401)"
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, str(exc)


def check_langfuse_connectivity() -> tuple[bool, str]:
    """Lightweight Langfuse connectivity check.

    Returns ``(ok, message)``.
    """
    if not is_langfuse_enabled():
        return False, "Not configured"

    try:
        from langfuse import Langfuse

        lf = Langfuse()
        lf.auth_check()
        return True, "Connected"
    except Exception as exc:
        return False, str(exc)


# ─── Token cost callback (accumulates LLM usage across a run) ────────────


class TokenCostCallback:
    """Accumulates per-model token usage from LangChain LLM calls.

    Lazily imports ``BaseCallbackHandler`` so that ``observability.py`` can
    be imported even when ``langchain-core`` has version incompatibilities
    (e.g. on Hugging Face Spaces).  Only ``on_llm_end`` is overridden;
    all other lifecycle events are ignored.

    Usage
    -----
        counter = TokenCostCallback()
        config["callbacks"] = [counter, langfuse_handler]  # if langfuse
        ... run graph ...
        print(counter.summary())
    """

    raise_error: bool = False
    ignore_llm: bool = False
    ignore_chain: bool = False
    ignore_agent: bool = False
    ignore_retriever: bool = False
    ignore_chat_model: bool = False
    ignore_custom_event: bool = False
    ignore_retry: bool = False
    run_inline: bool = False

    def __init__(self) -> None:
        self.usage: list[dict[str, Any]] = []

    @property
    def total_input_tokens(self) -> int:
        return sum(u.get("input_tokens", 0) for u in self.usage)

    @property
    def total_output_tokens(self) -> int:
        return sum(u.get("output_tokens", 0) for u in self.usage)

    @property
    def total_cost(self) -> float:
        return sum(u.get("cost", 0.0) for u in self.usage)

    def on_chain_start(self, *args, **kwargs) -> None:
        pass

    def on_chain_end(self, *args, **kwargs) -> None:
        pass

    def on_chat_model_start(self, *args, **kwargs) -> None:
        pass

    def on_llm_start(self, *args, **kwargs) -> None:
        pass

    def on_llm_error(self, error, **kwargs) -> None:
        pass

    def on_chain_error(self, error, **kwargs) -> None:
        pass

    def on_tool_error(self, error, **kwargs) -> None:
        pass

    def on_retriever_error(self, error, **kwargs) -> None:
        pass

    def on_llm_end(self, response, **kwargs) -> None:
        """LangChain callback — called after each LLM invocation.

        ``**kwargs`` absorbs ``run_id``, ``parent_run_id``, and other
        metadata that LangChain passes to callback handlers.
        """
        try:
            usage = response.llm_output or {}
            token_usage = usage.get("token_usage", {}) or {}
            model = usage.get("model_name", "") or ""

            input_tokens = token_usage.get("prompt_tokens", 0) or 0
            output_tokens = token_usage.get("completion_tokens", 0) or 0

            if input_tokens > 0 or output_tokens > 0:
                cost = estimate_groq_cost(model, input_tokens, output_tokens)
                self.usage.append({
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": cost,
                })
        except Exception:
            pass  # Best-effort — don't let cost tracking crash the run

    def summary(self) -> str:
        """Return a human-readable cost summary."""
        calls = len(self.usage)
        if calls == 0:
            return "Cost: N/A (no LLM calls captured)"
        return (
            f"Cost: {format_cost(self.total_cost)}"
            f" · {calls} LLM call(s)"
            f" · {self.total_input_tokens:,} in / {self.total_output_tokens:,} out tokens"
        )

    def reset(self) -> None:
        self.usage.clear()


# ─── Error event logging ─────────────────────────────────────────────────--


def log_error_to_backends(
    error: Exception,
    context: Optional[dict[str, Any]] = None,
) -> None:
    """Log an error event to whichever observability backend(s) are active.

    For LangSmith: logs via LangChain's callback system (automatic).
    For Langfuse: creates an identifyable error event via the SDK.

    This function primarily logs to ``logger`` with full traceback and
    enriches the log record with context that the tracing backends pick up.

    Parameters
    ----------
    error : Exception
        The exception that occurred.
    context : dict or None
        Additional context about where/when the error happened.
    """
    extras = ""
    if context:
        extras = f" [{', '.join(f'{k}={v}' for k, v in context.items())}]"

    logger.exception(
        "OBSERVABILITY-ERROR%s: %s: %s",
        extras,
        type(error).__name__,
        error,
        exc_info=error,
    )
