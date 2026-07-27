#!/usr/bin/env python3
"""Observability & monitoring for OpenCodeReview.

Provides a single-entry-point module for all tracing, cost tracking,
metadata attribution, health checks, and alerting concerns.

Backends
-------- * **LangSmith** — automatic via ``LANGSMITH_API_KEY`` / ``LANGSMITH_PROJECT``
 *                (also accepts legacy ``LANGCHAIN_API_KEY`` for back-compat)
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
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

# ─── LangSmith (automatic, env-var-driven) ──────────────────────────────────


def is_langsmith_enabled() -> bool:
    """Return True if LangSmith tracing is configured via env vars.

    Checks both the new ``LANGSMITH_API_KEY`` (preferred) and legacy
    ``LANGCHAIN_API_KEY`` env vars for backward compatibility.
    """
    return bool(
        os.environ.get("LANGSMITH_API_KEY", "").strip()
        or os.environ.get("LANGCHAIN_API_KEY", "").strip()
    )


def enable_langsmith(api_key: str, project: str = "") -> None:
    """Set the environment variables that LangSmith reads at runtime.

    Per the official LangSmith docs (https://docs.langchain.com/langsmith/trace-with-langgraph):
    the modern env vars are ``LANGSMITH_TRACING`` and ``LANGSMITH_API_KEY``.
    We also set the legacy ``LANGCHAIN_*`` vars for backward compatibility
    with older LangChain versions.

    Call this at startup **before** any LangChain/LangGraph imports so that
    the tracing is active from the first call.
    """
    # Modern env vars (required by langsmith-sdk >= 0.7)
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = os.environ.get(
        "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
    )
    os.environ["LANGSMITH_API_KEY"] = api_key
    # Legacy env vars (backward compat with older LangChain)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    if project:
        os.environ["LANGSMITH_PROJECT"] = project
        os.environ["LANGCHAIN_PROJECT"] = project
    logger.info(
        "LangSmith tracing enabled (project=%s)",
        project or os.environ.get("LANGSMITH_PROJECT", "default"),
    )


def disable_langsmith() -> None:
    """Remove LangSmith env vars so tracing stops."""
    # Modern vars
    os.environ.pop("LANGSMITH_TRACING", None)
    os.environ.pop("LANGSMITH_ENDPOINT", None)
    os.environ.pop("LANGSMITH_API_KEY", None)
    # Legacy vars
    os.environ.pop("LANGCHAIN_TRACING_V2", None)
    os.environ.pop("LANGCHAIN_API_KEY", None)
    # Keep project name in case user wants to set it separately


# ─── Langfuse (explicit callback) ──────────────────────────────────────────


def is_langfuse_enabled() -> bool:
    """Return True if Langfuse keys are present in the environment."""
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    return bool(pk and sk)


def get_langfuse_handler():
    """Return a bare Langfuse ``CallbackHandler`` if keys are configured, else None.

    In langfuse v4+ the ``CallbackHandler`` constructor only accepts
    ``public_key`` and ``trace_context`` — trace-level metadata (name, tags,
    user_id, session_id, metadata) is set via the ``propagate_attributes()``
    context manager instead.  Use :func:`langfuse_trace` to wrap your graph
    execution with proper trace attributes.
    """
    if not is_langfuse_enabled():
        return None

    try:
        from langfuse.langchain import CallbackHandler

        # CallbackHandler reads env vars automatically in v4+.
        # No trace metadata params are passed — those go through
        # propagate_attributes() instead.
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


@contextmanager
def langfuse_trace(
    trace_name: str = "opencodereview",
    tags: Optional[list[str]] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Generator[Any, None, None]:
    """Context manager that wraps a LangGraph execution block in Langfuse trace attributes.

    In langfuse v4+, trace-level metadata (name, tags, user_id, session_id,
    metadata) must be set via the ``propagate_attributes()`` context manager
    rather than passed to ``CallbackHandler()``.  This helper creates the
    handler and the ``propagate_attributes`` wrapper in one step.

    Usage
    -----
        with langfuse_trace(
            trace_name="opencodereview/review/psf/requests#42",
            tags=["gradio_ui", "psf/requests"],
            session_id=thread_id,
            metadata={"repo": "psf/requests", "pr_number": 42},
        ) as handler:
            if handler:
                config["callbacks"].append(handler)
            graph.stream(initial_state, config)
    """
    handler = None
    if not is_langfuse_enabled():
        yield None
        return

    try:
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler()
    except Exception:
        yield None
        return

    # Build propagate_attributes kwargs
    attrs: dict[str, Any] = {}
    if trace_name:
        attrs["trace_name"] = trace_name
    if tags:
        attrs["tags"] = tags
    if user_id:
        attrs["user_id"] = user_id
    if session_id:
        attrs["session_id"] = session_id
    if metadata:
        attrs["metadata"] = metadata

    if attrs:
        try:
            from langfuse import propagate_attributes as _propagate_attributes
        except Exception as exc:
            logger.debug(
                "propagate_attributes not available — yielding bare handler: %s",
                exc,
            )
            yield handler
            return

        # yield is OUTSIDE try/except so user-code exceptions propagate
        with _propagate_attributes(**attrs):
            yield handler
    else:
        yield handler


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
    """Accumulates per-model token usage & TTFT from LangChain LLM calls.

    Also tracks **TTFT** (Time to First Token) by recording a timestamp
    at ``on_chat_model_start`` and computing elapsed time at ``on_llm_end``.
    For non-streaming LLM calls this is effectively the full request duration;
    for streaming it would be the time to the first chunk.

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
        self._start_times: dict[str, float] = {}  # run_id -> timestamp

    @property
    def total_input_tokens(self) -> int:
        return sum(u.get("input_tokens", 0) for u in self.usage)

    @property
    def total_output_tokens(self) -> int:
        return sum(u.get("output_tokens", 0) for u in self.usage)

    @property
    def total_cost(self) -> float:
        return sum(u.get("cost", 0.0) for u in self.usage)

    @property
    def total_ttft(self) -> float:
        """Total TTFT across all LLM calls, in seconds."""
        ttfts = [u.get("ttft_seconds", 0.0) for u in self.usage if u.get("ttft_seconds") is not None]
        return sum(ttfts)

    @property
    def avg_ttft(self) -> Optional[float]:
        """Average TTFT across all LLM calls, in seconds (None if no calls)."""
        ttfts = [u.get("ttft_seconds", 0.0) for u in self.usage if u.get("ttft_seconds") is not None]
        if not ttfts:
            return None
        return sum(ttfts) / len(ttfts)

    def on_chain_start(self, *args, **kwargs) -> None:
        pass

    def on_chain_end(self, *args, **kwargs) -> None:
        pass

    def on_chat_model_start(self, *args, **kwargs) -> None:
        """Record start timestamp for TTFT measurement.

        ``**kwargs`` contains ``run_id`` which we use as the key.
        """
        run_id = kwargs.get("run_id")
        if run_id:
            self._start_times[str(run_id)] = time.time()

    def on_llm_start(self, *args, **kwargs) -> None:
        """Fallback: record start timestamp if ``on_chat_model_start`` wasn't called."""
        run_id = kwargs.get("run_id")
        if run_id and str(run_id) not in self._start_times:
            self._start_times[str(run_id)] = time.time()

    def on_llm_error(self, error, **kwargs) -> None:
        self._cleanup_run(kwargs.get("run_id"))

    def on_chain_error(self, error, **kwargs) -> None:
        pass

    def on_tool_error(self, error, **kwargs) -> None:
        pass

    def on_retriever_error(self, error, **kwargs) -> None:
        pass

    def on_llm_end(self, response, **kwargs) -> None:
        """LangChain callback — called after each LLM invocation.

        Computes TTFT from the recorded start time and captures token usage.
        ``**kwargs`` absorbs ``run_id``, ``parent_run_id``, and other
        metadata that LangChain passes to callback handlers.
        """
        try:
            run_key = str(kwargs.get("run_id", ""))

            # Compute TTFT (time from start to first token / completion)
            ttft: Optional[float] = None
            if run_key in self._start_times:
                ttft = round(time.time() - self._start_times.pop(run_key), 3)

            usage = response.llm_output or {}
            token_usage = usage.get("token_usage", {}) or {}
            model = usage.get("model_name", "") or ""

            input_tokens = token_usage.get("prompt_tokens", 0) or 0
            output_tokens = token_usage.get("completion_tokens", 0) or 0

            if input_tokens > 0 or output_tokens > 0:
                cost = estimate_groq_cost(model, input_tokens, output_tokens)
                entry: dict[str, Any] = {
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": cost,
                }
                if ttft is not None:
                    entry["ttft_seconds"] = ttft
                self.usage.append(entry)
            else:
                self._cleanup_run(run_key)
        except Exception:
            self._cleanup_run(run_key)
            pass  # Best-effort — don't let cost tracking crash the run

    def _cleanup_run(self, run_id: Any) -> None:
        """Remove a run_id from the start times map (cleanup on error)."""
        if run_id:
            self._start_times.pop(str(run_id), None)

    def summary(self) -> str:
        """Return a human-readable cost & TTFT summary."""
        calls = len(self.usage)
        if calls == 0:
            return "Cost: N/A (no LLM calls captured)"

        parts = [
            f"Cost: {format_cost(self.total_cost)}",
            f"{calls} LLM call(s)",
            f"{self.total_input_tokens:,} in / {self.total_output_tokens:,} out tokens",
        ]

        avg = self.avg_ttft
        if avg is not None:
            parts.append(f"TTFT: {self.total_ttft:.1f}s total / {avg:.2f}s avg")

        return " · ".join(parts)

    def reset(self) -> None:
        self.usage.clear()
        self._start_times.clear()


# ─── Error event logging ─────────────────────────────────────────────────--


# ─── Score logging ──────────────────────────────────────────────────────────


def _resolve_langfuse_trace_id(handler: Any) -> Optional[str]:
    """Resolve the current Langfuse trace ID from handler or context.

    Tries, in order:
    1. ``handler.trace_id`` (langfuse v3-style — likely None on v4+)
    2. ``handler.last_trace_id`` (langfuse v4+)
    3. ``Langfuse().get_current_trace_id()`` (langfuse v4+, works
       inside ``propagate_attributes`` context)
    """
    if handler is not None:
        try:
            tid = getattr(handler, "trace_id", None)
            if tid:
                return tid
        except Exception:
            pass
        try:
            tid = getattr(handler, "last_trace_id", None)
            if tid:
                return tid
        except Exception:
            pass

    try:
        from langfuse import Langfuse

        tid = Langfuse().get_current_trace_id()
        if tid:
            return tid
    except Exception:
        pass

    return None


def log_langfuse_score(
    name: str,
    value: float,
    comment: Optional[str] = None,
    trace_id: Optional[str] = None,
    handler: Optional[Any] = None,
) -> None:
    """Log a numeric score to Langfuse, linked to a specific trace.

    Requires Langfuse to be enabled (keys in environment).  The score is
    associated with a trace by ``trace_id``.  If neither ``trace_id`` nor
    a resolvable handler is provided, the score is logged without a trace
    link (appears in the Scores tab but not on the trace detail page).

    Parameters
    ----------
    name : str
        Score name (e.g. ``"verdict_score"``, ``"findings_count"``).
    value : float
        Numeric score value.
    comment : str or None
        Optional human-readable comment.
    trace_id : str or None
        Explicit trace ID.  If ``None``, attempts to resolve via handler.
    handler : Langfuse CallbackHandler or None
        Langfuse ``CallbackHandler`` from which to extract the trace ID.
    """
    if not is_langfuse_enabled():
        return

    resolved_trace_id = trace_id or _resolve_langfuse_trace_id(handler)

    try:
        from langfuse import Langfuse

        lf = Langfuse()
        kwargs: dict[str, Any] = {
            "name": name,
            "value": value,
        }
        if resolved_trace_id:
            kwargs["trace_id"] = resolved_trace_id
        if comment:
            kwargs["comment"] = comment

        lf.create_score(**kwargs)
        lf.flush()
    except Exception as exc:
        logger.debug("Failed to log Langfuse score: %s", exc)


def update_langfuse_trace(
    trace_name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    handler: Optional[Any] = None,
) -> None:
    """No-op in langfuse v4+ — trace attributes must be set upfront
    via ``propagate_attributes()`` (see :func:`langfuse_trace`).

    In langfuse v4 the trace API no longer supports ``update_trace``.
    Trace-level metadata (name, tags, user_id, session_id, metadata)
    must be provided at trace creation time via ``propagate_attributes``.
    Use :func:`log_langfuse_score` instead for post-hoc annotations
    like verdict score or findings count.
    """
    if resolved_trace_id := (trace_id or _resolve_langfuse_trace_id(handler)):
        logger.debug(
            "update_langfuse_trace is a no-op in langfuse v4 (trace=%s). "
            "Set attributes upfront via langfuse_trace() instead.",
            resolved_trace_id,
        )
    # Dedicated score logging is handled by log_langfuse_score() — nothing
    # to do here.


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
