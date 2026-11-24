"""Create chat models with Cloudflare, Groq, Gemini, and custom endpoints."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Any, Optional
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter

logger = logging.getLogger(__name__)

# Gemini 3.1 Flash model name as of 2026-07
# Verified via Google AI documentation: gemini-3.1-flash-lite is the correct name.
# The base 'gemini-3.1-flash' does not exist in the API.
GEMINI_MODEL = "gemini-3.1-flash-lite"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
CLOUDFLARE_MODEL = os.getenv("CLOUDFLARE_MODEL", "@cf/zai-org/glm-4.7-flash")
class CloudflareRateLimiter(BaseRateLimiter):
    """Conservative per-process pacing and rolling daily attempt budget."""

    def __init__(self, daily_limit: int = 50):
        self.pacer = InMemoryRateLimiter(requests_per_second=0.2, check_every_n_seconds=0.1, max_bucket_size=1)
        self.daily_limit = daily_limit
        self.attempts = deque()
        self.lock = threading.Lock()

    def _reserve(self):
        now = time.monotonic()
        with self.lock:
            while self.attempts and self.attempts[0] <= now - 86400:
                self.attempts.popleft()
            if len(self.attempts) >= self.daily_limit:
                raise ValueError("Cloudflare local daily request budget reached; try later or choose another provider.")
            self.attempts.append(now)

    def acquire(self, *, blocking: bool = True) -> bool:
        if not self.pacer.acquire(blocking=blocking):
            return False
        self._reserve()
        return True

    async def aacquire(self, *, blocking: bool = True) -> bool:
        if not await self.pacer.aacquire(blocking=blocking):
            return False
        self._reserve()
        return True


# Shared by reviewers and evaluation in this process. This request budget is
# a conservative application cap, not a measurement of Cloudflare neurons.
_cloudflare_limiter = CloudflareRateLimiter(max(1, int(os.getenv("CLOUDFLARE_DAILY_REQUEST_LIMIT", "50"))))
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REQUEST_TIMEOUT = float(os.getenv("OCR_MODEL_TIMEOUT_SECONDS", "60"))
DEFAULT_OUTPUT_TOKENS = int(os.getenv("OCR_MODEL_MAX_OUTPUT_TOKENS", "2000"))


def _get_gemini_chat(**kwargs: Any) -> Optional[Any]:
    """Create a ``ChatGoogleGenerativeAI`` instance if ``GEMINI_API_KEY`` is set.

    Returns ``None`` silently when the key is missing or the package is not
    installed — callers are expected to fall back.
    """
    kwargs.setdefault("timeout", DEFAULT_REQUEST_TIMEOUT)
    kwargs.setdefault("max_retries", 0)
    kwargs.setdefault("max_tokens", DEFAULT_OUTPUT_TOKENS)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=kwargs.pop("model", GEMINI_MODEL),
            temperature=kwargs.pop("temperature", DEFAULT_TEMPERATURE),
            google_api_key=api_key,
            **kwargs,
        )
    except ImportError:
        logger.warning("langchain-google-genai not installed — skipping Gemini")
        return None
    except Exception as exc:
        logger.warning("Failed to initialise Gemini: %s", exc)
        return None


def _get_groq_chat(**kwargs: Any) -> Optional[Any]:
    """Create a ``ChatGroq`` instance if ``GROQ_API_KEY`` is set."""
    kwargs.setdefault("timeout", DEFAULT_REQUEST_TIMEOUT)
    kwargs.setdefault("max_retries", 0)
    kwargs.setdefault("max_tokens", DEFAULT_OUTPUT_TOKENS)
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=kwargs.pop("model", GROQ_MODEL),
            temperature=kwargs.pop("temperature", DEFAULT_TEMPERATURE),
            api_key=api_key,
            **kwargs,
        )
    except ImportError:
        logger.warning("langchain-groq not installed")
        return None
    except Exception as exc:
        logger.warning("Failed to initialise Groq: %s", exc)
        return None


def _get_cloudflare_chat(**kwargs: Any) -> Optional[Any]:
    """Use Workers AI when an account ID and API token are configured."""
    api_key = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    if not api_key or not account_id:
        return None
    from langchain_openai import ChatOpenAI
    import httpx
    timeout = kwargs.pop("timeout", DEFAULT_REQUEST_TIMEOUT)
    if isinstance(timeout, (int, float)):
        timeout = httpx.Timeout(timeout, connect=min(float(timeout), 8.0))
    # Retry only failures while establishing a connection, before an HTTP
    # request is sent. Preserve normal TLS certificate and hostname checks.
    # IPv6 is unavailable on the hosted Space; keep DNS and TLS SNI intact.
    kwargs.setdefault("http_client", httpx.Client(transport=httpx.HTTPTransport(local_address="0.0.0.0", retries=2), timeout=timeout))
    kwargs.setdefault("http_async_client", httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(local_address="0.0.0.0", retries=2), timeout=timeout))
    kwargs["timeout"] = timeout
    kwargs.setdefault("max_retries", 0)
    kwargs["max_tokens"] = max(int(kwargs.get("max_tokens") or DEFAULT_OUTPUT_TOKENS), 4096)
    kwargs.setdefault("reasoning_effort", "low")
    kwargs.setdefault("rate_limiter", _cloudflare_limiter)
    return ChatOpenAI(model=kwargs.pop("model", CLOUDFLARE_MODEL), api_key=api_key,
                      base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
                      temperature=kwargs.pop("temperature", DEFAULT_TEMPERATURE), **kwargs)


def create_endpoint_llm(endpoint, **kwargs: Any) -> Any:
    """Create a chat model from a user-configured custom endpoint.

    Supports OpenAI-compatible APIs (DeepSeek, OpenRouter, vLLM, Ollama,
    Groq, Together, ...), Anthropic Claude, and Google Gemini.

    Parameters
    ----------
    endpoint : EndpointConfig
        The user's endpoint config (name, provider, api_key, base_url, model).
    **kwargs
        Passed through to the underlying LangChain constructor (e.g.
        ``temperature``).  ``model`` / ``api_key`` / ``base_url`` come from
        the endpoint and cannot be overridden here.

    Returns
    -------
    Any
        A LangChain chat model instance (``ChatOpenAI``, ``ChatAnthropic``
        or ``ChatGoogleGenerativeAI``).

    Raises
    ------
    ValueError
        If the endpoint's provider package is not installed or the config
        is incomplete.
    """
    if not endpoint.api_key or not endpoint.model:
        raise ValueError(
            f"Endpoint '{endpoint.name}' is missing an API key or model name."
        )

    kwargs.setdefault("timeout", DEFAULT_REQUEST_TIMEOUT)
    kwargs.setdefault("max_retries", 0)
    kwargs.setdefault("max_tokens", DEFAULT_OUTPUT_TOKENS)
    provider = endpoint.provider
    model = endpoint.model
    # MiniMax M3.1 requires reasoning and OpenRouter defaults to max effort.
    # Reserve room for reasoning + visible JSON rather than truncating at 1200.
    if 'minimax-m3.1' in model.lower():
        kwargs['max_tokens'] = max(int(kwargs.get('max_tokens') or 0), 8192)
        if endpoint.base_url and 'openrouter.ai' in endpoint.base_url:
            extra = dict(kwargs.get('extra_body') or {})
            extra['reasoning'] = {'effort': 'low'}
            kwargs['extra_body'] = extra
    temperature = kwargs.pop("temperature", DEFAULT_TEMPERATURE)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ValueError(
                "langchain-anthropic is not installed — cannot use endpoint "
                f"'{endpoint.name}'. Run: pip install langchain-anthropic"
            ) from None
        params: dict[str, Any] = {
            "model": model,
            "api_key": endpoint.api_key,
            "temperature": temperature,
        }
        if endpoint.base_url:
            params["base_url"] = endpoint.base_url
        params.update(kwargs)
        return ChatAnthropic(**params)

    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ValueError(
                "langchain-google-genai is not installed — cannot use endpoint "
                f"'{endpoint.name}'. Run: pip install langchain-google-genai"
            ) from None
        params = {
            "model": model,
            "google_api_key": endpoint.api_key,
            "temperature": temperature,
        }
        params.update(kwargs)
        return ChatGoogleGenerativeAI(**params)

    # Default: OpenAI-compatible
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ValueError(
            "langchain-openai is not installed — cannot use endpoint "
            f"'{endpoint.name}'. Run: pip install langchain-openai"
        ) from None
    params = {
        "model": model,
        "api_key": endpoint.api_key,
        "temperature": temperature,
    }
    if endpoint.base_url:
        params["base_url"] = endpoint.base_url
        if "api.cloudflare.com/client/v4/accounts/" in endpoint.base_url.lower():
            kwargs.setdefault("rate_limiter", _cloudflare_limiter)
    params.update(kwargs)
    return ChatOpenAI(**params)


def create_llm(endpoint=None, **kwargs: Any) -> Any:
    """Create a chat LLM — selected endpoint first, else Cloudflare → Groq → Gemini.

    Parameters
    ----------
    endpoint : EndpointConfig or str or None
        A custom endpoint to run on (``EndpointConfig`` object or its name).
        ``None`` (default) falls back to the built-in providers.
    **kwargs
        Passed through to the underlying LangChain constructor.
        Common overrides: ``model``, ``temperature``.

    Returns
    -------
    Any
        A LangChain chat model instance (``ChatOpenAI``/``ChatAnthropic``/
        ``ChatGoogleGenerativeAI`` from a custom endpoint, or the built-in
        ``ChatGoogleGenerativeAI`` / ``ChatGroq``).

    Raises
    ------
    ValueError
        If a custom endpoint is named but not found, or no provider key is set.
    """
    if endpoint:
        from backend.integrations.endpoints import EndpointConfig, get_endpoint

        # The browser historically sent lowercase builtin ids while the
        # endpoint registry uses display names. Accept both for API clients.
        if isinstance(endpoint, str):
            endpoint = {"gemini": "Gemini", "groq": "Grok", "cloudflare": "Cloudflare"}.get(endpoint.lower(), endpoint)
        ep = endpoint if isinstance(endpoint, EndpointConfig) else get_endpoint(endpoint)
        if ep is not None:
            if getattr(ep, "builtin", False):
                if ep.name == "Cloudflare":
                    llm = _get_cloudflare_chat(**kwargs)
                    if llm is not None:
                        return llm
                    raise ValueError("Cloudflare selected but CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN is not set.")
                # Built-in providers use their native constructors so the
                # behaviour is identical to the default (no-endpoint) path.
                if ep.name == "Gemini":
                    llm = _get_gemini_chat(**kwargs)
                    if llm is not None:
                        logger.info(
                            "Using built-in Gemini endpoint (model=%s)",
                            kwargs.get("model", GEMINI_MODEL),
                        )
                        return llm
                    raise ValueError(
                        "Gemini selected but GEMINI_API_KEY is not set."
                    )
                if ep.name == "Grok":
                    llm = _get_groq_chat(**kwargs)
                    if llm is not None:
                        logger.info(
                            "Using built-in Grok endpoint (model=%s)",
                            kwargs.get("model", GROQ_MODEL),
                        )
                        return llm
                    raise ValueError(
                        "Grok selected but GROQ_API_KEY is not set."
                    )
            try:
                llm = create_endpoint_llm(ep, **kwargs)
                logger.info("Using custom endpoint %s (%s/%s)", ep.name, ep.provider, ep.model)
                return llm
            except ValueError:
                raise
        else:
            raise ValueError(
                f"Custom endpoint '{endpoint}' not found. "
                "Configure it via OCR_ENDPOINT_* environment variables "
                "or add it in the UI."
            )

    llm = _get_cloudflare_chat(**kwargs)
    if llm is not None:
        return llm

    # Prefer Groq when configured: it has a separate quota pool and avoids
    # stalling the review pipeline on Gemini free-tier exhaustion.
    llm = _get_groq_chat(**kwargs)
    if llm is not None:
        logger.debug("Using Groq (model=%s)", kwargs.get("model", GROQ_MODEL))
        return llm

    llm = _get_gemini_chat(**kwargs)
    if llm is not None:
        logger.debug("Falling back to Gemini (model=%s)", kwargs.get("model", GEMINI_MODEL))
        return llm

    raise ValueError(
        "No LLM available. Set GEMINI_API_KEY (primary) or GROQ_API_KEY (fallback)."
    )
