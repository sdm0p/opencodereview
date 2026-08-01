"""LLM factory — creates LangChain chat models with primary + fallback support.

Priority
--------
1. **Google Gemini** (primary) — uses ``GEMINI_API_KEY`` env var.
   Model: ``gemini-3.1-flash-lite`` — generous free tier.
2. **Groq** (fallback) — uses ``GROQ_API_KEY`` env var.
   Model: ``llama-3.3-70b-versatile`` — fast inference, lower free limit.

If neither key is set, functions raise ``ValueError`` so callers can skip
gracefully (as they already do for Groq-only today).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Gemini 3.1 Flash model name as of 2026-07
# Verified via Google AI documentation: gemini-3.1-flash-lite is the correct name.
# The base 'gemini-3.1-flash' does not exist in the API.
GEMINI_MODEL = "gemini-3.1-flash-lite"
GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.0


def _get_gemini_chat(**kwargs: Any) -> Optional[Any]:
    """Create a ``ChatGoogleGenerativeAI`` instance if ``GEMINI_API_KEY`` is set.

    Returns ``None`` silently when the key is missing or the package is not
    installed — callers are expected to fall back.
    """
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

    provider = endpoint.provider
    model = endpoint.model
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
    params.update(kwargs)
    return ChatOpenAI(**params)


def create_llm(endpoint=None, **kwargs: Any) -> Any:
    """Create a chat LLM — custom endpoint first, else Gemini → Groq.

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
        from endpoints import EndpointConfig, get_endpoint

        ep = endpoint if isinstance(endpoint, EndpointConfig) else get_endpoint(endpoint)
        if ep is not None:
            try:
                llm = create_endpoint_llm(ep, **kwargs)
                logger.info("Using custom endpoint %s (%s/%s)", ep.name, ep.provider, ep.model)
                return llm
            except ValueError:
                raise
        else:
            raise ValueError(
                f"Custom endpoint '{endpoint}' not found. "
                "Configure it via OCR_ENDPOINT_* environment variables."
            )

    llm = _get_gemini_chat(**kwargs)
    if llm is not None:
        logger.debug("Using Gemini (model=%s)", kwargs.get("model", GEMINI_MODEL))
        return llm

    llm = _get_groq_chat(**kwargs)
    if llm is not None:
        logger.debug("Falling back to Groq (model=%s)", kwargs.get("model", GROQ_MODEL))
        return llm

    raise ValueError(
        "No LLM available. Set GEMINI_API_KEY (primary) or GROQ_API_KEY (fallback)."
    )
