"""LLM factory — creates LangChain chat models with primary + fallback support.

Priority
--------
1. **Google Gemini** (primary) — uses ``GEMINI_API_KEY`` env var.
   Model: ``gemini-2.0-flash`` — generous free tier (~1 500 req/day).
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

GEMINI_MODEL = "gemini-3.1-flash"
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


def create_llm(**kwargs: Any) -> Any:
    """Create a chat LLM with Gemini primary → Groq fallback.

    Parameters
    ----------
    **kwargs
        Passed through to the underlying LangChain constructor.
        Common overrides: ``model``, ``temperature``.

    Returns
    -------
    Any
        A LangChain chat model instance (``ChatGoogleGenerativeAI`` or
        ``ChatGroq``).

    Raises
    ------
    ValueError
        If neither ``GEMINI_API_KEY`` nor ``GROQ_API_KEY`` is set.
    """
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
