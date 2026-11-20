"""Backward-compatible import path for model provider setup."""
import sys as _sys
from backend.integrations import llm_factory as _implementation

_sys.modules[__name__] = _implementation
