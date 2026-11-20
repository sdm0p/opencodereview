"""Backward-compatible import path for LangGraph state models."""
import sys as _sys
from backend import state as _implementation

_sys.modules[__name__] = _implementation
