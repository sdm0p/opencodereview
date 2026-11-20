"""Backward-compatible import path for the LangGraph workflow."""
import sys as _sys
from backend import graph as _implementation

_sys.modules[__name__] = _implementation
