"""Backward-compatible Uvicorn entry point for the combined server."""
import sys as _sys
from backend import combined_server as _implementation

_sys.modules[__name__] = _implementation
