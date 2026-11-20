"""Backward-compatible import path for the packaged FastAPI application."""
import sys as _sys
from backend import api as _implementation

_sys.modules[__name__] = _implementation
