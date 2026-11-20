"""Backward-compatible import path for GitHub authentication helpers."""
import sys as _sys
from backend.integrations import auth as _implementation

_sys.modules[__name__] = _implementation
