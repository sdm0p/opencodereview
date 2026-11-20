"""Backward-compatible import path for model endpoint configuration."""
import sys as _sys
from backend.integrations import endpoints as _implementation

_sys.modules[__name__] = _implementation
