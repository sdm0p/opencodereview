"""Backward-compatible import path for configuration helpers."""
import sys as _sys
from backend.integrations import config as _implementation

_sys.modules[__name__] = _implementation
