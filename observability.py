"""Backward-compatible import path for telemetry helpers."""
import sys as _sys
from backend import observability as _implementation

_sys.modules[__name__] = _implementation
