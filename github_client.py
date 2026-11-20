"""Backward-compatible import path for GitHub API helpers."""
import sys as _sys
from backend.integrations import github_client as _implementation

_sys.modules[__name__] = _implementation
