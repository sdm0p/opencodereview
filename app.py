"""Backward-compatible import path and launcher for the Gradio UI."""
import sys as _sys
from backend import app as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    _sys.modules[__name__] = _implementation
