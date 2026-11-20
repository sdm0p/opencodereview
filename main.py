"""Backward-compatible CLI module and launcher."""
import sys as _sys
from backend import main as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    _sys.modules[__name__] = _implementation
