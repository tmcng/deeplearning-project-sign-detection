"""
Backward-compatibility wrapper.
Recommended professional entrypoints:
    python -m streaming.inference [OPTIONS]
    python -m streaming [OPTIONS]
"""
from streaming.inference import build_arg_parser, main
from streaming.tracker import IoUTracker

__all__ = ["build_arg_parser", "main", "IoUTracker"]

if __name__ == "__main__":
    raise SystemExit(main())
