"""
Top-level module execution point.
Allows running the streaming pipeline directly via:
    python -m streaming [OPTIONS]
"""
from streaming.inference import main

if __name__ == "__main__":
    raise SystemExit(main())
