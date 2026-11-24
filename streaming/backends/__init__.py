"""Detection backends package."""
from streaming.backends.base import BaseDetector
from streaming.backends.factory import DetectorFactory

__all__ = ["BaseDetector", "DetectorFactory"]
