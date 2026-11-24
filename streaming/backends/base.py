"""
Abstract base class definition for object detectors.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np


class BaseDetector(ABC):
    """Abstract interface defining the contract for detection backends."""

    @abstractmethod
    def predict(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Perform forward inference on a single BGR image.
        
        Returns:
            List of dicts: [
                {"box": [x1, y1, x2, y2], "score": float, "class_id": int, "label": str}, ...
            ]
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return engine metadata for reproducibility and audit logging."""
        pass
