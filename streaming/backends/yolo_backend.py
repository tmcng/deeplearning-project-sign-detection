"""
YOLO Backend wrapper supporting Ultralytics PyTorch (.pt), ONNX (.onnx), and TensorRT (.engine).
"""
from typing import List, Dict, Any
import numpy as np

from streaming.backends.base import BaseDetector
from streaming.config import PipelineConfig


class YoloDetector(BaseDetector):
    """YOLO detector backed by Ultralytics runtime."""

    def __init__(self, config: PipelineConfig):
        from ultralytics import YOLO
        import ultralytics

        self.config = config
        self.ultralytics_version = ultralytics.__version__
        self.model = YOLO(str(config.weights_path))
        self.conf_threshold = config.conf_threshold
        self.imgsz = config.imgsz
        self.device = config.device
        self.half = config.use_fp16 and (config.device != 'cpu')

    def predict(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.predict(
            image_bgr,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            device=self.device,
            half=self.half,
            verbose=False
        )[0]

        detections = []
        boxes = results.boxes.xyxy.cpu().tolist()
        scores = results.boxes.conf.cpu().tolist()
        classes = results.boxes.cls.cpu().tolist()

        for b, s, c in zip(boxes, scores, classes):
            c_int = int(c)
            label = str(self.model.names[c_int]) if hasattr(self.model, 'names') else f"class_{c_int}"
            detections.append({
                "box": b,
                "score": float(s),
                "class_id": c_int,
                "label": label
            })
        return detections

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "ultralytics": self.ultralytics_version,
            "imgsz": self.imgsz,
            "class_id_convention": "YOLO class indices (0..6)",
            "mode": "standard full-frame"
        }
