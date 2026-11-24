"""
Generic ONNX Runtime backend.
Supports CUDA, DirectML (DirectX 12 on Windows laptop RTX 3050), TensorRT, and CPU execution providers.
"""
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np

from streaming.backends.base import BaseDetector
from streaming.config import PipelineConfig


class OnnxRuntimeDetector(BaseDetector):
    """Executes exported ONNX detection models via ONNX Runtime."""

    def __init__(self, config: PipelineConfig):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for the ONNX backend. Run: pip install onnxruntime")

        self.config = config
        self.conf_threshold = config.conf_threshold

        # Select the best available execution provider
        available_providers = ort.get_available_providers()
        selected_providers = []

        if config.device != "cpu":
            # Priority: TensorRT -> CUDA -> DirectML (Perfect for Windows RTX 3050)
            if "TensorrtExecutionProvider" in available_providers:
                selected_providers.append("TensorrtExecutionProvider")
            if "CUDAExecutionProvider" in available_providers:
                selected_providers.append("CUDAExecutionProvider")
            if "DmlExecutionProvider" in available_providers:
                selected_providers.append("DmlExecutionProvider")

        selected_providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(config.weights_path),
            sess_options=sess_options,
            providers=selected_providers
        )
        self.actual_provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def predict(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        orig_h, orig_w = image_bgr.shape[:2]
        target_h = self.input_shape[2] if isinstance(self.input_shape[2], int) else 720
        target_w = self.input_shape[3] if isinstance(self.input_shape[3], int) else 1280

        resized = cv2.resize(image_bgr, (target_w, target_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        outputs = self.session.run(None, {self.input_name: blob})
        # Standard detection output parser (boxes, scores, labels)
        detections = []
        if len(outputs) >= 3:
            boxes = outputs[0][0] if outputs[0].ndim == 3 else outputs[0]
            labels = outputs[1][0] if outputs[1].ndim == 2 else outputs[1]
            scores = outputs[2][0] if outputs[2].ndim == 2 else outputs[2]

            scale_x = orig_w / target_w
            scale_y = orig_h / target_h

            for b, s, c in zip(boxes, scores, labels):
                if s >= self.conf_threshold:
                    c_int = int(c)
                    detections.append({
                        "box": [float(b[0] * scale_x), float(b[1] * scale_y),
                                float(b[2] * scale_x), float(b[3] * scale_y)],
                        "score": float(s),
                        "class_id": c_int,
                        "label": f"class_{c_int}"
                    })
        return detections

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "backend": "onnxruntime",
            "provider": self.actual_provider,
            "input_shape": list(self.input_shape)
        }
