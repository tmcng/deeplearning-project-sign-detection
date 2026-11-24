"""
Factory for creating object detector instances based on runtime configuration.
"""
from pathlib import Path
from streaming.backends.base import BaseDetector
from streaming.config import PipelineConfig


class DetectorFactory:
    """Instantiates the appropriate detection backend."""

    @staticmethod
    def create(config: PipelineConfig) -> BaseDetector:
        model_type = config.model_type.lower()
        suffix = config.weights_path.suffix.lower()

        # Route by explicit model_type or file extension
        if suffix in [".onnx", ".engine"] and model_type != "yolo":
            from streaming.backends.onnx_backend import OnnxRuntimeDetector
            return OnnxRuntimeDetector(config)

        if model_type == "yolo" or suffix in [".pt", ".engine"] and model_type != "faster-rcnn":
            from streaming.backends.yolo_backend import YoloDetector
            return YoloDetector(config)

        if model_type == "faster-rcnn":
            from streaming.backends.pytorch_frcnn import FasterRCNNPyTorchDetector
            return FasterRCNNPyTorchDetector(config)

        raise ValueError(
            f"Unsupported model type '{config.model_type}' or weights extension '{suffix}'. "
            "Supported: 'faster-rcnn' (.zip, .pth), 'yolo' (.pt, .engine), 'onnx' (.onnx)."
        )
