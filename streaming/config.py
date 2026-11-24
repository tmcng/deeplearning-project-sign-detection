"""
Configuration and data containers for the streaming inference pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class Detection:
    """Represents a single object detection."""
    box: List[float]  # [x1, y1, x2, y2]
    score: float
    class_id: int
    label: str
    track_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "box": self.box,
            "score": round(self.score, 4),
            "class_id": self.class_id,
            "label": self.label
        }
        if self.track_id is not None:
            data["track_id"] = self.track_id
        return data


@dataclass
class FrameMetrics:
    """Timing and performance metrics for a single processed frame."""
    frame_id: int
    is_warmup: bool
    capture_ms: float
    infer_ms: float
    pipeline_ms: float
    detection_count: int

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "frame": self.frame_id,
            "warmup": int(self.is_warmup),
            "capture_ms": self.capture_ms,
            "infer_ms": self.infer_ms,
            "pipeline_ms": self.pipeline_ms,
            "detections": self.detection_count,
        }


@dataclass
class PipelineConfig:
    """Runtime configuration for streaming video inference."""
    model_type: str  # 'faster-rcnn', 'yolo', 'onnx'
    weights_path: Path
    output_dir: Path
    config_path: Optional[Path] = None
    source: str = "rtsp://localhost:8554/live/traffic"
    device: str = "cpu"
    conf_threshold: float = 0.5
    max_frames: int = 300
    warmup_frames: int = 10
    headless: bool = False
    save_video: bool = False
    enable_tracking: bool = True
    read_timeout_ms: int = 5000
    threads: int = 4
    use_fp16: bool = False
    use_compile: bool = False
    roi: Optional[Tuple[float, float, float, float]] = None
    imgsz: int = 1280
