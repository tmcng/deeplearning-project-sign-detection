"""
Real-time streaming module for Traffic Sign Detection.
Provides pipeline, trackers, visualizers, and multi-backend engine wrappers.
"""
from streaming.config import PipelineConfig, Detection, FrameMetrics
from streaming.tracker import IoUTracker
from streaming.visualizer import StreamVisualizer
from streaming.pipeline import StreamingInferencePipeline
from streaming.backends.factory import DetectorFactory

__all__ = [
    "PipelineConfig",
    "Detection",
    "FrameMetrics",
    "IoUTracker",
    "StreamVisualizer",
    "StreamingInferencePipeline",
    "DetectorFactory",
]
