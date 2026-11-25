"""
End-to-end video streaming inference pipeline.
Orchestrates frame capture, ROI cropping, model prediction, tracking, visualization, and metrics export.
"""
import hashlib
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch

from streaming.backends.factory import DetectorFactory
from streaming.config import PipelineConfig
from streaming.metrics import MetricsCollector
from streaming.tracker import IoUTracker
from streaming.visualizer import StreamVisualizer


def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class StreamingInferencePipeline:
    """High-performance streaming engine for dashcam sign detection."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tracker = IoUTracker() if config.enable_tracking else None
        self.visualizer = StreamVisualizer()
        self.metrics = MetricsCollector(config.output_dir, config.warmup_frames)

    def _sync_cuda(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def run(self) -> int:
        """Execute the streaming loop. Returns exit status code (0 for success)."""
        torch.set_num_threads(self.config.threads)
        self.metrics.start()

        # Build metadata dictionary conforming to verification audit standard
        metadata = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "arguments": {
                "model": self.config.model_type,
                "weights": str(self.config.weights_path),
                "config": str(self.config.config_path) if self.config.config_path else None,
                "source": self.config.source,
                "device": self.config.device,
                "imgsz": self.config.imgsz,
                "conf": self.config.conf_threshold,
                "max_frames": self.config.max_frames,
                "warmup": self.config.warmup_frames,
                "output_dir": str(self.config.output_dir),
                "headless": self.config.headless,
                "save_video": self.config.save_video,
                "track": self.config.enable_tracking,
                "read_timeout_ms": self.config.read_timeout_ms,
                "threads": self.config.threads,
                "fp16": self.config.use_fp16,
                "compile": self.config.use_compile,
                "roi": list(self.config.roi) if self.config.roi else None,
            },
            "weights_sha256": get_file_sha256(self.config.weights_path),
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "device": str(self.device),
            "threads": self.config.threads,
            "gpu": torch.cuda.get_device_name(self.device) if (self.device.type == "cuda" and torch.cuda.is_available()) else None,
            "precision": "float16" if self.config.use_fp16 else "float32",
            "batch_size": 1,
            "roi": list(self.config.roi) if self.config.roi else None,
            "timing_contract": (
                "infer_ms: adapter preprocessing, transfer, inference, postprocessing, CPU results; "
                "pipeline_ms: capture through draw, optional encode/display and detection log, excludes CSV write; "
                "startup and warmup excluded. Sequential capture, no latest-frame dropping; no glass-to-glass latency claim."
            )
        }

        try:
            import torchvision
            metadata["torchvision"] = torchvision.__version__
        except ImportError:
            metadata["torchvision"] = None

        try:
            metadata["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            metadata["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            metadata["git_revision"] = None
            metadata["git_dirty"] = False

        metadata["implementation_sha256"] = get_file_sha256(Path(__file__))
        metadata["source_sha256"] = get_file_sha256(Path(self.config.source)) if Path(self.config.source).is_file() else None

        cap = None
        writer = None
        status = "completed"
        error_msg = None

        try:
            # 1. Instantiate detector backend
            detector = DetectorFactory.create(self.config)
            metadata.update(detector.get_metadata())

            # 2. Configure video capture (with TCP transport for RTSP stability)
            is_network = self.config.source.lower().startswith(("rtsp://", "http://", "https://"))
            if is_network and self.config.source.lower().startswith("rtsp://"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

            if is_network:
                cap = cv2.VideoCapture(
                    self.config.source,
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.config.read_timeout_ms,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.config.read_timeout_ms
                    ]
                )
            else:
                cap = cv2.VideoCapture(self.config.source)

            if not cap.isOpened():
                raise RuntimeError("Cannot open source")

            source_fps = cap.get(cv2.CAP_PROP_FPS)
            metadata["source_fps"] = source_fps if np.isfinite(source_fps) else None

            # 3. Processing loop
            with torch.inference_mode():
                for frame_id in range(self.config.max_frames):
                    self._sync_cuda()
                    t_frame_start = time.perf_counter()

                    ok, frame = cap.read()
                    t_captured = time.perf_counter()

                    if not ok:
                        if is_network:
                            raise RuntimeError("Stream read failed or timed out; stopping without stale frames")
                        status = "eof"
                        break

                    # Save resolution metadata on first valid frame
                    if frame_id == 0:
                        metadata["source_dimensions"] = [frame.shape[1], frame.shape[0]]

                    # Handle ROI Cropping
                    if self.config.roi:
                        h, w = frame.shape[:2]
                        rx1 = int(self.config.roi[0] * w)
                        ry1 = int(self.config.roi[1] * h)
                        rx2 = int(self.config.roi[2] * w)
                        ry2 = int(self.config.roi[3] * h)
                        crop_img = frame[ry1:ry2, rx1:rx2]
                    else:
                        rx1, ry1 = 0, 0
                        crop_img = frame

                    # Model forward pass
                    self._sync_cuda()
                    t_infer_start = time.perf_counter()
                    detections = detector.predict(crop_img)
                    self._sync_cuda()
                    t_infer_end = time.perf_counter()
                    infer_ms = 1000.0 * (t_infer_end - t_infer_start)

                    # Map coordinates back if ROI was used
                    if self.config.roi:
                        for d in detections:
                            d["box"] = [
                                d["box"][0] + rx1, d["box"][1] + ry1,
                                d["box"][2] + rx1, d["box"][3] + ry1
                            ]

                    # Object Tracking
                    if self.tracker:
                        self.tracker.update(detections, frame_id)

                    # Rendering
                    if self.config.roi:
                        self.visualizer.draw_roi(frame, (rx1, ry1, rx2, ry2))

                    self.visualizer.draw_detections(frame, detections, has_tracking=bool(self.tracker))

                    mode_desc = f"{'FP16' if self.config.use_fp16 else 'FP32'}{', ROI' if self.config.roi else ''}"
                    self.visualizer.draw_hud(frame, self.config.model_type, mode_desc, 1000.0 / max(infer_ms, 0.001), frame_id)

                    # Snapshot
                    if frame_id == 0:
                        snapshot_path = self.config.output_dir / "snapshot.jpg"
                        if not cv2.imwrite(str(snapshot_path), frame):
                            raise RuntimeError("Snapshot write failed")

                    # Recording
                    if self.config.save_video:
                        if writer is None:
                            fps_val = source_fps if (np.isfinite(source_fps) and source_fps > 0) else 25
                            writer = cv2.VideoWriter(
                                str(self.config.output_dir / "demo.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                fps_val,
                                (frame.shape[1], frame.shape[0])
                            )
                            if not writer.isOpened():
                                raise RuntimeError("Video writer failed")
                        writer.write(frame)

                    # UI Display
                    user_quit = False
                    if not self.config.headless:
                        cv2.imshow("Traffic Sign Detection Demo", frame)
                        user_quit = (cv2.waitKey(1) & 0xFF == ord("q"))

                    pipeline_ms = 1000.0 * (time.perf_counter() - t_frame_start)
                    capture_ms = 1000.0 * (t_captured - t_frame_start)

                    # Telemetry logging
                    self.metrics.record_frame(
                        frame_id=frame_id,
                        capture_ms=capture_ms,
                        infer_ms=infer_ms,
                        pipeline_ms=pipeline_ms,
                        detections=detections
                    )

                    if user_quit:
                        status = "user_stopped"
                        break

            if len([r for r in self.metrics.records if not r.is_warmup]) == 0:
                raise RuntimeError("No measured frames after warmup; increase source length or reduce warmup")

        except Exception as exc:
            status = "failed"
            error_msg = f"{type(exc).__name__}: {exc}"

        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            if not self.config.headless:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

            summary = self.metrics.finalize(metadata, status, error_msg)
            print(
                f"{{\n"
                f'  "status": "{summary["status"]}",\n'
                f'  "error": {("null" if not summary["error"] else repr(summary["error"]))},\n'
                f'  "frames": {summary["frames"]},\n'
                f'  "measured_frames": {summary["measured_frames"]},\n'
                f'  "infer_fps": {summary["infer_fps"]},\n'
                f'  "pipeline_fps": {summary["pipeline_fps"]}\n'
                f"}}"
            )

        return 1 if error_msg else 0
