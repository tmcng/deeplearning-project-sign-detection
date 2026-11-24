"""
Metrics collection and statistical analysis for streaming benchmark verification.
"""
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

from streaming.config import FrameMetrics


class MetricsCollector:
    """Collects per-frame performance timings and computes statistical summaries."""

    def __init__(self, output_dir: Path, warmup_frames: int):
        self.output_dir = output_dir
        self.warmup_frames = warmup_frames
        self.records: List[FrameMetrics] = []
        
        self.csv_path = output_dir / "timings.csv"
        self.jsonl_path = output_dir / "detections.jsonl"
        self.summary_path = output_dir / "summary.json"

        self._csv_file = None
        self._csv_writer = None
        self._jsonl_file = None

    def start(self) -> None:
        """Open file handles for persistent telemetry streaming."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=["frame", "warmup", "capture_ms", "infer_ms", "pipeline_ms", "detections"]
        )
        self._csv_writer.writeheader()
        self._jsonl_file = self.jsonl_path.open("w", encoding="utf-8")

    def record_frame(self, frame_id: int, capture_ms: float, infer_ms: float,
                     pipeline_ms: float, detections: List[Dict[str, Any]]) -> None:
        """Log telemetry for a processed frame."""
        is_warmup = (frame_id < self.warmup_frames)
        metric = FrameMetrics(
            frame_id=frame_id,
            is_warmup=is_warmup,
            capture_ms=capture_ms,
            infer_ms=infer_ms,
            pipeline_ms=pipeline_ms,
            detection_count=len(detections)
        )
        self.records.append(metric)

        if self._csv_writer:
            self._csv_writer.writerow(metric.to_csv_row())

        if self._jsonl_file:
            log_record = {
                "frame": frame_id,
                "warmup": is_warmup,
                "detections": detections
            }
            self._jsonl_file.write(json.dumps(log_record, ensure_ascii=False) + "\n")

    def finalize(self, metadata: Dict[str, Any], status: str,
                 error: Optional[str] = None) -> Dict[str, Any]:
        """Close handles, calculate final mean/p50/p95 latency metrics, and write summary.json."""
        if self._csv_file:
            self._csv_file.close()
        if self._jsonl_file:
            self._jsonl_file.close()

        measured = [r for r in self.records if not r.is_warmup]
        summary = dict(metadata)
        summary["status"] = status
        summary["error"] = error
        summary["frames"] = len(self.records)
        summary["measured_frames"] = len(measured)

        for kind in ["infer", "pipeline"]:
            values = [getattr(r, f"{kind}_ms") for r in measured]
            if values:
                summary[f"{kind}_fps"] = float(1000.0 * len(values) / sum(values))
                summary[f"{kind}_latency_ms"] = {
                    "mean": float(np.mean(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                }
            else:
                summary[f"{kind}_fps"] = None
                summary[f"{kind}_latency_ms"] = None

        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
