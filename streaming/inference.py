"""
CLI Entrypoint for real-time video detection and telemetry logging.
Usage:
    python -m streaming.inference [OPTIONS]
    or:
    python -m streaming [OPTIONS]
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

from streaming.config import PipelineConfig
from streaming.pipeline import StreamingInferencePipeline
from streaming.tracker import IoUTracker


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m streaming.inference",
        description="High-Performance Real-Time Video Streaming & Telemetry Benchmark CLI"
    )
    p.add_argument("--model", choices=["faster-rcnn", "yolo", "onnx"], required=True,
                   help="Detector backend family")
    p.add_argument("--weights", type=Path, required=True,
                   help="Path to checkpoint (.zip, .pth, .pt, .onnx, .engine)")
    p.add_argument("--config", type=Path,
                   help="Required for Faster R-CNN; must point to experiment YAML")
    p.add_argument("--source", default="rtsp://localhost:8554/live/traffic",
                   help="Video path or RTSP stream URL")
    p.add_argument("--device", default="cpu",
                   help="Execution device: 'cpu' or 'cuda:0'")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Image input size for YOLO models")
    p.add_argument("--conf", type=float, default=0.5,
                   help="Confidence score threshold")
    p.add_argument("--max-frames", type=int, default=300,
                   help="Maximum frames to process (including warmup)")
    p.add_argument("--warmup", type=int, default=10,
                   help="Warmup frame count excluded from statistical benchmark")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory (must be empty or newly created to preserve audit trails)")
    p.add_argument("--headless", action="store_true",
                   help="Disable OpenCV GUI window rendering")
    p.add_argument("--save-video", action="store_true",
                   help="Save processed output stream to demo.mp4")
    p.add_argument("--track", action="store_true",
                   help="Enable greedy class-aware IoU tracking")
    p.add_argument("--read-timeout-ms", type=int, default=5000,
                   help="Network read timeout in milliseconds")
    p.add_argument("--threads", type=int, default=4,
                   help="Torch CPU thread pool limit")
    p.add_argument("--fp16", action="store_true",
                   help="Enable Half-Precision (FP16 AMP) on CUDA")
    p.add_argument("--compile", action="store_true",
                   help="Enable PyTorch 2.x JIT graph compilation (torch.compile)")
    p.add_argument("--roi", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
                   help="Normalized ROI coordinates [0.0 - 1.0]: x1 y1 x2 y2")
    p.add_argument("--roi-preset", choices=["none", "right", "hood-cut"], default="right",
                   help="Preset ROI: 'right' (x:0.35->1.0, y:0.0->0.75), 'hood-cut' (x:0.0->1.0, y:0.0->0.75). Default: 'right'")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Validate arguments early
    if not args.weights.is_file():
        parser.error(f"Checkpoint does not exist; no automatic model download: {args.weights}")

    if args.model == "faster-rcnn" and (not args.config or not args.config.is_file()):
        parser.error("--config must point to a valid Faster R-CNN YAML configuration file")

    if not (0.0 <= args.conf <= 1.0):
        parser.error("Confidence threshold must be between 0.0 and 1.0")

    if not (0 <= args.warmup < args.max_frames):
        parser.error("Require 0 <= warmup < max-frames")

    if min(args.threads, args.imgsz, args.read_timeout_ms) <= 0:
        parser.error("threads, imgsz and read-timeout-ms must be positive")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be new or empty to preserve earlier evidence")

    # Resolve ROI
    roi_coords = None
    if args.roi:
        roi_coords = tuple(args.roi)
    elif args.roi_preset == "right":
        roi_coords = (0.35, 0.0, 1.0, 0.75)
    elif args.roi_preset == "hood-cut":
        roi_coords = (0.0, 0.0, 1.0, 0.75)

    config = PipelineConfig(
        model_type=args.model,
        weights_path=args.weights,
        output_dir=args.output_dir,
        config_path=args.config,
        source=args.source,
        device=args.device,
        conf_threshold=args.conf,
        max_frames=args.max_frames,
        warmup_frames=args.warmup,
        headless=args.headless,
        save_video=args.save_video,
        enable_tracking=args.track,
        read_timeout_ms=args.read_timeout_ms,
        threads=args.threads,
        use_fp16=args.fp16,
        use_compile=args.compile,
        roi=roi_coords,
        imgsz=args.imgsz
    )

    pipeline = StreamingInferencePipeline(config)
    return pipeline.run()


if __name__ == "__main__":
    raise SystemExit(main())
