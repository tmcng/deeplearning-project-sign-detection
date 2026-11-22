"""Batch-one video inference with auditable timings. Run: python -m streaming.demo --help."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['faster-rcnn', 'yolo'], required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--config', type=Path, help='Required for Faster R-CNN; must match training.')
    p.add_argument('--source', default='rtsp://localhost:8554/live/traffic')
    p.add_argument('--device', default='cpu', help='cpu or cuda:0')
    p.add_argument('--imgsz', type=int, default=1280, help='YOLO inference size')
    p.add_argument('--conf', type=float, default=0.3)
    p.add_argument('--max-frames', type=int, default=300, help='Includes warmup; finite by default')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--output-dir', type=Path, required=True, help='New or empty directory')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--save-video', action='store_true')
    p.add_argument('--track', action='store_true', help='Simple class-aware IoU association, not DeepSORT')
    p.add_argument('--read-timeout-ms', type=int, default=5000)
    p.add_argument('--threads', type=int, default=4, help='Torch CPU threads')
    return p


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class IoUTracker:
    """Greedy, class-aware one-to-one association; expires after 10 processed frames."""
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    @staticmethod
    def iou(a, b):
        area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
        return area / union if union > 0 else 0

    def update(self, detections, frame_id):
        self.tracks = {k: v for k, v in self.tracks.items() if frame_id-v[1] <= 10}
        available = set(self.tracks)
        for d in sorted(detections, key=lambda x: x['score'], reverse=True):
            candidates = [(self.iou(d['box'], self.tracks[k][0]['box']), k)
                          for k in available if self.tracks[k][0]['class_id'] == d['class_id']]
            score, key = max(candidates, default=(0, -1))
            if score < 0.3:
                key = self.next_id
                self.next_id += 1
            else:
                available.remove(key)
            d['track_id'] = key
            self.tracks[key] = (d.copy(), frame_id)


def load_detector(args, torch):
    import cv2
    if args.model == 'yolo':
        from ultralytics import YOLO
        import ultralytics
        model = YOLO(str(args.weights))
        def predict(frame):
            r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device,
                              half=False, verbose=False)[0]
            return [dict(box=b, score=s, class_id=int(c), label=str(model.names[int(c)]))
                    for b, s, c in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(),
                                       r.boxes.cls.cpu().tolist())]
        return predict, {'ultralytics': ultralytics.__version__, 'imgsz': args.imgsz,
                         'class_id_convention': 'checkpoint YOLO indices (usually 0..6)',
                         'mode': 'standard full-frame, no SAHI'}
    from faster_rcnn.configs.config_loader import load_config
    import torchvision.models.detection as detection
    from torchvision.models.detection.rpn import AnchorGenerator
    cfg = load_config(args.config)
    m = cfg['model']
    sizes = m['rpn']['anchor_sizes']
    sizes = tuple((s,) if isinstance(s, (int, float)) else tuple(s) for s in sizes)
    ratios = m['rpn']['aspect_ratios']
    ratios = tuple(tuple(ratios) for _ in sizes) if isinstance(ratios[0], (int, float)) else tuple(map(tuple, ratios))
    factory = getattr(detection, m['architecture'])
    # A complete checkpoint replaces all parameters: never download pretrained weights.
    model = factory(weights=None, weights_backbone=None, num_classes=cfg['NUM_CLASSES'],
                    rpn_anchor_generator=AnchorGenerator(sizes, ratios),
                    min_size=m['transform']['min_size'], max_size=m['transform']['max_size'])
    if m['architecture'] == 'fasterrcnn_resnet50_fpn':
        # Original COCO_V1 construction uses frozen BN and overwrite_eps(..., 0).
        # The weights=None constructor uses regular BN; restore the trained architecture.
        from torchvision.ops.misc import FrozenBatchNorm2d
        def freeze_bn(module):
            for name, child in list(module.named_children()):
                if isinstance(child, torch.nn.BatchNorm2d):
                    setattr(module, name, FrozenBatchNorm2d(child.num_features, eps=0.0))
                else:
                    freeze_bn(child)
        freeze_bn(model.backbone)
    state = torch.load(args.weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()
    names = cfg['CLASS_NAMES']
    def predict(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(args.device, dtype=torch.float32) / 255
        pred = model([tensor])[0]
        return [dict(box=b, score=s, class_id=int(c), label=names[int(c)])
                for b, s, c in zip(pred['boxes'].cpu().tolist(), pred['scores'].cpu().tolist(),
                                   pred['labels'].cpu().tolist()) if s >= args.conf]
    return predict, {'experiment': cfg['_name'], 'model_config': m,
                     'class_id_convention': 'COCO category indices 1..7 (0 background)',
                     'config_sha256': sha256(args.config)}


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not args.weights.is_file():
        p.error('Checkpoint does not exist; no automatic model download.')
    if args.model == 'faster-rcnn' and (not args.config or not args.config.is_file()):
        p.error('--config must point to the matching Faster R-CNN experiment YAML')
    if not 0 <= args.conf <= 1 or not 0 <= args.warmup < args.max_frames:
        p.error('Require 0 <= conf <= 1 and 0 <= warmup < max-frames')
    if min(args.threads, args.imgsz, args.read_timeout_ms) <= 0:
        p.error('threads, imgsz and read-timeout-ms must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('output-dir must be new or empty to preserve earlier evidence')
    import cv2
    import numpy as np
    import torch
    import torchvision
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable')
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {'started_utc': datetime.now(timezone.utc).isoformat(),
            'command': sys.argv, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'weights_sha256': sha256(args.weights), 'python': sys.version, 'platform': platform.platform(),
            'processor': platform.processor(), 'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'opencv': cv2.__version__, 'device': str(device), 'threads': args.threads,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'precision': 'float32', 'batch_size': 1,
            'timing_contract': 'infer_ms: adapter preprocessing, transfer, inference, postprocessing, CPU results; pipeline_ms: capture through draw, optional encode/display and detection log, excludes CSV write; startup and warmup excluded. Sequential capture, no latest-frame dropping; no glass-to-glass latency claim.'}
    try:
        meta['git_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        meta['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        meta['git_revision'] = None
    meta['implementation_sha256'] = sha256(__file__)
    meta['source_sha256'] = sha256(args.source) if Path(args.source).is_file() else None
    cap = None
    writer = None
    rows = []
    status = 'completed'
    error = None
    try:
        predict, extra = load_detector(args, torch)
        meta.update(extra)
        network = args.source.lower().startswith(('rtsp://', 'http://', 'https://'))
        if network and args.source.lower().startswith('rtsp://'):
            import os
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, args.read_timeout_ms,
                                cv2.CAP_PROP_READ_TIMEOUT_MSEC, args.read_timeout_ms]) if network else cv2.VideoCapture(args.source)
        if not cap.isOpened():
            raise RuntimeError('Cannot open source')
        fps = cap.get(cv2.CAP_PROP_FPS)
        meta['source_fps'] = fps if np.isfinite(fps) else None
        tracker = IoUTracker() if args.track else None
        with (args.output_dir/'detections.jsonl').open('w', encoding='utf-8') as log, \
                (args.output_dir/'timings.csv').open('w', newline='', encoding='utf-8') as timing:
            csv_writer = csv.DictWriter(timing, fieldnames=['frame', 'warmup', 'capture_ms', 'infer_ms', 'pipeline_ms', 'detections'])
            csv_writer.writeheader()
            with torch.inference_mode():
                for frame_id in range(args.max_frames):
                    sync()
                    start = time.perf_counter()
                    ok, frame = cap.read()
                    captured = time.perf_counter()
                    if not ok:
                        if network:
                            raise RuntimeError('Stream read failed or timed out; stopping without stale frames')
                        status = 'eof'
                        break
                    sync()
                    infer_start = time.perf_counter()
                    detections = predict(frame)
                    sync()
                    inferred = time.perf_counter()
                    infer_ms = 1000 * (inferred-infer_start)
                    if tracker:
                        tracker.update(detections, frame_id)
                    for d in detections:
                        x1, y1, x2, y2 = map(round, d['box'])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                        # OpenCV's default font cannot display Vietnamese; IDs remain authoritative.
                        label = f"class {d['class_id']} {d['score']:.2f}" + (f" ID {d['track_id']}" if tracker else '')
                        cv2.putText(frame, label, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 0), 1)
                    cv2.putText(frame, f'{args.model} | inference {1000/infer_ms:.2f} FPS | frame {frame_id}',
                                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
                    if frame_id == 0:
                        meta['source_dimensions'] = [frame.shape[1], frame.shape[0]]
                        if not cv2.imwrite(str(args.output_dir/'snapshot.jpg'), frame):
                            raise RuntimeError('Snapshot write failed')
                    if args.save_video:
                        if writer is None:
                            writer = cv2.VideoWriter(str(args.output_dir/'demo.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                                     fps if np.isfinite(fps) and fps > 0 else 25,
                                                     (frame.shape[1], frame.shape[0]))
                            if not writer.isOpened():
                                raise RuntimeError('Video writer failed')
                        writer.write(frame)
                    log.write(json.dumps({'frame': frame_id, 'warmup': frame_id < args.warmup,
                                          'detections': detections}, ensure_ascii=False) + '\n')
                    quit_requested = False
                    if not args.headless:
                        cv2.imshow('Traffic sign detection', frame)
                        quit_requested = cv2.waitKey(1) & 0xff == ord('q')
                    row = dict(frame=frame_id, warmup=int(frame_id < args.warmup),
                               capture_ms=1000*(captured-start), infer_ms=infer_ms,
                               pipeline_ms=1000*(time.perf_counter()-start), detections=len(detections))
                    rows.append(row)
                    csv_writer.writerow(row)
                    if quit_requested:
                        status = 'user_stopped'
                        break
        if not any(not r['warmup'] for r in rows):
            raise RuntimeError('No measured frames after warmup; increase source length/max-frames or reduce warmup')
    except Exception as exc:
        status, error = 'failed', f'{type(exc).__name__}: {exc}'
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        measured = [r for r in rows if not r['warmup']]
        summary = dict(meta, status=status, error=error, frames=len(rows), measured_frames=len(measured))
        for name in ['infer', 'pipeline']:
            values = [r[name+'_ms'] for r in measured]
            summary[name+'_fps'] = 1000*len(values)/sum(values) if values else None
            summary[name+'_latency_ms'] = {'mean': float(np.mean(values)), 'p50': float(np.percentile(values, 50)),
                                           'p95': float(np.percentile(values, 95))} if values else None
        (args.output_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k: summary[k] for k in ['status', 'error', 'frames', 'measured_frames', 'infer_fps', 'pipeline_fps']}, indent=2))
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
