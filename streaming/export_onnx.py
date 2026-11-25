"""
Export utility to convert Faster R-CNN or YOLO models to ONNX format.

Usage:
    # 1. Faster R-CNN:
    python -m streaming.export_onnx --config faster_rcnn/configs/experiment/exp03_fpn_aug.yaml \
                                    --weights faster_rcnn/checkpoints/fasterrcnn_fpn_aug_best.zip \
                                    --output faster_rcnn/checkpoints/fasterrcnn_fpn_aug.onnx

    # 2. YOLO:
    python -m streaming.export_onnx --model yolo \
                                    --weights yolov8s.pt \
                                    --output yolov8s.onnx
"""
import argparse
from pathlib import Path
import shutil
import torch
import torchvision.models.detection as detection
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.misc import FrozenBatchNorm2d

from faster_rcnn.configs.config_loader import load_config


def export_faster_rcnn_to_onnx(config_path: Path, weights_path: Path, output_path: Path,
                               height: int = 720, width: int = 1280, opset: int = 14):
    print(f"[*] Loading Faster R-CNN config from: {config_path}")
    raw_cfg = load_config(config_path)
    m_cfg = raw_cfg['model']

    raw_sizes = m_cfg['rpn']['anchor_sizes']
    anchor_sizes = tuple((s,) if isinstance(s, (int, float)) else tuple(s) for s in raw_sizes)
    raw_ratios = m_cfg['rpn']['aspect_ratios']
    if isinstance(raw_ratios[0], (int, float)):
        aspect_ratios = tuple(tuple(raw_ratios) for _ in anchor_sizes)
    else:
        aspect_ratios = tuple(map(tuple, raw_ratios))

    arch_name = m_cfg['architecture']
    factory = getattr(detection, arch_name)
    model = factory(
        weights=None,
        weights_backbone=None,
        num_classes=raw_cfg['NUM_CLASSES'],
        rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
        min_size=m_cfg['transform']['min_size'],
        max_size=m_cfg['transform']['max_size']
    )

    if arch_name == 'fasterrcnn_resnet50_fpn':
        for name, child in list(model.backbone.named_children()):
            if isinstance(child, torch.nn.BatchNorm2d):
                setattr(model.backbone, name, FrozenBatchNorm2d(child.num_features, eps=0.0))

    print(f"[*] Loading state dict from: {weights_path}")
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy_input = [torch.rand(3, height, width, dtype=torch.float32)]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Exporting ONNX model to: {output_path} (opset {opset})...")
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['boxes', 'labels', 'scores'],
        dynamic_axes={'images': {1: 'height', 2: 'width'}},
        dynamo=False
    )
    print(f"[✓] Export completed successfully! Saved to: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def export_yolo_to_onnx(weights_path: Path, output_path: Path = None, opset: int = 14,
                        imgsz: int = 640, dynamic: bool = True):
    from ultralytics import YOLO

    print(f"[*] Loading YOLO model from: {weights_path}")
    model = YOLO(str(weights_path))
    print(f"[*] Exporting YOLO model to ONNX (opset {opset}, imgsz={imgsz}, dynamic={dynamic})...")
    exported_file = model.export(format="onnx", opset=opset, dynamic=dynamic, imgsz=imgsz)
    exported_path = Path(exported_file)

    if output_path and exported_path.resolve() != output_path.resolve():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported_path), str(output_path))
        print(f"[✓] Moved exported ONNX to: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"[✓] Export completed successfully! Saved to: {exported_path} ({exported_path.stat().st_size / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description="Export PyTorch models (Faster R-CNN / YOLO) to ONNX format")
    p.add_argument("--model", type=str, choices=["auto", "faster-rcnn", "yolo"], default="auto",
                   help="Model architecture: 'faster-rcnn', 'yolo', or 'auto' (default: auto)")
    p.add_argument("--weights", type=Path, required=True, help="Path to checkpoint (.zip, .pth, .pt)")
    p.add_argument("--config", type=Path, default=None, help="Path to Faster R-CNN YAML config (required for faster-rcnn)")
    p.add_argument("--output", type=Path, default=None, help="Output .onnx file path")
    p.add_argument("--opset", type=int, default=14, help="ONNX opset version (default: 14)")
    p.add_argument("--imgsz", type=int, default=640, help="Image size for YOLO export (default: 640)")
    p.add_argument("--no-dynamic", action="store_true", help="Disable dynamic batch/resolution for YOLO")
    args = p.parse_args()

    model_type = args.model
    if model_type == "auto":
        if args.weights.suffix == ".pt" or "yolo" in args.weights.name.lower():
            model_type = "yolo"
        else:
            model_type = "faster-rcnn"

    if model_type == "faster-rcnn":
        if args.config is None:
            p.error("--config is required when exporting Faster R-CNN models.")
        if args.output is None:
            p.error("--output is required when exporting Faster R-CNN models.")
        export_faster_rcnn_to_onnx(args.config, args.weights, args.output, opset=args.opset)
    elif model_type == "yolo":
        export_yolo_to_onnx(args.weights, args.output, opset=args.opset, imgsz=args.imgsz,
                            dynamic=not args.no_dynamic)


if __name__ == "__main__":
    main()
