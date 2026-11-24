"""
Export utility to convert Faster R-CNN or YOLO models to ONNX format.
Usage:
    python -m streaming.export_onnx --config faster_rcnn/configs/experiment/exp03_fpn_aug.yaml \
                                    --weights faster_rcnn/checkpoints/fasterrcnn_fpn_aug_best.zip \
                                    --output faster_rcnn/checkpoints/fasterrcnn_fpn_aug.onnx
"""
import argparse
from pathlib import Path
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
        dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'}}
    )
    print(f"[✓] Export completed successfully! Saved to: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description="Export PyTorch models to ONNX format")
    p.add_argument("--config", type=Path, required=True, help="Path to Faster R-CNN YAML config")
    p.add_argument("--weights", type=Path, required=True, help="Path to checkpoint (.zip, .pth)")
    p.add_argument("--output", type=Path, required=True, help="Output .onnx file path")
    p.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = p.parse_args()

    export_faster_rcnn_to_onnx(args.config, args.weights, args.output, opset=args.opset)


if __name__ == "__main__":
    main()
