# faster_rcnn/models.py
# Xây dựng Faster R-CNN từ config dict (YAML-driven).

import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


def build_model(cfg: dict) -> torch.nn.Module:
    """Xây dựng Faster R-CNN từ flat config dict."""
    model_cfg   = cfg.get("model", {})
    num_classes = cfg.get("NUM_CLASSES", 8)
    arch        = model_cfg.get("architecture", "fasterrcnn_resnet50_fpn")

    backbone_cfg     = model_cfg.get("backbone", {})
    trainable_layers = backbone_cfg.get("trainable_layers", 0)

    rpn_cfg      = model_cfg.get("rpn", {})
    raw_sizes    = rpn_cfg.get("anchor_sizes", [[32], [64], [128], [256], [512]])
    if raw_sizes and isinstance(raw_sizes[0], (int, float)):
        anchor_sizes = tuple((s,) for s in raw_sizes)
    else:
        anchor_sizes = tuple(tuple(s) for s in raw_sizes)

    raw_ratios = rpn_cfg.get("aspect_ratios", [[0.5, 1.0, 2.0]] * len(anchor_sizes))
    if raw_ratios and isinstance(raw_ratios[0], (int, float)):
        aspect_ratios = tuple((tuple(raw_ratios),) * len(anchor_sizes))
    else:
        aspect_ratios = tuple(tuple(r) for r in raw_ratios)

    rpn_anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    transform_cfg = model_cfg.get("transform", {})
    min_size = transform_cfg.get("min_size", 800)
    max_size = transform_cfg.get("max_size", 1333)

    if arch == "fasterrcnn_resnet50_fpn":
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            trainable_backbone_layers=trainable_layers,
            rpn_anchor_generator=rpn_anchor_generator,
            min_size=min_size,
            max_size=max_size,
        )
    elif arch == "fasterrcnn_resnet50_fpn_v2":
        try:
            from torchvision.models.detection import (
                fasterrcnn_resnet50_fpn_v2,
                FasterRCNN_ResNet50_FPN_V2_Weights,
            )
            model = fasterrcnn_resnet50_fpn_v2(
                weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                trainable_backbone_layers=trainable_layers,
                rpn_anchor_generator=rpn_anchor_generator,
                min_size=min_size,
                max_size=max_size,
            )
        except ImportError:
            raise ImportError("fasterrcnn_resnet50_fpn_v2 yêu cầu torchvision >= 0.14.")
    else:
        raise ValueError(f"Kiến trúc không được hỗ trợ: '{arch}'")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

