# faster_rcnn/models.py
"""Xây dựng mô hình Faster R-CNN từ cấu hình YAML (YAML-driven architecture builder)."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

# Các hằng số mặc định
DEFAULT_ARCHITECTURE: str = "fasterrcnn_resnet50_fpn"
DEFAULT_NUM_CLASSES: int = 8
DEFAULT_MIN_SIZE: int = 800
DEFAULT_MAX_SIZE: int = 1333
DEFAULT_ANCHOR_SIZES: Tuple[Tuple[int, ...], ...] = ((32,), (64,), (128,), (256,), (512,))
DEFAULT_ASPECT_RATIOS: Tuple[Tuple[float, ...], ...] = ((0.5, 1.0, 2.0),) * 5


def _normalize_anchor_sizes(
    raw_sizes: Optional[Sequence[Union[int, float, Sequence[Union[int, float]]]]]
) -> Tuple[Tuple[int, ...], ...]:
    """Chuẩn hóa kích thước anchor sizes thành tuple lồng nhau theo chuẩn torchvision AnchorGenerator."""
    if not raw_sizes:
        return DEFAULT_ANCHOR_SIZES

    first_item = raw_sizes[0]
    if isinstance(first_item, (int, float)):
        return tuple((int(s),) for s in raw_sizes)

    return tuple(tuple(int(s) for s in sublist) for sublist in raw_sizes)  # type: ignore


def _normalize_aspect_ratios(
    raw_ratios: Optional[Sequence[Union[float, Sequence[float]]]],
    num_scales: int,
) -> Tuple[Tuple[float, ...], ...]:
    """Chuẩn hóa tỉ lệ anchor aspect ratios cho từng pyramid level."""
    if not raw_ratios:
        return ((0.5, 1.0, 2.0),) * num_scales

    first_item = raw_ratios[0]
    if isinstance(first_item, (int, float)):
        return tuple((tuple(float(r) for r in raw_ratios),) * num_scales)

    return tuple(tuple(float(r) for r in sublist) for sublist in raw_ratios)  # type: ignore


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Xây dựng mô hình Faster R-CNN từ flat config dict.

    Args:
        cfg: Dict cấu hình tổng hợp chứa các section 'model', 'NUM_CLASSES', v.v.

    Returns:
        Mô hình PyTorch Faster R-CNN đã tùy biến FastRCNNPredictor theo NUM_CLASSES.

    Raises:
        ValueError: Nếu kiến trúc mô hình không được hỗ trợ.
        ImportError: Nếu torchvision phiên bản hiện tại không hỗ trợ kiến trúc yêu cầu.
    """
    model_cfg: Dict[str, Any] = cfg.get("model", {})
    num_classes: int = cfg.get("NUM_CLASSES", DEFAULT_NUM_CLASSES)
    arch: str = model_cfg.get("architecture", DEFAULT_ARCHITECTURE)

    backbone_cfg: Dict[str, Any] = model_cfg.get("backbone", {})
    trainable_layers: int = backbone_cfg.get("trainable_layers", 0)

    rpn_cfg: Dict[str, Any] = model_cfg.get("rpn", {})
    raw_sizes = rpn_cfg.get("anchor_sizes", None)
    anchor_sizes = _normalize_anchor_sizes(raw_sizes)

    raw_ratios = rpn_cfg.get("aspect_ratios", None)
    aspect_ratios = _normalize_aspect_ratios(raw_ratios, len(anchor_sizes))

    rpn_anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    transform_cfg: Dict[str, Any] = model_cfg.get("transform", {})
    min_size: int = transform_cfg.get("min_size", DEFAULT_MIN_SIZE)
    max_size: int = transform_cfg.get("max_size", DEFAULT_MAX_SIZE)
    box_score_thresh: float = cfg.get("CONF_THRESHOLD", 0.001)

    if arch == "fasterrcnn_resnet50_fpn":
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            trainable_backbone_layers=trainable_layers,
            rpn_anchor_generator=rpn_anchor_generator,
            min_size=min_size,
            max_size=max_size,
            box_score_thresh=box_score_thresh,
        )
    elif arch == "fasterrcnn_resnet50_fpn_v2":
        try:
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_V2_Weights,
                fasterrcnn_resnet50_fpn_v2,
            )

            model = fasterrcnn_resnet50_fpn_v2(
                weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                trainable_backbone_layers=trainable_layers,
                rpn_anchor_generator=rpn_anchor_generator,
                min_size=min_size,
                max_size=max_size,
                box_score_thresh=box_score_thresh,
            )
        except ImportError as exc:
            raise ImportError(
                "Kiến trúc 'fasterrcnn_resnet50_fpn_v2' yêu cầu torchvision >= 0.14."
            ) from exc
    else:
        raise ValueError(
            f"Kiến trúc không được hỗ trợ: '{arch}'. "
            "Các kiến trúc hợp lệ: ['fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2']"
        )

    # Thay thế đầu ra (box predictor) để khớp với số lớp của bài toán nhận diện biển báo
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def resolve_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    return torch.device(requested)
