"""
PyTorch backend for Faster R-CNN with ResNet-50 FPN.
Supports FP16 AMP, torch.compile, and CUDA/CPU synchronization.
"""
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
import torch
import torchvision.models.detection as detection
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.misc import FrozenBatchNorm2d

from faster_rcnn.configs.config_loader import load_config
from streaming.backends.base import BaseDetector
from streaming.config import PipelineConfig


class FasterRCNNPyTorchDetector(BaseDetector):
    """Faster R-CNN detector implemented on top of native PyTorch."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.conf_threshold = config.conf_threshold

        # Load YAML architecture specification
        if not config.config_path or not config.config_path.is_file():
            raise FileNotFoundError(f"Config path does not exist: {config.config_path}")
        self.raw_cfg = load_config(config.config_path)
        m_cfg = self.raw_cfg['model']

        # Parse Anchor Generator params
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
            num_classes=self.raw_cfg['NUM_CLASSES'],
            rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
            min_size=m_cfg['transform']['min_size'],
            max_size=m_cfg['transform']['max_size']
        )

        if arch_name == 'fasterrcnn_resnet50_fpn':
            # Restore FrozenBatchNorm2d convention from COCO_V1 pretraining
            def freeze_bn(module):
                for name, child in list(module.named_children()):
                    if isinstance(child, torch.nn.BatchNorm2d):
                        setattr(module, name, FrozenBatchNorm2d(child.num_features, eps=0.0))
                    else:
                        freeze_bn(child)
            freeze_bn(model.backbone)

        # Load weights
        state = torch.load(config.weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)

        if config.use_compile and hasattr(torch, 'compile'):
            model = torch.compile(model)

        model.to(self.device).eval()
        self.model = model
        self.class_names = self.raw_cfg['CLASS_NAMES']

    def predict(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device, dtype=torch.float32) / 255.0

        if self.config.use_fp16 and self.device.type == 'cuda':
            autocast_ctx = (
                torch.amp.autocast('cuda', dtype=torch.float16)
                if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast')
                else torch.cuda.amp.autocast(dtype=torch.float16)
            )
            with autocast_ctx:
                outputs = self.model([tensor])[0]
        else:
            outputs = self.model([tensor])[0]

        boxes = outputs['boxes'].cpu().tolist()
        scores = outputs['scores'].cpu().tolist()
        labels = outputs['labels'].cpu().tolist()

        detections = []
        for b, s, c in zip(boxes, scores, labels):
            if s >= self.conf_threshold:
                c_int = int(c)
                detections.append({
                    "box": b,
                    "score": float(s),
                    "class_id": c_int,
                    "label": self.class_names[c_int]
                })
        return detections

    def get_metadata(self) -> Dict[str, Any]:
        with open(self.config.config_path, 'rb') as f:
            cfg_sha256 = hashlib.sha256(f.read()).hexdigest()
        return {
            "experiment": self.raw_cfg['_name'],
            "model_config": self.raw_cfg['model'],
            "class_id_convention": "COCO category indices 1..7 (0 background)",
            "config_sha256": cfg_sha256
        }
