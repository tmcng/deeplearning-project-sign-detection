"""Inference CLI: checkpoint and images -> COCO predictions JSON."""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple, Optional
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faster_rcnn.configs.config_loader import load_config


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config")
    parser.add_argument("--weights")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--gt", help="COCO JSON containing image IDs and filenames")
    parser.add_argument("--img-dir")
    parser.add_argument("--out-pred")
    parser.add_argument("--conf", "--conf-thresh", dest="conf_thresh", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def _resolve_checkpoint_path(cfg: Dict[str, Any], custom_weights: Optional[str]) -> str:
    """Tìm đường dẫn checkpoint theo thứ tự: custom -> best (.pth/.zip) -> last (.pth/.zip)."""
    if custom_weights:
        return custom_weights

    candidates: List[Tuple[str, str]] = [
        (cfg["MODEL_BEST_WEIGHTS"], "Best checkpoint"),
        (cfg["MODEL_BEST_WEIGHTS"].replace(".pth", ".zip"), "Best checkpoint (.zip)"),
        (cfg["MODEL_WEIGHTS"], "Last checkpoint"),
        (cfg["MODEL_WEIGHTS"].replace(".pth", ".zip"), "Last checkpoint (.zip)"),
    ]

    for path_str, label in candidates:
        if os.path.isfile(path_str):
            print(f"[WEIGHTS] {label}: {path_str}")
            return path_str

    raise FileNotFoundError(
        f"Không tìm thấy checkpoint:\n"
        f"  best: {cfg['MODEL_BEST_WEIGHTS']} (hoặc .zip)\n"
        f"  last: {cfg['MODEL_WEIGHTS']} (hoặc .zip)\n"
        f"Hãy chạy train.py trước hoặc chỉ định --weights."
    )

def predict(
    model: Any,
    data_loader: Any,
    device: Any,
    conf_thresh: float = 0.001,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Inference và trích xuất danh sách predictions theo định dạng chuẩn COCO JSON.

    Args:
        model: Mô hình PyTorch Faster R-CNN.
        data_loader: DataLoader chứa ảnh cần inference.
        device: CPU hoặc GPU device.
        conf_thresh: Ngưỡng lọc confidence score tối thiểu.
        dry_run: True nếu chỉ chạy sanity check 1 batch.

    Returns:
        Danh sách các dictionary dự đoán theo chuẩn COCO (image_id, category_id, bbox, score).
    """
    import torch
    from tqdm import tqdm

    model.eval()
    if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "score_thresh"):
        model.roi_heads.score_thresh = conf_thresh
    results: List[Dict[str, Any]] = []

    pbar = tqdm(data_loader, desc="[PREDICT] Inference", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        predictions = model(images)

        for pred, target in zip(predictions, targets):
            img_id = target["image_id"] if isinstance(target, dict) else target
            img_id = int(img_id.item()) if isinstance(img_id, torch.Tensor) else int(img_id)

            boxes = pred["boxes"].detach().cpu().numpy()
            labels = pred["labels"].detach().cpu().numpy()
            scores = pred["scores"].detach().cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                if score >= conf_thresh:
                    xmin, ymin, xmax, ymax = box
                    results.append({
                        "image_id": img_id,
                        "category_id": int(label),
                        "bbox": [
                            float(xmin),
                            float(ymin),
                            float(xmax - xmin),
                            float(ymax - ymin),
                        ],
                        "score": float(score),
                    })

        if dry_run:
            break

    unique_images = len(set(r["image_id"] for r in results))
    total_images = len(data_loader.dataset) if hasattr(data_loader, "dataset") else "?"
    print(f"[PREDICT] Hoàn tất inference: {len(results)} boxes trên {unique_images}/{total_images} ảnh (ngưỡng conf >= {conf_thresh})")
    return results

def main():
    args = get_parser().parse_args()
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from faster_rcnn.cli import put
    from faster_rcnn.dataset import ZaloCOCODataset, collate_fn
    from faster_rcnn.models import build_model, resolve_device

    overrides = {}
    for key, value in [
        ("trainer.batch_size", args.batch_size),
        ("trainer.num_workers", args.num_workers),
        ("trainer.dry_run", args.dry_run),
        ("paths.data.img_dir", args.img_dir),
        (f"paths.data.{args.split}_json", args.gt),
        ("data.conf_threshold", args.conf_thresh),
    ]:
        if value is not None:
            put(overrides, key, value)
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError("--set requires KEY=VALUE")
        put(overrides, key, yaml.safe_load(value))
    cfg = load_config(args.config, overrides)
    device = resolve_device(args.device)
    model = build_model(cfg)
    model.load_state_dict(torch.load(_resolve_checkpoint_path(cfg, args.weights), map_location=device))
    model.to(device)
    dataset = ZaloCOCODataset(
        cfg[f"{args.split.upper()}_JSON_PATH"], cfg["IMG_DIR"], is_train=False,
    )
    loader = DataLoader(
        dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False,
        num_workers=0 if cfg["DRY_RUN"] else cfg["NUM_WORKERS"], collate_fn=collate_fn,
    )
    with torch.no_grad():
        predictions = predict(model, loader, device, cfg["CONF_THRESHOLD"], cfg["DRY_RUN"])
    default_path = cfg["OUTPUT_PRED_JSON"]
    if args.split == "test":
        default_path = default_path.replace("_val_", "_test_")
    path = Path(args.out_pred or default_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(f"Saved {len(predictions)} predictions: {path}")


if __name__ == "__main__":
    main()
