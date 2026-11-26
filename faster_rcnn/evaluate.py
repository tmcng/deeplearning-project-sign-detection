# faster_rcnn/evaluate.py
"""Inference Faster R-CNN, xuất file prediction COCO JSON và gọi scripts/evaluate.py để chấm mAP.

Cách chạy:
    # 1. Chấm điểm từ checkpoint model (Inference -> COCO Eval):
    python faster_rcnn/evaluate.py -c faster_rcnn/configs/experiment/exp01_baseline.yaml --split val
    python faster_rcnn/evaluate.py -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml --split test

    # 2. Chấm điểm trực tiếp từ file predictions JSON có sẵn (không cần GPU/PyTorch):
    python faster_rcnn/evaluate.py --pred preds/predictions_val_exp01_baseline.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faster_rcnn import config
from scripts.evaluate import main as run_eval_coco

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from faster_rcnn.dataset import ZaloCOCODataset, collate_fn
    from faster_rcnn.models import build_model

    HAS_TORCH: bool = True
except ImportError:
    HAS_TORCH = False

    class _NoGrad:
        def __call__(self, *args, **kwargs):
            if args and callable(args[0]):
                return args[0]

            def decorator(fn):
                return fn

            return decorator

    torch = type("TorchDummy", (), {"no_grad": _NoGrad()})()  # type: ignore

DEFAULT_CONF_THRESHOLD: float = 0.001
DEFAULT_REPORT_DIR: str = "reports/eval"


def get_parser() -> argparse.ArgumentParser:
    """Tạo parser đối số dòng lệnh cho evaluate."""
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN")
    parser.add_argument("--config", "-c", type=str, default=None, metavar="YAML")
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Custom checkpoint path. Mặc định: best -> last.",
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["val", "test"],
        help="Dataset split để evaluate (mặc định: val).",
    )
    parser.add_argument(
        "--img-dir", type=str, default=None,
        help="Đường dẫn thư mục chứa ảnh (override config, ví dụ: /kaggle/input/<dataset>/images).",
    )
    parser.add_argument(
        "--gt", type=str, default=None,
        help="Đường dẫn file annotation COCO JSON (override split config, ví dụ: /kaggle/input/<dataset>/val.json).",
    )
    parser.add_argument(
        "--pred", type=str, default=None,
        help="File JSON predictions đã có (nếu truyền, bỏ qua inference và chấm điểm ngay).",
    )
    parser.add_argument(
        "--out-pred", type=str, default=None,
        help="Đường dẫn file predictions JSON xuất ra (mặc định theo config: preds/...).",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Tên thí nghiệm ghi vào báo cáo (mặc định lấy theo config _name).",
    )
    parser.add_argument("--conf-thresh", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


@torch.no_grad()
def evaluate_loss(
    model: Any,
    data_loader: Any,
    device: Any,
    dry_run: bool = False,
) -> Dict[str, float]:
    """Tính validation loss (model được chuyển tạm sang train mode để lấy loss_dict).

    Args:
        model: Mô hình PyTorch Faster R-CNN.
        data_loader: DataLoader của tập validation.
        device: CPU hoặc GPU device.
        dry_run: True nếu chỉ chạy 1 batch sanity check.

    Returns:
        Dict chứa loss tổng hợp và từng loss thành phần.
    """
    was_training = model.training
    model.train()

    total_loss = 0.0
    loss_components: Dict[str, float] = {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_rpn_box_reg": 0.0,
        "loss_objectness": 0.0,
    }
    num_batches = 0

    pbar = tqdm(data_loader, desc="[VAL] Val Loss", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        total_loss += losses.item()
        for k in loss_components:
            if k in loss_dict:
                loss_components[k] += loss_dict[k].item()

        num_batches += 1
        pbar.set_postfix(val_loss=f"{losses.item():.4f}")
        if dry_run:
            break

    if not was_training:
        model.eval()

    if num_batches == 0:
        return {"val/loss": 0.0}

    return {
        "val/loss": total_loss / num_batches,
        **{f"val/{k}": v / num_batches for k, v in loss_components.items()},
    }


@torch.no_grad()
def predict(
    model: Any,
    data_loader: Any,
    device: Any,
    conf_thresh: float = DEFAULT_CONF_THRESHOLD,
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
    model.eval()
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

    return results


@torch.no_grad()
def evaluate_coco_metrics(
    model: Any,
    data_loader: Any,
    device: Any,
    gt_path: Union[str, Path],
    conf_thresh: float = DEFAULT_CONF_THRESHOLD,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Inference và chấm metrics COCO bằng pycocotools, trả về dict kết quả mAP.

    Args:
        model: Mô hình PyTorch Faster R-CNN.
        data_loader: DataLoader cho tập đánh giá.
        device: Device tính toán.
        gt_path: Đường dẫn tới ground-truth COCO JSON.
        conf_thresh: Ngưỡng lọc score.
        dry_run: True nếu chạy thử nhanh 1 batch.

    Returns:
        Dict gồm mAP50, mAP50_95, AP_small, AP_medium, AP_large, predictions.
    """
    preds = predict(model, data_loader, device, conf_thresh=conf_thresh, dry_run=dry_run)
    empty_result = {
        "mAP50": 0.0,
        "mAP50_95": 0.0,
        "AP_small": 0.0,
        "AP_medium": 0.0,
        "AP_large": 0.0,
        "predictions": preds,
    }
    if not preds:
        return empty_result

    import pycocotools.coco as coco_mod
    from pycocotools.cocoeval import COCOeval

    from scripts.evaluate import check_preds

    gt = coco_mod.COCO()
    gt.dataset = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    gt.createIndex()

    check_preds(preds, gt)
    dt = gt.loadRes(preds)

    evaluator = COCOeval(gt, dt, iouType="bbox")
    evaluator.params.areaRng = [
        [0, 1e10],
        [0, 32 ** 2],
        [32 ** 2, 96 ** 2],
        [96 ** 2, 1e10],
    ]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    st = evaluator.stats

    return {
        "mAP50": float(st[1]) if st[1] >= 0 else 0.0,
        "mAP50_95": float(st[0]) if st[0] >= 0 else 0.0,
        "AP_small": float(st[3]) if st[3] >= 0 else 0.0,
        "AP_medium": float(st[4]) if st[4] >= 0 else 0.0,
        "AP_large": float(st[5]) if st[5] >= 0 else 0.0,
        "predictions": preds,
    }


def _invoke_coco_evaluator(
    gt_path: Union[str, Path],
    pred_path: Union[str, Path],
    exp_name: str,
    report_dir: Union[str, Path],
) -> None:
    """Gọi runner đánh giá scripts/evaluate.py một cách an toàn."""
    eval_argv = [
        "evaluate.py",
        "--gt", str(gt_path),
        "--pred", str(pred_path),
        "--name", str(exp_name),
        "--out", str(report_dir),
    ]
    sys_argv_backup = sys.argv
    try:
        sys.argv = eval_argv
        run_eval_coco()
    finally:
        sys.argv = sys_argv_backup


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


def _evaluate_from_pred_file(args: argparse.Namespace) -> None:
    """Chế độ 1: Đánh giá mAP trực tiếp từ file predictions JSON có sẵn."""
    pred_path = Path(args.pred)
    if not pred_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file prediction: '{pred_path}'")

    exp_name = args.name
    if not exp_name:
        stem = pred_path.stem
        for prefix in ["predictions_val_", "predictions_test_", "predictions_"]:
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        exp_name = stem

    report_dir = DEFAULT_REPORT_DIR
    gt_path = args.gt if args.gt else f"data/splits/{args.split}.json"

    if args.config:
        cfg = config.load(args.config)
        if not args.gt:
            gt_path = cfg["VAL_JSON_PATH"] if args.split == "val" else cfg["TEST_JSON_PATH"]
        report_dir = cfg.get("REPORT_EVAL_DIR", report_dir)

    print("=" * 54)
    print(f"  EVALUATE TỪ FILE PREDICTIONS: {pred_path}")
    print(f"  Thí nghiệm: {exp_name} | split={args.split}")
    print(f"  Ground truth: {gt_path}")
    print("=" * 54)

    _invoke_coco_evaluator(gt_path, pred_path, exp_name, report_dir)

    print("=" * 54)
    print("  EVALUATE HOÀN TẤT")
    print(f"  [OK] File Predictions: {os.path.abspath(pred_path)}")
    print(f"  [OK] Báo cáo chi tiết: {os.path.abspath(report_dir)}/{exp_name}.json")
    print(f"  [OK] Bảng tổng hợp:    {os.path.abspath(report_dir)}/results.md")
    print("=" * 54)


def _evaluate_from_model(args: argparse.Namespace) -> None:
    """Chế độ 2: Tải model checkpoint, inference và chấm điểm."""
    if not HAS_TORCH:
        raise ImportError(
            "Cần cài đặt PyTorch để chạy model inference.\n"
            "Nếu bạn đã có sẵn file predictions JSON (ví dụ tải từ Kaggle về), "
            "hãy chạy với cờ --pred: python faster_rcnn/evaluate.py --pred <đường_dẫn_file.json>"
        )

    cli_overrides: Dict[str, Any] = {}
    if args.batch_size is not None:
        cli_overrides.setdefault("trainer", {})["batch_size"] = args.batch_size
    if args.dry_run:
        cli_overrides.setdefault("trainer", {})["dry_run"] = True
    if args.img_dir:
        cli_overrides.setdefault("paths", {}).setdefault("data", {})["img_dir"] = args.img_dir
    if args.gt:
        target_key = "val_json" if args.split == "val" else "test_json"
        cli_overrides.setdefault("paths", {}).setdefault("data", {})[target_key] = args.gt

    cfg = config.load(args.config, cli_overrides or None)

    exp_name: str = args.name or cfg.get("_name", "unnamed")
    is_dry_run: bool = cfg["DRY_RUN"]
    conf_thresh: float = (
        args.conf_thresh if args.conf_thresh is not None else cfg["CONF_THRESHOLD"]
    )

    print("=" * 54)
    print(f"  EVALUATE: {exp_name.upper()} | split={args.split}")
    print("=" * 54)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    model = build_model(cfg)
    weights_path = _resolve_checkpoint_path(cfg, args.weights)

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    gt_path = args.gt if args.gt else (cfg["VAL_JSON_PATH"] if args.split == "val" else cfg["TEST_JSON_PATH"])
    img_dir = args.img_dir if args.img_dir else cfg["IMG_DIR"]
    dataset = ZaloCOCODataset(gt_path, img_dir, is_train=True, use_augment=False)
    loader = DataLoader(
        dataset,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=0 if is_dry_run else cfg["NUM_WORKERS"],
        collate_fn=collate_fn,
    )
    print(f"[DATASET] {args.split.upper()} — {len(dataset)} ảnh")

    # 1. Validation Loss (chỉ tính khi đánh giá tập val)
    if args.split == "val":
        loss_metrics = evaluate_loss(model, loader, device, dry_run=is_dry_run)
        print(f"[VAL LOSS] {loss_metrics['val/loss']:.4f}")
        for k, v in loss_metrics.items():
            if k != "val/loss":
                print(f"  {k}: {v:.4f}")
    else:
        print("[TEST MODE] Bỏ qua bước tính loss, bắt đầu inference lấy predictions & chấm mAP.")

    # 2. Inference & lưu predictions JSON
    preds = predict(model, loader, device, conf_thresh=conf_thresh, dry_run=is_dry_run)
    if args.out_pred:
        pred_path = args.out_pred
    else:
        pred_path = cfg["OUTPUT_PRED_JSON"]
        if args.split == "test" and "_val_" in pred_path:
            pred_path = pred_path.replace("_val_", "_test_")

    os.makedirs(os.path.dirname(os.path.abspath(pred_path)), exist_ok=True)
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)
    print(f"[OUTPUT] Predictions: {pred_path}")

    if is_dry_run:
        print("[DRY-RUN] Hoàn tất sanity check.")
        return

    # 3. Chấm điểm COCO: gọi hàm main của scripts/evaluate.py
    report_dir = cfg["REPORT_EVAL_DIR"]
    _invoke_coco_evaluator(gt_path, pred_path, exp_name, report_dir)

    print("=" * 54)
    print("  EVALUATE HOÀN TẤT")
    print(f"  [OK] File Predictions: {os.path.abspath(pred_path)}")
    print(f"  [OK] Báo cáo chi tiết: {os.path.abspath(report_dir)}/{exp_name}.json")
    print(f"  [OK] Bảng tổng hợp:    {os.path.abspath(report_dir)}/results.md")
    print("=" * 54)


def main() -> None:
    """Hàm điều phối đánh giá chính."""
    cli_parser = get_parser()
    args = cli_parser.parse_args()

    if args.pred:
        _evaluate_from_pred_file(args)
    else:
        _evaluate_from_model(args)


if __name__ == "__main__":
    main()
