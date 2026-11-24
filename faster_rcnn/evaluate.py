# faster_rcnn/evaluate.py
# Inference Faster R-CNN, xuất file prediction và gọi scripts/evaluate.py để chấm điểm.

import os
import sys
import json
import argparse
from pathlib import Path

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
    from torch.utils.data import DataLoader
    from faster_rcnn.dataset import ZaloCOCODataset, collate_fn
    from faster_rcnn.models import build_model
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    class _NoGrad:
        def __call__(self, *args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            def decorator(fn):
                return fn
            return decorator
    torch = type("TorchDummy", (), {"no_grad": _NoGrad()})()


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN")
    parser.add_argument("--config", "-c", type=str, default=None, metavar="YAML")
    parser.add_argument("--weights",    type=str,   default=None,
                        help="Custom checkpoint path. Mặc định: best → last.")
    parser.add_argument("--split",      type=str,   default="val", choices=["val", "test"],
                        help="Dataset split để evaluate (mặc định: val).")
    parser.add_argument("--pred",       type=str,   default=None,
                        help="File JSON predictions đã có (nếu truyền, bỏ qua inference và chấm điểm ngay).")
    parser.add_argument("--out-pred",   type=str,   default=None,
                        help="Đường dẫn file predictions JSON xuất ra (mặc định theo config: preds/...).")
    parser.add_argument("--name",       type=str,   default=None,
                        help="Tên thí nghiệm ghi vào báo cáo (mặc định lấy theo config _name).")
    parser.add_argument("--conf-thresh",type=float, default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--dry-run",    action="store_true")
    return parser


@torch.no_grad()
def evaluate_loss(model, data_loader, device, dry_run=False) -> dict:
    """Tính validation loss (model ở train mode để có loss_dict)."""
    was_training = model.training
    model.train()

    total_loss = 0.0
    loss_components = {
        "loss_classifier":  0.0,
        "loss_box_reg":     0.0,
        "loss_rpn_box_reg": 0.0,
        "loss_objectness":  0.0,
    }
    num_batches = 0

    pbar = tqdm(data_loader, desc="[VAL] Val Loss", leave=False)
    for images, targets in pbar:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss for loss in loss_dict.values())

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
def predict(model, data_loader, device, conf_thresh=0.001, dry_run=False) -> list:
    """Inference và trích xuất danh sách predictions theo chuẩn COCO JSON."""
    model.eval()
    results = []

    pbar = tqdm(data_loader, desc="[PREDICT] Inference", leave=False)
    for images, targets in pbar:
        images      = [img.to(device) for img in images]
        predictions = model(images)

        for pred, target in zip(predictions, targets):
            img_id = target["image_id"] if isinstance(target, dict) else target
            img_id = int(img_id.item()) if isinstance(img_id, torch.Tensor) else int(img_id)

            boxes  = pred["boxes"].detach().cpu().numpy()
            labels = pred["labels"].detach().cpu().numpy()
            scores = pred["scores"].detach().cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                if score >= conf_thresh:
                    xmin, ymin, xmax, ymax = box
                    results.append({
                        "image_id":    img_id,
                        "category_id": int(label),
                        "bbox":  [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)],
                        "score": float(score),
                    })

        if dry_run:
            break

    return results


@torch.no_grad()
def evaluate_coco_metrics(model, data_loader, device, gt_path, conf_thresh=0.001, dry_run=False) -> dict:
    """Inference và chấm metrics COCO bằng scripts/evaluate.py, trả về dict kết quả."""
    preds = predict(model, data_loader, device, conf_thresh=conf_thresh, dry_run=dry_run)
    if not preds:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0}

    import pycocotools.coco as coco_mod
    from pycocotools.cocoeval import COCOeval
    from scripts.evaluate import check_preds

    gt = coco_mod.COCO()
    gt.dataset = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    gt.createIndex()

    check_preds(preds, gt)
    dt = gt.loadRes(preds)

    E = COCOeval(gt, dt, iouType="bbox")
    E.params.areaRng = [[0, 1e10], [0, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e10]]
    E.evaluate()
    E.accumulate()
    E.summarize()
    st = E.stats

    return {
        "mAP50":       float(st[1]) if st[1] >= 0 else 0.0,
        "mAP50_95":    float(st[0]) if st[0] >= 0 else 0.0,
        "AP_small":    float(st[3]) if st[3] >= 0 else 0.0,
        "AP_medium":   float(st[4]) if st[4] >= 0 else 0.0,
        "AP_large":    float(st[5]) if st[5] >= 0 else 0.0,
        "predictions": preds,
    }


def main():
    parser = get_parser()
    args   = parser.parse_args()

    # Chế độ 1: Đánh giá trực tiếp từ file predictions JSON đã có (tiện lợi khi tải file từ Kaggle về)
    if args.pred:
        pred_path = Path(args.pred)
        if not pred_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file prediction: {pred_path}")

        # Tự động suy ra tên thí nghiệm nếu không truyền --name
        exp_name = args.name
        if not exp_name:
            stem = pred_path.stem
            for prefix in ["predictions_val_", "predictions_test_", "predictions_"]:
                if stem.startswith(prefix):
                    stem = stem[len(prefix):]
                    break
            exp_name = stem

        report_dir = "reports/eval"
        gt_path = f"data/splits/{args.split}.json"

        # Nếu có truyền config thì lấy đường dẫn chuẩn từ config
        if args.config:
            cfg = config.load(args.config)
            gt_path = cfg["VAL_JSON_PATH"] if args.split == "val" else cfg["TEST_JSON_PATH"]
            report_dir = cfg.get("REPORT_EVAL_DIR", report_dir)

        print("=" * 54)
        print(f"  EVALUATE TỪ FILE PREDICTIONS: {pred_path}")
        print(f"  Thí nghiệm: {exp_name} | split={args.split}")
        print(f"  Ground truth: {gt_path}")
        print("=" * 54)

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

        print("=" * 54)
        print("  EVALUATE HOÀN TẤT")
        print("=" * 54)
        return

    # Chế độ 2: Chạy inference từ checkpoint model
    if not HAS_TORCH:
        raise ImportError(
            "Cần cài đặt PyTorch để chạy model inference.\n"
            "Nếu bạn đã có sẵn file predictions JSON (ví dụ tải từ Kaggle về), "
            "hãy chạy với cờ --pred: python faster_rcnn/evaluate.py --pred <đường_dẫn_file.json>"
        )

    cli_overrides = {}
    if args.batch_size is not None:
        cli_overrides.setdefault("trainer", {})["batch_size"] = args.batch_size
    if args.dry_run:
        cli_overrides.setdefault("trainer", {})["dry_run"] = True

    cfg = config.load(args.config, cli_overrides or None)

    exp_name    = args.name or cfg.get("_name", "unnamed")
    is_dry_run  = cfg["DRY_RUN"]
    conf_thresh = args.conf_thresh if args.conf_thresh is not None else cfg["CONF_THRESHOLD"]

    print("=" * 54)
    print(f"  EVALUATE: {exp_name.upper()} | split={args.split}")
    print("=" * 54)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    model = build_model(cfg)

    if args.weights:
        weights_path = args.weights
    elif os.path.isfile(cfg["MODEL_BEST_WEIGHTS"]):
        weights_path = cfg["MODEL_BEST_WEIGHTS"]
        print(f"[WEIGHTS] Best checkpoint: {weights_path}")
    elif os.path.isfile(cfg["MODEL_BEST_WEIGHTS"].replace(".pth", ".zip")):
        weights_path = cfg["MODEL_BEST_WEIGHTS"].replace(".pth", ".zip")
        print(f"[WEIGHTS] Best checkpoint (.zip): {weights_path}")
    elif os.path.isfile(cfg["MODEL_WEIGHTS"]):
        weights_path = cfg["MODEL_WEIGHTS"]
        print(f"[WEIGHTS] Last checkpoint: {weights_path}")
    elif os.path.isfile(cfg["MODEL_WEIGHTS"].replace(".pth", ".zip")):
        weights_path = cfg["MODEL_WEIGHTS"].replace(".pth", ".zip")
        print(f"[WEIGHTS] Last checkpoint (.zip): {weights_path}")
    else:
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint:\n"
            f"  best: {cfg['MODEL_BEST_WEIGHTS']} (hoặc .zip)\n"
            f"  last: {cfg['MODEL_WEIGHTS']} (hoặc .zip)\n"
            f"Hãy chạy train.py trước."
        )

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    gt_path = cfg["VAL_JSON_PATH"] if args.split == "val" else cfg["TEST_JSON_PATH"]
    dataset = ZaloCOCODataset(gt_path, cfg["IMG_DIR"], is_train=True, use_augment=False)
    loader  = DataLoader(
        dataset,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=0 if is_dry_run else cfg["NUM_WORKERS"],
        collate_fn=collate_fn,
    )
    print(f"[DATASET] {args.split.upper()} — {len(dataset)} ảnh")

    # 1. Validation Loss
    loss_metrics = evaluate_loss(model, loader, device, dry_run=is_dry_run)
    print(f"[VAL LOSS] {loss_metrics['val/loss']:.4f}")
    for k, v in loss_metrics.items():
        if k != "val/loss":
            print(f"  {k}: {v:.4f}")

    # 2. Inference & lưu predictions JSON
    preds = predict(model, loader, device, conf_thresh=conf_thresh, dry_run=is_dry_run)
    if args.out_pred:
        pred_path = args.out_pred
    else:
        pred_path = cfg["OUTPUT_PRED_JSON"]
        if args.split == "test" and "_val_" in pred_path:
            pred_path = pred_path.replace("_val_", "_test_")
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)
    print(f"[OUTPUT] Predictions: {pred_path}")

    if is_dry_run:
        print("[DRY-RUN] Hoàn tất sanity check.")
        return

    # 3. Chấm điểm COCO: gọi hàm main của scripts/evaluate.py
    report_dir = cfg["REPORT_EVAL_DIR"]
    eval_argv = [
        "evaluate.py",
        "--gt", gt_path,
        "--pred", pred_path,
        "--name", exp_name,
        "--out", report_dir,
    ]
    sys_argv_backup = sys.argv
    try:
        sys.argv = eval_argv
        run_eval_coco()
    finally:
        sys.argv = sys_argv_backup

    print("=" * 54)
    print("  EVALUATE HOÀN TẤT")
    print("=" * 54)


if __name__ == "__main__":
    main()


