# faster_rcnn/train.py
"""Module huấn luyện Faster R-CNN cho bài toán phát hiện biển báo giao thông.

Cách chạy:
    python faster_rcnn/train.py --config faster_rcnn/configs/experiment/exp01_baseline.yaml
    python faster_rcnn/train.py --config faster_rcnn/configs/experiment/exp01_baseline.yaml --trainer debug_cpu --dry-run --no-wandb
    python faster_rcnn/train.py --config faster_rcnn/configs/experiment/exp03_fpn_aug.yaml --epochs 30 --lr 0.001
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Thiết lập đường dẫn root dự án vào sys.path để các import nội bộ hoạt động nhất quán
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faster_rcnn import config
from faster_rcnn.configs.config_loader import _deep_merge, _load_yaml
from faster_rcnn.dataset import ZaloCOCODataset, collate_fn
from faster_rcnn.evaluate import evaluate_coco_metrics, evaluate_loss
from faster_rcnn.models import build_model
from scripts.tracker import WandbTracker

LOG_STEP_INTERVAL: int = 20
DEFAULT_CONF_THRESH: float = 0.001


def get_parser() -> argparse.ArgumentParser:
    """Tạo parser cho các đối số dòng lệnh khi huấn luyện."""
    parser = argparse.ArgumentParser(description="Train Faster R-CNN")
    parser.add_argument(
        "--config", "-c", type=str, default=None, metavar="YAML",
        help="Experiment yaml (mặc định: configs/train.yaml)",
    )
    parser.add_argument(
        "--trainer", type=str, default=None,
        choices=["gpu_fp32", "gpu_fp16", "debug_cpu"],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--img-dir", type=str, default=None,
        help="Đường dẫn thư mục ảnh (override config, ví dụ: /kaggle/input/<dataset>/images).",
    )
    parser.add_argument(
        "--train-json", type=str, default=None,
        help="Đường dẫn file annotation train COCO JSON (override config).",
    )
    parser.add_argument(
        "--val-json", type=str, default=None,
        help="Đường dẫn file annotation val COCO JSON (override config).",
    )
    parser.add_argument("--wandb-key", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _build_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Chuyển đổi các cờ dòng lệnh thành dictionary overrides cho config loader."""
    overrides: Dict[str, Any] = {}
    trainer_overrides: Dict[str, Any] = {}

    paths_overrides: Dict[str, Any] = {}
    if args.img_dir:
        paths_overrides["img_dir"] = args.img_dir
    if args.train_json:
        paths_overrides["train_json"] = args.train_json
    if args.val_json:
        paths_overrides["val_json"] = args.val_json
    if paths_overrides:
        overrides.setdefault("paths", {})["data"] = paths_overrides

    if args.epochs is not None:
        trainer_overrides["num_epochs"] = args.epochs
    if args.batch_size is not None:
        trainer_overrides["batch_size"] = args.batch_size
    if args.num_workers is not None:
        trainer_overrides["num_workers"] = args.num_workers
    if args.dry_run:
        trainer_overrides["dry_run"] = True
    if args.lr is not None:
        trainer_overrides.setdefault("optimizer", {})["lr"] = args.lr

    if args.trainer:
        trainer_file = (
            Path(__file__).resolve().parent / "configs" / "trainer" / f"{args.trainer}.yaml"
        )
        if trainer_file.exists():
            base_trainer = _load_yaml(trainer_file)
            overrides["trainer"] = _deep_merge(base_trainer, trainer_overrides)
        else:
            overrides["trainer"] = args.trainer
    elif trainer_overrides:
        overrides["trainer"] = trainer_overrides

    wandb_overrides: Dict[str, Any] = {}
    if args.wandb_key:
        wandb_overrides["api_key"] = args.wandb_key
    if args.no_wandb or args.dry_run:
        wandb_overrides["enabled"] = False
    if args.run_name:
        wandb_overrides["run_name"] = args.run_name
    if wandb_overrides:
        overrides["wandb"] = wandb_overrides

    return overrides


def _setup_dataloaders(
    cfg: Dict[str, Any],
    device: torch.device,
    is_dry_run: bool,
) -> Tuple[ZaloCOCODataset, ZaloCOCODataset, DataLoader, DataLoader]:
    """Khởi tạo Datasets và DataLoaders cho tập train và validation."""
    train_dataset = ZaloCOCODataset(
        cfg["TRAIN_JSON_PATH"],
        cfg["IMG_DIR"],
        is_train=True,
        use_augment=cfg["USE_AUGMENT"],
    )
    val_dataset = ZaloCOCODataset(
        cfg["VAL_JSON_PATH"],
        cfg["IMG_DIR"],
        is_train=True,
        use_augment=False,
    )
    num_workers = 0 if is_dry_run else cfg["NUM_WORKERS"]
    pin_mem = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def _setup_optimizer_and_scheduler(
    model: nn.Module,
    cfg: Dict[str, Any],
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.MultiStepLR]:
    """Cấu hình SGD optimizer và MultiStepLR scheduler theo tham số cấu hình."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg["LEARNING_RATE"],
        momentum=cfg["MOMENTUM"],
        weight_decay=cfg["WEIGHT_DECAY"],
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["LR_MILESTONES"],
        gamma=cfg["LR_GAMMA"],
    )
    return optimizer, lr_scheduler


def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    cfg: Dict[str, Any],
    tracker: WandbTracker,
    global_step: int,
    epoch: int,
    num_epochs: int,
    is_dry_run: bool,
) -> Tuple[float, int, bool]:
    """Thực thi một epoch huấn luyện.

    Returns:
        Tuple (epoch_loss_average, updated_global_step, is_dry_run_finished).
    """
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for index, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(losses).backward()

        if cfg["GRAD_CLIP"]:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["GRAD_CLIP_NORM"])

        scaler.step(optimizer)
        scaler.update()

        epoch_loss += losses.item()
        pbar.set_postfix(loss=f"{losses.item():.4f}")

        if index % LOG_STEP_INTERVAL == 0:
            tracker.log_metrics(
                {
                    "batch_total_loss": losses.item(),
                    "loss_classifier": loss_dict.get("loss_classifier", torch.zeros(1)).item(),
                    "loss_box_reg": loss_dict.get("loss_box_reg", torch.zeros(1)).item(),
                    "loss_rpn_box_reg": loss_dict.get("loss_rpn_box_reg", torch.zeros(1)).item(),
                    "loss_objectness": loss_dict.get("loss_objectness", torch.zeros(1)).item(),
                },
                step=global_step,
            )
        global_step += 1

        if is_dry_run and index >= 0:
            return epoch_loss, global_step, True

    avg_train_loss = epoch_loss / len(train_loader)
    return avg_train_loss, global_step, False


def _save_weights(model: nn.Module, path: str) -> None:
    """Lưu state_dict của mô hình ra file an toàn."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)


def train(args: argparse.Namespace) -> nn.Module:
    """Quy trình huấn luyện chính cho Faster R-CNN."""
    cli_overrides = _build_cli_overrides(args)
    cfg = config.load(args.config, cli_overrides or None)

    exp_name: str = cfg.get("_name", "unnamed")
    desc: str = cfg.get("_description", "")

    print("=" * 54)
    print(f"  TRAINING: {exp_name.upper()}")
    if desc:
        print(f"  {desc}")
    print("=" * 54)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    model = build_model(cfg)
    model.to(device)
    print(
        f"[MODEL] {cfg['model'].get('architecture', 'unknown')} "
        f"| trainable_layers={cfg['model'].get('backbone', {}).get('trainable_layers', 0)} "
        f"| num_classes={cfg['NUM_CLASSES']}"
    )

    is_dry_run: bool = cfg["DRY_RUN"]
    tracker = WandbTracker(
        api_key=cfg["WANDB_API_KEY"],
        project_name=cfg["WANDB_PROJECT"],
        run_name=cfg["WANDB_RUN_NAME"],
        config_dict={
            "experiment": exp_name,
            "epochs": cfg["NUM_EPOCHS"],
            "batch_size": cfg["BATCH_SIZE"],
            "lr": cfg["LEARNING_RATE"],
            "weight_decay": cfg["WEIGHT_DECAY"],
            "use_augment": cfg["USE_AUGMENT"],
            "mixed_precision": cfg["MIXED_PRECISION"],
            "description": desc,
        },
        enabled=cfg["WANDB_ENABLED"] and not is_dry_run,
    )

    train_ds, val_ds, train_loader, val_loader = _setup_dataloaders(cfg, device, is_dry_run)
    num_workers = 0 if is_dry_run else cfg["NUM_WORKERS"]
    print(
        f"[DATASET] Train: {len(train_ds)} ảnh | Val: {len(val_ds)} ảnh "
        f"| augmentation={cfg['USE_AUGMENT']} | workers={num_workers}"
    )

    optimizer, lr_scheduler = _setup_optimizer_and_scheduler(model, cfg)

    use_amp = bool(cfg["MIXED_PRECISION"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("[AMP] Mixed precision FP16 được bật")

    num_epochs: int = 1 if is_dry_run else cfg["NUM_EPOCHS"]
    eval_interval: int = cfg.get("EVAL_INTERVAL", 1)
    best_map: float = -1.0
    global_step: int = 0

    print("-" * 54)
    print(
        f"[TRAIN] epochs={num_epochs} | batch={cfg['BATCH_SIZE']} "
        f"| lr={cfg['LEARNING_RATE']} | dry_run={is_dry_run}"
    )

    for epoch in range(num_epochs):
        train_loss, global_step, dry_run_finished = _train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            cfg=cfg,
            tracker=tracker,
            global_step=global_step,
            epoch=epoch,
            num_epochs=num_epochs,
            is_dry_run=is_dry_run,
        )

        if dry_run_finished:
            print(f"\n[DRY-RUN] 1 batch thành công! ({exp_name})")
            tracker.finish()
            return model

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Validation Loss chi tiết
        val_losses = evaluate_loss(model, val_loader, device, dry_run=is_dry_run)

        epoch_metrics: Dict[str, Any] = {
            "epoch": epoch + 1,
            "train/loss_total": train_loss,
            "train/lr": current_lr,
            "val/loss_total": val_losses.get("val/loss", 0.0),
            "val/loss_classifier": val_losses.get("val/loss_classifier", 0.0),
            "val/loss_box_reg": val_losses.get("val/loss_box_reg", 0.0),
            "val/loss_rpn_box_reg": val_losses.get("val/loss_rpn_box_reg", 0.0),
            "val/loss_objectness": val_losses.get("val/loss_objectness", 0.0),
        }

        # Đánh giá mAP định kỳ theo EVAL_INTERVAL hoặc epoch cuối
        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == num_epochs:
            coco_res = evaluate_coco_metrics(
                model,
                val_loader,
                device,
                cfg["VAL_JSON_PATH"],
                conf_thresh=cfg["CONF_THRESHOLD"],
                dry_run=is_dry_run,
            )
            epoch_metrics["val/mAP50"] = coco_res["mAP50"]
            epoch_metrics["val/mAP50_95"] = coco_res["mAP50_95"]
            epoch_metrics["val/AP_small"] = coco_res["AP_small"]
            epoch_metrics["val/AP_medium"] = coco_res["AP_medium"]
            epoch_metrics["val/AP_large"] = coco_res["AP_large"]

            current_map = coco_res["mAP50"]
            print(
                f"[EP {epoch + 1:03d}] train_loss={train_loss:.4f} | "
                f"val_loss={val_losses.get('val/loss', 0.0):.4f} | "
                f"mAP50={current_map*100:.2f}% | AP_small={coco_res['AP_small']*100:.2f}% | "
                f"lr={current_lr:.6f}"
            )

            # Lưu checkpoint tốt nhất theo mAP50
            if cfg.get("SAVE_BEST", True) and current_map > best_map:
                best_map = current_map
                best_ckpt_path = cfg["MODEL_BEST_WEIGHTS"]
                _save_weights(model, best_ckpt_path)
                print(
                    f"  >>> [BEST] Cập nhật best checkpoint (mAP50={best_map*100:.2f}%): {best_ckpt_path}"
                )
        else:
            print(
                f"[EP {epoch + 1:03d}] train_loss={train_loss:.4f} | "
                f"val_loss={val_losses.get('val/loss', 0.0):.4f} | lr={current_lr:.6f}"
            )

        tracker.log_metrics(epoch_metrics, step=global_step)

    last_ckpt_path = cfg["MODEL_WEIGHTS"]
    _save_weights(model, last_ckpt_path)

    print("=" * 54)
    print(f"[SAVED] Last Checkpoint: {last_ckpt_path}")
    if best_map >= 0:
        print(f"[SAVED] Best Checkpoint: {cfg['MODEL_BEST_WEIGHTS']} (mAP50: {best_map*100:.2f}%)")
    print("=" * 54)

    tracker.log_artifact(
        file_path=last_ckpt_path,
        artifact_name=f"fasterrcnn_{exp_name}_weights",
        artifact_type="model",
        description=f"Model weights for {exp_name}",
    )
    tracker.finish()
    return model


if __name__ == "__main__":
    cli_parser = get_parser()
    parsed_args = cli_parser.parse_args()
    train(parsed_args)
