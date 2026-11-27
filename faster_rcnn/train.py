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
from faster_rcnn.cli import get_train_parser as get_parser, build_overrides as _build_cli_overrides
from faster_rcnn.engine import _train_one_epoch, evaluate_loss, _setup_optimizer_and_scheduler
from faster_rcnn.dataset import _setup_dataloaders
from faster_rcnn.models import build_model, resolve_device
from faster_rcnn.tracker import WandbTracker

LOG_STEP_INTERVAL: int = 20
DEFAULT_CONF_THRESH: float = 0.001


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

    device = resolve_device(args.device)
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
    best_loss: float = float("inf")
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

        current_loss = val_losses["val/loss"]
        print(
            f"[EP {epoch + 1:03d}] train_loss={train_loss:.4f} | "
            f"val_loss={current_loss:.4f} | lr={current_lr:.6f}"
        )
        if cfg["SAVE_BEST"] and current_loss < best_loss:
            best_loss = current_loss
            _save_weights(model, cfg["MODEL_BEST_WEIGHTS"])

        tracker.log_metrics(epoch_metrics, step=global_step)

    last_ckpt_path = cfg["MODEL_WEIGHTS"]
    _save_weights(model, last_ckpt_path)

    print("=" * 54)
    print(f"[SAVED] Last Checkpoint: {last_ckpt_path}")
    if best_loss < float("inf"):
        print(f"[SAVED] Best Checkpoint: {cfg['MODEL_BEST_WEIGHTS']} (val_loss: {best_loss:.4f})")
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
    train(get_parser().parse_args())
