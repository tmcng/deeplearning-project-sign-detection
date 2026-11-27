"""Training computation: optimizer, one training epoch and validation loss."""
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from faster_rcnn.tracker import WandbTracker

LOG_STEP_INTERVAL = 20

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
