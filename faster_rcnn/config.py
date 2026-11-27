# faster_rcnn/config.py
"""Wrapper load YAML config, expose globals để backward-compatibility.

Cung cấp cấu hình tập trung cho Faster R-CNN pipeline từ hệ thống modular YAML.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from faster_rcnn.configs.config_loader import load_config

# Khởi tạo giá trị mặc định cho module-level variables (hỗ trợ type checking và IDE navigation)
PROJECT_ROOT: Path
IMG_DIR: str
TRAIN_JSON_PATH: str
VAL_JSON_PATH: str
TEST_JSON_PATH: str
CHECKPOINT_DIR: Path
OUTPUT_DIR: Path
REPORT_EVAL_DIR: str
MODEL_WEIGHTS: str
MODEL_BEST_WEIGHTS: str
OUTPUT_PRED_JSON: str
NUM_CLASSES: int
CLASS_NAMES: List[str]
CONF_THRESHOLD: float
BATCH_SIZE: int
NUM_EPOCHS: int
LEARNING_RATE: float
MOMENTUM: float
WEIGHT_DECAY: float
NUM_WORKERS: int
EVAL_INTERVAL: int
DRY_RUN: bool
MIXED_PRECISION: bool
SAVE_BEST: bool
LR_MILESTONES: List[int]
LR_GAMMA: float
GRAD_CLIP: bool
GRAD_CLIP_NORM: float
USE_AUGMENT: bool
WANDB_PROJECT: str
WANDB_ENABLED: bool
WANDB_API_KEY: Optional[str]
WANDB_RUN_NAME: str
CFG: Dict[str, Any]

_cfg: Dict[str, Any] = load_config()


def _ensure_directories(checkpoint_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
    """Tạo thư mục checkpoint và output an toàn nếu chưa tồn tại."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def _apply(cfg: Dict[str, Any]) -> None:
    """Ánh xạ dict cấu hình phẳng vào namespace toàn cục của module."""
    g = globals()
    g["PROJECT_ROOT"] = Path(cfg["PROJECT_ROOT"])
    g["IMG_DIR"] = cfg["IMG_DIR"]
    g["TRAIN_JSON_PATH"] = cfg["TRAIN_JSON_PATH"]
    g["VAL_JSON_PATH"] = cfg["VAL_JSON_PATH"]
    g["TEST_JSON_PATH"] = cfg["TEST_JSON_PATH"]
    g["CHECKPOINT_DIR"] = Path(cfg["CHECKPOINT_DIR"])
    g["OUTPUT_DIR"] = Path(cfg["OUTPUT_DIR"])
    g["REPORT_EVAL_DIR"] = cfg["REPORT_EVAL_DIR"]
    g["MODEL_WEIGHTS"] = cfg["MODEL_WEIGHTS"]
    g["MODEL_BEST_WEIGHTS"] = cfg["MODEL_BEST_WEIGHTS"]
    g["OUTPUT_PRED_JSON"] = cfg["OUTPUT_PRED_JSON"]
    g["NUM_CLASSES"] = cfg["NUM_CLASSES"]
    g["CLASS_NAMES"] = cfg["CLASS_NAMES"]
    g["CONF_THRESHOLD"] = cfg["CONF_THRESHOLD"]
    g["BATCH_SIZE"] = cfg["BATCH_SIZE"]
    g["NUM_EPOCHS"] = cfg["NUM_EPOCHS"]
    g["LEARNING_RATE"] = cfg["LEARNING_RATE"]
    g["MOMENTUM"] = cfg["MOMENTUM"]
    g["WEIGHT_DECAY"] = cfg["WEIGHT_DECAY"]
    g["NUM_WORKERS"] = cfg["NUM_WORKERS"]
    g["EVAL_INTERVAL"] = cfg["EVAL_INTERVAL"]
    g["DRY_RUN"] = cfg["DRY_RUN"]
    g["MIXED_PRECISION"] = cfg["MIXED_PRECISION"]
    g["SAVE_BEST"] = cfg["SAVE_BEST"]
    g["LR_MILESTONES"] = cfg["LR_MILESTONES"]
    g["LR_GAMMA"] = cfg["LR_GAMMA"]
    g["GRAD_CLIP"] = cfg["GRAD_CLIP"]
    g["GRAD_CLIP_NORM"] = cfg["GRAD_CLIP_NORM"]
    g["USE_AUGMENT"] = cfg["USE_AUGMENT"]
    g["WANDB_PROJECT"] = cfg["WANDB_PROJECT"]
    g["WANDB_ENABLED"] = cfg["WANDB_ENABLED"]
    g["WANDB_API_KEY"] = cfg["WANDB_API_KEY"]
    g["WANDB_RUN_NAME"] = cfg["WANDB_RUN_NAME"]
    g["CFG"] = cfg


_apply(_cfg)


def load(
    config_path: Optional[Union[str, Path]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Tải lại config và cập nhật globals của module.

    Args:
        config_path: Đường dẫn file cấu hình YAML (tùy chọn).
        cli_overrides: Dict chứa các tham số ghi đè từ CLI (tùy chọn).

    Returns:
        Dict cấu hình phẳng đã được resolve đầy đủ.
    """
    global _cfg
    _cfg = load_config(config_path, cli_overrides)
    _apply(_cfg)
    return _cfg
