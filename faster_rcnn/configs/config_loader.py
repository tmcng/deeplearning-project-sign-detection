# faster_rcnn/configs/config_loader.py
"""Load và merge YAML config theo thứ tự phân cấp ưu tiên:

sub-configs (paths, data, aug, trainer, model) -> train.yaml -> experiment.yaml -> CLI overrides.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

_THIS_FILE: Path = Path(__file__).resolve()
CONFIGS_DIR: Path = _THIS_FILE.parent  # .../faster_rcnn/configs/
PROJECT_ROOT: Path = _THIS_FILE.parent.parent.parent  # .../sign-detection/


def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Đọc file YAML trả về dict (trả về rỗng nếu file trống hoặc lỗi)."""
    resolved_path = Path(path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình YAML: '{resolved_path}'")
    with open(resolved_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge đệ quy hai từ điển, override sẽ ghi đè lên base."""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def _resolve_subconfig(key: str, value: Any, subdir: str) -> Dict[str, Any]:
    """Phân giải subconfig theo tên file hoặc trả về dict được chỉ định trực tiếp."""
    if isinstance(value, str):
        sub_path = CONFIGS_DIR / subdir / f"{value}.yaml"
        return _load_yaml(sub_path)
    if isinstance(value, dict):
        if key == "model" and "architecture" in value:
            arch = value["architecture"]
            base = _load_yaml(CONFIGS_DIR / "model" / f"{arch}.yaml")
            return _deep_merge(base, value)
        return deepcopy(value)
    return {}


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Tải và merge toàn bộ hệ thống cấu hình YAML.

    Args:
        config_path: Đường dẫn file thí nghiệm YAML (vd: configs/experiment/exp01_baseline.yaml).
        cli_overrides: Các tham số ghi đè từ dòng lệnh CLI.

    Returns:
        Dict cấu hình phẳng (flat dict) với tất cả tham số đã resolve đầy đủ.
    """
    base = _load_yaml(CONFIGS_DIR / "train.yaml")

    if config_path is not None:
        base = _deep_merge(base, _load_yaml(Path(config_path)))

    cfg: Dict[str, Any] = {
        "paths": _resolve_subconfig(
            "paths", base.get("paths", "local"), "paths"
        ),
        "data": _resolve_subconfig(
            "data", base.get("data", "dataset_v1"), "data"
        ),
        "augmentation": _resolve_subconfig(
            "augmentation", base.get("augmentation", "no_aug"), "augmentation"
        ),
        "trainer": _resolve_subconfig(
            "trainer", base.get("trainer", "gpu_fp32"), "trainer"
        ),
        "model": _resolve_subconfig(
            "model", base.get("model", "fasterrcnn_resnet50_fpn"), "model"
        ),
        "_name": base.get("_name", "unnamed"),
        "_description": base.get("_description", ""),
        "wandb": base.get("wandb", {}),
        "output": base.get("output", {}),
    }

    if cli_overrides:
        cfg = _deep_merge(cfg, cli_overrides)

    _flatten(cfg)
    return cfg


def _to_absolute(base_dir: Path, raw_path: Union[str, Path]) -> str:
    """Chuyển đổi đường dẫn tương đối thành tuyệt đối từ base_dir (giữ nguyên nếu đã là đường dẫn tuyệt đối)."""
    import os
    path_str = str(raw_path)
    if os.path.isabs(path_str) or path_str.startswith(("/", "\\")):
        return path_str
    return str(base_dir / path_str)


def _flatten(cfg: Dict[str, Any]) -> None:
    """Đưa các tham số hay dùng lên top-level của cfg dict (in-place)."""
    paths = cfg.get("paths", {})
    data_dirs = paths.get("data", {})
    out_dirs = paths.get("output", {})

    cfg["PROJECT_ROOT"] = str(PROJECT_ROOT)
    cfg["IMG_DIR"] = _to_absolute(PROJECT_ROOT, data_dirs.get("img_dir", "data/images"))
    cfg["TRAIN_JSON_PATH"] = _to_absolute(
        PROJECT_ROOT, data_dirs.get("train_json", "data/splits/train.json")
    )
    cfg["VAL_JSON_PATH"] = _to_absolute(
        PROJECT_ROOT, data_dirs.get("val_json", "data/splits/val.json")
    )
    cfg["TEST_JSON_PATH"] = _to_absolute(
        PROJECT_ROOT, data_dirs.get("test_json", "data/splits/test.json")
    )
    cfg["CHECKPOINT_DIR"] = _to_absolute(
        PROJECT_ROOT, out_dirs.get("checkpoint_dir", "faster_rcnn/checkpoints")
    )
    cfg["OUTPUT_DIR"] = _to_absolute(
        PROJECT_ROOT, out_dirs.get("output_dir", "faster_rcnn/outputs")
    )
    cfg["REPORT_EVAL_DIR"] = _to_absolute(
        PROJECT_ROOT, out_dirs.get("report_dir", "reports/eval")
    )

    data_cfg = cfg.get("data", {})
    cfg["NUM_CLASSES"] = data_cfg.get("num_classes", 8)
    cfg["CLASS_NAMES"] = data_cfg.get("class_names", [])
    cfg["CONF_THRESHOLD"] = data_cfg.get("conf_threshold", 0.001)

    # Nếu trainer vẫn là string (do CLI override), resolve nó thành dict
    trainer_raw = cfg.get("trainer", {})
    if isinstance(trainer_raw, str):
        cfg["trainer"] = _load_yaml(CONFIGS_DIR / "trainer" / f"{trainer_raw}.yaml")
    trainer_cfg = cfg.get("trainer", {})
    opt_cfg = trainer_cfg.get("optimizer", {})
    sched_cfg = trainer_cfg.get("lr_scheduler", {})
    grad_cfg = trainer_cfg.get("gradient_clip", {})

    cfg["NUM_WORKERS"] = trainer_cfg.get(
        "num_workers", paths.get("num_workers", 2)
    )
    cfg["BATCH_SIZE"] = trainer_cfg.get("batch_size", 4)
    cfg["NUM_EPOCHS"] = trainer_cfg.get("num_epochs", 20)
    cfg["EVAL_INTERVAL"] = trainer_cfg.get("eval_interval", 1)
    cfg["DRY_RUN"] = trainer_cfg.get("dry_run", False)
    cfg["MIXED_PRECISION"] = trainer_cfg.get("mixed_precision", False)
    cfg["SAVE_BEST"] = trainer_cfg.get("save_best", True)
    cfg["LEARNING_RATE"] = opt_cfg.get("lr", 0.005)
    cfg["MOMENTUM"] = opt_cfg.get("momentum", 0.9)
    cfg["WEIGHT_DECAY"] = opt_cfg.get("weight_decay", 0.0005)
    cfg["LR_MILESTONES"] = sched_cfg.get("milestones", [12, 16])
    cfg["LR_GAMMA"] = sched_cfg.get("gamma", 0.1)
    cfg["GRAD_CLIP"] = grad_cfg.get("enabled", False)
    cfg["GRAD_CLIP_NORM"] = grad_cfg.get("max_norm", 1.0)

    aug_cfg = cfg.get("augmentation", {})
    cfg["USE_AUGMENT"] = aug_cfg.get("use_augment", False)

    exp_name = cfg.get("_name", "unnamed")
    out_cfg = cfg.get("output", {})
    ckpt_name = out_cfg.get("checkpoint_name", exp_name)
    pred_json = out_cfg.get("pred_json", f"predictions_val_{exp_name}.json")
    ckpt_dir = Path(cfg["CHECKPOINT_DIR"])
    out_dir = Path(cfg["OUTPUT_DIR"])

    cfg["MODEL_WEIGHTS"] = str(ckpt_dir / f"{ckpt_name}_last.pth")
    cfg["MODEL_BEST_WEIGHTS"] = str(ckpt_dir / f"{ckpt_name}_best.pth")
    cfg["OUTPUT_PRED_JSON"] = str(out_dir / pred_json)

    wandb_cfg = cfg.get("wandb", {})
    cfg["WANDB_PROJECT"] = wandb_cfg.get("project", "Zalo-Traffic-Sign-FasterRCNN")
    cfg["WANDB_ENABLED"] = wandb_cfg.get("enabled", True)
    cfg["WANDB_API_KEY"] = wandb_cfg.get("api_key", None)
    cfg["WANDB_RUN_NAME"] = wandb_cfg.get("run_name", exp_name)
