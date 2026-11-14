# faster_rcnn/configs/config_loader.py
# Load và merge YAML config theo thứ tự ưu tiên:
#   sub-configs → train.yaml → experiment yaml → CLI overrides

import yaml
from pathlib import Path
from copy import deepcopy

_THIS_FILE   = Path(__file__).resolve()
CONFIGS_DIR  = _THIS_FILE.parent           # .../faster_rcnn/configs/
PROJECT_ROOT = _THIS_FILE.parent.parent.parent  # .../sign-detection/


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def _resolve_subconfig(key: str, value, subdir: str) -> dict:
    if isinstance(value, str):
        return _load_yaml(CONFIGS_DIR / subdir / f"{value}.yaml")
    if isinstance(value, dict):
        if key == "model" and "architecture" in value:
            arch = value["architecture"]
            base = _load_yaml(CONFIGS_DIR / "model" / f"{arch}.yaml")
            return _deep_merge(base, value)
        return deepcopy(value)
    return {}


def load_config(config_path=None, cli_overrides: dict = None) -> dict:
    """Tải và merge config. Trả về flat dict với tất cả tham số đã resolve."""
    base = _load_yaml(CONFIGS_DIR / "train.yaml")

    if config_path is not None:
        base = _deep_merge(base, _load_yaml(Path(config_path)))

    cfg = {
        "paths":        _resolve_subconfig("paths",        base.get("paths",        "local"),                   "paths"),
        "data":         _resolve_subconfig("data",          base.get("data",          "dataset_v1"),              "data"),
        "augmentation": _resolve_subconfig("augmentation",  base.get("augmentation",  "no_aug"),                  "augmentation"),
        "trainer":      _resolve_subconfig("trainer",       base.get("trainer",       "gpu_fp32"),                "trainer"),
        "model":        _resolve_subconfig("model",         base.get("model",         "fasterrcnn_resnet50_fpn"), "model"),
        "_name":        base.get("_name", "unnamed"),
        "_description": base.get("_description", ""),
        "wandb":        base.get("wandb", {}),
        "output":       base.get("output", {}),
    }

    if cli_overrides:
        cfg = _deep_merge(cfg, cli_overrides)

    _flatten(cfg)
    return cfg


def _flatten(cfg: dict) -> None:
    """Đưa các tham số hay dùng lên top-level cfg (in-place)."""
    paths    = cfg.get("paths", {})
    data_dirs = paths.get("data", {})
    out_dirs  = paths.get("output", {})

    cfg["PROJECT_ROOT"]    = str(PROJECT_ROOT)
    cfg["IMG_DIR"]         = str(PROJECT_ROOT / data_dirs.get("img_dir",     "data/images"))
    cfg["TRAIN_JSON_PATH"] = str(PROJECT_ROOT / data_dirs.get("train_json",  "data/splits/train.json"))
    cfg["VAL_JSON_PATH"]   = str(PROJECT_ROOT / data_dirs.get("val_json",    "data/splits/val.json"))
    cfg["TEST_JSON_PATH"]  = str(PROJECT_ROOT / data_dirs.get("test_json",   "data/splits/test.json"))
    cfg["CHECKPOINT_DIR"]  = str(PROJECT_ROOT / out_dirs.get("checkpoint_dir", "faster_rcnn/checkpoints"))
    cfg["OUTPUT_DIR"]      = str(PROJECT_ROOT / out_dirs.get("output_dir",     "faster_rcnn/outputs"))
    cfg["REPORT_EVAL_DIR"] = str(PROJECT_ROOT / out_dirs.get("report_dir",     "reports/eval"))

    data_cfg = cfg.get("data", {})
    cfg["NUM_CLASSES"]    = data_cfg.get("num_classes",    8)
    cfg["CLASS_NAMES"]    = data_cfg.get("class_names",    [])
    cfg["CONF_THRESHOLD"] = data_cfg.get("conf_threshold", 0.001)

    # Nếu trainer vẫn là string (do CLI override), resolve nó
    trainer_raw = cfg.get("trainer", {})
    if isinstance(trainer_raw, str):
        cfg["trainer"] = _load_yaml(CONFIGS_DIR / "trainer" / f"{trainer_raw}.yaml")
    trainer_cfg = cfg.get("trainer", {})
    opt_cfg     = trainer_cfg.get("optimizer", {})
    sched_cfg   = trainer_cfg.get("lr_scheduler", {})
    grad_cfg    = trainer_cfg.get("gradient_clip", {})

    cfg["NUM_WORKERS"]     = trainer_cfg.get("num_workers",    paths.get("num_workers", 2))
    cfg["BATCH_SIZE"]      = trainer_cfg.get("batch_size",     4)
    cfg["NUM_EPOCHS"]      = trainer_cfg.get("num_epochs",     20)
    cfg["EVAL_INTERVAL"]   = trainer_cfg.get("eval_interval",  1)
    cfg["DRY_RUN"]         = trainer_cfg.get("dry_run",        False)
    cfg["MIXED_PRECISION"] = trainer_cfg.get("mixed_precision", False)
    cfg["SAVE_BEST"]       = trainer_cfg.get("save_best",       True)
    cfg["LEARNING_RATE"]   = opt_cfg.get("lr",           0.005)
    cfg["MOMENTUM"]        = opt_cfg.get("momentum",     0.9)
    cfg["WEIGHT_DECAY"]    = opt_cfg.get("weight_decay", 0.0005)
    cfg["LR_MILESTONES"]   = sched_cfg.get("milestones", [12, 16])
    cfg["LR_GAMMA"]        = sched_cfg.get("gamma",      0.1)
    cfg["GRAD_CLIP"]       = grad_cfg.get("enabled",  False)
    cfg["GRAD_CLIP_NORM"]  = grad_cfg.get("max_norm", 1.0)

    aug_cfg = cfg.get("augmentation", {})
    cfg["USE_AUGMENT"] = aug_cfg.get("use_augment", False)

    exp_name  = cfg.get("_name", "unnamed")
    out_cfg   = cfg.get("output", {})
    ckpt_name = out_cfg.get("checkpoint_name", exp_name)
    pred_json = out_cfg.get("pred_json", f"predictions_val_{exp_name}.json")
    ckpt_dir  = Path(cfg["CHECKPOINT_DIR"])
    out_dir   = Path(cfg["OUTPUT_DIR"])

    cfg["MODEL_WEIGHTS"]      = str(ckpt_dir / f"{ckpt_name}_last.pth")
    cfg["MODEL_BEST_WEIGHTS"] = str(ckpt_dir / f"{ckpt_name}_best.pth")
    cfg["OUTPUT_PRED_JSON"]   = str(out_dir  / pred_json)

    wandb_cfg = cfg.get("wandb", {})
    cfg["WANDB_PROJECT"]  = wandb_cfg.get("project",  "Zalo-Traffic-Sign-FasterRCNN")
    cfg["WANDB_ENABLED"]  = wandb_cfg.get("enabled",  True)
    cfg["WANDB_API_KEY"]  = wandb_cfg.get("api_key",  None)
    cfg["WANDB_RUN_NAME"] = wandb_cfg.get("run_name", exp_name)
