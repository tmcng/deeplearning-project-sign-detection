# faster_rcnn/config.py
# Wrapper load YAML config, expose globals để backward-compat.

from pathlib import Path
from faster_rcnn.configs.config_loader import load_config

_cfg: dict = load_config()


def _apply(cfg: dict) -> None:
    g = globals()
    g["PROJECT_ROOT"]    = Path(cfg["PROJECT_ROOT"])
    g["IMG_DIR"]         = cfg["IMG_DIR"]
    g["TRAIN_JSON_PATH"] = cfg["TRAIN_JSON_PATH"]
    g["VAL_JSON_PATH"]   = cfg["VAL_JSON_PATH"]
    g["TEST_JSON_PATH"]  = cfg["TEST_JSON_PATH"]
    g["CHECKPOINT_DIR"]  = Path(cfg["CHECKPOINT_DIR"])
    g["OUTPUT_DIR"]      = Path(cfg["OUTPUT_DIR"])
    g["REPORT_EVAL_DIR"] = cfg["REPORT_EVAL_DIR"]
    g["MODEL_WEIGHTS"]      = cfg["MODEL_WEIGHTS"]
    g["MODEL_BEST_WEIGHTS"] = cfg["MODEL_BEST_WEIGHTS"]
    g["OUTPUT_PRED_JSON"]   = cfg["OUTPUT_PRED_JSON"]
    g["NUM_CLASSES"]    = cfg["NUM_CLASSES"]
    g["CLASS_NAMES"]    = cfg["CLASS_NAMES"]
    g["CONF_THRESHOLD"] = cfg["CONF_THRESHOLD"]
    g["BATCH_SIZE"]      = cfg["BATCH_SIZE"]
    g["NUM_EPOCHS"]      = cfg["NUM_EPOCHS"]
    g["LEARNING_RATE"]   = cfg["LEARNING_RATE"]
    g["MOMENTUM"]        = cfg["MOMENTUM"]
    g["WEIGHT_DECAY"]    = cfg["WEIGHT_DECAY"]
    g["NUM_WORKERS"]     = cfg["NUM_WORKERS"]
    g["EVAL_INTERVAL"]   = cfg["EVAL_INTERVAL"]
    g["DRY_RUN"]         = cfg["DRY_RUN"]
    g["MIXED_PRECISION"] = cfg["MIXED_PRECISION"]
    g["SAVE_BEST"]       = cfg["SAVE_BEST"]
    g["LR_MILESTONES"]   = cfg["LR_MILESTONES"]
    g["LR_GAMMA"]        = cfg["LR_GAMMA"]
    g["GRAD_CLIP"]       = cfg["GRAD_CLIP"]
    g["GRAD_CLIP_NORM"]  = cfg["GRAD_CLIP_NORM"]
    g["USE_AUGMENT"]     = cfg["USE_AUGMENT"]
    g["WANDB_PROJECT"]   = cfg["WANDB_PROJECT"]
    g["WANDB_ENABLED"]   = cfg["WANDB_ENABLED"]
    g["WANDB_API_KEY"]   = cfg["WANDB_API_KEY"]
    g["WANDB_RUN_NAME"]  = cfg["WANDB_RUN_NAME"]
    g["CFG"] = cfg


_apply(_cfg)
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)  # noqa: F821
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)       # noqa: F821


def load(config_path=None, cli_overrides: dict = None) -> dict:
    """Tải lại config và cập nhật globals. Trả về flat cfg dict."""
    global _cfg
    _cfg = load_config(config_path, cli_overrides)
    _apply(_cfg)
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)  # noqa: F821
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)       # noqa: F821
    return _cfg

