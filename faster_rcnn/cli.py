"""CLI adapters. Missing flags preserve YAML defaults; --set covers nested settings."""
import argparse
from typing import Any, Dict
from pathlib import Path
import yaml
from faster_rcnn.configs.config_loader import _deep_merge, _load_yaml

FIELDS = {
    "momentum": ("trainer.optimizer.momentum", float),
    "weight_decay": ("trainer.optimizer.weight_decay", float),
    "eval_interval": ("trainer.eval_interval", int),
    "lr_gamma": ("trainer.lr_scheduler.gamma", float),
    "grad_clip_norm": ("trainer.gradient_clip.max_norm", float),
    "min_size": ("model.transform.min_size", int),
    "max_size": ("model.transform.max_size", int),
    "trainable_layers": ("model.backbone.trainable_layers", int),
    "num_classes": ("data.num_classes", int),
    "checkpoint_dir": ("paths.output.checkpoint_dir", str),
    "output_dir": ("paths.output.output_dir", str),
    "report_dir": ("paths.output.report_dir", str),
}
BOOL_FIELDS = {
    "mixed_precision": "trainer.mixed_precision", "save_best": "trainer.save_best",
    "grad_clip": "trainer.gradient_clip.enabled", "augment": "augmentation.use_augment",
}
def add_config_arguments(parser):
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=YAML",
                        help="Override any nested config value; repeatable. Applied last.")
    for flag, (_, kind) in FIELDS.items():
        parser.add_argument("--" + flag.replace("_", "-"), type=kind, default=None)
    for flag in BOOL_FIELDS:
        parser.add_argument("--" + flag.replace("_", "-"), action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lr-milestones", nargs="+", type=int)
    if "--num-workers" not in parser._option_string_actions:
        parser.add_argument("--num-workers", type=int)
    return parser

def get_train_parser() -> argparse.ArgumentParser:
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
    add_config_arguments(parser)
    return parser


def _legacy_overrides(args: argparse.Namespace) -> Dict[str, Any]:
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



def put(mapping, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        mapping = mapping.setdefault(part, {})
    mapping[parts[-1]] = value

def build_overrides(args):
    # Train has the legacy aliases; inference uses only its own path/split flags.
    overrides = _legacy_overrides(args) if hasattr(args, "epochs") else {}
    for flag, (path, _) in FIELDS.items():
        value = getattr(args, flag, None)
        if value is not None:
            put(overrides, path, value)
    for flag, path in BOOL_FIELDS.items():
        value = getattr(args, flag, None)
        if value is not None:
            put(overrides, path, value)
    if args.lr_milestones is not None:
        put(overrides, "trainer.lr_scheduler.milestones", args.lr_milestones)
    if args.num_workers is not None:
        put(overrides, "trainer.num_workers", args.num_workers)
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep or not key or any(not part for part in key.split(".")):
            raise ValueError("--set requires a dotted key=value")
        put(overrides, key, yaml.safe_load(value))
    return overrides
