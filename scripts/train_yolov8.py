"""Script huß║Ñn luyß╗çn YOLOv8 chuß║⌐n h├│a cho b├ái to├ín ph├ít hiß╗çn biß╗ân b├ío (Th├ánh vi├¬n B, b├án giao #5 & #7).

Hß╗ù trß╗ú:
    - Chß║íy thß╗¡ nhanh (smoke test tuß║ºn 1) tr├¬n ~200 ß║únh: --model yolov8n.pt --fraction 0.07 --epochs 5
    - Baseline tuß║ºn 2: --model yolov8s.pt --imgsz 640 --epochs 50
    - Ph╞░╞íng ph├íp ch├¡nh tuß║ºn 3 (ß║únh lß╗¢n / ─æß║ºu ra P2 cho biß╗ân nhß╗Å):
        + --model yolov8s.pt --imgsz 1280 --batch 8
        + --model yolov8s-p2.yaml --pretrained yolov8s.pt --imgsz 1280 --batch 8
    - Tß╗▒ ─æß╗Öng ghi log l├¬n Weights & Biases (W&B) nß║┐u ─æ├ú ─æ─âng nhß║¡p, hoß║╖c tß║»t bß║▒ng --no-wandb.

C├ích chß║íy:
    # 1. Baseline YOLOv8s ß╗ƒ imgsz=640
    python scripts/train_yolov8.py --data data/yolo/data.yaml --model yolov8s.pt --imgsz 640 --name yolov8s_640

    # 2. YOLOv8s ß╗ƒ imgsz=1280 (giß║úm batch ─æß╗â vß╗½a VRAM 16GB tr├¬n Kaggle T4/P100)
    python scripts/train_yolov8.py --data data/yolo/data.yaml --model yolov8s.pt --imgsz 1280 --batch 8 --name yolov8s_1280

    # 3. Kiß║┐n tr├║c YOLOv8s-P2 (th├¬m tß║ºng stride 4 bß║»t biß╗ân b├ío rß║Ñt nhß╗Å < 10px)
    python scripts/train_yolov8.py --data data/yolo/data.yaml --model yolov8s-p2.yaml --pretrained yolov8s.pt --imgsz 1280 --batch 8 --name yolov8s_p2_1280
"""
import argparse
import os
from pathlib import Path


def setup_wandb(project_name, disable_wandb):
    """Bß║¡t hoß║╖c tß║»t t├¡ch hß╗úp Weights & Biases trong Ultralytics."""
    from ultralytics import settings

    if disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        settings.update({"wandb": False})
        print("[W&B] ─É├ú tß║»t ghi log Weights & Biases (--no-wandb).")
        return

    os.environ.setdefault("WANDB_PROJECT", project_name)
    settings.update({"wandb": True})
    print(f"[W&B] Bß║¡t ghi log v├áo project: {os.environ['WANDB_PROJECT']}")


def main():
    ap = argparse.ArgumentParser(description="Huß║Ñn luyß╗çn YOLOv8 cho b├ái to├ín ph├ít hiß╗çn biß╗ân b├ío giao th├┤ng.")
    ap.add_argument("--data", default="data/yolo/data.yaml", help="─É╞░ß╗¥ng dß║½n tß╗¢i file data.yaml")
    ap.add_argument("--model", default="yolov8s.pt",
                    help="T├¬n m├┤ h├¼nh hoß║╖c file cß║Ñu h├¼nh (vd: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8s-p2.yaml)")
    ap.add_argument("--pretrained", default=None,
                    help="Trß╗ìng sß╗æ pretrained nß║íp v├áo khi d├╣ng file cß║Ñu h├¼nh .yaml (vd: yolov8s.pt cho yolov8s-p2.yaml)")
    ap.add_argument("--imgsz", type=int, default=640, help="K├¡ch th╞░ß╗¢c cß║ính d├ái ß║únh khi train (640 hoß║╖c 1280)")
    ap.add_argument("--epochs", type=int, default=50, help="Sß╗æ epoch huß║Ñn luyß╗çn")
    ap.add_argument("--batch", type=int, default=None,
                    help="Batch size (mß║╖c ─æß╗ïnh: 16 cho imgsz<=640, 8 cho imgsz>=1280)")
    ap.add_argument("--workers", type=int, default=4, help="Sß╗æ luß╗ông nß║íp dß╗» liß╗çu")
    ap.add_argument("--device", default=None, help="Thiß║┐t bß╗ï chß║íy (vd: '0', '0,1', hoß║╖c 'cpu'; mß║╖c ─æß╗ïnh tß╗▒ chß╗ìn)")
    ap.add_argument("--project", default=str(Path("runs/detect").resolve()), help="Th╞░ mß╗Ñc gß╗æc l╞░u kß║┐t quß║ú train")
    ap.add_argument("--name", default=None, help="T├¬n th╞░ mß╗Ñc th├¡ nghiß╗çm (mß║╖c ─æß╗ïnh tß╗▒ tß║ío tß╗½ model v├á imgsz)")
    ap.add_argument("--wandb-project", default="dl-traffic-sign-detection", help="T├¬n project tr├¬n Weights & Biases")
    ap.add_argument("--no-wandb", action="store_true", help="Tß║»t ghi log l├¬n W&B")

    # Si├¬u tham sß╗æ ph├╣ hß╗úp vß╗¢i ─æß║╖c th├╣ dß╗» liß╗çu Zalo AI 2020
    ap.add_argument("--fliplr", type=float, default=0.5,
                    help="X├íc suß║Ñt lß║¡t ngang (giß╗» 0.5 v├¼ lß╗¢p 'Cß║Ñm rß║╜' gß╗Öp cß║ú tr├íi lß║½n phß║úi)")
    ap.add_argument("--flipud", type=float, default=0.0,
                    help="X├íc suß║Ñt lß║¡t dß╗ìc (lu├┤n ─æß╗â 0.0 cho ß║únh camera h├ánh tr├¼nh)")
    ap.add_argument("--mosaic", type=float, default=1.0, help="X├íc suß║Ñt gh├⌐p ß║únh Mosaic")
    ap.add_argument("--close-mosaic", type=int, default=10, help="Tß║»t Mosaic ß╗ƒ N epoch cuß╗æi ─æß╗â ß╗òn ─æß╗ïnh tß╗ìa ─æß╗Ö box")
    ap.add_argument("--copy-paste", type=float, default=0.0, help="X├íc suß║Ñt augmentation copy-paste")
    ap.add_argument("--scale", type=float, default=0.5, help="Bi├¬n ─æß╗Ö co gi├ún tß╗ë lß╗ç ß║únh khi augment")
    ap.add_argument("--iou", type=float, default=0.65,
                    help="Ng╞░ß╗íng IoU cß╗ºa NMS khi validate (giß╗» 0.6-0.7 ─æß╗â tr├ính x├│a nhß║ºm biß╗ân ─æß╗⌐ng cß╗Ñm)")
    ap.add_argument("--patience", type=int, default=15, help="Sß╗æ epoch chß╗¥ early stopping")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="Tß╗ë lß╗ç tß║¡p train sß╗¡ dß╗Ñng (vd: 0.07 ~ 200 ß║únh ─æß╗â chß║íy thß╗¡ nhanh ß╗ƒ tuß║ºn 1)")
    ap.add_argument("--seed", type=int, default=0, help="Seed cß╗æ ─æß╗ïnh ─æß╗â t├íi tß║ío kß║┐t quß║ú")
    ap.add_argument("--exist-ok", action="store_true", help="Ghi ─æ├¿ v├áo th╞░ mß╗Ñc th├¡ nghiß╗çm nß║┐u ─æ├ú tß╗ôn tß║íi")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Kh├┤ng t├¼m thß║Ñy {data_path}. H├úy chß║íy b╞░ß╗¢c chuyß╗ân ─æß╗òi tr╞░ß╗¢c:\n"
            f"  python scripts/coco_to_yolo.py --splits data/splits --images za_traffic_2020/traffic_train/images --out data/yolo"
        )

    batch_size = args.batch if args.batch is not None else (8 if args.imgsz >= 1280 else 16)
    model_stem = Path(args.model).stem.replace("-", "_")
    exp_name = args.name or f"{model_stem}_{args.imgsz}"

    setup_wandb(args.wandb_project, args.no_wandb)

    from ultralytics import YOLO

    model = YOLO(args.model)
    if args.pretrained and args.model.endswith(".yaml"):
        print(f"Nß║íp trß╗ìng sß╗æ pretrained tß╗½ {args.pretrained} v├áo cß║Ñu h├¼nh {args.model}...")
        model = model.load(args.pretrained)

    train_kwargs = dict(
        data=str(data_path.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=batch_size,
        workers=args.workers,
        project=args.project,
        name=exp_name,
        fliplr=args.fliplr,
        flipud=args.flipud,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        copy_paste=args.copy_paste,
        scale=args.scale,
        iou=args.iou,
        agnostic_nms=False,  # Giß╗» NMS theo tß╗½ng lß╗¢p v├¼ c├│ nhiß╗üu cß║╖p biß╗ân kh├íc lß╗¢p ─æß╗⌐ng s├ít nhau
        patience=args.patience,
        fraction=args.fraction,
        seed=args.seed,
        exist_ok=args.exist_ok,
        plots=True,
        save=True,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    print(f"\n=== Bß║»t ─æß║ºu huß║Ñn luyß╗çn {exp_name} ===")
    print(f"  Model : {args.model}" + (f" (pretrained: {args.pretrained})" if args.pretrained else ""))
    print(f"  Data  : {data_path.resolve()}")
    print(f"  imgsz : {args.imgsz}, batch: {batch_size}, epochs: {args.epochs}, fraction: {args.fraction}")
    model.train(**train_kwargs)

    best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nHo├án tß║Ñt! Trß╗ìng sß╗æ tß╗æt nhß║Ñt l╞░u tß║íi: {best_pt}")
    print("B╞░ß╗¢c tiß║┐p theo ΓÇô xuß║Ñt dß╗▒ ─æo├ín tr├¬n tß║¡p val v├á chß║Ñm bß║▒ng pycocotools:")
    print(f"  python scripts/predict_yolo.py --weights {best_pt} --imgsz {args.imgsz} "
          f"--out preds/{exp_name}_val.json --eval --name {exp_name}")


if __name__ == "__main__":
    main()
