"""Script huấn luyện YOLOv8 chuẩn hóa cho bài toán phát hiện biển báo (Thành viên B, bàn giao #5 & #7).

Hỗ trợ:
    - Chạy thử nhanh (smoke test tuần 1) trên ~200 ảnh: --model yolov8n.pt --fraction 0.07 --epochs 5
    - Baseline tuần 2: --model yolov8s.pt --imgsz 640 --epochs 50
    - Phương pháp chính tuần 3 (ảnh lớn / đầu ra P2 cho biển nhỏ):
        + --model yolov8s.pt --imgsz 1280 --batch 8
        + --model yolov8s-p2.yaml --pretrained yolov8s.pt --imgsz 1280 --batch 8
    - Tự động ghi log lên Weights & Biases (W&B) nếu đã đăng nhập, hoặc tắt bằng --no-wandb.

Cách chạy:
    # 1. Baseline YOLOv8s ở imgsz=640
    python scripts/train_yolo.py --data data/yolo/data.yaml --model yolov8s.pt --imgsz 640 --name yolov8s_640

    # 2. YOLOv8s ở imgsz=1280 (giảm batch để vừa VRAM 16GB trên Kaggle T4/P100)
    python scripts/train_yolo.py --data data/yolo/data.yaml --model yolov8s.pt --imgsz 1280 --batch 8 --name yolov8s_1280

    # 3. Kiến trúc YOLOv8s-P2 (thêm tầng stride 4 bắt biển báo rất nhỏ < 10px)
    python scripts/train_yolo.py --data data/yolo/data.yaml --model yolov8s-p2.yaml --pretrained yolov8s.pt --imgsz 1280 --batch 8 --name yolov8s_p2_1280
"""
import argparse
import os
from pathlib import Path


def setup_wandb(project_name, disable_wandb):
    """Bật hoặc tắt tích hợp Weights & Biases trong Ultralytics."""
    from ultralytics import settings

    if disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        settings.update({"wandb": False})
        print("[W&B] Đã tắt ghi log Weights & Biases (--no-wandb).")
        return

    os.environ.setdefault("WANDB_PROJECT", project_name)
    settings.update({"wandb": True})
    print(f"[W&B] Bật ghi log vào project: {os.environ['WANDB_PROJECT']}")


def main():
    ap = argparse.ArgumentParser(description="Huấn luyện YOLOv8 cho bài toán phát hiện biển báo giao thông.")
    ap.add_argument("--data", default="data/yolo/data.yaml", help="Đường dẫn tới file data.yaml")
    ap.add_argument("--model", default="yolov8s.pt",
                    help="Tên mô hình hoặc file cấu hình (vd: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8s-p2.yaml)")
    ap.add_argument("--pretrained", default=None,
                    help="Trọng số pretrained nạp vào khi dùng file cấu hình .yaml (vd: yolov8s.pt cho yolov8s-p2.yaml)")
    ap.add_argument("--imgsz", type=int, default=640, help="Kích thước cạnh dài ảnh khi train (640 hoặc 1280)")
    ap.add_argument("--epochs", type=int, default=50, help="Số epoch huấn luyện")
    ap.add_argument("--batch", type=int, default=None,
                    help="Batch size (mặc định: 16 cho imgsz<=640, 8 cho imgsz>=1280)")
    ap.add_argument("--workers", type=int, default=4, help="Số luồng nạp dữ liệu")
    ap.add_argument("--device", default=None, help="Thiết bị chạy (vd: '0', '0,1', hoặc 'cpu'; mặc định tự chọn)")
    ap.add_argument("--project", default=str(Path("runs/detect").resolve()), help="Thư mục gốc lưu kết quả train")
    ap.add_argument("--name", default=None, help="Tên thư mục thí nghiệm (mặc định tự tạo từ model và imgsz)")
    ap.add_argument("--wandb-project", default="dl-traffic-sign-detection", help="Tên project trên Weights & Biases")
    ap.add_argument("--no-wandb", action="store_true", help="Tắt ghi log lên W&B")

    # Siêu tham số phù hợp với đặc thù dữ liệu Zalo AI 2020
    ap.add_argument("--fliplr", type=float, default=0.5,
                    help="Xác suất lật ngang (giữ 0.5 vì lớp 'Cấm rẽ' gộp cả trái lẫn phải)")
    ap.add_argument("--flipud", type=float, default=0.0,
                    help="Xác suất lật dọc (luôn để 0.0 cho ảnh camera hành trình)")
    ap.add_argument("--mosaic", type=float, default=1.0, help="Xác suất ghép ảnh Mosaic")
    ap.add_argument("--close-mosaic", type=int, default=10, help="Tắt Mosaic ở N epoch cuối để ổn định tọa độ box")
    ap.add_argument("--copy-paste", type=float, default=0.0, help="Xác suất augmentation copy-paste")
    ap.add_argument("--scale", type=float, default=0.5, help="Biên độ co giãn tỉ lệ ảnh khi augment")
    ap.add_argument("--iou", type=float, default=0.65,
                    help="Ngưỡng IoU của NMS khi validate (giữ 0.6-0.7 để tránh xóa nhầm biển đứng cụm)")
    ap.add_argument("--patience", type=int, default=15, help="Số epoch chờ early stopping")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="Tỉ lệ tập train sử dụng (vd: 0.07 ~ 200 ảnh để chạy thử nhanh ở tuần 1)")
    ap.add_argument("--seed", type=int, default=0, help="Seed cố định để tái tạo kết quả")
    ap.add_argument("--exist-ok", action="store_true", help="Ghi đè vào thư mục thí nghiệm nếu đã tồn tại")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Không tìm thấy {data_path}. Hãy chạy bước chuyển đổi trước:\n"
            f"  python scripts/coco_to_yolo.py --splits data/splits --images za_traffic_2020/traffic_train/images --out data/yolo"
        )

    batch_size = args.batch if args.batch is not None else (8 if args.imgsz >= 1280 else 16)
    model_stem = Path(args.model).stem.replace("-", "_")
    exp_name = args.name or f"{model_stem}_{args.imgsz}"

    setup_wandb(args.wandb_project, args.no_wandb)

    from ultralytics import YOLO

    model = YOLO(args.model)
    if args.pretrained and args.model.endswith(".yaml"):
        print(f"Nạp trọng số pretrained từ {args.pretrained} vào cấu hình {args.model}...")
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
        agnostic_nms=False,  # Giữ NMS theo từng lớp vì có nhiều cặp biển khác lớp đứng sát nhau
        patience=args.patience,
        fraction=args.fraction,
        seed=args.seed,
        exist_ok=args.exist_ok,
        plots=True,
        save=True,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    print(f"\n=== Bắt đầu huấn luyện {exp_name} ===")
    print(f"  Model : {args.model}" + (f" (pretrained: {args.pretrained})" if args.pretrained else ""))
    print(f"  Data  : {data_path.resolve()}")
    print(f"  imgsz : {args.imgsz}, batch: {batch_size}, epochs: {args.epochs}, fraction: {args.fraction}")
    model.train(**train_kwargs)

    best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nHoàn tất! Trọng số tốt nhất lưu tại: {best_pt}")
    print("Bước tiếp theo – xuất dự đoán trên tập val và chấm bằng pycocotools:")
    print(f"  python scripts/predict_yolo.py --weights {best_pt} --imgsz {args.imgsz} "
          f"--out preds/{exp_name}_val.json --eval --name {exp_name}")


if __name__ == "__main__":
    main()
