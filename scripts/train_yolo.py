"""Train một mô hình YOLO (Ultralytics) rồi xuất dự đoán theo định dạng chung để chấm bằng evaluate.py.

Dùng cho thí nghiệm rò rỉ dữ liệu (tuần 2) và mất cân bằng lớp (tuần 4). Chạy trên Kaggle/Colab có GPU.

Cách chạy:
    python scripts/train_yolo.py --yolo data/yolo --splits data/splits --name yolov8n_street
    python scripts/train_yolo.py --yolo data/yolo_random --splits data/splits_random --name yolov8n_random

Kết quả:
    <project>/<name>/                : log, biểu đồ, weights/best.pt của Ultralytics
    preds/<name>_val.json            : dự đoán trên val, đúng định dạng của evaluate.py
    preds/<name>_test.json           : chỉ khi có --export-test (để dành tuần 5, KHÔNG chấm trước)

W&B: đặt biến môi trường WANDB_API_KEY (Kaggle: Add-ons > Secrets) rồi thêm --wandb.
Tên project W&B = --project, tên run = --name.

Hai thí nghiệm muốn so được với nhau thì phải giữ nguyên model, epochs, imgsz, batch, seed;
chỉ đổi dữ liệu (hoặc đúng một thứ đang thử nghiệm).
"""
import argparse
import json
from pathlib import Path


def export_preds(model, coco_json, img_dir, class_map, imgsz, out_file):
    """Dự đoán với conf rất thấp (mAP cần cả dự đoán điểm thấp), đổi lớp YOLO về category_id gốc,
    đổi box [x1, y1, x2, y2] về [x, y, w, h] pixel trên ảnh gốc."""
    gt = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    name2id = {Path(im["file_name"]).stem: im["id"] for im in gt["images"]}
    preds = []
    for r in model.predict(str(img_dir), imgsz=imgsz, conf=0.001, max_det=300, stream=True, verbose=False):
        img_id = name2id[Path(r.path).stem]
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            preds.append({"image_id": img_id, "category_id": class_map[int(c)],
                          "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text(json.dumps(preds), encoding="utf-8")
    print(f"Đã ghi {len(preds)} dự đoán vào {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", required=True, help="thư mục do coco_to_yolo.py tạo (có data.yaml)")
    ap.add_argument("--splits", required=True, help="thư mục COCO JSON tương ứng (để lấy image_id)")
    ap.add_argument("--name", required=True, help="tên thí nghiệm, vd. yolov8n_street")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="sign-detection", help="tên project W&B, cũng là thư mục lưu run")
    ap.add_argument("--wandb", action="store_true", help="ghi log lên W&B")
    ap.add_argument("--export-test", action="store_true", help="xuất thêm dự đoán trên test (tuần 5)")
    args = ap.parse_args()

    from ultralytics import YOLO, settings
    settings.update({"wandb": args.wandb})

    yolo_dir = Path(args.yolo)
    class_map = {int(k): v for k, v in
                 json.loads((yolo_dir / "class_map.json").read_text(encoding="utf-8")).items()}

    model = YOLO(args.model)
    # fliplr giữ mặc định (0.5): 7 lớp đã gộp trái/phải nên lật ngang không đổi lớp
    model.train(data=str(yolo_dir / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                seed=args.seed, deterministic=True, project=args.project, name=args.name,
                exist_ok=True)

    best = YOLO(str(Path(model.trainer.save_dir) / "weights" / "best.pt"))
    sets = ["val"] + (["test"] if args.export_test else [])
    for s in sets:
        export_preds(best, Path(args.splits) / f"{s}.json", yolo_dir / "images" / s, class_map,
                     args.imgsz, Path("preds") / f"{args.name}_{s}.json")


if __name__ == "__main__":
    main()
