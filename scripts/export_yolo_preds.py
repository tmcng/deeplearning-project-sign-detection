"""Xuất dự đoán của một mô hình YOLO (Ultralytics) ra file JSON đúng định dạng của evaluate.py.

Cách chạy:
    python scripts/export_yolo_preds.py --weights runs/detect/train/weights/best.pt \
        --gt data/splits/val.json --images data/yolo/images/val \
        --class-map data/yolo/class_map.json --imgsz 1280 --out preds/yolov8s_1280_val.json

Những điều script lo sẵn (đã thống nhất ở bàn giao #1):
    - category_id: đổi chỉ số YOLO (0..6) về id gốc của COCO bằng class_map.json
    - bbox: đổi [x1, y1, x2, y2] sang [x, y, w, h], pixel trên ảnh gốc
    - conf=0.001 để mAP tính đủ đường precision-recall
    - imgsz phải BẰNG imgsz lúc train
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gt", required=True, help="COCO JSON của đúng tập đang dự đoán")
    ap.add_argument("--images", required=True, help="thư mục ảnh của tập đó")
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    name2id = {Path(im["file_name"]).stem: im["id"] for im in gt["images"]}
    cls_map = {int(k): v for k, v in json.loads(Path(args.class_map).read_text()).items()}

    model = YOLO(args.weights)
    preds, n_img = [], 0
    for r in model.predict(args.images, imgsz=args.imgsz, conf=args.conf, max_det=300,
                           device=args.device, stream=True, verbose=False):
        stem = Path(r.path).stem
        if stem not in name2id:
            continue
        n_img += 1
        img_id = name2id[stem]
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                                          r.boxes.conf.tolist()):
            preds.append({"image_id": img_id, "category_id": cls_map[int(c)],
                          "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})

    if n_img != len(gt["images"]):
        print(f"CẢNH BÁO: dự đoán {n_img} ảnh nhưng tập có {len(gt['images'])} ảnh. "
              f"--images có đúng thư mục của tập này không?")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(preds), encoding="utf-8")
    print(f"{n_img} ảnh, {len(preds)} dự đoán -> {args.out}")


if __name__ == "__main__":
    main()
