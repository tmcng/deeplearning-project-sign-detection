"""Xuất dự đoán từ YOLOv8 ra định dạng COCO JSON để chấm bằng scripts/evaluate.py (bàn giao #5).
Hỗ trợ cả suy luận chuẩn (Standard Inference) và suy luận cắt mảnh SAHI (Slicing Aided Hyper Inference - tuần 4).

Quy ước bắt buộc (đã thống nhất ở #1):
    [{"image_id": 12, "category_id": 3, "bbox": [x, y, w, h], "score": 0.91}, ...]
    - image_id    : đúng id trong data/splits/val.json (không dùng tên file)
    - category_id : id gốc của COCO (1..7), ánh xạ qua data/yolo/class_map.json
    - bbox        : [x, y, w, h] pixel trên ảnh gốc 1622x626
    - score       : xuất với ngưỡng thấp conf=0.001 để pycocotools tính đủ đường cong PR

Cách chạy:
    # 1. Xuất dự đoán baseline YOLOv8s (imgsz=640) và chấm điểm luôn
    python scripts/predict_yolo.py --weights runs/detect/yolov8s_640/weights/best.pt \
        --imgsz 640 --out preds/yolov8s_640_val.json --eval --name yolov8s_640

    # 2. Xuất dự đoán ở imgsz=1280
    python scripts/predict_yolo.py --weights runs/detect/yolov8s_1280/weights/best.pt \
        --imgsz 1280 --out preds/yolov8s_1280_val.json --eval --name yolov8s_1280

    # 3. Suy luận kết hợp SAHI (cắt mảnh 640x640 chồng lấp 20% + ghép với dự đoán toàn ảnh)
    python scripts/predict_yolo.py --weights runs/detect/yolov8s_640/weights/best.pt \
        --imgsz 640 --sahi --slice-size 640 --overlap 0.2 \
        --out preds/yolov8s_640_sahi_val.json --eval --name yolov8s_640_sahi
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def predict_standard(model, img_dir, name2id, cls_map, imgsz, conf, iou, max_det, device):
    """Suy luận chuẩn bằng Ultralytics YOLO.predict(stream=True)."""
    preds = []
    kwargs = dict(
        source=str(img_dir),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        agnostic_nms=False,
        max_det=max_det,
        stream=True,
        verbose=False,
    )
    if device is not None:
        kwargs["device"] = device

    for r in model.predict(**kwargs):
        stem = Path(r.path).stem
        if stem not in name2id:
            continue
        img_id = name2id[stem]
        for (x1, y1, x2, y2), c, s in zip(
            r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()
        ):
            preds.append({
                "image_id": img_id,
                "category_id": cls_map[int(c)],
                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                "score": round(float(s), 5),
            })
    return preds


def generate_slices(W, H, slice_w, slice_h, overlap):
    """Sinh danh sách cửa sổ cắt [x0, y0, x1, y1] phủ kín ảnh WxH."""
    step_x = max(1, int(slice_w * (1.0 - overlap)))
    step_y = max(1, int(slice_h * (1.0 - overlap)))
    windows = []
    y0 = 0
    while y0 < H:
        y1 = min(H, y0 + slice_h)
        x0 = 0
        while x0 < W:
            x1 = min(W, x0 + slice_w)
            windows.append((max(0, x1 - slice_w), max(0, y1 - slice_h), x1, y1))
            if x1 == W:
                break
            x0 += step_x
        if y1 == H:
            break
        y0 += step_y
    # Loại các cửa sổ trùng lặp nếu ảnh nhỏ hơn kích thước slice
    return list(dict.fromkeys(windows))


def predict_sahi(weights_path, model, img_dir, gt_images, cls_map, imgsz, conf, iou,
                 slice_size, overlap, max_det, device):
    """Suy luận cắt mảnh (SAHI): kết hợp dự đoán trên toàn ảnh và trên các mảnh cắt chồng lấp,
    sau đó gộp bằng NMS theo từng lớp (class-wise NMS)."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.ops import batched_nms

    preds = []
    img_paths = {p.stem: p for p in Path(img_dir).iterdir() if p.is_file()}

    print(f"[SAHI] Chạy suy luận cắt mảnh {slice_size}x{slice_size} (overlap={overlap}) trên {len(gt_images)} ảnh...")
    for idx, im_info in enumerate(gt_images, 1):
        stem = Path(im_info["file_name"]).stem
        img_path = img_paths.get(stem)
        if img_path is None:
            continue
        img_id = im_info["id"]
        pil_img = Image.open(img_path).convert("RGB")
        W, H = pil_img.size
        np_img = np.asarray(pil_img)[:, :, ::-1]  # RGB -> BGR cho Ultralytics

        windows = generate_slices(W, H, slice_size, slice_size, overlap)
        crops = [np_img[y0:y1, x0:x1] for (x0, y0, x1, y1) in windows]
        # Thêm cả ảnh gốc vào cuối danh sách để không mất ngữ cảnh với biển vừa/lớn
        crops.append(np_img)
        offsets = [(x0, y0) for (x0, y0, _, _) in windows] + [(0, 0)]

        pred_kwargs = dict(imgsz=imgsz, conf=conf, iou=iou, agnostic_nms=False, verbose=False)
        if device is not None:
            pred_kwargs["device"] = device
        results = model.predict(crops, **pred_kwargs)

        all_boxes, all_scores, all_cls = [], [], []
        for r, (ox, oy) in zip(results, offsets):
            if len(r.boxes) == 0:
                continue
            b = r.boxes.xyxy.cpu().clone()
            b[:, [0, 2]] += ox
            b[:, [1, 3]] += oy
            all_boxes.append(b)
            all_scores.append(r.boxes.conf.cpu())
            all_cls.append(r.boxes.cls.cpu().to(torch.int64))

        if not all_boxes:
            continue

        boxes_t = torch.cat(all_boxes, dim=0)
        scores_t = torch.cat(all_scores, dim=0)
        cls_t = torch.cat(all_cls, dim=0)

        keep = batched_nms(boxes_t, scores_t, cls_t, iou_threshold=iou)[:max_det]
        for i in keep.tolist():
            x1, y1, x2, y2 = boxes_t[i].tolist()
            c = int(cls_t[i].item())
            s = float(scores_t[i].item())
            preds.append({
                "image_id": img_id,
                "category_id": cls_map[c],
                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                "score": round(s, 5),
            })

        if idx % 100 == 0 or idx == len(gt_images):
            print(f"  Đã xử lý {idx}/{len(gt_images)} ảnh...")
    return preds


def main():
    ap = argparse.ArgumentParser(description="Xuất dự đoán YOLOv8 ra COCO JSON và chấm điểm.")
    ap.add_argument("--weights", required=True, help="Đường dẫn tới file trọng số best.pt")
    ap.add_argument("--gt", default="data/splits/val.json", help="File COCO JSON của tập cần dự đoán (val hoặc test)")
    ap.add_argument("--images", default="data/yolo/images/val",
                    help="Thư mục chứa ảnh của tập cần dự đoán (vd: data/yolo/images/val hoặc za_traffic_2020/traffic_train/images)")
    ap.add_argument("--class-map", default="data/yolo/class_map.json",
                    help="File ánh xạ chỉ số lớp YOLO (0..6) -> category_id COCO (1..7)")
    ap.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh suy luận (khớp với lúc train: 640 hoặc 1280)")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="Ngưỡng độ tin cậy tối thiểu (giữ 0.001 khi chấm mAP)")
    ap.add_argument("--iou", type=float, default=0.65,
                    help="Ngưỡng IoU của NMS (giữ 0.6-0.7 để tránh xóa nhầm biển đứng cụm)")
    ap.add_argument("--max-det", type=int, default=300, help="Số box tối đa mỗi ảnh")
    ap.add_argument("--device", default=None, help="Thiết bị chạy (vd: '0' hoặc 'cpu')")
    ap.add_argument("--out", required=True, help="Đường dẫn file JSON đầu ra (vd: preds/yolov8s_640_val.json)")

    # Tùy chọn mở rộng tuần 4: Suy luận cắt mảnh (SAHI)
    ap.add_argument("--sahi", action="store_true", help="Bật suy luận cắt mảnh SAHI cho ảnh độ phân giải cao")
    ap.add_argument("--slice-size", type=int, default=640, help="Kích thước mảnh cắt vuông khi dùng --sahi")
    ap.add_argument("--overlap", type=float, default=0.2, help="Tỉ lệ chồng lấp giữa các mảnh cắt khi dùng --sahi")

    # Tùy chọn chấm điểm ngay bằng scripts/evaluate.py
    ap.add_argument("--eval", action="store_true", help="Gọi luôn scripts/evaluate.py sau khi xuất JSON")
    ap.add_argument("--name", default=None, help="Tên thí nghiệm truyền vào evaluate.py (mặc định lấy từ tên file --out)")
    args = ap.parse_args()

    gt_data = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    name2id = {Path(im["file_name"]).stem: im["id"] for im in gt_data["images"]}

    cls_map_path = Path(args.class_map)
    if cls_map_path.exists():
        cls_map = {int(k): int(v) for k, v in json.loads(cls_map_path.read_text(encoding="utf-8")).items()}
    else:
        # Dự phòng lấy trực tiếp từ gt['categories'] nếu chưa có class_map.json
        cats = sorted(gt_data["categories"], key=lambda c: c["id"])
        cls_map = {i: int(c["id"]) for i, c in enumerate(cats)}

    from ultralytics import YOLO

    weights_path = Path(args.weights)
    if not weights_path.exists():
        # Tự động tìm lại trong runs/detect/runs/detect/... hoặc /kaggle/input nếu Ultralytics lưu lồng thư mục
        alt = Path("runs/detect") / weights_path
        if alt.exists():
            weights_path = alt
        else:
            exp_dir = weights_path.parent.parent.name
            for root in [Path("runs"), Path("/kaggle/input")]:
                if root.exists():
                    matches = list(root.rglob(f"{exp_dir}/weights/{weights_path.name}"))
                    if matches:
                        weights_path = matches[0]
                        break
        if weights_path.exists():
            print(f"[Auto-resolve] Đã tìm thấy trọng số tại: {weights_path}")

    model = YOLO(str(weights_path))

    if args.sahi:
        preds = predict_sahi(
            weights_path=args.weights,
            model=model,
            img_dir=args.images,
            gt_images=gt_data["images"],
            cls_map=cls_map,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            slice_size=args.slice_size,
            overlap=args.overlap,
            max_det=args.max_det,
            device=args.device,
        )
    else:
        print(f"Đang chạy dự đoán bằng {args.weights} trên {args.images} (imgsz={args.imgsz}, conf={args.conf})...")
        preds = predict_standard(
            model=model,
            img_dir=args.images,
            name2id=name2id,
            cls_map=cls_map,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(preds), encoding="utf-8")
    print(f"Đã ghi {len(preds)} dự đoán vào {out_path}")

    if args.eval:
        exp_name = args.name or out_path.stem.replace("_val", "").replace("_test", "")
        eval_script = Path(__file__).parent / "evaluate.py"
        subprocess.run(
            [sys.executable, str(eval_script), "--gt", args.gt, "--pred", str(out_path), "--name", exp_name],
            check=True,
        )


if __name__ == "__main__":
    main()
