"""Script đánh giá chung cho CẢ HAI mô hình (bàn giao #4). Chấm bằng pycocotools.

Cách chạy:
    python scripts/evaluate.py --gt data/splits/val.json --pred preds/yolov8s_640_val.json \
        --name yolov8s_640 --out reports/eval

Định dạng file dự đoán (B và C phải xuất đúng như này, đã thống nhất ở #1):
    [
      {"image_id": 12, "category_id": 3, "bbox": [x, y, w, h], "score": 0.91},
      ...
    ]
    - image_id    : đúng id trong file COCO của tập đang chấm
    - category_id : id lớp GỐC của COCO (YOLO phải đổi ngược qua data/yolo/class_map.json)
    - bbox        : góc trên-trái + rộng, cao, tính bằng PIXEL trên ảnh GỐC (không chuẩn hóa)
    - score       : độ tin cậy; nên giữ ngưỡng thấp (vd. conf=0.001) khi xuất, đừng lọc 0.25

Kết quả:
    in ra bảng chỉ số; ghi reports/eval/<name>.json và nối một dòng vào reports/eval/results.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def check_preds(preds, gt):
    """Bắt các lỗi định dạng hay gặp trước khi chấm."""
    img_ids = set(gt.getImgIds())
    cat_ids = set(gt.getCatIds())
    bad_img = sum(p["image_id"] not in img_ids for p in preds)
    bad_cat = {p["category_id"] for p in preds} - cat_ids
    if bad_img:
        raise SystemExit(f"Lỗi: {bad_img} dự đoán có image_id không có trong ground truth.")
    if bad_cat:
        raise SystemExit(f"Lỗi: category_id {sorted(bad_cat)} không tồn tại. "
                         f"Các id hợp lệ: {sorted(cat_ids)}. YOLO đã đổi qua class_map.json chưa?")
    bb = np.array([p["bbox"] for p in preds], dtype=float)
    if len(bb) and bb[:, 2:].max() <= 1.0:
        raise SystemExit("Lỗi: bbox có vẻ đã chuẩn hóa về 0-1. Cần tọa độ pixel trên ảnh gốc.")


def ap_from_precision(precision, k, iou_index=None, area_index=0):
    """precision có shape [IoU(10), recall(101), lớp K, vùng(4), maxDets(3)]."""
    p = precision[:, :, k, area_index, -1] if iou_index is None else precision[iou_index, :, k, area_index, -1]
    p = p[p > -1]
    return float(p.mean()) if p.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="COCO JSON của tập val hoặc test")
    ap.add_argument("--pred", required=True, help="file JSON dự đoán")
    ap.add_argument("--name", required=True, help="tên thí nghiệm, vd. yolov8s_640")
    ap.add_argument("--out", default="reports/eval")
    ap.add_argument("--small-thr", type=float, default=32,
                    help="cạnh (px) dưới mức này tính là 'nhỏ'; mặc định 32 theo COCO")
    args = ap.parse_args()

    # Tự đọc JSON bằng UTF-8: COCO(path) dùng bảng mã mặc định của máy, trên Windows
    # (cp1252) sẽ lỗi với tên lớp tiếng Việt.
    gt = COCO()
    gt.dataset = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    gt.createIndex()
    preds = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    if not preds:
        raise SystemExit("File dự đoán rỗng.")
    check_preds(preds, gt)
    dt = gt.loadRes(preds)

    E = COCOeval(gt, dt, iouType="bbox")
    s, m = args.small_thr ** 2, 96 ** 2
    E.params.areaRng = [[0, 1e10], [0, s], [s, m], [m, 1e10]]
    E.evaluate()
    E.accumulate()
    E.summarize()
    st = E.stats  # 0:mAP .5:.95, 1:mAP .5, 3/4/5: AP small/medium/large

    precision = E.eval["precision"]
    cat_names = {int(c["id"]): c["name"] for c in gt.dataset["categories"]}
    per_class = {}
    for k, cid in enumerate(E.params.catIds):
        # cid có thể là kiểu số của numpy, nên tra bằng int() thay vì gt.loadCats()
        name = cat_names[int(cid)]
        per_class[name] = {
            "AP50": ap_from_precision(precision, k, iou_index=0),
            "AP50_95": ap_from_precision(precision, k),
            "AP50_small": ap_from_precision(precision, k, iou_index=0, area_index=1),
        }

    result = {
        "name": args.name,
        "gt": args.gt,
        "small_thr_px": args.small_thr,
        "mAP50": st[1], "mAP50_95": st[0],
        "AP_small": st[3], "AP_medium": st[4], "AP_large": st[5],
        "AR100": st[8],
        "per_class": per_class,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                           encoding="utf-8")

    fmt = lambda v: "–" if v is None or v < 0 or np.isnan(v) else f"{v * 100:.1f}"
    print(f"\n{'Lớp':<35} AP50  AP50:95  AP50(nhỏ)")
    for name, r in per_class.items():
        print(f"{name:<35} {fmt(r['AP50']):>5} {fmt(r['AP50_95']):>7} {fmt(r['AP50_small']):>9}")

    table = out / "results.md"
    if not table.exists():
        table.write_text("| Thí nghiệm | Tập | mAP@0.5 | mAP@0.5:0.95 | AP_small | AP_medium | AP_large |\n"
                         "| --- | --- | --- | --- | --- | --- | --- |\n", encoding="utf-8")
    with table.open("a", encoding="utf-8") as f:
        f.write(f"| {args.name} | {Path(args.gt).stem} | {fmt(st[1])} | {fmt(st[0])} | "
                f"{fmt(st[3])} | {fmt(st[4])} | {fmt(st[5])} |\n")
    print(f"\nĐã lưu {out}/{args.name}.json và thêm một dòng vào {table}")


if __name__ == "__main__":
    main()