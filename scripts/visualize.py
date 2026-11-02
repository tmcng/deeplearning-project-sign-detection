"""Vẽ nhãn thật (xanh lá) và dự đoán (đỏ) lên ảnh để xem tận mắt, phục vụ EDA và phân tích lỗi.

Cách chạy:
    # Chỉ xem nhãn (tuần 1, kiểm tra dữ liệu và script chuyển đổi)
    python scripts/visualize.py --gt data/splits/val.json --images data/raw/images --n 20

    # Nhãn + dự đoán, chỉ lấy ảnh có biển nhỏ bị bỏ sót (tuần 3, phân tích lỗi)
    python scripts/visualize.py --gt data/splits/val.json --images data/raw/images \
        --pred preds/yolov8s_640_val.json --only-missed-small --n 20

Mỗi ảnh lưu thêm một bản phóng to quanh từng biển, vì biển báo rất nhỏ so với cả ảnh.
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--pred", default=None)
    ap.add_argument("--score", type=float, default=0.3, help="chỉ vẽ dự đoán có score >= ngưỡng")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--only-missed-small", action="store_true",
                    help="chỉ lấy ảnh có biển nhỏ (<32px) không được dự đoán đúng (IoU>=0.5)")
    ap.add_argument("--out", default="reports/vis")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    coco = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    gts, preds = {}, {}
    for a in coco["annotations"]:
        gts.setdefault(a["image_id"], []).append(a)
    if args.pred:
        for p in json.loads(Path(args.pred).read_text(encoding="utf-8")):
            if p["score"] >= args.score:
                preds.setdefault(p["image_id"], []).append(p)

    def missed_small(img_id):
        for g in gts.get(img_id, []):
            if g["bbox"][2] * g["bbox"][3] >= 32 ** 2:
                continue
            ok = any(p["category_id"] == g["category_id"] and iou(p["bbox"], g["bbox"]) >= 0.5
                     for p in preds.get(img_id, []))
            if not ok:
                return True
        return False

    candidates = [im for im in coco["images"] if im["id"] in gts]
    if args.only_missed_small:
        candidates = [im for im in candidates if missed_small(im["id"])]
        print(f"{len(candidates)} ảnh có biển nhỏ bị bỏ sót")
    random.Random(args.seed).shuffle(candidates)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for im in candidates[:args.n]:
        path = Path(args.images) / im["file_name"]
        if not path.exists():
            print(f"Thiếu ảnh {path}")
            continue
        img = Image.open(path).convert("RGB")
        d = ImageDraw.Draw(img)
        for g in gts.get(im["id"], []):
            x, y, w, h = g["bbox"]
            d.rectangle([x, y, x + w, y + h], outline=(0, 200, 0), width=2)
            d.text((x, y + h + 2), cats[g["category_id"]], fill=(0, 200, 0))
        for p in preds.get(im["id"], []):
            x, y, w, h = p["bbox"]
            d.rectangle([x, y, x + w, y + h], outline=(230, 0, 0), width=2)
            d.text((x, max(0, y - 12)), f"{cats.get(p['category_id'], '?')} {p['score']:.2f}",
                   fill=(230, 0, 0))
        stem = Path(im["file_name"]).stem
        img.save(out / f"{stem}.jpg", quality=90)

        # Bản phóng to quanh từng biển (crop gấp 4 lần kích thước biển, phóng lên 256px)
        for i, g in enumerate(gts.get(im["id"], [])):
            x, y, w, h = g["bbox"]
            cx, cy, r = x + w / 2, y + h / 2, max(w, h, 16) * 2
            crop = img.crop((cx - r, cy - r, cx + r, cy + r)).resize((256, 256), Image.NEAREST)
            crop.save(out / f"{stem}_zoom{i}.jpg", quality=90)
    print(f"Đã lưu ảnh vào {out}/  (xanh lá = nhãn thật, đỏ = dự đoán)")


if __name__ == "__main__":
    main()
