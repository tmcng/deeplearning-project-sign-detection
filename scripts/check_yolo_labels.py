"""Vẽ lại box từ file nhãn YOLO (.txt) để chắc chắn bước chuyển COCO -> YOLO không sai.

Cách chạy:
    python scripts/check_yolo_labels.py --yolo data/yolo --split train --n 5

Mở ảnh trong reports/check_yolo/: box phải ôm khít biển báo. Nếu box lệch, sai kích thước
hoặc nằm chỗ trống thì script chuyển đổi có lỗi, KHÔNG được giao cho B.
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def read_names(yaml_path):
    """Đọc phần names trong data.yaml (không cần cài thư viện yaml)."""
    names, inside = {}, False
    for line in Path(yaml_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("names:"):
            inside = True
            continue
        if inside and line.startswith("  ") and ":" in line:
            k, v = line.strip().split(":", 1)
            names[int(k)] = v.strip().strip('"')
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="data/yolo")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="reports/check_yolo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.yolo)
    names = read_names(root / "data.yaml")
    labels = sorted((root / "labels" / args.split).glob("*.txt"))
    labels = [p for p in labels if p.read_text().strip()]  # bỏ ảnh không có box
    random.Random(args.seed).shuffle(labels)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for lbl in labels[:args.n]:
        imgs = list((root / "images" / args.split).glob(lbl.stem + ".*"))
        if not imgs:
            print(f"Thiếu ảnh cho {lbl.name}")
            continue
        img = Image.open(imgs[0]).convert("RGB")
        W, H = img.size
        d = ImageDraw.Draw(img)
        for line in lbl.read_text().splitlines():
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
            d.rectangle(box, outline=(255, 0, 255), width=2)
            d.text((box[0], box[3] + 2), names.get(int(c), c), fill=(255, 0, 255))
        img.save(out / f"{lbl.stem}.jpg", quality=90)
    print(f"Đã lưu {min(args.n, len(labels))} ảnh vào {out}/ (box tím vẽ từ nhãn YOLO)")


if __name__ == "__main__":
    main()
