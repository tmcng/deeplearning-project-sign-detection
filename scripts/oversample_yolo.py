"""Oversampling lớp hiếm cho tập train YOLO (thí nghiệm mất cân bằng lớp, tuần 4).

Cách chạy (trên thư mục do coco_to_yolo.py hoặc tile_yolo_dataset.py tạo ra):
    python scripts/oversample_yolo.py --yolo data/yolo_tiled --thr 0.3
    # rồi train như bình thường, chỉ đổi file data:
    python scripts/train_yolov8.py --data data/yolo_tiled/data_rfs.yaml ...

Kết quả (ghi vào chính thư mục --yolo, không sửa ảnh hay nhãn):
    train_rfs.txt   : danh sách ảnh train, ảnh chứa lớp hiếm xuất hiện nhiều lần
    data_rfs.yaml   : giống data.yaml nhưng train trỏ tới train_rfs.txt (val/test giữ nguyên)
Ultralytics không loại đường dẫn trùng trong file danh sách, nên lặp dòng = lặp ảnh trong mỗi epoch.

Cách tính số lần lặp (Repeat Factor Sampling, Gupta et al., LVIS, CVPR 2019):
    f_c = tỉ lệ ảnh train có chứa lớp c
    r_c = max(1, sqrt(thr / f_c))          lớp có f_c >= thr không được lặp
    r_i = max r_c trên các lớp có trong ảnh i (ảnh nền: 1)
    r_i lẻ thì làm tròn ngẫu nhiên (vd. 1.4 -> 2 với xác suất 0.4), cố định bằng --seed.
Lặp ảnh thì cũng lặp luôn các lớp phổ biến đứng chung ảnh, nên script in số box mỗi lớp
trước và sau để biết mức cân bằng thực sự đạt được.
"""
import argparse
import math
import random
from collections import Counter
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", required=True, help="thư mục YOLO có data.yaml, images/train, labels/train")
    ap.add_argument("--thr", type=float, default=0.3,
                    help="lớp xuất hiện trong ít hơn tỉ lệ ảnh này thì được lặp; lớn hơn = lặp mạnh hơn")
    ap.add_argument("--name", default="rfs", help="hậu tố file ra: train_<name>.txt, data_<name>.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.yolo)
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    names = names if isinstance(names, dict) else dict(enumerate(names))

    img_dir, lbl_dir = root / "images" / "train", root / "labels" / "train"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    classes_of = []
    for p in images:
        lbl = lbl_dir / f"{p.stem}.txt"
        rows = lbl.read_text(encoding="utf-8").split("\n") if lbl.exists() else []
        classes_of.append([int(r.split()[0]) for r in rows if r.strip()])

    n_img = len(images)
    img_freq = Counter(c for cls in classes_of for c in set(cls))
    r_c = {c: max(1.0, math.sqrt(args.thr / (img_freq[c] / n_img))) for c in names if img_freq[c]}

    rng = random.Random(args.seed)
    lines, before, after = [], Counter(), Counter()
    for p, cls in zip(images, classes_of):
        r = max((r_c[c] for c in set(cls)), default=1.0)
        k = int(r) + (rng.random() < r - int(r))
        lines += [f"./images/train/{p.name}"] * k
        for c in cls:
            before[c] += 1
            after[c] += k

    (root / f"train_{args.name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_yaml = dict(data, train=f"train_{args.name}.txt")
    (root / f"data_{args.name}.yaml").write_text(yaml.safe_dump(out_yaml, allow_unicode=True, sort_keys=False),
                                                 encoding="utf-8")

    print(f"thr = {args.thr}: {n_img} ảnh train -> {len(lines)} dòng ({len(lines) / n_img:.2f} lần)\n")
    print(f"{'Lớp':<20} {'% ảnh có lớp':>13} {'r_c':>6} {'box trước':>10} {'box sau':>9} {'tăng':>6}")
    for c in sorted(names):
        f = img_freq[c] / n_img
        print(f"{names[c]:<20} {f:>12.1%} {r_c.get(c, 1.0):>6.2f} {before[c]:>10} {after[c]:>9} "
              f"{after[c] / max(before[c], 1):>5.2f}x")
    ratio = lambda cnt: max(cnt.values()) / max(min(cnt.values()), 1)
    print(f"\nTỉ lệ lớp nhiều nhất / ít nhất (theo box): {ratio(before):.2f} -> {ratio(after):.2f}")
    print(f"Đã ghi {root / f'train_{args.name}.txt'} và {root / f'data_{args.name}.yaml'}")


if __name__ == "__main__":
    main()
