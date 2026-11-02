"""EDA cho bộ dữ liệu biển báo (định dạng COCO JSON).

Cách chạy:
    python scripts/eda.py --ann data/raw/train_traffic_sign_dataset.json --out reports/eda

Kết quả (trong thư mục --out):
    eda_summary.txt      : tóm tắt số liệu, dán thẳng vào proposal
    class_counts.png     : số box theo từng lớp (thấy mất cân bằng)
    box_size_hist.png    : phân phối cạnh box (px) ở ảnh gốc và sau resize
    boxes_per_image.png  : số box trên mỗi ảnh
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ngưỡng diện tích theo chuẩn COCO (tính bằng pixel^2 trên ảnh GỐC)
SMALL, MEDIUM = 32 ** 2, 96 ** 2


def size_group(area):
    if area < SMALL:
        return "small"
    if area < MEDIUM:
        return "medium"
    return "large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="file COCO JSON của tập train")
    ap.add_argument("--out", default="reports/eda")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="kích thước resize khi huấn luyện, để ước lượng box co lại bao nhiêu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    coco = json.loads(Path(args.ann).read_text(encoding="utf-8"))

    images = {im["id"]: im for im in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns = coco["annotations"]

    # ---- 1. Số box theo lớp ----
    cls_count = Counter(a["category_id"] for a in anns)

    # ---- 2. Kích thước box ----
    sides, sides_resized, groups = [], [], Counter()
    groups_by_cls = {cid: Counter() for cid in cats}
    for a in anns:
        x, y, w, h = a["bbox"]
        im = images[a["image_id"]]
        scale = args.imgsz / max(im["width"], im["height"])
        side = math.sqrt(w * h)
        sides.append(side)
        sides_resized.append(side * scale)
        g = size_group(w * h)
        groups[g] += 1
        groups_by_cls[a["category_id"]][g] += 1

    # ---- 3. Số box mỗi ảnh ----
    per_img = Counter(a["image_id"] for a in anns)
    n_per_img = [per_img.get(i, 0) for i in images]
    n_empty = sum(1 for n in n_per_img if n == 0)

    resolutions = Counter((im["width"], im["height"]) for im in images.values())
    sides, sides_resized = np.array(sides), np.array(sides_resized)

    # ---- Ghi tóm tắt ----
    lines = [
        f"Số ảnh: {len(images)}  (không có box: {n_empty})",
        f"Số box: {len(anns)}  (trung bình {len(anns) / len(images):.2f} box/ảnh, nhiều nhất {max(n_per_img)})",
        f"Độ phân giải phổ biến: {resolutions.most_common(3)}",
        "",
        "Số box theo lớp (nhiều -> ít):",
    ]
    for cid, n in cls_count.most_common():
        g = groups_by_cls[cid]
        lines.append(f"  [{cid}] {cats[cid]:<35} {n:>6}  "
                     f"(nhỏ {g['small']}, vừa {g['medium']}, lớn {g['large']})")
    most, least = cls_count.most_common()[0][1], cls_count.most_common()[-1][1]
    lines += [
        f"Tỉ lệ lớp nhiều nhất / ít nhất: {most / least:.1f} lần",
        "",
        "Nhóm kích thước (chuẩn COCO, ảnh gốc): "
        + ", ".join(f"{k} {groups[k]} ({groups[k] / len(anns):.1%})" for k in ("small", "medium", "large")),
        f"Cạnh box (sqrt(w*h)) ảnh gốc: trung vị {np.median(sides):.1f}px, "
        f"p10 {np.percentile(sides, 10):.1f}px, p90 {np.percentile(sides, 90):.1f}px",
        f"Sau resize về {args.imgsz}: trung vị {np.median(sides_resized):.1f}px, "
        f"{(sides_resized < 10).mean():.1%} box có cạnh < 10px",
    ]
    text = "\n".join(lines)
    (out / "eda_summary.txt").write_text(text, encoding="utf-8")
    print(text)

    # ---- Vẽ hình ----
    order = [cid for cid, _ in cls_count.most_common()]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([cats[c] for c in order][::-1], [cls_count[c] for c in order][::-1], color="#4c72b0")
    ax.set_xlabel("Số box")
    ax.set_title("Số box theo lớp")
    fig.tight_layout()
    fig.savefig(out / "class_counts.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, max(sides.max(), 1), 50)
    ax.hist(sides, bins=bins, alpha=0.6, label="Ảnh gốc")
    ax.hist(sides_resized, bins=bins, alpha=0.6, label=f"Sau resize {args.imgsz}")
    ax.axvline(32, color="k", ls="--", lw=1)
    ax.text(33, ax.get_ylim()[1] * 0.9, "32px (ngưỡng nhỏ COCO)", fontsize=8)
    ax.set_xlabel("Cạnh box sqrt(w*h) (px)")
    ax.set_ylabel("Số box")
    ax.set_title("Phân phối kích thước biển báo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "box_size_hist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    c = Counter(n_per_img)
    ks = sorted(c)
    ax.bar(ks, [c[k] for k in ks], color="#55a868")
    ax.set_xlabel("Số box trên ảnh")
    ax.set_ylabel("Số ảnh")
    ax.set_title("Số box mỗi ảnh")
    fig.tight_layout()
    fig.savefig(out / "boxes_per_image.png", dpi=150)
    plt.close(fig)

    print(f"\nĐã lưu tóm tắt và 3 hình vào {out}/")


if __name__ == "__main__":
    main()
