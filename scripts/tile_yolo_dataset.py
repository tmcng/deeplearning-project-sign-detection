"""Tạo tập huấn luyện YOLO đã cắt mảnh (Tiling) cho ảnh độ phân giải cao 1622x626 (Phần mở rộng của B – Tuần 4).

Ý tưởng:
    Ảnh gốc 1622x626 khi thu nhỏ về 640 làm 68.9% biển báo còn dưới 10 px.
    Khi cắt ảnh gốc thành các ô (tile) 640x640 chồng lấp 20%:
    - Một biển báo 16x16 px trên ảnh gốc vẫn giữ nguyên kích thước 16x16 px trong tile 640x640.
    - Chỉ giữ các box trong tile nếu diện tích phần nằm trong tile >= min_area_ratio (mặc định 0.6)
      để tránh học các mẩu biển bị cắt cụt ở mép.
    - Giữ tất cả các tile có chứa biển báo, và chỉ lấy ngẫu nhiên một tỉ lệ nhỏ (neg_ratio=0.15)
      các tile trống làm mẫu nền (background) để không làm phình kích thước tập train.
    - Tập val và test giữ nguyên ảnh gốc để đánh giá công bằng (hoặc kết hợp suy luận SAHI ở predict_yolo.py).

Cách chạy:
    python scripts/tile_yolo_dataset.py --splits data/splits \
        --images za_traffic_2020/traffic_train/images --out data/yolo_tiled
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image

from coco_to_yolo import link_or_copy


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
    return list(dict.fromkeys(windows))


def main():
    ap = argparse.ArgumentParser(description="Cắt ảnh độ phân giải cao thành các tile huấn luyện YOLOv8.")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--images", required=True, help="Thư mục chứa ảnh gốc")
    ap.add_argument("--out", default="data/yolo_tiled")
    ap.add_argument("--tile-size", type=int, default=640, help="Kích thước cạnh ô vuông cắt ra (mặc định 640)")
    ap.add_argument("--overlap", type=float, default=0.2, help="Tỉ lệ chồng lấp giữa các ô (mặc định 0.2)")
    ap.add_argument("--min-area-ratio", type=float, default=0.6,
                    help="Chỉ giữ box nếu phần diện tích nằm trong tile >= tỉ lệ này so với box gốc")
    ap.add_argument("--neg-ratio", type=float, default=0.15,
                    help="Tỉ lệ giữ lại các tile không có biển báo làm ảnh nền")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    splits, out, img_root = Path(args.splits), Path(args.out), Path(args.images).resolve()

    train_coco = json.loads((splits / "train.json").read_text(encoding="utf-8"))
    categories = sorted(train_coco["categories"], key=lambda c: c["id"])
    cat2idx = {c["id"]: i for i, c in enumerate(categories)}

    # 1. Cắt mảnh (Tiling) trên tập train
    train_img_dir = out / "images" / "train"
    train_lbl_dir = out / "labels" / "train"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)

    by_img = {}
    for a in train_coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    n_pos_tiles, n_neg_tiles, n_boxes = 0, 0, 0
    for im in train_coco["images"]:
        src = img_root / im["file_name"]
        if not src.exists():
            continue
        pil_img = Image.open(src).convert("RGB")
        W, H = pil_img.size
        stem = Path(im["file_name"]).stem
        anns = by_img.get(im["id"], [])

        windows = generate_slices(W, H, args.tile_size, args.tile_size, args.overlap)
        for ti, (tx0, ty0, tx1, ty1) in enumerate(windows):
            tw, th = tx1 - tx0, ty1 - ty0
            rows = []
            for a in anns:
                x, y, w, h = a["bbox"]
                orig_area = w * h
                if orig_area <= 0:
                    continue
                ix0, iy0 = max(tx0, x), max(ty0, y)
                ix1, iy1 = min(tx1, x + w), min(ty1, y + h)
                iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
                if (iw * ih) / orig_area < args.min_area_ratio:
                    continue
                cx = ((ix0 + ix1) / 2.0 - tx0) / tw
                cy = ((iy0 + iy1) / 2.0 - ty0) / th
                bw, bh = iw / tw, ih / th
                rows.append(f"{cat2idx[a['category_id']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not rows and rng.random() > args.neg_ratio:
                continue

            tile_name = f"{stem}_t{ti}_{tx0}_{ty0}"
            crop = pil_img.crop((tx0, ty0, tx1, ty1))
            crop.save(train_img_dir / f"{tile_name}.jpg", quality=92)
            (train_lbl_dir / f"{tile_name}.txt").write_text("\n".join(rows), encoding="utf-8")
            if rows:
                n_pos_tiles += 1
                n_boxes += len(rows)
            else:
                n_neg_tiles += 1

    print(f"train (tiled): {n_pos_tiles} tile có biển + {n_neg_tiles} tile nền, tổng {n_boxes} box")

    # 2. Giữ nguyên ảnh gốc cho tập val và test
    for name in ("val", "test"):
        coco = json.loads((splits / f"{name}.json").read_text(encoding="utf-8"))
        img_dir, lbl_dir = out / "images" / name, out / "labels" / name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        by_img_split = {}
        for a in coco["annotations"]:
            by_img_split.setdefault(a["image_id"], []).append(a)
        for im in coco["images"]:
            src = img_root / im["file_name"]
            if not src.exists():
                continue
            dst = img_dir / Path(im["file_name"]).name
            if not dst.exists():
                link_or_copy(src, dst)
            W, H = im["width"], im["height"]
            rows = []
            for a in by_img_split.get(im["id"], []):
                x, y, w, h = a["bbox"]
                x0, y0 = max(0.0, x), max(0.0, y)
                x1, y1 = min(W, x + w), min(H, y + h)
                if x1 <= x0 or y1 <= y0:
                    continue
                rows.append(
                    f"{cat2idx[a['category_id']]} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                    f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"
                )
            (lbl_dir / (dst.stem + ".txt")).write_text("\n".join(rows), encoding="utf-8")

    names = "\n".join(f"  {i}: {json.dumps(c['name'], ensure_ascii=False)}" for i, c in enumerate(categories))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n",
        encoding="utf-8",
    )
    (out / "class_map.json").write_text(
        json.dumps({i: c["id"] for i, c in enumerate(categories)}, indent=2), encoding="utf-8"
    )
    print(f"Đã ghi {out}/data.yaml và {out}/class_map.json")


if __name__ == "__main__":
    main()
