"""Chuyển các file split COCO JSON sang định dạng YOLO cho Ultralytics (bàn giao #3 cho B).

Cách chạy:
    python scripts/coco_to_yolo.py --splits data/splits --images data/raw/images --out data/yolo

Kết quả:
    data/yolo/images/{train,val,test}/   : liên kết (symlink) tới ảnh gốc, không tốn dung lượng
    data/yolo/labels/{train,val,test}/   : mỗi ảnh một file .txt, mỗi dòng "cls cx cy w h" (chuẩn hóa 0-1)
    data/yolo/data.yaml                  : B dùng trực tiếp: yolo train data=data/yolo/data.yaml
    data/yolo/class_map.json             : chỉ số YOLO (0..K-1) -> category_id COCO

QUAN TRỌNG: YOLO đánh số lớp từ 0, còn COCO dùng category_id gốc (có thể từ 1).
Khi B xuất dự đoán để chấm, phải đổi ngược bằng class_map.json, nếu không mAP sẽ gần 0.
Dùng --copy nếu môi trường (vd. Windows, Kaggle dataset chỉ đọc) không tạo được symlink.
"""
import argparse
import json
import os
import shutil
from pathlib import Path


_warned = set()


def link_or_copy(src, dst):
    """Thử symlink; Windows không cho thì dùng hard link (không tốn dung lượng,
    không cần quyền admin, nhưng ảnh và thư mục đích phải cùng ổ đĩa); cuối cùng mới sao chép."""
    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass
    try:
        os.link(src, dst)
        if "hard" not in _warned:
            print("  (không tạo được symlink, dùng hard link thay thế: không tốn thêm dung lượng)")
            _warned.add("hard")
        return
    except OSError:
        pass
    if "copy" not in _warned:
        print("  (không tạo được link, chuyển sang sao chép ảnh: tốn thêm dung lượng)")
        _warned.add("copy")
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--images", required=True, help="thư mục chứa ảnh gốc")
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--copy", action="store_true", help="sao chép ảnh thay vì symlink")
    args = ap.parse_args()

    splits, out, img_root = Path(args.splits), Path(args.out), Path(args.images).resolve()
    categories = None

    for name in ("train", "val", "test"):
        coco = json.loads((splits / f"{name}.json").read_text(encoding="utf-8"))
        if categories is None:
            categories = sorted(coco["categories"], key=lambda c: c["id"])
            cat2idx = {c["id"]: i for i, c in enumerate(categories)}

        img_dir, lbl_dir = out / "images" / name, out / "labels" / name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        by_img = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)

        n_box, n_missing = 0, 0
        for im in coco["images"]:
            src = img_root / im["file_name"]
            if not src.exists():
                n_missing += 1
                continue
            dst = img_dir / Path(im["file_name"]).name
            if not dst.exists():
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    link_or_copy(src, dst)

            W, H = im["width"], im["height"]
            rows = []
            for a in by_img.get(im["id"], []):
                x, y, w, h = a["bbox"]
                # cắt box về trong ảnh để tránh tọa độ âm hoặc vượt 1
                x0, y0 = max(0.0, x), max(0.0, y)
                x1, y1 = min(W, x + w), min(H, y + h)
                if x1 <= x0 or y1 <= y0:
                    continue
                cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
                rows.append(f"{cat2idx[a['category_id']]} {cx:.6f} {cy:.6f} "
                            f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
            # ảnh không có box vẫn cần file rỗng để YOLO dùng làm ảnh nền
            (lbl_dir / (dst.stem + ".txt")).write_text("\n".join(rows), encoding="utf-8")
            n_box += len(rows)

        msg = f"{name}: {len(coco['images']) - n_missing} ảnh, {n_box} box"
        if n_missing:
            msg += f"  (CẢNH BÁO: thiếu {n_missing} file ảnh, kiểm tra lại --images)"
        print(msg)

    names = "\n".join(f"  {i}: {json.dumps(c['name'], ensure_ascii=False)}"
                      for i, c in enumerate(categories))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names}\n", encoding="utf-8")
    (out / "class_map.json").write_text(
        json.dumps({i: c["id"] for i, c in enumerate(categories)}, indent=2), encoding="utf-8")
    print(f"Đã ghi {out}/data.yaml và {out}/class_map.json")


if __name__ == "__main__":
    main()
