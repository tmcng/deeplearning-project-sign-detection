"""Chia train/val/test cố định, có stratify theo lớp. Chạy MỘT lần rồi commit kết quả.

Cách chạy:
    python scripts/make_split.py --ann data/raw/train_traffic_sign_dataset.json --out data/splits

Kết quả:
    data/splits/train.json, val.json, test.json   : COCO JSON của từng tập (bàn giao #3)
    data/splits/split_stats.txt                   : số ảnh, số box mỗi lớp trong từng tập

Cách stratify: mỗi ảnh được gán "khóa" = lớp HIẾM NHẤT có trong ảnh (ảnh không có box
có khóa "empty"). Chia theo từng khóa với cùng tỉ lệ, nên lớp hiếm có mặt đủ ở cả 3 tập.

Rò rỉ dữ liệu: nếu nhiều ảnh là các khung hình liên tiếp trên cùng một con đường, ảnh
gần giống nhau có thể rơi vào cả train và test, làm mAP cao giả. Nếu JSON có trường
nhóm (vd. street_id), truyền --group-key street_id để cả nhóm luôn nằm chung một tập.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def subset(coco, img_ids):
    img_ids = set(img_ids)
    return {
        "info": coco.get("info", {"description": "traffic sign split"}),
        "licenses": coco.get("licenses", []),
        "images": [im for im in coco["images"] if im["id"] in img_ids],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in img_ids],
        "categories": coco["categories"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-key", default=None,
                    help="tên trường trong images[] dùng để nhóm ảnh (tránh rò rỉ)")
    args = ap.parse_args()

    coco = json.loads(Path(args.ann).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Đảm bảo mọi annotation có trường mà pycocotools cần
    for i, a in enumerate(coco["annotations"]):
        a.setdefault("id", i + 1)
        a.setdefault("iscrowd", 0)
        a.setdefault("area", a["bbox"][2] * a["bbox"][3])

    cls_freq = Counter(a["category_id"] for a in coco["annotations"])
    cls_of_img = defaultdict(set)
    for a in coco["annotations"]:
        cls_of_img[a["image_id"]].add(a["category_id"])

    # Gom ảnh thành "đơn vị chia": mỗi ảnh một đơn vị, hoặc theo nhóm nếu có --group-key
    units = defaultdict(list)
    for im in coco["images"]:
        key = im.get(args.group_key, im["id"]) if args.group_key else im["id"]
        units[key].append(im["id"])

    # Khóa stratify của mỗi đơn vị = lớp hiếm nhất xuất hiện trong đơn vị đó
    strata = defaultdict(list)
    for key, ids in units.items():
        classes = set().union(*(cls_of_img[i] for i in ids))
        s = min(classes, key=lambda c: cls_freq[c]) if classes else "empty"
        strata[s].append(key)

    rng = random.Random(args.seed)
    split_ids = {"train": [], "val": [], "test": []}
    for s in sorted(strata, key=str):
        keys = sorted(strata[s], key=str)
        rng.shuffle(keys)
        n = len(keys)
        n_test = round(n * args.test)
        n_val = round(n * args.val)
        parts = {"test": keys[:n_test], "val": keys[n_test:n_test + n_val],
                 "train": keys[n_test + n_val:]}
        for name, ks in parts.items():
            for k in ks:
                split_ids[name].extend(units[k])

    cats = {c["id"]: c["name"] for c in coco["categories"]}
    lines = []
    for name, ids in split_ids.items():
        sub = subset(coco, ids)
        (out / f"{name}.json").write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
        cc = Counter(a["category_id"] for a in sub["annotations"])
        lines.append(f"{name:<5}: {len(sub['images'])} ảnh, {len(sub['annotations'])} box")
        lines.append("       " + ", ".join(f"{cats[c]}={cc.get(c, 0)}" for c in sorted(cats)))

    # Kiểm tra không trùng ảnh giữa các tập
    a, b, c = (set(split_ids[k]) for k in ("train", "val", "test"))
    assert not (a & b or a & c or b & c), "Có ảnh bị trùng giữa các tập!"

    text = "\n".join(lines)
    (out / "split_stats.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nĐã ghi {out}/train.json, val.json, test.json. Nhớ commit thư mục này.")


if __name__ == "__main__":
    main()
