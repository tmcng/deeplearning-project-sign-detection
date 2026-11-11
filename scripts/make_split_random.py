"""Tạo split NGẪU NHIÊN theo từng ảnh cho thí nghiệm rò rỉ dữ liệu. KHÔNG dùng cho thí nghiệm chính.

Cách chạy:
    python scripts/make_split_random.py --splits data/splits --out data/splits_random

Thiết kế để so sánh công bằng với split theo street_id (data/splits):
    - Gộp train + val của data/splits lại, rồi chia ngẫu nhiên theo ẢNH (không theo đường) thành
      train/val với ĐÚNG số ảnh như data/splits. Stratify theo lớp hiếm nhất như make_split.py.
    - test.json chép y nguyên từ data/splits: chứa 5 con đường mà cả hai mô hình chưa từng thấy.
Như vậy hai mô hình có cùng số ảnh train, chỉ khác cách chia. Kết quả mong đợi:
    mAP trên val ngẫu nhiên cao hơn hẳn, nhưng trên test (đường mới) thì không cao hơn
    -> val ngẫu nhiên đánh giá quá cao khả năng tổng quát hóa.
Theo quy ước nhóm, test chỉ chấm một lần ở tuần 5.

Script cũng in mức độ rò rỉ: bao nhiêu ảnh val có khung hình liền kề (ảnh trước/sau trên
cùng đường, xếp theo id) nằm trong train.
"""
import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from make_split import subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits", help="thư mục split theo street_id")
    ap.add_argument("--out", default="data/splits_random")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.splits)
    load = lambda name: json.loads((src / f"{name}.json").read_text(encoding="utf-8"))
    train, val = load("train"), load("val")
    pool = dict(train, images=train["images"] + val["images"],
                annotations=train["annotations"] + val["annotations"])
    n_val = len(val["images"])

    cls_freq = Counter(a["category_id"] for a in pool["annotations"])
    cls_of_img = defaultdict(set)
    for a in pool["annotations"]:
        cls_of_img[a["image_id"]].add(a["category_id"])
    strata = defaultdict(list)
    for im in pool["images"]:
        classes = cls_of_img[im["id"]]
        strata[min(classes, key=lambda c: cls_freq[c]) if classes else "empty"].append(im["id"])

    # Chia từng tầng theo tỉ lệ val, rồi chỉnh vài ảnh cuối để val đúng bằng n_val
    rng = random.Random(args.seed)
    frac = n_val / len(pool["images"])
    val_ids, rest = [], []
    for s in sorted(strata, key=str):
        ids = sorted(strata[s])
        rng.shuffle(ids)
        k = round(len(ids) * frac)
        val_ids += ids[:k]
        rest += ids[k:]
    rng.shuffle(rest)
    if len(val_ids) < n_val:
        k = n_val - len(val_ids)
        val_ids += rest[:k]
        rest = rest[k:]
    elif len(val_ids) > n_val:
        rng.shuffle(val_ids)
        rest += val_ids[n_val:]
        val_ids = val_ids[:n_val]
    train_ids = rest
    assert not set(train_ids) & set(val_ids) and len(train_ids) + len(val_ids) == len(pool["images"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cats = {c["id"]: c["name"] for c in pool["categories"]}
    lines = [f"Split ngẫu nhiên theo ảnh (seed {args.seed}) từ train+val của {src.as_posix()}; test chép từ {src.as_posix()}."]
    for name, ids in (("train", train_ids), ("val", val_ids)):
        sub = subset(pool, ids)
        (out / f"{name}.json").write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
        cc = Counter(a["category_id"] for a in sub["annotations"])
        lines.append(f"{name:<5}: {len(sub['images'])} ảnh, {len(sub['annotations'])} box, "
                     f"{len({im['street_id'] for im in sub['images']})} đường")
        lines.append("       " + ", ".join(f"{cats[c]}={cc.get(c, 0)}" for c in sorted(cats)))
    shutil.copyfile(src / "test.json", out / "test.json")
    lines.append(f"test : chép từ {(src / 'test.json').as_posix()}")

    # Mức độ rò rỉ: khung hình liền kề trên cùng đường
    street = {im["id"]: im["street_id"] for im in pool["images"]}
    seq = defaultdict(list)
    for i in sorted(street):
        seq[street[i]].append(i)
    neighbors = defaultdict(set)
    for ids in seq.values():
        for a, b in zip(ids, ids[1:]):
            neighbors[a].add(b)
            neighbors[b].add(a)
    tr = set(train_ids)
    leaked = sum(bool(neighbors[i] & tr) for i in val_ids)
    shared_streets = {street[i] for i in val_ids} & {street[i] for i in train_ids}
    lines.append(f"\nRò rỉ: {leaked}/{len(val_ids)} ảnh val ({leaked / len(val_ids):.1%}) có khung hình "
                 f"liền kề trong train; {len(shared_streets)} đường có mặt ở cả train và val.")

    text = "\n".join(lines)
    (out / "split_stats.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
