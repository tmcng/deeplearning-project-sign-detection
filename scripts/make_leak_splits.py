"""Tạo dữ liệu cho thí nghiệm rò rỉ dữ liệu: chia NGẪU NHIÊN theo ảnh và chia THEO TUYẾN ĐƯỜNG.

Ý tưởng thí nghiệm:
    - Lấy data/splits/train.json làm "kho" (pool), chia lại thành train'/val' theo hai cách:
        random : chia ngẫu nhiên từng ảnh (ảnh cùng tuyến đường có thể nằm cả ở train' và val')
        street : chia theo street_id (cả tuyến đường nằm trọn trong train' hoặc val')
    - data/splits/val.json (các tuyến đường mô hình CHƯA từng thấy) là tập kiểm tra chung,
      ghi ra dưới tên test.json trong mỗi thư mục. KHÔNG dùng data/splits/test.json,
      để tập test thật vẫn chỉ được chấm một lần ở tuần 5.
    - Train cùng một cấu hình trên hai train', rồi chấm mỗi mô hình trên val' của nó và trên
      tập kiểm tra chung. Nếu chia ngẫu nhiên bị rò rỉ, mAP trên val' của nó sẽ cao hơn hẳn
      mAP trên tuyến đường mới; còn chia theo tuyến đường thì hai con số gần nhau.

Cách chạy (trên máy, không cần GPU; chạy một lần rồi commit data/leak/):
    python scripts/make_leak_splits.py --pool data/splits/train.json --heldout data/splits/val.json --out data/leak

Kết quả:
    data/leak/random/{train,val,test}.json
    data/leak/street/{train,val,test}.json      (test.json = tuyến đường chưa thấy)
    data/leak/leak_stats.txt
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def subset(coco, img_ids):
    img_ids = set(img_ids)
    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": [im for im in coco["images"] if im["id"] in img_ids],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in img_ids],
        "categories": coco["categories"],
    }


def rarest_class_key(classes, freq):
    return min(classes, key=lambda c: freq[c]) if classes else "empty"


def street_split(coco, frac, rng, freq, cls_of_img):
    """Chia theo street_id, phân tầng theo lớp hiếm nhất của mỗi tuyến đường."""
    streets = defaultdict(list)
    for im in coco["images"]:
        streets[im["street_id"]].append(im["id"])
    strata = defaultdict(list)
    for sid, ids in streets.items():
        classes = set().union(*(cls_of_img[i] for i in ids))
        strata[rarest_class_key(classes, freq)].append(sid)

    val_ids = []
    for key in sorted(strata, key=str):
        sids = sorted(strata[key])
        rng.shuffle(sids)
        target = frac * sum(len(streets[s]) for s in sids)
        taken = 0
        for s in sids:  # thêm từng tuyến đường đến khi đủ tỉ lệ ảnh trong tầng này
            if taken >= target:
                break
            val_ids.extend(streets[s])
            taken += len(streets[s])
    return val_ids


def random_split(coco, n_val, rng, freq, cls_of_img):
    """Chia ngẫu nhiên từng ảnh, phân tầng theo lớp hiếm nhất, lấy đúng n_val ảnh."""
    strata = defaultdict(list)
    for im in coco["images"]:
        strata[rarest_class_key(cls_of_img[im["id"]], freq)].append(im["id"])
    total = len(coco["images"])
    val_ids, leftovers = [], []
    for key in sorted(strata, key=str):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        k = int(len(ids) * n_val / total)
        val_ids.extend(ids[:k])
        leftovers.extend(ids[k:])
    rng.shuffle(leftovers)
    val_ids.extend(leftovers[:n_val - len(val_ids)])  # bù phần lẻ để đúng n_val
    return val_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/splits/train.json")
    ap.add_argument("--heldout", default="data/splits/val.json")
    ap.add_argument("--out", default="data/leak")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    heldout = json.loads(Path(args.heldout).read_text(encoding="utf-8"))
    if "street_id" not in pool["images"][0]:
        raise SystemExit("Ảnh không có trường street_id.")

    freq = Counter(a["category_id"] for a in pool["annotations"])
    cls_of_img = defaultdict(set)
    for a in pool["annotations"]:
        cls_of_img[a["image_id"]].add(a["category_id"])
    street_of = {im["id"]: im["street_id"] for im in pool["images"]}
    all_ids = [im["id"] for im in pool["images"]]

    # Chia theo tuyến đường trước, rồi chia ngẫu nhiên với ĐÚNG số ảnh val đó để hai cách cân bằng
    val_street = street_split(pool, args.val_frac, random.Random(args.seed), freq, cls_of_img)
    val_random = random_split(pool, len(val_street), random.Random(args.seed), freq, cls_of_img)

    # Không được có tuyến đường nào của tập kiểm tra chung nằm trong pool
    pool_streets = set(street_of.values())
    assert not pool_streets & {im["street_id"] for im in heldout["images"]}, \
        "Tập kiểm tra chung trùng tuyến đường với pool!"

    out = Path(args.out)
    cats = {c["id"]: c["name"] for c in pool["categories"]}
    lines = [f"Pool: {len(all_ids)} ảnh, {len(pool_streets)} tuyến đường. "
             f"Tập kiểm tra chung (tuyến đường chưa thấy): {len(heldout['images'])} ảnh", ""]
    for name, val_ids in (("street", val_street), ("random", val_random)):
        vset = set(val_ids)
        train_ids = [i for i in all_ids if i not in vset]
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        tr, va = subset(pool, train_ids), subset(pool, val_ids)
        (d / "train.json").write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
        (d / "val.json").write_text(json.dumps(va, ensure_ascii=False), encoding="utf-8")
        (d / "test.json").write_text(json.dumps(heldout, ensure_ascii=False), encoding="utf-8")

        train_streets = {street_of[i] for i in train_ids}
        leaked = sum(street_of[i] in train_streets for i in val_ids)
        cc = Counter(a["category_id"] for a in va["annotations"])
        lines += [
            f"[{name}] train {len(train_ids)} ảnh / {len(tr['annotations'])} box, "
            f"val {len(val_ids)} ảnh / {len(va['annotations'])} box",
            f"   ảnh val có tuyến đường xuất hiện trong train: {leaked}/{len(val_ids)} "
            f"({leaked / len(val_ids):.1%})",
            "   box val theo lớp: " + ", ".join(f"{cats[c]}={cc.get(c, 0)}" for c in sorted(cats)),
        ]
        if name == "street":
            assert leaked == 0, "Chia theo tuyến đường mà vẫn trùng tuyến!"

    text = "\n".join(lines)
    (out / "leak_stats.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nĐã ghi {out}/random và {out}/street. Commit thư mục {out}/.")


if __name__ == "__main__":
    main()
