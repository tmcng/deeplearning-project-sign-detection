"""Gộp các box trùng nhau (cùng một biển bị gán nhiều box) trong file nhãn COCO gốc.

Vì sao cần: kiểm tra dữ liệu cho thấy khoảng 15% box train và 23% box test là box lặp
(cùng lớp, chồng lên nhau). Khi đánh giá, mỗi biển chỉ được một dự đoán khớp, nên mọi
box lặp bị tính là "bỏ sót" và recall tối đa trên test chỉ còn khoảng 77% dù mô hình
hoàn hảo. Hai biển thật đứng cạnh nhau gần như không chồng nhau (IoU ~ 0), nên gộp
theo ngưỡng IoU không làm mất biển thật.

Cách chạy (chạy trên file GỐC, trước make_split):
    python scripts/dedup_boxes.py --ann za_traffic_2020/traffic_train/train_traffic_sign_dataset.json \
        --out data/clean/train_dedup.json

Cách gộp: trong mỗi ảnh, các box CÙNG LỚP có IoU >= ngưỡng được nối thành một cụm
(nối bắc cầu: A trùng B, B trùng C thì A, B, C chung cụm). Mỗi cụm thay bằng một box
có tọa độ trung bình. Box khác lớp không bao giờ bị gộp.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def clusters(boxes, thr):
    """Union-find: trả về danh sách cụm, mỗi cụm là list chỉ số box."""
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if iou(boxes[i], boxes[j]) >= thr:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(boxes)):
        groups[find(i)].append(i)
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="file COCO JSON gốc")
    ap.add_argument("--out", default="data/clean/train_dedup.json")
    ap.add_argument("--iou", type=float, default=0.45, help="ngưỡng IoU để coi là box trùng")
    args = ap.parse_args()

    coco = json.loads(Path(args.ann).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}

    by_key = defaultdict(list)  # (image_id, category_id) -> annotations
    for a in coco["annotations"]:
        by_key[(a["image_id"], a["category_id"])].append(a)

    new_anns, removed, imgs_touched = [], Counter(), set()
    size_hist = Counter()
    for (img_id, cid), anns in by_key.items():
        for group in clusters([a["bbox"] for a in anns], args.iou):
            members = [anns[i] for i in group]
            keep = dict(members[0])
            if len(members) > 1:
                n = len(members)
                keep["bbox"] = [sum(m["bbox"][k] for m in members) / n for k in range(4)]
                removed[cid] += n - 1
                imgs_touched.add(img_id)
                size_hist[n] += 1
            keep["area"] = keep["bbox"][2] * keep["bbox"][3]
            keep.setdefault("iscrowd", 0)
            new_anns.append(keep)

    new_anns.sort(key=lambda a: a["id"])
    out = dict(coco, annotations=new_anns)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    total_removed = sum(removed.values())
    before, after = len(coco["annotations"]), len(new_anns)
    print(f"Box: {before} -> {after} (bỏ {total_removed} box trùng, {total_removed / before:.1%})")
    print(f"Ảnh có box trùng: {len(imgs_touched)} / {len(coco['images'])}")
    print("Kích thước cụm (số box cho một biển): "
          + ", ".join(f"{k} box: {v} biển" for k, v in sorted(size_hist.items())))
    print("Số box bỏ theo lớp:")
    for cid in sorted(cats):
        print(f"  {cats[cid]:<25} {removed.get(cid, 0)}")
    print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
