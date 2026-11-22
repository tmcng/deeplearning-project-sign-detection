"""Kiểm tra chênh lệch AP50 giữa các mô hình có ý nghĩa thống kê không (paired bootstrap theo ảnh).

Cách chạy:
    python scripts/bootstrap_ap.py --gt data/splits/val.json --base yolov8s_1280 --runs imb_rfs imb_clspw \
        --out reports/imbalance/bootstrap.json

Đọc preds/<tên>_val.json của từng mô hình (đổi bằng --pred-pattern nếu tên file khác).

Vì sao cần: val chỉ có khoảng 100–400 biển mỗi lớp (Cấm rẽ chỉ 98), nên AP của một lớp có thể lệch
vài điểm chỉ vì tập val tình cờ có những ảnh nào.

Cách làm:
    1. Ghép dự đoán với nhãn MỘT lần cho mỗi (ảnh, lớp), giống COCO (IoU >= 0.5, tối đa 100 dự đoán/ảnh).
    2. Lặp B lần: lấy ngẫu nhiên có hoàn lại 696 ảnh val, tính lại AP50 từng lớp cho mọi mô hình
       trên CÙNG bộ ảnh đó ("paired"), ghi lại chênh lệch so với --base.
    3. Khoảng 2.5%–97.5% của các chênh lệch là khoảng tin cậy 95%. Khoảng không chứa 0 = có ý nghĩa.
AP50 tự tính ở đây khớp với pycocotools (script in ra để kiểm tra).

Giới hạn: chỉ đo độ dao động do chọn ảnh val, KHÔNG đo độ dao động giữa các lần train (seed).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_errors import iou


def match(preds, gts, img_ids, cat_ids):
    """(ảnh, lớp) -> (scores, cờ TP, số box thật), ghép giống COCO ở IoU 0.5."""
    P = defaultdict(list)
    for p in preds:
        P[p["image_id"]].append(p)
    out = {}
    for i in img_ids:
        ps = sorted(P.get(i, []), key=lambda p: -p["score"])[:100]
        for c in cat_ids:
            g = gts.get((i, c), [])
            used, sc, tp = set(), [], []
            for p in (p for p in ps if p["category_id"] == c):
                best, bj = 0.5, None
                for j, b in enumerate(g):
                    if j not in used and iou(p["bbox"], b) >= best:
                        best, bj = iou(p["bbox"], b), j
                if bj is not None:
                    used.add(bj)
                sc.append(p["score"])
                tp.append(bj is not None)
            out[(i, c)] = (np.array(sc), np.array(tp, dtype=bool), len(g))
    return out


def ap101(entries):
    """AP nội suy 101 điểm recall như COCO."""
    n = sum(e[2] for e in entries)
    if n == 0:
        return np.nan
    sc = np.concatenate([e[0] for e in entries])
    tp = np.concatenate([e[1] for e in entries])[np.argsort(-sc, kind="mergesort")]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    rec = ctp / n
    prec = np.maximum.accumulate((ctp / np.maximum(ctp + cfp, 1e-9))[::-1])[::-1]
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    q = np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0)
    return float(q.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/splits/val.json")
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--pred-pattern", default="preds/{}_val.json")
    ap.add_argument("--n", type=int, default=1000, help="số lần lấy mẫu lại")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="ghi kết quả ra file JSON")
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in gt["categories"]}
    img_ids = sorted(im["id"] for im in gt["images"])
    gts = defaultdict(list)
    for a in gt["annotations"]:
        gts[(a["image_id"], a["category_id"])].append(a["bbox"])

    models = [args.base] + args.runs
    M = {m: match(json.loads(Path(args.pred_pattern.format(m)).read_text(encoding="utf-8")), gts, img_ids, cats)
         for m in models}
    class_ap = lambda m, sample: {c: ap101([M[m][(i, c)] for i in sample]) for c in cats}

    full = {m: class_ap(m, img_ids) for m in models}
    print("AP50 trên toàn bộ val (khớp pycocotools):")
    for m in models:
        print(f"  {m:<20} mAP50 {100 * np.nanmean(list(full[m].values())):.1f}  "
              + " ".join(f"{cats[c]}={100 * full[m][c]:.1f}" for c in cats))

    rng = np.random.default_rng(args.seed)
    diffs = {m: defaultdict(list) for m in args.runs}
    for _ in range(args.n):
        s = rng.choice(img_ids, size=len(img_ids), replace=True)
        base = class_ap(args.base, s)
        for m in args.runs:
            a = class_ap(m, s)
            for c in cats:
                diffs[m][c].append(a[c] - base[c])
            diffs[m]["mAP50"].append(np.nanmean(list(a.values())) - np.nanmean(list(base.values())))

    print(f"\nChênh lệch AP50 so với {args.base}: điểm [khoảng tin cậy 95%], bootstrap {args.n} lần theo ảnh")
    result = {"base": args.base, "n": args.n, "runs": {}}
    for m in args.runs:
        print(f"  {m}")
        result["runs"][m] = {}
        for c in list(cats) + ["mAP50"]:
            if c == "mAP50":
                point = np.nanmean(list(full[m].values())) - np.nanmean(list(full[args.base].values()))
                name = "mAP50"
            else:
                point, name = full[m][c] - full[args.base][c], cats[c]
            lo, hi = np.nanpercentile(np.array(diffs[m][c]), [2.5, 97.5])
            sig = lo > 0 or hi < 0
            result["runs"][m][name] = {"diff": round(100 * point, 1), "ci95": [round(100 * lo, 1), round(100 * hi, 1)],
                                       "co_y_nghia": bool(sig)}
            print(f"    {name:<17} {100 * point:+5.1f}  [{100 * lo:+5.1f}, {100 * hi:+5.1f}]"
                  + ("  CÓ Ý NGHĨA" if sig else ""))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
