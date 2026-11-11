"""Phân tích lỗi cho CẢ HAI mô hình (YOLO và Faster R-CNN), dùng chung định dạng dự đoán của evaluate.py.

Cách chạy:
    python scripts/analyze_errors.py --gt data/splits/val.json --pred preds/yolov8s_640_val.json \
        --name yolov8s_640 --out reports/errors

Kết quả (trong reports/errors/<name>/):
    confusion.png   : ma trận nhầm lẫn 7 lớp + cột "bỏ sót" + hàng "báo nhầm nền"
    summary.md      : bảng recall theo kích thước, theo cụm/đứng riêng, theo lớp
    summary.json    : cùng số liệu, để vẽ lại hoặc so sánh nhiều mô hình

Cách ghép dự đoán với nhãn thật (IoU >= --iou, chỉ xét dự đoán có score >= ngưỡng):
    1. Ghép đúng lớp trước: trong mỗi ảnh, mỗi lớp, dự đoán xếp theo score giảm dần, ghép với
       box thật CÙNG LỚP chưa được ghép có IoU cao nhất (giống cách COCO tính TP).
    2. Dự đoán còn thừa ghép tiếp với box thật còn lại BẤT KỂ LỚP -> "sai lớp".
    3. Box thật vẫn chưa ghép -> "bỏ sót". Dự đoán vẫn thừa:
       - chồng lên một box thật đã được ghép -> "dự đoán lặp" (NMS chưa lọc hết), đếm riêng
       - còn lại -> "báo nhầm nền"
    Ghép đúng lớp trước nên đường chéo của ma trận đúng bằng số TP dùng để tính recall.

Ngưỡng score: mặc định "f1" = chọn ngưỡng cho F1 cao nhất trên tập đang chấm, để so hai mô hình
công bằng (score của YOLO và Faster R-CNN không cùng thang). Có thể đặt số cố định, vd. 0.25.

Kích thước biển = sqrt(w*h) trên ảnh gốc (giống eda.py). "Cụm" = box có ít nhất một box thật
khác trong ảnh nằm sát nó (khe hở cả ngang lẫn dọc < nửa cạnh nhỏ nhất của hai box), giống
định nghĩa dùng trong docs/notes_du_lieu.md.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIZE_BINS = [(0, 8, "<8"), (8, 16, "8-16"), (16, 32, "16-32"), (32, 1e9, ">=32")]


def iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def near(a, b):
    s = 0.5 * min(a[2], a[3], b[2], b[3])
    gx = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
    gy = max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3])
    return gx < s and gy < s


def size_bin(box):
    s = (box[2] * box[3]) ** 0.5
    return next(name for lo, hi, name in SIZE_BINS if lo <= s < hi)


def greedy(preds, gts, iou_thr, same_class):
    """Ghép tham lam theo score giảm dần. preds/gts là list chỉ số vào p_all/g_all của ảnh.
    Trả về dict pred_idx -> gt_idx. Hàm gọi phải truyền preds đã xếp theo score."""
    pairs, used = {}, set()
    for pi, pb, pc in preds:
        best, best_iou = None, iou_thr
        for gi, gb, gc in gts:
            if gi in used or (same_class and gc != pc):
                continue
            v = iou(pb, gb)
            if v >= best_iou:
                best, best_iou = gi, v
        if best is not None:
            pairs[pi] = best
            used.add(best)
    return pairs


def class_aware_tp(gts_by_img, preds_by_img, iou_thr):
    """Cờ TP (ghép đúng lớp) cho từng dự đoán, dùng để chọn ngưỡng F1.
    Ghép tham lam theo score nên kết quả với ngưỡng t trùng với chạy lại chỉ trên dự đoán >= t."""
    scores, tps = [], []
    for img_id, preds in preds_by_img.items():
        gts = gts_by_img.get(img_id, [])
        items = sorted(((i, p["bbox"], p["category_id"]) for i, p in enumerate(preds)),
                       key=lambda t: -preds[t[0]]["score"])
        pairs = greedy(items, [(j, g["bbox"], g["category_id"]) for j, g in enumerate(gts)],
                       iou_thr, same_class=True)
        for i, p in enumerate(preds):
            scores.append(p["score"])
            tps.append(i in pairs)
    return np.array(scores), np.array(tps, dtype=bool)


def pick_f1_threshold(scores, tps, n_gt):
    order = np.argsort(-scores)
    s, tp = scores[order], np.cumsum(tps[order])
    k = np.arange(1, len(s) + 1)
    f1 = 2 * tp / (k + n_gt)
    # Chỉ xét vị trí cuối của mỗi mức score (các dự đoán bằng score phải cùng vào hoặc cùng ra)
    last = np.r_[s[1:] != s[:-1], True]
    i = int(np.argmax(np.where(last, f1, -1)))
    return float(s[i]), float(f1[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="COCO JSON của tập val hoặc test")
    ap.add_argument("--pred", required=True, help="file JSON dự đoán (định dạng như evaluate.py)")
    ap.add_argument("--name", required=True, help="tên thí nghiệm, vd. yolov8s_640")
    ap.add_argument("--out", default="reports/errors")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--score", default="f1", help='"f1" (tự chọn) hoặc một số, vd. 0.25')
    args = ap.parse_args()

    coco = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    preds = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    cat_ids = sorted(c["id"] for c in coco["categories"])
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    img_ids = {im["id"] for im in coco["images"]}
    bad = sum(p["image_id"] not in img_ids for p in preds) + sum(p["category_id"] not in cats for p in preds)
    if bad:
        raise SystemExit(f"Lỗi: {bad} dự đoán có image_id/category_id lạ. Chạy evaluate.py để xem chi tiết.")

    gts_by_img, preds_by_img = defaultdict(list), defaultdict(list)
    for a in coco["annotations"]:
        gts_by_img[a["image_id"]].append(a)
    for p in preds:
        preds_by_img[p["image_id"]].append(p)
    n_gt = len(coco["annotations"])

    if args.score == "f1":
        thr, best_f1 = pick_f1_threshold(*class_aware_tp(gts_by_img, preds_by_img, args.iou), n_gt)
        thr_note = f"{thr:.3f} (tự chọn, F1 = {best_f1:.3f})"
    else:
        thr = float(args.score)
        thr_note = f"{thr:.3f} (cố định)"

    K = len(cat_ids)
    col = {cid: k for k, cid in enumerate(cat_ids)}
    conf = np.zeros((K + 1, K + 1), dtype=int)  # hàng: lớp thật + "nền"; cột: lớp dự đoán + "bỏ sót"
    dup_preds = 0
    records = []  # mỗi box thật: lớp, nhóm kích thước, có trong cụm không, kết quả

    for img_id in sorted(img_ids):
        gts = gts_by_img.get(img_id, [])
        ps = sorted((p for p in preds_by_img.get(img_id, []) if p["score"] >= thr), key=lambda p: -p["score"])
        g_items = [(j, g["bbox"], g["category_id"]) for j, g in enumerate(gts)]
        p_items = [(i, p["bbox"], p["category_id"]) for i, p in enumerate(ps)]

        right = greedy(p_items, g_items, args.iou, same_class=True)
        taken = set(right.values())
        wrong = greedy([t for t in p_items if t[0] not in right],
                       [t for t in g_items if t[0] not in taken], args.iou, same_class=False)

        status = {}
        for pi, gj in right.items():
            status[gj] = ("dung", ps[pi]["category_id"])
        for pi, gj in wrong.items():
            status[gj] = ("sai_lop", ps[pi]["category_id"])
        for i, p in enumerate(ps):
            if i in right or i in wrong:
                continue
            if any(iou(p["bbox"], gts[j]["bbox"]) >= args.iou for j in status):
                dup_preds += 1
            else:
                conf[K, col[p["category_id"]]] += 1

        for j, g in enumerate(gts):
            kind, pc = status.get(j, ("bo_sot", None))
            conf[col[g["category_id"]], K if pc is None else col[pc]] += 1
            in_cluster = any(near(g["bbox"], o["bbox"]) for k, o in enumerate(gts) if k != j)
            records.append((g["category_id"], size_bin(g["bbox"]), in_cluster, kind))

    def recall(rs):
        n = len(rs)
        return {"n": n,
                "recall": sum(r[3] == "dung" for r in rs) / n if n else None,
                "sai_lop": sum(r[3] == "sai_lop" for r in rs) / n if n else None,
                "bo_sot": sum(r[3] == "bo_sot" for r in rs) / n if n else None}

    bins = [name for _, _, name in SIZE_BINS]
    by_size = {b: recall([r for r in records if r[1] == b]) for b in bins}
    by_cluster = {}
    for label, flag in (("cum", True), ("rieng", False)):
        rs = [r for r in records if r[2] == flag]
        by_cluster[label] = dict(recall(rs), theo_kich_thuoc={b: recall([r for r in rs if r[1] == b]) for b in bins})
    by_class = {cats[c]: recall([r for r in records if r[0] == c]) for c in cat_ids}
    top_conf = Counter()
    for i in range(K):
        for j in range(K):
            if i != j and conf[i, j]:
                top_conf[(cats[cat_ids[i]], cats[cat_ids[j]])] = int(conf[i, j])

    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    result = {"name": args.name, "gt": args.gt, "pred": args.pred, "iou": args.iou, "score_thr": thr,
              "tong": recall(records), "du_doan_lap": dup_preds, "bao_nham_nen": int(conf[K, :K].sum()),
              "theo_kich_thuoc": by_size, "cum_va_rieng": by_cluster, "theo_lop": by_class,
              "nham_lan_nhieu_nhat": [[a, b, n] for (a, b), n in top_conf.most_common(10)],
              "confusion": {"hang": [cats[c] for c in cat_ids] + ["nền"],
                            "cot": [cats[c] for c in cat_ids] + ["bỏ sót"],
                            "so_lieu": conf.tolist()}}
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Ma trận nhầm lẫn: tô màu theo tỉ lệ trong từng hàng, ghi số đếm trong ô
    names = [cats[c] for c in cat_ids]
    rows = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
    for i in range(K + 1):
        for j in range(K + 1):
            if conf[i, j] and not (i == K and j == K):
                ax.text(j, i, conf[i, j], ha="center", va="center", fontsize=9,
                        color="white" if rows[i, j] > 0.5 else "black")
    ax.set_xticks(range(K + 1), names + ["bỏ sót"], rotation=40, ha="right")
    ax.set_yticks(range(K + 1), names + ["báo nhầm nền"])
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title(f"{args.name}: IoU >= {args.iou}, score >= {thr:.3f} (màu = tỉ lệ theo hàng)")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=150)
    plt.close(fig)

    pct = lambda v: "–" if v is None else f"{v * 100:.1f}"
    L = [f"# Phân tích lỗi: {args.name}", "",
         f"- Nhãn: `{args.gt}`, dự đoán: `{args.pred}`",
         f"- IoU >= {args.iou}, ngưỡng score = {thr_note}",
         f"- Tổng: {len(records)} biển, recall {pct(result['tong']['recall'])}%, "
         f"sai lớp {pct(result['tong']['sai_lop'])}%, bỏ sót {pct(result['tong']['bo_sot'])}%",
         f"- Báo nhầm nền: {result['bao_nham_nen']} box; dự đoán lặp trên cùng một biển: {dup_preds} box", "",
         "## Recall theo kích thước biển (sqrt(w*h), px ảnh gốc)", "",
         "| Kích thước | Số biển | Recall % | Sai lớp % | Bỏ sót % |", "| --- | --- | --- | --- | --- |"]
    L += [f"| {b} | {r['n']} | {pct(r['recall'])} | {pct(r['sai_lop'])} | {pct(r['bo_sot'])} |"
          for b, r in by_size.items()]
    L += ["", "## Biển đứng cụm và đứng riêng", "",
          "Biển trong cụm thường nhỏ hơn, nên so cả trong từng nhóm kích thước để không nhầm "
          "ảnh hưởng của kích thước với ảnh hưởng của cụm.", "",
          "| Nhóm | " + " | ".join(f"Recall {b} (n)" for b in bins) + " | Recall chung (n) |",
          "| --- |" + " --- |" * (len(bins) + 1)]
    for label, r in by_cluster.items():
        cells = [f"{pct(x['recall'])} ({x['n']})" for x in r["theo_kich_thuoc"].values()]
        L.append(f"| {'Cụm' if label == 'cum' else 'Đứng riêng'} | " + " | ".join(cells)
                 + f" | {pct(r['recall'])} ({r['n']}) |")
    L += ["", "## Theo lớp", "", "| Lớp | Số biển | Recall % | Sai lớp % | Bỏ sót % |",
          "| --- | --- | --- | --- | --- |"]
    L += [f"| {c} | {r['n']} | {pct(r['recall'])} | {pct(r['sai_lop'])} | {pct(r['bo_sot'])} |"
          for c, r in by_class.items()]
    L += ["", "## Cặp nhầm lẫn nhiều nhất (nhãn thật -> dự đoán)", ""]
    L += [f"- {a} -> {b}: {n}" for (a, b), n in top_conf.most_common(6)] or ["- Không có"]
    L += ["", "![confusion](confusion.png)", ""]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:-2]))
    print(f"\nĐã lưu {out}/confusion.png, summary.md, summary.json")


if __name__ == "__main__":
    main()
