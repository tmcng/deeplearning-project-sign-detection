"""Tạo các hình cho mục Phân tích của báo cáo (docs/report/figures/).

Cách chạy (từ thư mục gốc repo, sau khi đã chạy evaluation/analyze_errors.py cho các mô hình trên test):
    python docs/report/make_figures.py --images za_traffic_2020/traffic_train/images

Kết quả:
    figures/recall_theo_kich_thuoc.pdf : recall theo kích thước biển, 3 mô hình cuối, tập test
    figures/vi_du_du_doan.pdf          : ví dụ đúng / bỏ sót / nhầm lớp / báo nhầm nền của YOLOv8s 1280
"""
import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))
from analyze_errors import greedy, iou  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
MODELS = [("yolov8s_1280", "YOLOv8s 1280"), ("yolov8s_tiled_640_sahi", "YOLOv8s cắt ô + SAHI"),
          ("fasterrcnn_exp03", "Faster R-CNN")]
COLORS = ["#257de0", "#f47230", "#00b894"]  # 3 màu đầu của bảng màu phân loại, đã kiểm tra mù màu
INK, INK2, GRID = "#1a1a1a", "#5f5f5f", "#e3e3e3"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def recall_by_size():
    bins = ["<8", "8-16", "16-32", ">=32"]
    data = {m: json.loads((ROOT / f"reports/errors/{m}_test/summary.json").read_text(encoding="utf-8"))
            for m, _ in MODELS}
    n = [data[MODELS[0][0]]["theo_kich_thuoc"][b]["n"] for b in bins]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    w, gap = 0.26, 0.02
    for k, ((m, label), c) in enumerate(zip(MODELS, COLORS)):
        xs = [i + (k - 1) * (w + gap) for i in range(len(bins))]
        ys = [100 * data[m]["theo_kich_thuoc"][b]["recall"] for b in bins]
        ax.bar(xs, ys, width=w, color=c, label=label, zorder=2)
    ax.set_xticks(range(len(bins)), [f"{b} px\n({k} biển)" for b, k in zip(bins, n)])
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              handlelength=1, columnspacing=0.8, labelcolor=INK)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "recall_theo_kich_thuoc.pdf")
    fig.savefig(OUT / "recall_theo_kich_thuoc.png", dpi=200)
    plt.close(fig)


def examples(images_dir, model="yolov8s_1280", seed=3):
    gt = json.loads((ROOT / "data/splits/test.json").read_text(encoding="utf-8"))
    preds = json.loads((ROOT / f"preds/{model}_test.json").read_text(encoding="utf-8"))
    thr = json.loads((ROOT / f"reports/errors/{model}_test/summary.json").read_text(encoding="utf-8"))["score_thr"]
    cats = {c["id"]: c["name"] for c in gt["categories"]}
    files = {im["id"]: im["file_name"] for im in gt["images"]}
    G, P = {}, {}
    for a in gt["annotations"]:
        G.setdefault(a["image_id"], []).append(a)
    for p in preds:
        if p["score"] >= thr:
            P.setdefault(p["image_id"], []).append(p)

    # Ghép giống analyze_errors.py: đúng lớp trước, rồi bất kể lớp
    cases = {"dung": [], "bo_sot": [], "sai_lop": [], "nen": []}
    for img, gts in G.items():
        ps = sorted(P.get(img, []), key=lambda p: -p["score"])
        gi = [(j, g["bbox"], g["category_id"]) for j, g in enumerate(gts)]
        pi = [(i, p["bbox"], p["category_id"]) for i, p in enumerate(ps)]
        right = greedy(pi, gi, 0.5, same_class=True)
        taken = set(right.values())
        wrong = greedy([t for t in pi if t[0] not in right], [t for t in gi if t[0] not in taken], 0.5, False)
        size = lambda b: (b[2] * b[3]) ** 0.5
        for i, j in right.items():
            if size(gts[j]["bbox"]) < 14:
                cases["dung"].append((img, gts[j], ps[i]))
        for i, j in wrong.items():
            cases["sai_lop"].append((img, gts[j], ps[i]))
        for j, g in enumerate(gts):
            if j not in taken and j not in wrong.values() and 8 <= size(g["bbox"]) < 14:
                cases["bo_sot"].append((img, g, None))
        for i, p in enumerate(ps):
            if i not in right and i not in wrong and p["score"] >= 0.6 \
                    and not any(iou(p["bbox"], g["bbox"]) > 0 for g in gts):
                cases["nen"].append((img, None, p))

    rng = random.Random(seed)
    titles = {"dung": "Đúng (biển < 14 px)", "bo_sot": "Bỏ sót", "sai_lop": "Nhầm lớp", "nen": "Báo nhầm nền"}
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.6))
    for col, key in enumerate(titles):
        picks = rng.sample(cases[key], 2)
        for row, (img, g, p) in enumerate(picks):
            ax = axes[row, col]
            box = (g or p)["bbox"]
            cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            r = max(box[2], box[3], 12) * 2.2
            im = Image.open(Path(images_dir) / files[img]).convert("RGB")
            ax.imshow(im.crop((cx - r, cy - r, cx + r, cy + r)).resize((240, 240), Image.LANCZOS))
            s = 240 / (2 * r)
            for b, c, ls in ((g, "#00c000", "-"), (p, "#ff2d2d", "--")):
                if b:
                    x, y, w, h = b["bbox"]
                    ax.add_patch(Rectangle(((x - cx + r) * s, (y - cy + r) * s), w * s, h * s,
                                           fill=False, edgecolor=c, linewidth=1.4, linestyle=ls))
            lines = []
            if g:
                lines.append(f"Thật: {cats[g['category_id']]} ({(box[2] * box[3]) ** 0.5:.0f} px)")
            if p:
                lines.append(f"Đoán: {cats[p['category_id']]} ({p['score']:.2f})")
            ax.set_xlabel("\n".join(lines), fontsize=6.5, color=INK)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[key], fontsize=8, color=INK)
    fig.tight_layout(pad=0.6, h_pad=1.2, w_pad=0.3)
    fig.savefig(OUT / "vi_du_du_doan.pdf")
    fig.savefig(OUT / "vi_du_du_doan.png", dpi=200)
    plt.close(fig)
    return {k: len(v) for k, v in cases.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="za_traffic_2020/traffic_train/images")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    recall_by_size()
    print("Số ứng viên mỗi loại ví dụ:", examples(ROOT / args.images, seed=args.seed))
    print(f"Đã ghi hình vào {OUT}")


if __name__ == "__main__":
    main()
