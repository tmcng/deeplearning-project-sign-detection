"""So sánh từng lớp giữa một thí nghiệm gốc và các biến thể (thí nghiệm mất cân bằng lớp, tuần 4).

Cần chạy trước cho mỗi thí nghiệm:
    python scripts/evaluate.py       --gt data/splits/val.json --pred preds/<tên>_val.json --name <tên>
    python scripts/analyze_errors.py --gt data/splits/val.json --pred preds/<tên>_val.json --name <tên>

Cách chạy:
    python scripts/compare_runs.py --base yolo_tiled_base --runs yolo_tiled_rfs yolo_tiled_clspw \
        --out reports/imbalance/so_sanh.md

Đọc reports/eval/<tên>.json (AP) và reports/errors/<tên>/summary.json (recall, sai lớp), in một bảng
markdown: mỗi lớp một dòng, mỗi biến thể ghi giá trị và chênh lệch so với gốc. Nếu chưa chạy
analyze_errors.py cho thí nghiệm nào thì các cột recall/sai lớp của nó để trống.
"""
import argparse
import json
from pathlib import Path


def load(name, eval_dir, err_dir):
    ev = json.loads((Path(eval_dir) / f"{name}.json").read_text(encoding="utf-8"))
    err_file = Path(err_dir) / name / "summary.json"
    err = json.loads(err_file.read_text(encoding="utf-8")) if err_file.exists() else None
    return ev, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="tên thí nghiệm gốc (như --name của evaluate.py)")
    ap.add_argument("--runs", nargs="+", required=True, help="tên các biến thể")
    ap.add_argument("--eval-dir", default="reports/eval")
    ap.add_argument("--err-dir", default="reports/errors")
    ap.add_argument("--out", default=None, help="ghi bảng ra file .md (mặc định chỉ in ra)")
    args = ap.parse_args()

    names = [args.base] + args.runs
    data = {n: load(n, args.eval_dir, args.err_dir) for n in names}
    base_ev, base_err = data[args.base]
    classes = list(base_ev["per_class"])

    def cell(v, b):
        if v is None:
            return "–"
        return f"{v:.1f}" if b is None else f"{v:.1f} ({v - b:+.1f})"

    def metrics(name, cls):
        ev, err = data[name]
        ap50 = ev["per_class"][cls]["AP50"] * 100
        rec = err["theo_lop"][cls] if err else None
        return (ap50,
                rec["recall"] * 100 if rec else None,
                rec["sai_lop"] * 100 if rec else None)

    head = ["Lớp"]
    for n in names:
        head += [f"AP50 {n}", f"Recall {n}", f"Sai lớp {n}"]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for cls in classes:
        b = metrics(args.base, cls)
        row = [cls]
        for n in names:
            m = metrics(n, cls)
            row += [cell(v, None if n == args.base else bv) for v, bv in zip(m, b)]
        lines.append("| " + " | ".join(row) + " |")

    # Dòng tổng: mAP50, AP_small, recall chung
    tot = ["**Chung (mAP50 / AP_small / recall)**"]
    for n in names:
        ev, err = data[n]
        v = f"{ev['mAP50'] * 100:.1f} / {ev['AP_small'] * 100:.1f}"
        if n != args.base:
            v += f" ({(ev['mAP50'] - base_ev['mAP50']) * 100:+.1f})"
        tot += [v, f"{err['tong']['recall'] * 100:.1f}" if err else "–", ""]
    lines.append("| " + " | ".join(tot) + " |")

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
