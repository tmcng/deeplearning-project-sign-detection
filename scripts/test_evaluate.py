"""Tự kiểm tra evaluate.py trước khi giao cho B và C (bàn giao #4).

Cách chạy:
    python scripts/test_evaluate.py --gt data/splits/val.json

Tạo 3 file dự đoán giả từ chính nhãn thật rồi chấm:
    perfect  : chép y nguyên nhãn            -> mAP phải ~100
    shifted  : dời mỗi box 30% chiều rộng    -> mAP@0.5 vẫn cao, mAP@0.5:0.95 tụt mạnh
    halfmiss : bỏ ngẫu nhiên 50% số box      -> mAP khoảng 50
Nếu kết quả không đúng như trên thì evaluate.py hoặc định dạng có vấn đề.

Kết quả thử nghiệm KHÔNG ghi vào reports/eval/results.md (ghi vào reports/eval_test/).
Ba ví dụ này cũng là cách tốt để tự hiểu mAP@0.5 khác mAP@0.5:0.95 thế nào khi vấn đáp.
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/splits/val.json")
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    rng = random.Random(0)
    tmp = Path("preds/_test")
    tmp.mkdir(parents=True, exist_ok=True)

    perfect = [{"image_id": a["image_id"], "category_id": a["category_id"],
                "bbox": list(a["bbox"]), "score": 1.0} for a in gt["annotations"]]
    shifted = [dict(p, bbox=[p["bbox"][0] + 0.3 * p["bbox"][2], *p["bbox"][1:]]) for p in perfect]
    halfmiss = [p for p in perfect if rng.random() < 0.5]

    here = Path(__file__).parent
    for name, preds in [("perfect", perfect), ("shifted", shifted), ("halfmiss", halfmiss)]:
        f = tmp / f"{name}.json"
        f.write_text(json.dumps(preds), encoding="utf-8")
        print(f"\n===================== {name} =====================")
        subprocess.run([sys.executable, str(here / "evaluate.py"), "--gt", args.gt,
                        "--pred", str(f), "--name", f"test_{name}", "--out", "reports/eval_test"],
                       check=True)


if __name__ == "__main__":
    main()
