"""Snapshot source artifacts without claiming that missing experiments are complete."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra", type=Path, action="append", default=[],
                        help="Additional checkpoint, video, benchmark summary or trace to hash (repeatable).")
    args = parser.parse_args()
    paths = set()
    for pattern in ("reports/eval/*.json", "reports/eval/results.md",
                    "faster_rcnn/configs/**/*.yaml", "preds/*exp0*.json",
                    "preds/yolov8s*.json", "data/splits/*.json"):
        paths.update(ROOT.glob(pattern))
    for extra in args.extra:
        path = extra if extra.is_absolute() else ROOT / extra
        if not path.is_file():
            parser.error(f"Required extra artifact does not exist: {path}")
        paths.add(path)
    records = []
    for path in sorted(paths, key=str):
        path = path.resolve()
        try:
            name = path.relative_to(ROOT).as_posix()
        except ValueError:
            name = str(path)
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": digest(path)})
    payload = {
        "schema_version": 1,
        "status": "source_snapshot_only_not_final_results_freeze",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_worktree_status": git("status", "--porcelain"),
        "limitations": ["Hashing establishes artifact identity, not scientific validity.",
                        "Historical JSON metrics are not rerun by this script.",
                        "Final performance freeze requires verified benchmarks for the claimed scope and demo evidence; review manually."],
        "files": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves an existing snapshot rather than silently replacing it.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Saved {len(records)} artifact hashes to {args.output}; status={payload['status']}")


if __name__ == "__main__":
    main()

