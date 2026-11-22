"""Independently verify produced video evidence and FPS arithmetic, never infer a model."""
import csv
import json
import math
import sys
from pathlib import Path

import cv2


def verify(directory):
    root = Path(directory)
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    rows = list(csv.DictReader((root / 'timings.csv').open(encoding='utf-8')))
    detections = [json.loads(line) for line in (root / 'detections.jsonl').read_text(encoding='utf-8').splitlines()]
    assert summary['status'] in ('completed', 'eof', 'user_stopped'), summary
    assert len(rows) == len(detections) == summary['frames']
    measured = [r for r in rows if int(r['warmup']) == 0]
    assert len(measured) == summary['measured_frames'] > 0
    for i, (row, det) in enumerate(zip(rows, detections)):
        assert int(row['frame']) == det['frame'] == i
        assert bool(int(row['warmup'])) == det['warmup'] == (i < summary['arguments']['warmup'])
        assert int(row['detections']) == len(det['detections'])
        assert float(row['pipeline_ms']) >= float(row['infer_ms']) > 0
    fps = {}
    for kind in ('infer', 'pipeline'):
        values = sorted(float(r[kind + '_ms']) for r in measured)
        fps[kind] = len(measured) * 1000 / sum(values)
        assert math.isclose(fps[kind], summary[kind + '_fps'], rel_tol=1e-10)
        assert math.isclose(sum(values) / len(values), summary[kind + '_latency_ms']['mean'], rel_tol=1e-10)
        for percentile in (50, 95):
            index = (len(values) - 1) * percentile / 100
            lower = math.floor(index)
            upper = math.ceil(index)
            interpolated = values[lower] + (values[upper] - values[lower]) * (index - lower)
            assert math.isclose(interpolated, summary[kind + '_latency_ms']['p' + str(percentile)], rel_tol=1e-10)
    image = cv2.imread(str(root / 'snapshot.jpg'))
    assert image is not None
    assert [image.shape[1], image.shape[0]] == summary['source_dimensions']
    video_frames = None
    if summary['arguments']['save_video']:
        cap = cv2.VideoCapture(str(root / 'demo.mp4'))
        assert cap.isOpened()
        video_frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert [frame.shape[1], frame.shape[0]] == summary['source_dimensions']
            video_frames += 1
        cap.release()
        assert video_frames == len(rows)
    return dict(path=str(root), status='PASS', frames=len(rows), measured=len(measured),
                decoded_video_frames=video_frames, recomputed_fps=fps,
                detections=sum(len(d['detections']) for d in detections))


if __name__ == '__main__':
    results = [verify(path) for path in sys.argv[1:]]
    output = Path(__file__).with_name('artifact-checks.json')
    output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(output.read_text(encoding='utf-8'))
