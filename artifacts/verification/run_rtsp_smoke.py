"""Local physical RTSP check; only stop processes created by this script."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/verification/rtsp_transport'
OUT.mkdir(exist_ok=True)
config = OUT / 'mediamtx.yml'
config.write_text('rtspAddress: :8554\nrtspTransports: [tcp]\nrtmp: no\nhls: no\nwebrtc: no\nsrt: no\npaths:\n  all_others:\n', encoding='utf-8')
children = []
demo_dir = ROOT / 'artifacts/verification/demo_rtsp_smoke'
if demo_dir.exists():
    import shutil
    shutil.rmtree(demo_dir)
with socket.socket() as sock:
    if sock.connect_ex(('127.0.0.1', 8554)) == 0:
        raise SystemExit('Port 8554 already in use; existing service left untouched.')
try:
    with (OUT/'mediamtx.txt').open('w') as server_log, (OUT/'ffmpeg.txt').open('w') as publisher_log, (OUT/'inference.txt').open('w') as infer_log:
        server = subprocess.Popen(['D:/Downloads/mediamtx_v1.21.3_windows_amd64/mediamtx.exe', str(config)], stdout=server_log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        children.append(server)
        for _ in range(50):
            if server.poll() is not None:
                raise RuntimeError('MediaMTX exited; inspect mediamtx.txt')
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1', 8554)) == 0:
                    break
            time.sleep(.2)
        else:
            raise RuntimeError('MediaMTX readiness timeout')
        command = ['ffmpeg', '-hide_banner', '-nostdin', '-re', '-stream_loop', '-1', '-i', str(ROOT/'data/DashCam/traffic.mp4'), '-c', 'copy', '-rtsp_transport', 'tcp', '-f', 'rtsp', 'rtsp://127.0.0.1:8554/live/traffic']
        publisher = subprocess.Popen(command, stdout=publisher_log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        children.append(publisher)
        time.sleep(3)
        if publisher.poll() is not None:
            raise RuntimeError('FFmpeg publisher exited; inspect ffmpeg.txt')
        inference = [sys.executable, '-m', 'streaming.demo', '--model', 'faster-rcnn', '--weights', 'faster_rcnn/checkpoints/fasterrcnn_fpn_aug_best.zip', '--config', 'faster_rcnn/configs/experiment/exp03_fpn_aug.yaml', '--source', 'rtsp://127.0.0.1:8554/live/traffic', '--device', 'cpu', '--max-frames', '5', '--warmup', '1', '--read-timeout-ms', '10000', '--track', '--save-video', '--headless', '--output-dir', 'artifacts/verification/demo_rtsp_smoke']
        (OUT/'commands.json').write_text(json.dumps({'publisher': command, 'inference': inference}, indent=2), encoding='utf-8')
        result = subprocess.run(inference, cwd=ROOT, stdout=infer_log, stderr=subprocess.STDOUT, timeout=180)
        (OUT/'exit.json').write_text(json.dumps({'inference_exit_code': result.returncode}), encoding='utf-8')
        raise SystemExit(result.returncode)
finally:
    for child in reversed(children):
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
