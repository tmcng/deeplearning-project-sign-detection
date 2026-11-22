# Demo và đo tốc độ Faster R-CNN

Chạy từ thư mục gốc dự án trên nhánh `faster-rcnn`. Module mới không thay đổi
`test_view.py`, code train hay evaluate. Dùng checkpoint `.zip` trực tiếp;
đây là state dict PyTorch, không cần giải nén. Phải chọn YAML đúng checkpoint.

## Chạy video có box, ID và log

```powershell
.venv/Scripts/python.exe -m streaming.demo --model faster-rcnn --weights faster_rcnn/checkpoints/fasterrcnn_fpn_aug_best.zip --config faster_rcnn/configs/experiment/exp03_fpn_aug.yaml --source data/DashCam/traffic.mp4 --device cpu --max-frames 300 --warmup 10 --track --save-video --output-dir artifacts/verification/demo_aug
```

Bấm `q` để thoát; thêm `--headless` khi không cần cửa sổ. Mỗi lần chạy dùng
thư mục đầu ra mới để không ghi đè bằng chứng. `--max-frames` tính cả warmup.
`--device cuda:0` chỉ dùng khi môi trường PyTorch thực sự hỗ trợ CUDA.
`--threads 4` là mặc định cho CPU; giữ cố định khi so sánh.

Đầu ra: `demo.mp4`, `snapshot.jpg`, `detections.jsonl`, `timings.csv`,
`summary.json`. Video xuất giữ FPS của nguồn để xem lại; tốc độ phát video
không phải tốc độ xử lý. JSONL chứa nhãn tiếng Việt; cửa sổ dùng class ID vì
font mặc định OpenCV không hiển thị đúng tiếng Việt. Tracking là ghép IoU cùng
lớp đơn giản, không phải đánh giá MOT hay đảm bảo giữ ID qua che khuất.

## Luồng RTSP qua MediaMTX

Mở hai terminal riêng:

```powershell
& 'D:/Downloads/mediamtx_v1.21.3_windows_amd64/mediamtx.exe'
```

```powershell
ffmpeg -re -stream_loop -1 -i data/DashCam/traffic.mp4 -c copy -f rtsp rtsp://localhost:8554/live/traffic
```

Trong terminal thứ ba, chạy lệnh demo ở trên nhưng thay `--source` bằng
`rtsp://localhost:8554/live/traffic` và chọn output-dir mới.
Luồng hiện đọc tuần tự: máy xử lý chậm hơn nguồn có thể tích lũy trễ. Đây là
demo inference trên RTSP, chưa chứng minh đáp ứng thời gian thực hoặc độ trễ
camera tới màn hình. Mất luồng sẽ dừng với trạng thái lỗi có log; không phát lại
frame cũ. Không có tự reconnect trong phiên bản này.

## Benchmark và ablation

Đo từ cùng video cục bộ để tránh nhầm giới hạn tốc độ nguồn RTSP với khả năng
xử lý. Chạy từng cấu hình tuần tự, cùng thiết bị, threads, đoạn video, confidence,
warmup và tùy chọn draw/track/save. Không chạy benchmark đồng thời.

| Cấu hình | Checkpoint trong `faster_rcnn/checkpoints` | YAML trong `faster_rcnn/configs/experiment` |
| --- | --- | --- |
| exp01 | fasterrcnn_baseline_best1.zip | exp01_baseline.yaml |
| exp02 | fasterrcnn_fpn_small_anchor_best.zip | exp02_fpn_small_anchor.yaml |
| exp03 | fasterrcnn_fpn_aug_best.zip | exp03_fpn_aug.yaml |

`infer_fps` gồm chuyển đổi ảnh, truyền tensor, model, hậu xử lý và lấy kết quả
về CPU. `pipeline_fps` còn gồm đọc frame, vẽ, tracking nếu bật, ghi video nếu
bật, hiển thị nếu bật và ghi detection log; không gồm ghi CSV. Cả hai loại
đều bỏ warmup và thời gian nạp model. CUDA được synchronize khi đo.
FPS tổng hợp là số frame chia tổng thời gian, không lấy trung bình FPS từng frame.
P50/P95 là thời gian mỗi frame, không phải độ trễ đầu cuối RTSP.

Trước khi chốt tốc độ cho báo cáo: tối thiểu 300 frame đo sau warmup và ba lần
lặp trên thiết bị đích, ghi mean/P95 và biến thiên giữa các lần. Chạy ngắn hơn
chỉ là smoke test. So sánh accuracy sử dụng JSON hiện có, không lấy confidence
0.3 của demo để thay thế đánh giá COCO. Chi tiết ablation và bản thảo báo cáo:
[realtime_report.md](../docs/realtime_report.md).

## Verifier và snapshot

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_streaming_demo.py -v
.venv/Scripts/python.exe scripts/freeze_results.py --help
```

Bằng chứng kiểm tra nằm ở `artifacts/verification/`. Manifest SHA256 xác định
đúng nguồn số liệu và cấu hình; nó không tự chứng nhận kết quả benchmark đủ
để đóng băng. Không commit checkpoint hoặc video nặng. YOLO thuộc phần bàn giao
của thành viên khác; adapter `--model yolo --weights <path> --imgsz 1280` chỉ
dành cho inference toàn ảnh, không thay thế phép đo SAHI và chưa được kiểm thử
bằng checkpoint YOLO của nhóm.
