# Bằng chứng thực thi

Các artifact được tạo từ lệnh chạy thật trên máy Windows của dự án. Các video
MP4 được giữ cục bộ (đã nằm trong quy tắc gitignore của repository); snapshot,
trace CSV/JSONL, summary và log nhỏ được commit để reviewer kiểm tra.

| Thư mục | Nội dung |
| --- | --- |
| `setup` | Đồng bộ Git, phần cứng, Python packages và stdout các lượt chạy |
| `demo_baseline_smoke` | Checkpoint exp01: 12 frame đầu nguồn gốc, bỏ 2 warmup |
| `demo_anchors_smoke` | Checkpoint exp02: cùng điều kiện |
| `demo_aug_smoke` | Checkpoint exp03: cùng điều kiện |
| `demo_aug_offline` | Clip 16 giây lấy mẫu 2 FPS, xử lý offline, không mô phỏng tốc độ realtime |
| `demo_rtsp_smoke` | Kiểm tra inference qua MediaMTX/FFmpeg, không đo độ trễ đầu cuối |
| `rtsp_transport` | Cấu hình MediaMTX localhost, lệnh, log server/publisher/inference |
| `verifier` | Kiểm tra độc lập checkpoint, tests, số liệu báo cáo và artifact đầu ra |
| `report` | Hash nguồn số liệu và snapshot bàn giao |

`summary.json` lưu lệnh thực thi, SHA256 checkpoint/video/config/implementation,
thiết bị, phiên bản thư viện và trạng thái. CSV giữ cả warmup để đối chiếu;
summary chỉ tổng hợp frame sau warmup. Git revision của lượt đo có thể trước
commit bàn giao; `git_dirty` và hash implementation được giữ nguyên, không
viết lại lịch sử bằng chứng.

Verifier tính lại FPS, mean/p50/p95 từ CSV, đối chiếu JSONL và giải mã toàn bộ
video để đếm frame. Snapshot là khung hình đã vẽ từ chương trình, không phải
ảnh chụp một cửa sổ desktop. Các lần chạy này dùng `--headless`; chưa chứng nhận
GUI bằng ảnh chụp màn hình. Box hiển thị chứng minh bước vẽ hoạt động, không
chứng minh detection đúng vì video không có ground truth.

`setup/install.txt` là log cài đặt ban đầu đã dừng sau khi người dùng thu hẹp
phạm vi về Faster R-CNN. `setup/install_frcnn.txt` ghi lượt cài đặt hoàn tất;
`setup/environment.txt` ghi môi trường thực dùng. Không cài YOLO để suy diễn
kết quả checkpoint của thành viên khác.

Chạy lại verifier (đường dẫn video phải còn tồn tại cục bộ):

```powershell
.venv/Scripts/python.exe artifacts/verification/verifier/verify_run.py artifacts/verification/demo_baseline_smoke artifacts/verification/demo_anchors_smoke artifacts/verification/demo_aug_smoke artifacts/verification/demo_aug_offline artifacts/verification/demo_rtsp_smoke
```

Các phép đo ngắn là smoke test, không thay thế benchmark dài có lặp trên thiết
bị triển khai. Snapshot nguồn không đồng nghĩa đóng băng kết luận realtime.
