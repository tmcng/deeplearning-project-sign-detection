# Bảng ánh xạ nguồn tài liệu → `docs/proposal.tex`

Tài liệu này ánh xạ các file ghi chép, báo cáo và cấu hình (`.md`, `.txt`) trong dự án vào từng mục tương ứng của báo cáo ACL (`docs/proposal.tex`).

| File nguồn | Mục nạp trong `proposal.tex` | Loại thông tin được sử dụng |
| :--- | :--- | :--- |
| `README.md` | Author/Contributions, Method (Bảng 1), Experiments | Vai trò thành viên, tổng quan kiến trúc repo và danh sách script tự cài đặt |
| `requirements.txt` | Method (Bảng 1) | Tên các thư viện nền tảng (PyTorch, Ultralytics, pycocotools, OpenCV, ONNX) |
| `docs/notes_du_lieu.md` | Experiments (Mục 5.1), Analysis (Mục 6.2), Limitations | Hiện trạng lỗi nhãn trùng, biển cụm, cặp lớp dễ nhầm và nguồn hình `bo_sot_mo_hinh_tot_nhat.png` |
| `docs/realtime_report.md` | Abstract, Experiments (Mục 5.2–5.3), Analysis (Mục 6.1), Conclusion, Limitations | Chuỗi ablation Faster R-CNN, lưu ý phương pháp luận, giao thức đo FPS và nhận xét triển khai |
| `docs/thi_nghiem_mat_can_bang.md` | Abstract, Related Work, Method (Mục 4.1–4.2), Experiments (Mục 5.2), Analysis (Mục 6.1, Bảng 6), Limitations | Thiết kế RFS/`cls_pw`, kết quả cân bằng lớp, kiểm định bootstrap và 4 lý do không cải thiện |
| `docs/thi_nghiem_ro_ri.md` | Abstract, Experiments (Mục 5.1, Bảng 3), Limitations | Thiết kế thí nghiệm `random` vs `street`, kết quả chênh lệch +5,7 mAP@0.5 và giới hạn số tuyến đường |
| `docs/tom_tat_FPN.md` | Related Work, Method (Mục 4.1) | Lý thuyết FPN, lý do tinh chỉnh anchor nhỏ (8–128 px) cho Faster R-CNN |
| `docs/tom_tat_YOLO.md` | Related Work, Method (Mục 4.1–4.2), `custom.bib` | Kiến trúc YOLOv8 (PAN-FPN, Decoupled Head, CIoU/DFL), lý do tăng `imgsz=1280` & Tiling/SAHI, BibTeX Redmon 2016 |
| `reports/eval/results.md` | Abstract, Experiments (Mục 5.3, Bảng 4), Conclusion, Appendix (Bảng 7) | Bảng điểm tổng hợp 9 cấu hình trên tập `val` |
| `reports/eval_leak/results.md` | Experiments (Mục 5.1, Bảng 3) | Số liệu mAP@0.5 và mAP@0.5:0.95 của thí nghiệm rò rỉ dữ liệu |
| `reports/imbalance/so_sanh.md` | Analysis (Mục 6.1–6.2, Bảng 6) | Đối chiếu AP50, Recall và tỉ lệ sai lớp chi tiết cho `imb_rfs`, `imb_clspw` |
| `reports/latency_fps/benchmark_faster_r_cnn.md` | Abstract, Experiments (Mục 5.3, Bảng 5), Analysis (Mục 6.2), Conclusion | FPS/Latency (Mean, P50, P95) của Faster R-CNN `exp03` trên Tesla T4 và hành vi nhảy nhãn/trùng track trên video |
| `reports/latency_fps/bench_mark_yolov8.md` | Abstract, Experiments (Mục 5.3, Bảng 5), Conclusion | FPS/Latency (Mean, P50, P95) của YOLOv8s `imgsz=1280` trên Tesla T4 |
| `reports/errors/phan_tich_loi.md` | Analysis (Mục 6.2, Hình 2–3), Limitations | Phân tích lỗi tổng hợp theo kích thước, biển cụm, cặp nhầm lớp biển tròn viền đỏ và lưu ý `conf=0.05` |
| `reports/errors/fasterrcnn_exp01_baseline/summary.md` | Analysis (Mục 6.2) | Số liệu bỏ sót, sai lớp và cặp nhầm lẫn chi tiết của Faster R-CNN `exp01` |
| `reports/errors/fasterrcnn_exp03_fpn_aug/summary.md` | Analysis (Mục 6.2) | Số liệu recall theo kích thước, biển cụm và tỉ lệ sai lớp của Faster R-CNN `exp03` |
| `reports/errors/yolov8s_640/summary.md` | Analysis (Mục 6.2) | Số liệu recall nhóm `<8` px và tỉ lệ bỏ sót của baseline YOLOv8s 640 |
| `reports/errors/yolov8s_640_sahi/summary.md` | Analysis (Mục 6.1–6.2) | Số liệu đối chiếu mức độ trùng lặp dự đoán và sai lớp khi bật SAHI trên 640 |
| `reports/errors/yolov8s_1280/summary.md` | Analysis (Mục 6.1–6.2, Hình 2) | Số liệu recall, sai lớp của `yolov8s_1280` và ma trận nhầm lẫn `confusion.png` |
| `reports/errors/yolov8s_tiled_640_sahi/summary.md` | Analysis (Mục 6.1–6.2) | Số liệu dự đoán lặp (71 box) và recall của cấu hình Tiling + SAHI |
| `reports/errors/imb_clspw/summary.md` | Analysis (Mục 6.1) | Hỗ trợ đối chiếu số liệu sai lớp của `imb_clspw` (tham chiếu chéo với `thi_nghiem_mat_can_bang.md`) |
| `reports/errors/imb_rfs/summary.md` | Analysis (Mục 6.1) | Hỗ trợ đối chiếu số liệu sai lớp của `imb_rfs` (tham chiếu chéo với `thi_nghiem_mat_can_bang.md`) |
| `reports/eda/eda_summary.txt` | Experiments (Mục 5.1) | Thống kê gốc trước làm sạch (11.000 box, tỉ lệ lệch 5,5 lần) |
| `reports/eda_dedup/eda_summary.txt` | Abstract, Introduction, Experiments (Mục 5.1, Bảng 2) | Thống kê sau làm sạch (9.077 box, 77,6% box nhỏ, trung vị 16,5 px, phân bố 7 lớp) |
| `faster_rcnn/README.md` | Method (Bảng 1), Experiments (Mục 5.2) | Quy trình train/eval Faster R-CNN và lưu ý ngưỡng `conf` đánh giá |
| `faster_rcnn/configs/README.md` | Method (Mục 4.3, Bảng 1) | Kiến trúc hệ thống cấu hình modular YAML cho các thí nghiệm `exp01`–`exp03` |
| `streaming/README.md` | Method (Mục 4.3, Bảng 1), Experiments (Mục 5.3), Limitations | Kiến trúc `streaming/`, lộ trình tối ưu FPS (CPU→GPU→FP16→ROI), ONNX/TensorRT và giả lập RTSP |
| `data/leak/leak_stats.txt` | Experiments (Mục 5.1) | Số liệu thống kê chi tiết số ảnh/box của hai tập chia rò rỉ (`street` vs `random`) |
| `data/splits_raw/split_stats.txt` | Experiments (Mục 5.1) | Đối chiếu số lượng box gốc trước khi gộp trùng ở từng tập `train`/`val`/`test` |
| `data/splits_random/split_stats.txt` | *(Không nạp trực tiếp)* | Tài liệu tham chiếu phụ về phân bố chia ngẫu nhiên ban đầu (đã thay thế bằng `data/splits/` và `data/leak/`) |
