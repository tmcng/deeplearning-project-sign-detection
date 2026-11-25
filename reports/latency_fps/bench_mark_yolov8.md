# Báo cáo Đánh giá Hiệu năng (FPS & Latency) — Hệ thống Nhận diện & Theo dõi Biển báo Giao thông

## 1. Tổng quan & Môi trường Thực nghiệm

Đánh giá được thực hiện trên tập dữ liệu video thực tế (`video_da_cat.mp4`, độ phân giải $1280 \times 720$, 30 FPS gốc) nhằm đo lường khả năng xử lý thời gian thực (real-time) của mô hình nhận diện kết hợp theo dõi đa đối tượng (Multi-Object Tracking).

* **Phần cứng & Môi trường:**
  * **GPU:** NVIDIA Tesla T4 (15.64 GB VRAM)
  * **Nền tảng phần mềm:** Python 3.12.13, PyTorch 2.10.0+cu128, Ultralytics 8.4.37
* **Cấu hình Mô hình & Pipeline:**
  * **Kiến trúc:** YOLO (`best.pt`), độ phân giải đầu vào `imgsz = 1280`
  * **Độ chính xác tính toán:** Half-precision (`FP16 = True`), `batch_size = 1`
  * **Tác vụ đi kèm:** Bật theo dõi đối tượng (`track = True`) và xuất video kết quả (`save_video = True`)
* **Quy chuẩn đo lường (Timing Contract):**
  * Tổng số khung hình video: **5,400 frames**.
  * Loại bỏ **10 frames đầu tiên** (`warmup_frames = 10`) để ổn định bộ nhớ đệm GPU; số liệu thống kê được tính chính xác trên **5,390 frames** thực đo.
  * Đo lường theo phương pháp xử lý tuần tự từng khung hình (offline video benchmark), **chưa bao gồm** độ trễ phần cứng cảm biến camera vật lý (*glass-to-glass latency*).

---

## 2. Kết quả Đo lường Chi tiết

### 2.1. Bảng tổng hợp FPS và Độ trễ (Latency)

| Thành phần đo lường | Tốc độ xử lý (FPS) | Mean (ms) | P50 / Median (ms) | P95 (ms) | P99 (ms) | Max (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Capture (Đọc & giải mã frame)** | ~1063.8 FPS | 0.94 | 0.90 | 1.15 | 1.38 | 8.12 |
| **Inference (Tiền xử lý + Mô hình + NMS)** | **71.23 FPS** | **14.04** | **13.91** | **15.28** | **16.09** | **40.45** |
| **End-to-End Pipeline (Toàn quy trình)** | **46.28 FPS** | **21.61** | **21.46** | **23.35** | **25.26** | **49.20** |

### 2.2. Bóc tách Chi phí Thời gian (Latency Breakdown)

Dựa trên độ trễ trung bình toàn quy trình là **21.61 ms/frame**, thời gian thực thi được phân bổ qua các giai đoạn như sau:

1. **Giải mã khung hình (`capture_ms`):** **0.94 ms** (~4.3%) — Tốc độ đọc dữ liệu từ luồng video rất nhanh, không gây nghẽn đầu vào.
2. **Suy luận mô hình (`infer_ms`):** **14.04 ms** (~65.0%) — Bao gồm resize/chuẩn hóa ảnh $1280 \times 720 \rightarrow 1280$, chạy mạng nơ-ron trên Tesla T4 ở chế độ FP16 và lọc Non-Maximum Suppression (NMS).
3. **Hậu xử lý & Ghi xuất dữ liệu (Overhead):** **~6.63 ms** (~30.7%) — Bao gồm thuật toán cập nhật ID theo dõi (`tracking`), vẽ khung bao (bounding box/nhãn) lên ảnh, ghi log và mã hóa (encode) xuất file video đầu ra.

---

## 3. Đánh giá & Nhận xét Kỹ thuật

* **Đạt chuẩn thời gian thực (Real-time):** Với tốc độ toàn quy trình đạt **46.28 FPS** (và suy luận thuần đạt **71.23 FPS**), hệ thống vượt xa ngưỡng tiêu chuẩn 30 FPS của camera giao thông thông thường, đảm bảo khả năng phản hồi mượt mà ngay cả ở độ phân giải đầu vào cao (`1280px`).
* **Độ ổn định cao (Low Jitter):** Khoảng cách giữa độ trễ trung vị (**P50 = 21.46 ms**) và phân vị 99% (**P99 = 25.26 ms**) chỉ chênh lệch **3.80 ms**. Điều này cho thấy hệ thống hoạt động rất ổn định, ít bị trồi sụt khung hình ngay cả khi số lượng biển báo xuất hiện dày đặc trong một khung hình (lên tới 8–9 đối tượng cùng lúc). Các đỉnh trễ cực đại (`Max = 49.20 ms`) chỉ xuất hiện cục bộ ở vài khung hình đơn lẻ và vẫn nằm trong ngưỡng chấp nhận được.
* **Tiềm năng tối ưu hóa khi triển khai thực tế:** Trong bài benchmark này, cờ `--save-video` được kích hoạt khiến hệ thống tốn thêm chi phí encode video vào `pipeline_ms`. Khi triển khai trên môi trường sản phẩm (production) chỉ truyền cảnh báo hoặc hiển thị trực tiếp mà không cần ghi file video ra ổ đĩa, tốc độ toàn hệ thống dự kiến sẽ tăng lên mức **~60–65 FPS**.

---

## 4. Lưu ý khi Trình bày trong Báo cáo

* **Tính minh bạch của số liệu:** Nên trích dẫn song song cả hai chỉ số **Inference FPS (71.23 FPS)** để thể hiện năng lực tính toán của mô hình tối ưu FP16 ở `imgsz=1280`, và **Pipeline FPS (46.28 FPS)** để phản ánh đúng hiệu năng thực tế khi chạy kèm thuật toán tracking và xuất hình ảnh.
* **Ghi chú cấu hình:** Đảm bảo chú thích rõ việc sử dụng `FP16` và kích thước ảnh đầu vào `1280` vì đây là yếu tố then chốt giúp cân bằng giữa độ chính xác nhận diện biển báo kích thước nhỏ (small objects) và tốc độ xử lý trên GPU Tesla T4.