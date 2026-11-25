# Báo cáo Phân tích Benchmark & Hiệu năng Pipeline (Faster R-CNN)

Báo cáo này tổng hợp và phân tích kết quả thực nghiệm từ bộ dữ liệu log (`summary1.json`, `timings1.csv`, và file JSONL detection log) cho mô hình **Faster R-CNN (`exp03_fpn_aug`)** trên môi trường GPU Tesla T4.

---

## 1. Cấu hình Thực nghiệm & Môi trường (`summary1.json`)

* **Model & Kiến trúc:** `fasterrcnn_resnet50_fpn` (3 `trainable_layers`), tinh chỉnh RPN anchor sizes `[8, 16, 32, 64, 128]` nhằm tối ưu hóa việc phát hiện các biển báo kích thước nhỏ.
* **Dữ liệu đầu vào & Tiền xử lý:**
  * Video nguồn: `video_da_cat.mp4` (`1280x720` @ `30.0 FPS`).
  * Tổng số khung hình: `5,400` frames (`10` frames warmup + `5,390` frames đo đạc thực tế).
  * Vùng quan tâm (ROI): `[0.35, 0.0, 1.0, 0.75]` (loại bỏ 35% lề trái và 25% phần mặt đường phía dưới), sau đó áp dụng transform resize trong khoảng `min_size: 800, max_size: 1800`.
* **Cấu hình Inference & Tracking:**
  * Phần cứng: `Tesla T4` chạy ở độ chính xác bán phần `float16` (`--fp16`), `batch_size: 1`.
  * Ngưỡng tin cậy (Confidence threshold): `0.3`, bật chế độ theo dõi đối tượng (`--track`) và xuất video trực quan (`--save-video`).

---

## 2. Hiệu năng Latency & FPS (`summary1.json` & `timings1.csv`)

| Chỉ số đo lường | Mean (Trung bình) | P50 (Trung vị) | P95 | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: |
| **Inference (`infer_ms`)** | 82.06 ms | 82.30 ms | 85.56 ms | **12.19 FPS** |
| **End-to-End (`pipeline_ms`)** | 89.71 ms | 89.75 ms | 93.59 ms | **11.15 FPS** |

### Nhận xét chi tiết:
1. **Độ ổn định hệ thống cao:** Chênh lệch giữa `P50` và `P95` của `infer_ms` chỉ khoảng **3.26 ms** (`82.30 ms` so với `85.56 ms`), chứng tỏ GPU Tesla T4 xử lý rất mượt mà, không gặp hiện tượng nghẽn cổ chai hay tăng vọt độ trễ (latency spike) bất thường.
2. **Chi phí xử lý ngoại vi (System Overhead) thấp:** Phần chênh lệch giữa `pipeline_ms` và `infer_ms` chỉ tiêu tốn trung bình **~7.65 ms/frame** cho toàn bộ quy trình đọc video, cắt ROI, cập nhật tracker, vẽ bounding box và mã hóa (encode) video đầu ra.
3. **Khả năng đáp ứng Real-time:** Với tốc độ xử lý toàn trình đạt **11.15 FPS** (so với tốc độ gốc 30 FPS của video), hệ thống hiện đạt khoảng **37.2% tốc độ thời gian thực** (xử lý 3 phút video mất khoảng ~8 phút).

---

## 3. Chất lượng Nhận diện & Hành vi Tracking (Detection Log)

Phân tích sâu dữ liệu từng khung hình (đặc biệt tại các đoạn có cụm 4 biển báo dày đặc từ frame `579` đến `659`) cho thấy các đặc điểm sau:

### Ưu điểm:
* **Khả năng bắt biển báo siêu nhỏ ấn tượng:** Nhờ cấu hình anchor size nhỏ và ảnh đầu vào độ phân giải cao, mô hình phát hiện được các biển báo ở khoảng cách rất xa với kích thước bounding box chỉ từ **`7x7` đến `12x12` pixels** (ví dụ: `track_id: 2` tại frame 139 có kích thước `11x7.8 px`, `track_id: 5` tại frame 301 chỉ `6.2x5.9 px`).
* **Độ tin cậy cao ở tầm trung và gần:** Khi xe tiến lại gần mục tiêu (giai đoạn frame 141–321), điểm số confidence của `track_id: 2` ("Nguy hiểm") duy trì cực kỳ ổn định ở mức **0.97 – 0.99**.

### Hạn chế cần lưu ý:
* **Hiện tượng nhảy nhãn (Class Flipping) ở khoảng cách xa:** Tại cùng một vị trí vật lý (cụm biển báo quanh tọa độ `x ≈ 725–737, y ≈ 440–456`), khi kích thước biển báo còn quá nhỏ (~`8x8 px`), mô hình có xu hướng dao động giữa hai lớp có hình dáng tương đồng là **`Giới hạn tốc độ`** (`track_id: 15/19`) và **`Cấm còn lại`** (`track_id: 17`).
* **Trùng lặp Track ID (Duplicate Tracks):** Ở nhiều khung hình (như `531`, `546–548`, `550–561`, `568–571`), cả hai dự đoán cho cùng một biển báo đều vượt ngưỡng `conf = 0.3`. Do thuật toán NMS mặc định lọc độc lập theo từng class, tracker đã khởi tạo **2 `track_id` song song** (`17` và `19`) đè lên nhau cho đến khi xe tiến lại gần hơn và nhãn `Cấm còn lại` (`track_id: 17`) chiếm ưu thế tuyệt đối (`score > 0.97`). Hiện tượng tương tự cũng lặp lại với cặp `track_id: 21` và `23` quanh vùng `x ≈ 814–823`.

---

## 4. Đề xuất Cải thiện Tiếp theo

* **Khắc phục trùng lặp box:** Áp dụng **Class-Agnostic NMS** (lọc NMS không phân biệt nhãn) ở bước hậu xử lý để loại bỏ hoàn toàn tình trạng hai bounding box đè lên nhau trên cùng một biển báo ở tầm xa.
* **Tối ưu tốc độ Inference:** Cân nhắc giảm `max_size` của ảnh đầu vào (ví dụ từ `1800` xuống `1280` hoặc `1333`) hoặc xuất mô hình sang **TensorRT** để đẩy tốc độ xử lý trên Tesla T4 tiệm cận ngưỡng thời gian thực (25–30 FPS).