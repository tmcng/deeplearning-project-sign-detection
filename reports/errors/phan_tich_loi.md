# Phân tích lỗi baseline (bàn giao #6, tuần 3)

Tập: `data/splits/val.json` (696 ảnh, 1636 biển, đã gộp box trùng). Chi tiết từng mô hình nằm trong `reports/errors/<mô hình>/` (`summary.md`, `confusion.png`).

- **Faster R-CNN chưa phân tích được.** Repo mới có file điểm (`reports/eval/fasterrcnn_baseline_check.json`), chưa có file dự đoán trên val. Cần C đưa file ở đường dẫn `OUTPUT_PRED_JSON` vào `preds/fasterrcnn_baseline_val.json`. Các số AP của Faster R-CNN dưới đây lấy từ file điểm.
- **Cách tính recall:** ngưỡng score được tự chọn cho F1 cao nhất ở từng mô hình, vì score của các mô hình không cùng thang. Một biển được tính là tìm thấy khi có dự đoán đúng lớp với IoU ≥ 0.5.

## Tổng quan

| Mô hình | mAP@0.5 | AP_small | AP_medium | AP_large | Recall | Sai lớp | Bỏ sót | Báo nhầm nền | Dự đoán lặp |
|---|---|---|---|---|---|---|---|---|---|
| yolov8s_640 (baseline) | 61.7 | 29.8 | 70.5 | 54.5 | 62.3% | 4.9% | 32.8% | 131 | 26 |
| yolov8s_640_sahi | 78.1 | 37.3 | 70.8 | 53.0 | 75.3% | 2.3% | 22.4% | 91 | 66 |
| yolov8s_1280 | 82.2 | 50.2 | 76.1 | 53.9 | 80.1% | 3.2% | 16.7% | 74 | 17 |
| yolov8s_tiled_640_sahi | **83.9** | **52.3** | 75.2 | 54.9 | 80.1% | 2.1% | 17.8% | 86 | 71 |
| fasterrcnn_baseline | 56.6 | 27.1 | 60.3 | 45.3 | – | – | – | – | – |

Lỗi chính là **bỏ sót**, không phải nhầm lớp hay báo nhầm nền. Ở baseline, 32.8% biển bị bỏ sót, còn chỉ 4.9% bị nhận sai lớp.

## 1. AP_small kém AP_large bao nhiêu?

- **Baseline:** YOLO 640 có AP_small 29.8 so với AP_medium 70.5. Faster R-CNN có 27.1 so với 60.3. Biển nhỏ kém khoảng **33–41 điểm**.
- **Tăng độ phân giải (1280) hoặc cắt ảnh thành ô (tiling)** gần như chỉ cải thiện biển nhỏ: AP_small tăng từ 29.8 lên 50.2–52.3, trong khi AP_medium chỉ tăng khoảng 5 điểm.
- **AP_large không đáng tin.** Val chỉ có 23 biển lớn (≥ 96 px), nên AP_large dao động mạnh, và việc nó thấp hơn AP_medium không có ý nghĩa. **Nên so small với medium.**

Recall theo kích thước biển (sqrt(w*h), px trên ảnh gốc):

| Kích thước (số biển) | <8 (167) | 8–16 (543) | 16–32 (522) | ≥32 (404) |
|---|---|---|---|---|
| yolov8s_640 | 1.2% | 37.8% | 84.3% | 92.1% |
| yolov8s_1280 | 25.1% | 74.6% | 92.1% | 94.6% |
| yolov8s_tiled_640_sahi | 25.1% | 74.6% | 91.6% | 95.3% |

- Ở ảnh 640, biển dưới 8 px gần như không bao giờ được tìm thấy. Biển 8–16 px, nhóm đông nhất, chỉ được tìm thấy 38%.
- Lý do: ảnh gốc rộng 1622 px, khi resize về 640 thì biển 8 px chỉ còn khoảng 3 px.

## 2. Lớp nào AP thấp nhất? "Cấm ngược chiều" có thấp như dự đoán không?

AP@0.5 theo lớp:

| Lớp | % box nhỏ (val) | yolo 640 | yolo 1280 | tiled+SAHI | Faster R-CNN |
|---|---|---|---|---|---|
| Cấm ngược chiều | **92%** | 68.2 | **89.2** | 86.4 | 64.6 |
| Cấm dừng và đỗ | 69% | **77.2** | **92.0** | **93.3** | 69.9 |
| Cấm rẽ | 65% | 52.7 | **75.0** | 80.6 | **37.6** |
| Giới hạn tốc độ | 73% | **49.5** | 77.7 | **77.9** | 51.3 |
| Cấm còn lại | 76% | 55.0 | 75.4 | 80.3 | 44.6 |
| Nguy hiểm | 74% | 74.2 | 87.6 | 88.2 | 71.2 |
| Hiệu lệnh | 85% | 54.9 | 78.6 | 80.7 | 56.9 |

- **Không đúng như dự đoán.** "Cấm ngược chiều" có tỉ lệ box nhỏ cao nhất (92%) nhưng AP thuộc nhóm cao. Ở YOLO 1280, nó đứng thứ hai.
- Lý do có thể: biển nền đỏ đặc với vạch trắng ngang rất khác các lớp còn lại, nên dù chỉ vài pixel vẫn nhận ra được.
- **Lớp thấp nhất luôn thuộc ba lớp biển tròn viền đỏ:** Giới hạn tốc độ (YOLO 640, tiled+SAHI), Cấm rẽ (YOLO 1280, Faster R-CNN). Trên cả 4 mô hình trong bảng, ba lớp này đều nằm trong nhóm 4 lớp có AP thấp nhất.
- **Kết luận:** độ khó phụ thuộc vào việc lớp đó có giống lớp khác không, chứ không chỉ vào kích thước biển.

## 3. Ba lớp biển tròn viền đỏ có nhầm với nhau không?

**Có, và gần như toàn bộ lỗi nhầm lớp nằm trong ba lớp này.** Ở YOLO 640, các cặp nhầm nhiều nhất (nhãn thật → dự đoán):

- Giới hạn tốc độ → Cấm còn lại: 25
- Cấm rẽ → Cấm còn lại: 14
- Cấm còn lại → Giới hạn tốc độ: 13
- Giới hạn tốc độ → Cấm rẽ: 6; Cấm còn lại → Cấm rẽ: 6

Tỉ lệ biển bị nhận sai lớp theo từng lớp:

| Mô hình | Cấm rẽ | Giới hạn tốc độ | Cấm còn lại | Bốn lớp còn lại |
|---|---|---|---|---|
| YOLO 640 | 18.4% | 21.6% | 5.3% | ≤ 1.3% |
| Tiled + SAHI | 11.2% | 7.2% | 2.8% | ≤ 1.3% |

- "Cấm còn lại" là lớp hay bị dự đoán nhầm vào nhất. Đây là lớp gom nhiều loại biển cấm khác nhau.
- Ngay cả người gán nhãn cũng nhầm giữa các lớp này: trong 20 ảnh xem tay ở tuần 1, có 2 box sai lớp đúng kiểu này.

## 4. Biển trong cụm có bị bỏ sót nhiều hơn biển đứng riêng không?

46% biển ở val (753/1636) đứng trong cụm.

| YOLO 640 | <8 | 8–16 | 16–32 | ≥32 | Chung |
|---|---|---|---|---|---|
| Cụm | 1.1% | 35.8% | 84.3% | 95.1% | **58.0%** |
| Đứng riêng | 1.3% | 39.8% | 84.3% | 90.4% | **65.9%** |

- Nếu chỉ nhìn số chung, biển trong cụm có vẻ bị bỏ sót nhiều hơn (recall 58.0% so với 65.9%).
- **Nhưng trong cùng một nhóm kích thước, hai nhóm gần như bằng nhau.** Chênh lệch chung là do biển trong cụm nhỏ hơn: cụm có 49.7% biển dưới 16 px, đứng riêng chỉ có 38.1%.
- Ở YOLO 1280 và tiled+SAHI, recall của biển cụm và biển đứng riêng chỉ chênh 1–2 điểm.
- **Kết luận:** chưa thấy NMS xóa nhầm biển đứng sát nhau. Không cần ưu tiên hướng mở rộng xử lý cụm.

## Các phát hiện khác

- **SAHI tạo nhiều dự đoán lặp.** Mô hình có SAHI có 66–71 dự đoán lặp trên cùng một biển, mô hình không dùng SAHI chỉ có 17–26.
  - Nguyên nhân: các ô cắt chồng lên nhau, rồi bước gộp kết quả không gộp hết.
  - Mỗi box lặp bị tính là một báo nhầm, làm giảm precision.
  - B có thể thử chỉnh tham số gộp của SAHI (`postprocess_type`, `postprocess_match_metric="IOS"`, `postprocess_match_threshold`).
- **Đặc điểm của biển bị bỏ sót.** Xem 40 biển bị mô hình tốt nhất bỏ sót (`bo_sot_mo_hinh_tot_nhat.png`, mô hình này không tìm đúng lớp 201 biển ≥ 8 px):
  - Khoảng 28/40 biển nhỏ hơn hoặc bằng 13 px, nhòe hoặc bạc màu, nhiều biển nằm trên nền trời sáng.
  - Một số biển nằm sát nhau trên thanh ngang, hoặc bị biển khác che một phần.
  - Một số biển lớn bị bỏ sót thực ra là **nhãn sai**. Ví dụ 12405 là biển hướng đi tròn màu xanh (thuộc Hiệu lệnh) nhưng bị ghi "Cấm còn lại".
- Ảnh từng biển nhỏ bị bỏ sót, có cả dự đoán: `reports/missed_yolov8s_640/` và `reports/missed_yolov8s_tiled_640_sahi/`. Thư mục này không commit, chạy lại bằng `visualize.py --only-missed-small`.

## Gợi ý hướng mở rộng tuần 4

1. **Biển nhỏ (8–16 px)** vẫn là nguồn lỗi lớn nhất. Ở mô hình tốt nhất vẫn bỏ sót 23% nhóm này, và 74% nhóm dưới 8 px.
2. **Mất cân bằng lớp (phần của A).** Cấm rẽ và Giới hạn tốc độ vừa ít mẫu vừa hay bị nhầm lớp. Oversampling hai lớp này có thể giảm nhầm sang "Cấm còn lại". Cần theo dõi cả AP và tỉ lệ sai lớp, không chỉ mAP.
3. Không cần ưu tiên xử lý biển đứng cụm (xem câu 4).
