# Phân tích lỗi: yolov8s_tiled_640_sahi

- Nhãn: `data/splits/val.json`, dự đoán: `preds/yolov8s_tiled_640_sahi_val.json`
- IoU >= 0.5, ngưỡng score = 0.542 (tự chọn, F1 = 0.835)
- Tổng: 1636 biển, recall 80.1%, sai lớp 2.1%, bỏ sót 17.8%
- Báo nhầm nền: 86 box; dự đoán lặp trên cùng một biển: 71 box

## Recall theo kích thước biển (sqrt(w*h), px ảnh gốc)

| Kích thước | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| <8 | 167 | 25.1 | 1.2 | 73.7 |
| 8-16 | 543 | 74.6 | 2.6 | 22.8 |
| 16-32 | 522 | 91.6 | 2.3 | 6.1 |
| >=32 | 404 | 95.3 | 1.7 | 3.0 |

## Biển đứng cụm và đứng riêng

Biển trong cụm thường nhỏ hơn, nên so cả trong từng nhóm kích thước để không nhầm ảnh hưởng của kích thước với ảnh hưởng của cụm.

| Nhóm | Recall <8 (n) | Recall 8-16 (n) | Recall 16-32 (n) | Recall >=32 (n) | Recall chung (n) |
| --- | --- | --- | --- | --- | --- |
| Cụm | 31.5 (92) | 75.2 (282) | 92.4 (236) | 95.1 (143) | 79.0 (753) |
| Đứng riêng | 17.3 (75) | 73.9 (261) | 90.9 (286) | 95.4 (261) | 81.0 (883) |

## Theo lớp

| Lớp | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| Cấm ngược chiều | 160 | 81.9 | 0.0 | 18.1 |
| Cấm dừng và đỗ | 408 | 90.0 | 0.0 | 10.0 |
| Cấm rẽ | 98 | 72.4 | 11.2 | 16.3 |
| Giới hạn tốc độ | 153 | 71.9 | 7.2 | 20.9 |
| Cấm còn lại | 395 | 71.6 | 2.8 | 25.6 |
| Nguy hiểm | 272 | 85.3 | 0.0 | 14.7 |
| Hiệu lệnh | 150 | 77.3 | 1.3 | 21.3 |

## Cặp nhầm lẫn nhiều nhất (nhãn thật -> dự đoán)

- Cấm rẽ -> Cấm còn lại: 9
- Giới hạn tốc độ -> Cấm còn lại: 9
- Cấm còn lại -> Giới hạn tốc độ: 8
- Cấm rẽ -> Giới hạn tốc độ: 2
- Giới hạn tốc độ -> Cấm rẽ: 2
- Cấm còn lại -> Cấm rẽ: 2

![confusion](confusion.png)
