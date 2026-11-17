# Phân tích lỗi: yolov8s_1280

- Nhãn: `data/splits/val.json`, dự đoán: `preds/yolov8s_1280_val.json`
- IoU >= 0.5, ngưỡng score = 0.313 (tự chọn, F1 = 0.848)
- Tổng: 1636 biển, recall 80.1%, sai lớp 3.2%, bỏ sót 16.7%
- Báo nhầm nền: 74 box; dự đoán lặp trên cùng một biển: 17 box

## Recall theo kích thước biển (sqrt(w*h), px ảnh gốc)

| Kích thước | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| <8 | 167 | 25.1 | 1.2 | 73.7 |
| 8-16 | 543 | 74.6 | 4.8 | 20.6 |
| 16-32 | 522 | 92.1 | 2.5 | 5.4 |
| >=32 | 404 | 94.6 | 2.7 | 2.7 |

## Biển đứng cụm và đứng riêng

Biển trong cụm thường nhỏ hơn, nên so cả trong từng nhóm kích thước để không nhầm ảnh hưởng của kích thước với ảnh hưởng của cụm.

| Nhóm | Recall <8 (n) | Recall 8-16 (n) | Recall 16-32 (n) | Recall >=32 (n) | Recall chung (n) |
| --- | --- | --- | --- | --- | --- |
| Cụm | 31.5 (92) | 75.5 (282) | 93.2 (236) | 95.1 (143) | 79.4 (753) |
| Đứng riêng | 17.3 (75) | 73.6 (261) | 91.3 (286) | 94.3 (261) | 80.6 (883) |

## Theo lớp

| Lớp | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| Cấm ngược chiều | 160 | 85.6 | 0.0 | 14.4 |
| Cấm dừng và đỗ | 408 | 88.7 | 0.0 | 11.3 |
| Cấm rẽ | 98 | 71.4 | 14.3 | 14.3 |
| Giới hạn tốc độ | 153 | 76.5 | 11.1 | 12.4 |
| Cấm còn lại | 395 | 71.6 | 4.6 | 23.8 |
| Nguy hiểm | 272 | 84.2 | 0.0 | 15.8 |
| Hiệu lệnh | 150 | 74.7 | 2.0 | 23.3 |

## Cặp nhầm lẫn nhiều nhất (nhãn thật -> dự đoán)

- Giới hạn tốc độ -> Cấm còn lại: 14
- Cấm còn lại -> Giới hạn tốc độ: 12
- Cấm rẽ -> Cấm còn lại: 8
- Cấm rẽ -> Giới hạn tốc độ: 6
- Cấm còn lại -> Cấm rẽ: 4
- Giới hạn tốc độ -> Cấm rẽ: 3

![confusion](confusion.png)
