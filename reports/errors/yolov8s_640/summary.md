# Phân tích lỗi: yolov8s_640

- Nhãn: `data/splits/val.json`, dự đoán: `preds/yolov8s_640_val.json`
- IoU >= 0.5, ngưỡng score = 0.220 (tự chọn, F1 = 0.705)
- Tổng: 1636 biển, recall 62.3%, sai lớp 4.9%, bỏ sót 32.8%
- Báo nhầm nền: 131 box; dự đoán lặp trên cùng một biển: 26 box

## Recall theo kích thước biển (sqrt(w*h), px ảnh gốc)

| Kích thước | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| <8 | 167 | 1.2 | 0.6 | 98.2 |
| 8-16 | 543 | 37.8 | 4.2 | 58.0 |
| 16-32 | 522 | 84.3 | 7.3 | 8.4 |
| >=32 | 404 | 92.1 | 4.5 | 3.5 |

## Biển đứng cụm và đứng riêng

Biển trong cụm thường nhỏ hơn, nên so cả trong từng nhóm kích thước để không nhầm ảnh hưởng của kích thước với ảnh hưởng của cụm.

| Nhóm | Recall <8 (n) | Recall 8-16 (n) | Recall 16-32 (n) | Recall >=32 (n) | Recall chung (n) |
| --- | --- | --- | --- | --- | --- |
| Cụm | 1.1 (92) | 35.8 (282) | 84.3 (236) | 95.1 (143) | 58.0 (753) |
| Đứng riêng | 1.3 (75) | 39.8 (261) | 84.3 (286) | 90.4 (261) | 65.9 (883) |

## Theo lớp

| Lớp | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| Cấm ngược chiều | 160 | 63.1 | 0.6 | 36.2 |
| Cấm dừng và đỗ | 408 | 74.0 | 1.2 | 24.8 |
| Cấm rẽ | 98 | 52.0 | 18.4 | 29.6 |
| Giới hạn tốc độ | 153 | 53.6 | 21.6 | 24.8 |
| Cấm còn lại | 395 | 53.9 | 5.3 | 40.8 |
| Nguy hiểm | 272 | 68.8 | 0.0 | 31.2 |
| Hiệu lệnh | 150 | 55.3 | 1.3 | 43.3 |

## Cặp nhầm lẫn nhiều nhất (nhãn thật -> dự đoán)

- Giới hạn tốc độ -> Cấm còn lại: 25
- Cấm rẽ -> Cấm còn lại: 14
- Cấm còn lại -> Giới hạn tốc độ: 13
- Giới hạn tốc độ -> Cấm rẽ: 6
- Cấm còn lại -> Cấm rẽ: 6
- Cấm rẽ -> Giới hạn tốc độ: 3

![confusion](confusion.png)
