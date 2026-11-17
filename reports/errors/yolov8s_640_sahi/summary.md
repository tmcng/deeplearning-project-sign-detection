# Phân tích lỗi: yolov8s_640_sahi

- Nhãn: `data/splits/val.json`, dự đoán: `preds/yolov8s_640_sahi_val.json`
- IoU >= 0.5, ngưỡng score = 0.616 (tự chọn, F1 = 0.804)
- Tổng: 1636 biển, recall 75.3%, sai lớp 2.3%, bỏ sót 22.4%
- Báo nhầm nền: 91 box; dự đoán lặp trên cùng một biển: 66 box

## Recall theo kích thước biển (sqrt(w*h), px ảnh gốc)

| Kích thước | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| <8 | 167 | 22.2 | 0.0 | 77.8 |
| 8-16 | 543 | 66.9 | 2.4 | 30.8 |
| 16-32 | 522 | 88.7 | 2.1 | 9.2 |
| >=32 | 404 | 91.3 | 3.5 | 5.2 |

## Biển đứng cụm và đứng riêng

Biển trong cụm thường nhỏ hơn, nên so cả trong từng nhóm kích thước để không nhầm ảnh hưởng của kích thước với ảnh hưởng của cụm.

| Nhóm | Recall <8 (n) | Recall 8-16 (n) | Recall 16-32 (n) | Recall >=32 (n) | Recall chung (n) |
| --- | --- | --- | --- | --- | --- |
| Cụm | 22.8 (92) | 70.6 (282) | 89.4 (236) | 93.0 (143) | 74.9 (753) |
| Đứng riêng | 21.3 (75) | 62.8 (261) | 88.1 (286) | 90.4 (261) | 75.7 (883) |

## Theo lớp

| Lớp | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| Cấm ngược chiều | 160 | 77.5 | 0.0 | 22.5 |
| Cấm dừng và đỗ | 408 | 85.0 | 0.2 | 14.7 |
| Cấm rẽ | 98 | 62.2 | 11.2 | 26.5 |
| Giới hạn tốc độ | 153 | 71.9 | 5.9 | 22.2 |
| Cấm còn lại | 395 | 65.3 | 3.8 | 30.9 |
| Nguy hiểm | 272 | 82.7 | 0.0 | 17.3 |
| Hiệu lệnh | 150 | 71.3 | 1.3 | 27.3 |

## Cặp nhầm lẫn nhiều nhất (nhãn thật -> dự đoán)

- Cấm còn lại -> Giới hạn tốc độ: 11
- Cấm rẽ -> Cấm còn lại: 8
- Giới hạn tốc độ -> Cấm còn lại: 8
- Cấm rẽ -> Giới hạn tốc độ: 3
- Cấm còn lại -> Cấm rẽ: 2
- Hiệu lệnh -> Nguy hiểm: 2

![confusion](confusion.png)
