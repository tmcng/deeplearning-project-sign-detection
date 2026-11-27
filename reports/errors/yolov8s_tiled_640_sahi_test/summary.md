# Phân tích lỗi: yolov8s_tiled_640_sahi_test

- Nhãn: `data/splits/test.json`, dự đoán: `preds/yolov8s_tiled_640_sahi_test.json`
- IoU >= 0.5, ngưỡng score = 0.530 (tự chọn, F1 = 0.823)
- Tổng: 1748 biển, recall 79.2%, sai lớp 1.7%, bỏ sót 19.1%
- Báo nhầm nền: 123 box; dự đoán lặp trên cùng một biển: 78 box

## Recall theo kích thước biển (sqrt(w*h), px ảnh gốc)

| Kích thước | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| <8 | 367 | 38.7 | 0.8 | 60.5 |
| 8-16 | 597 | 84.1 | 2.0 | 13.9 |
| 16-32 | 454 | 93.0 | 2.4 | 4.6 |
| >=32 | 330 | 96.4 | 1.2 | 2.4 |

## Biển đứng cụm và đứng riêng

Biển trong cụm thường nhỏ hơn, nên so cả trong từng nhóm kích thước để không nhầm ảnh hưởng của kích thước với ảnh hưởng của cụm.

| Nhóm | Recall <8 (n) | Recall 8-16 (n) | Recall 16-32 (n) | Recall >=32 (n) | Recall chung (n) |
| --- | --- | --- | --- | --- | --- |
| Cụm | 45.4 (227) | 88.2 (279) | 93.2 (147) | 96.8 (95) | 77.3 (748) |
| Đứng riêng | 27.9 (140) | 80.5 (318) | 92.8 (307) | 96.2 (235) | 80.6 (1000) |

## Theo lớp

| Lớp | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| Cấm ngược chiều | 253 | 81.8 | 0.0 | 18.2 |
| Cấm dừng và đỗ | 209 | 86.6 | 0.0 | 13.4 |
| Cấm rẽ | 81 | 59.3 | 11.1 | 29.6 |
| Giới hạn tốc độ | 79 | 65.8 | 8.9 | 25.3 |
| Cấm còn lại | 176 | 68.8 | 7.4 | 23.9 |
| Nguy hiểm | 690 | 84.2 | 0.0 | 15.8 |
| Hiệu lệnh | 260 | 74.6 | 0.4 | 25.0 |

## Cặp nhầm lẫn nhiều nhất (nhãn thật -> dự đoán)

- Cấm còn lại -> Giới hạn tốc độ: 10
- Giới hạn tốc độ -> Cấm còn lại: 6
- Cấm rẽ -> Giới hạn tốc độ: 5
- Cấm rẽ -> Cấm còn lại: 4
- Cấm còn lại -> Cấm dừng và đỗ: 2
- Giới hạn tốc độ -> Cấm rẽ: 1

![confusion](confusion.png)
