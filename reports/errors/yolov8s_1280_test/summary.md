# Phân tích lỗi: yolov8s_1280_test

- Nhãn: `data/splits/test.json`, dự đoán: `preds/yolov8s_1280_test.json`
- IoU >= 0.5, ngưỡng score = 0.374 (tự chọn, F1 = 0.831)
- Tổng: 1748 biển, recall 77.1%, sai lớp 1.9%, bỏ sót 21.1%
- Báo nhầm nền: 97 box; dự đoán lặp trên cùng một biển: 17 box

## Recall theo kích thước biển (sqrt(w*h), px ảnh gốc)

| Kích thước | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| <8 | 367 | 33.0 | 0.3 | 66.8 |
| 8-16 | 597 | 81.6 | 3.0 | 15.4 |
| 16-32 | 454 | 92.7 | 2.2 | 5.1 |
| >=32 | 330 | 96.4 | 1.2 | 2.4 |

## Biển đứng cụm và đứng riêng

Biển trong cụm thường nhỏ hơn, nên so cả trong từng nhóm kích thước để không nhầm ảnh hưởng của kích thước với ảnh hưởng của cụm.

| Nhóm | Recall <8 (n) | Recall 8-16 (n) | Recall 16-32 (n) | Recall >=32 (n) | Recall chung (n) |
| --- | --- | --- | --- | --- | --- |
| Cụm | 37.0 (227) | 86.0 (279) | 92.5 (147) | 96.8 (95) | 73.8 (748) |
| Đứng riêng | 26.4 (140) | 77.7 (318) | 92.8 (307) | 96.2 (235) | 79.5 (1000) |

## Theo lớp

| Lớp | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| Cấm ngược chiều | 253 | 81.0 | 0.0 | 19.0 |
| Cấm dừng và đỗ | 209 | 85.6 | 0.0 | 14.4 |
| Cấm rẽ | 81 | 64.2 | 9.9 | 25.9 |
| Giới hạn tốc độ | 79 | 63.3 | 11.4 | 25.3 |
| Cấm còn lại | 176 | 64.8 | 8.5 | 26.7 |
| Nguy hiểm | 690 | 82.3 | 0.0 | 17.7 |
| Hiệu lệnh | 260 | 68.8 | 0.4 | 30.8 |

## Cặp nhầm lẫn nhiều nhất (nhãn thật -> dự đoán)

- Cấm còn lại -> Giới hạn tốc độ: 10
- Giới hạn tốc độ -> Cấm còn lại: 8
- Cấm rẽ -> Giới hạn tốc độ: 4
- Cấm rẽ -> Cấm còn lại: 4
- Cấm còn lại -> Cấm rẽ: 3
- Cấm còn lại -> Cấm dừng và đỗ: 2

![confusion](confusion.png)
