# Phân tích lỗi: fasterrcnn_exp03_test

- Nhãn: `data/splits/test.json`, dự đoán: `preds/fasterrcnn_exp03_test.json`
- IoU >= 0.5, ngưỡng score = 0.679 (tự chọn, F1 = 0.759)
- Tổng: 1748 biển, recall 68.4%, sai lớp 2.6%, bỏ sót 29.0%
- Báo nhầm nền: 156 box; dự đoán lặp trên cùng một biển: 7 box

## Recall theo kích thước biển (sqrt(w*h), px ảnh gốc)

| Kích thước | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| <8 | 367 | 21.0 | 0.8 | 78.2 |
| 8-16 | 597 | 72.0 | 2.0 | 26.0 |
| 16-32 | 454 | 83.7 | 5.1 | 11.2 |
| >=32 | 330 | 93.6 | 2.1 | 4.2 |

## Biển đứng cụm và đứng riêng

Biển trong cụm thường nhỏ hơn, nên so cả trong từng nhóm kích thước để không nhầm ảnh hưởng của kích thước với ảnh hưởng của cụm.

| Nhóm | Recall <8 (n) | Recall 8-16 (n) | Recall 16-32 (n) | Recall >=32 (n) | Recall chung (n) |
| --- | --- | --- | --- | --- | --- |
| Cụm | 26.0 (227) | 75.3 (279) | 85.0 (147) | 90.5 (95) | 64.2 (748) |
| Đứng riêng | 12.9 (140) | 69.2 (318) | 83.1 (307) | 94.9 (235) | 71.6 (1000) |

## Theo lớp

| Lớp | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| Cấm ngược chiều | 253 | 75.1 | 0.8 | 24.1 |
| Cấm dừng và đỗ | 209 | 78.9 | 0.0 | 21.1 |
| Cấm rẽ | 81 | 32.1 | 28.4 | 39.5 |
| Giới hạn tốc độ | 79 | 45.6 | 10.1 | 44.3 |
| Cấm còn lại | 176 | 55.7 | 6.2 | 38.1 |
| Nguy hiểm | 690 | 75.1 | 0.0 | 24.9 |
| Hiệu lệnh | 260 | 62.7 | 0.4 | 36.9 |

## Cặp nhầm lẫn nhiều nhất (nhãn thật -> dự đoán)

- Cấm rẽ -> Cấm còn lại: 18
- Giới hạn tốc độ -> Cấm còn lại: 8
- Cấm còn lại -> Giới hạn tốc độ: 6
- Cấm rẽ -> Giới hạn tốc độ: 3
- Cấm ngược chiều -> Cấm dừng và đỗ: 2
- Cấm còn lại -> Cấm dừng và đỗ: 2

![confusion](confusion.png)
