# Phân tích lỗi: fasterrcnn_exp03_fpn_aug

- Nhãn: `data/splits/val.json`, dự đoán: `preds/predictions_val_exp03_fpn_aug.json`
- IoU >= 0.5, ngưỡng score = 0.651 (tự chọn, F1 = 0.762)
- Tổng: 1636 biển, recall 68.4%, sai lớp 3.7%, bỏ sót 27.9%
- Báo nhầm nền: 115 box; dự đoán lặp trên cùng một biển: 5 box

## Recall theo kích thước biển (sqrt(w*h), px ảnh gốc)

| Kích thước | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| <8 | 167 | 15.6 | 1.2 | 83.2 |
| 8-16 | 543 | 55.8 | 4.6 | 39.6 |
| 16-32 | 522 | 82.8 | 3.4 | 13.8 |
| >=32 | 404 | 88.6 | 4.0 | 7.4 |

## Biển đứng cụm và đứng riêng

Biển trong cụm thường nhỏ hơn, nên so cả trong từng nhóm kích thước để không nhầm ảnh hưởng của kích thước với ảnh hưởng của cụm.

| Nhóm | Recall <8 (n) | Recall 8-16 (n) | Recall 16-32 (n) | Recall >=32 (n) | Recall chung (n) |
| --- | --- | --- | --- | --- | --- |
| Cụm | 20.7 (92) | 57.8 (282) | 87.3 (236) | 88.8 (143) | 68.4 (753) |
| Đứng riêng | 9.3 (75) | 53.6 (261) | 79.0 (286) | 88.5 (261) | 68.4 (883) |

## Theo lớp

| Lớp | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| Cấm ngược chiều | 160 | 73.8 | 1.9 | 24.4 |
| Cấm dừng và đỗ | 408 | 81.4 | 0.5 | 18.1 |
| Cấm rẽ | 98 | 48.0 | 17.3 | 34.7 |
| Giới hạn tốc độ | 153 | 54.2 | 16.3 | 29.4 |
| Cấm còn lại | 395 | 57.2 | 2.8 | 40.0 |
| Nguy hiểm | 272 | 74.6 | 0.0 | 25.4 |
| Hiệu lệnh | 150 | 73.3 | 2.0 | 24.7 |

## Cặp nhầm lẫn nhiều nhất (nhãn thật -> dự đoán)

- Giới hạn tốc độ -> Cấm còn lại: 23
- Cấm rẽ -> Cấm còn lại: 14
- Cấm còn lại -> Cấm rẽ: 7
- Cấm rẽ -> Giới hạn tốc độ: 3
- Cấm ngược chiều -> Cấm dừng và đỗ: 2
- Cấm dừng và đỗ -> Hiệu lệnh: 2

![confusion](confusion.png)
