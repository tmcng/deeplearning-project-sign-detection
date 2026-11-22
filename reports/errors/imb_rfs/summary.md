# Phân tích lỗi: imb_rfs

- Nhãn: `data/splits/val.json`, dự đoán: `preds/imb_rfs_val.json`
- IoU >= 0.5, ngưỡng score = 0.378 (tự chọn, F1 = 0.835)
- Tổng: 1636 biển, recall 78.4%, sai lớp 3.9%, bỏ sót 17.7%
- Báo nhầm nền: 77 box; dự đoán lặp trên cùng một biển: 12 box

## Recall theo kích thước biển (sqrt(w*h), px ảnh gốc)

| Kích thước | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| <8 | 167 | 27.5 | 2.4 | 70.1 |
| 8-16 | 543 | 71.1 | 5.0 | 23.9 |
| 16-32 | 522 | 90.6 | 4.0 | 5.4 |
| >=32 | 404 | 93.6 | 3.0 | 3.5 |

## Biển đứng cụm và đứng riêng

Biển trong cụm thường nhỏ hơn, nên so cả trong từng nhóm kích thước để không nhầm ảnh hưởng của kích thước với ảnh hưởng của cụm.

| Nhóm | Recall <8 (n) | Recall 8-16 (n) | Recall 16-32 (n) | Recall >=32 (n) | Recall chung (n) |
| --- | --- | --- | --- | --- | --- |
| Cụm | 33.7 (92) | 69.9 (282) | 91.5 (236) | 94.4 (143) | 76.9 (753) |
| Đứng riêng | 20.0 (75) | 72.4 (261) | 89.9 (286) | 93.1 (261) | 79.7 (883) |

## Theo lớp

| Lớp | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| Cấm ngược chiều | 160 | 83.8 | 0.0 | 16.2 |
| Cấm dừng và đỗ | 408 | 87.5 | 0.5 | 12.0 |
| Cấm rẽ | 98 | 68.4 | 19.4 | 12.2 |
| Giới hạn tốc độ | 153 | 72.5 | 14.4 | 13.1 |
| Cấm còn lại | 395 | 70.4 | 4.8 | 24.8 |
| Nguy hiểm | 272 | 80.9 | 0.0 | 19.1 |
| Hiệu lệnh | 150 | 77.3 | 1.3 | 21.3 |

## Cặp nhầm lẫn nhiều nhất (nhãn thật -> dự đoán)

- Giới hạn tốc độ -> Cấm còn lại: 18
- Cấm rẽ -> Cấm còn lại: 15
- Cấm còn lại -> Giới hạn tốc độ: 12
- Giới hạn tốc độ -> Cấm rẽ: 4
- Cấm còn lại -> Cấm rẽ: 4
- Cấm rẽ -> Giới hạn tốc độ: 3

![confusion](confusion.png)
