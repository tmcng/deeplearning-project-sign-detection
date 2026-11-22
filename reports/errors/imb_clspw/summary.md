# Phân tích lỗi: imb_clspw

- Nhãn: `data/splits/val.json`, dự đoán: `preds/imb_clspw_val.json`
- IoU >= 0.5, ngưỡng score = 0.328 (tự chọn, F1 = 0.846)
- Tổng: 1636 biển, recall 79.3%, sai lớp 3.3%, bỏ sót 17.4%
- Báo nhầm nền: 68 box; dự đoán lặp trên cùng một biển: 13 box

## Recall theo kích thước biển (sqrt(w*h), px ảnh gốc)

| Kích thước | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| <8 | 167 | 24.6 | 1.2 | 74.3 |
| 8-16 | 543 | 73.5 | 4.2 | 22.3 |
| 16-32 | 522 | 91.4 | 3.4 | 5.2 |
| >=32 | 404 | 94.3 | 2.7 | 3.0 |

## Biển đứng cụm và đứng riêng

Biển trong cụm thường nhỏ hơn, nên so cả trong từng nhóm kích thước để không nhầm ảnh hưởng của kích thước với ảnh hưởng của cụm.

| Nhóm | Recall <8 (n) | Recall 8-16 (n) | Recall 16-32 (n) | Recall >=32 (n) | Recall chung (n) |
| --- | --- | --- | --- | --- | --- |
| Cụm | 30.4 (92) | 74.1 (282) | 93.2 (236) | 95.1 (143) | 78.8 (753) |
| Đứng riêng | 17.3 (75) | 72.8 (261) | 89.9 (286) | 93.9 (261) | 79.8 (883) |

## Theo lớp

| Lớp | Số biển | Recall % | Sai lớp % | Bỏ sót % |
| --- | --- | --- | --- | --- |
| Cấm ngược chiều | 160 | 84.4 | 0.0 | 15.6 |
| Cấm dừng và đỗ | 408 | 88.2 | 0.2 | 11.5 |
| Cấm rẽ | 98 | 74.5 | 10.2 | 15.3 |
| Giới hạn tốc độ | 153 | 73.9 | 11.8 | 14.4 |
| Cấm còn lại | 395 | 70.1 | 5.8 | 24.1 |
| Nguy hiểm | 272 | 83.1 | 0.0 | 16.9 |
| Hiệu lệnh | 150 | 76.0 | 1.3 | 22.7 |

## Cặp nhầm lẫn nhiều nhất (nhãn thật -> dự đoán)

- Cấm còn lại -> Giới hạn tốc độ: 14
- Giới hạn tốc độ -> Cấm còn lại: 12
- Cấm rẽ -> Cấm còn lại: 8
- Cấm còn lại -> Cấm rẽ: 8
- Giới hạn tốc độ -> Cấm rẽ: 6
- Cấm rẽ -> Giới hạn tốc độ: 2

![confusion](confusion.png)
