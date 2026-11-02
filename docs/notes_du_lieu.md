# Ghi chú dữ liệu (Việc 3 – xem ảnh)

## Tóm tắt (dùng cho milestone)

- Biển quá nhỏ: **9 / 20 ảnh**
- Biển bị che hoặc đứng cụm: **3 / 20 ảnh**. Cả 3 đều là đứng cụm (12442, 10183, 930), không ảnh nào bị che.
- Lỗi nhãn: **8 chỗ**
  - 2 chỗ sai lớp: ở ảnh 12442.png, biển "30" bị ghi là Cấm rẽ, biển "10" bị ghi là Cấm còn lại.
  - 6 chỗ một biển bị gán 2–3 box trùng nhau: 10160, 10183 (2 biển), 10697, 11803, 9458.
  - Không gặp chỗ thiếu nhãn hay box lệch.
  - Trên toàn bộ dữ liệu, box trùng chiếm 15.5% box train và 22.6% box test. Cần gộp box trùng trước khi đánh giá, xem chi tiết bên dưới.
- 3 ảnh tiêu biểu: **5070.png** (biển xa), **12442.png** (biển cụm và sai lớp), **10183.png** (biển cụm và box trùng). Ảnh đã lưu ở `reports/vis_baocao/`.

Phần dưới là chi tiết, không bắt buộc.

Mẫu: 20 ảnh ngẫu nhiên từ `data/splits/train.json` (seed 0), 45 box.
Lệnh: `python scripts/visualize.py --gt data/splits/train.json --images za_traffic_2020/traffic_train/images --n 20 --out reports/vis_gt`

Quy ước:
- `xa` = cạnh box ≤ 13 px trên ảnh gốc 1622×626 (≈ 5 px sau khi resize về 640).
- Từ khóa bổ sung (không có trong danh sách gốc, nhưng gặp nhiều nên tách riêng để đếm được):
  - `trung_box`: cùng một biển được gán 2–3 box gần như trùng nhau (IoU ≥ 0.45, cùng lớp).
  - `nghieng`: biển quay gần vuông góc với camera, chỉ thấy cạnh, không đọc được nội dung.

## Bảng chi tiết

| Ảnh | Lớp | Vấn đề | Ghi chú |
|---|---|---|---|
| 4049.png | Cấm dừng và đỗ | xa, mo | 7 px, chỉ thấy một chấm xanh trên đỉnh cột |
| 5070.png | Cấm dừng và đỗ | xa | 6 px, cuối đường. Cùng ảnh có biển Cấm rẽ 34 px rất rõ |
| 5271.png | Nguy hiểm | xa | 8 px, chỉ còn một đốm vàng |
| 7412.png | Cấm ngược chiều | xa | 7 px, sát điểm tụ của đường |
| 748.png | Cấm ngược chiều | xa | 11 px, chỉ còn chấm đỏ có vạch trắng |
| 10697.png | Cấm ngược chiều | xa, trung_box | 9 px, có 2 box cho một biển |
| 6876.png | Nguy hiểm | xa | 12 px, đường quê, nền đất và cây |
| 3154.png | Nguy hiểm | xa | 13 px, nền rất rối (bảng quảng cáo xanh, vàng ngay phía sau) |
| 12442.png | Giới hạn tốc độ ×3 | xa, cum | 3 biển 10–11 px đứng ngang nhau trên giá long môn, khe hở chỉ khoảng 4 px |
| 12442.png | Cấm dừng và đỗ, Cấm rẽ, Nguy hiểm | cum | 3 biển xếp dọc trên một cột (x≈332) |
| 12442.png | Cấm rẽ (box 3) | nhan_sai | Thực ra là biển giới hạn tốc độ "30" (xem `reports/vis_baocao/12442_nhan_sai.png`) |
| 12442.png | Cấm còn lại (box 11) | nhan_sai | Biển "10" trong vòng đỏ, không thấy chữ "t", nhiều khả năng là Giới hạn tốc độ |
| 12442.png | Cấm ngược chiều + Hiệu lệnh | cum | 2 biển xếp dọc, 14 px |
| 12442.png | Cấm dừng và đỗ + Giới hạn tốc độ | xa, cum | 2 biển xếp dọc, 11 px |
| 10183.png | Cấm còn lại + Cấm dừng và đỗ | cum, trung_box | 2 biển xếp dọc, sát nhau. Mỗi biển bị gán 3 box (ảnh có 6 box cho 2 biển) |
| 930.png | Giới hạn tốc độ + Cấm còn lại | cum | 2 biển đứng ngang, chạm cạnh nhau |
| 10160.png | Nguy hiểm | mo, trung_box | Ảnh nhòe, 3 box cho một biển |
| 10160.png | Cấm còn lại | mo | Nhòe, biển trắng gần như hòa vào nền |
| 6088.png | Cấm ngược chiều | mo | Nằm sát mép trái ảnh, nhòe do ống kính góc rộng |
| 9458.png | Cấm dừng và đỗ | nguoc_sang, trung_box | Biển nằm trong bóng cây nên tối, độ tương phản thấp. Có 2 box |
| 11803.png | Cấm dừng và đỗ | trung_box | 3 box cho một biển |
| 11803.png | Cấm dừng và đỗ (box 1) | nghieng | 11×30 px, chỉ thấy cạnh biển, mắt người không đọc được loại biển |

Có nhưng không phải lỗi nhãn (dễ làm mô hình báo nhầm):
- 491.png: logo đỏ trắng hình tròn trên cột, trông giống biển cấm.
- 3154.png: mặt sau của một biển tròn (xám).
- 10160, 11803, 748: biển chỉ dẫn hoặc thông tin màu xanh, không thuộc 7 lớp.

## Đếm (trên 20 ảnh)

| Từ khóa | Số ảnh | Ảnh |
|---|---|---|
| xa | **9/20** | 4049, 5070, 5271, 7412, 748, 10697, 6876, 3154, 12442 |
| trung_box | **5/20** | 10160, 10183, 10697, 11803, 9458 |
| cum | 3/20 | 12442, 10183, 930 |
| mo | 3/20 | 4049, 10160, 6088 |
| nhan_sai | 1/20 (2 box) | 12442 |
| nguoc_sang | 1/20 | 9458 |
| nghieng | 1/20 | 11803 |
| che | 0/20 | |
| thieu_nhan | 0/20 | |
| box_lech | 0/20 | Ngoài các box trùng, các box đều ôm khá khít |

## Kiểm tra trên toàn bộ split (bằng script, không chỉ 20 ảnh)

Box trùng (cùng một biển gán nhiều box, IoU ≥ 0.7):

| Split | Ảnh có box trùng | Box thừa / tổng box |
|---|---|---|
| train | 447/2933 (15.2%) | 1063/6844 (**15.5%**) |
| val | 80/696 (11.5%) | 188/1836 (10.2%) |
| test | 197/871 (22.6%) | 525/2320 (**22.6%**) |

- 99% các cặp trùng là cùng lớp, tức là lặp lại chứ không phải hai người gán khác nhau.
- Các cặp có IoU 0.45–0.7 (125 cặp ở train) cũng là box trùng, chỉ lệch 1–2 px trên biển nhỏ (124/125 cùng lớp).
- ID ảnh có box trùng thường đi liền nhau (ví dụ 5954–5963), có vẻ là lỗi theo đợt gán nhãn.

Biển đứng cụm (2 box gần như chạm nhau, khe hở < nửa cạnh nhỏ hơn):

| Split | Ảnh có cụm | Cặp sát nhau | Cùng lớp |
|---|---|---|---|
| train | 802/2933 (27.3%) | 2059 | 772 (≈ 650 nếu bỏ các cặp thực ra là box trùng) |
| val | 270/696 (38.8%) | 527 | 78 |
| test | 333/871 (38.2%) | 988 | 331 |

- Cặp hay đứng cụm nhất: **Cấm ngược chiều + Hiệu lệnh** (khác lớp), sau đó là Giới hạn tốc độ + Giới hạn tốc độ, Nguy hiểm + Nguy hiểm, Cấm còn lại + Cấm dừng và đỗ.
- Hai biển thật đứng cạnh nhau gần như **không chồng lên nhau** (IoU ≈ 0). Mọi cặp có IoU ≥ 0.45 đều là box trùng.

### Lưu ý cho B, C (NMS)
- NMS chỉ xóa nhầm khi box **dự đoán** của hai biển kề nhau chồng lên nhau quá ngưỡng. Box nhãn thật kề nhau nhưng không chồng, nên nếu mô hình định vị tốt thì rủi ro thấp.
- Nên giữ NMS **theo từng lớp** (mặc định của YOLO, `agnostic_nms=False`). Cặp hay gặp nhất là Cấm ngược chiều + Hiệu lệnh khác lớp nên sẽ không bị xóa lẫn nhau. Nếu bật agnostic thì cặp này có nguy cơ.
- Rủi ro thật nằm ở cụm **cùng lớp và rất nhỏ**, ví dụ 3 biển tốc độ 11 px cách nhau 4 px trên giá long môn (12442). Lệch 2–3 px là box dự đoán đã chồng nhau. **Không nên hạ ngưỡng IoU của NMS xuống thấp** (giữ khoảng 0.6–0.7). Khi phân tích lỗi, nên đếm riêng số biển bị bỏ sót nằm trong cụm.
- **Quan trọng hơn NMS là box trùng trong nhãn test:** mỗi biển chỉ được một dự đoán khớp, nên các box thừa luôn bị tính là bỏ sót. Recall tối đa trên test chỉ khoảng 1795/2320 ≈ **77%** dù mô hình hoàn hảo. Cần gộp box trùng (cùng lớp, IoU ≥ 0.45) trong cả train/val/test trước khi train và đánh giá, và ghi rõ việc này trong báo cáo.

## Trả lời 4 câu

**1. Vấn đề hay gặp nhất?**
`xa` là nhiều nhất (9/20 ảnh). Khớp với EDA: trung vị cạnh box là 16 px, sau khi resize về 640 thì 70% box nhỏ hơn 10 px. `cum` chỉ 3/20 trong mẫu, nhưng đếm trên toàn bộ dữ liệu thì 27% ảnh train và 38% ảnh val/test có biển đứng cụm. `che` không gặp lần nào trong 20 ảnh.

**2. Nhãn có nhiều lỗi không?**
Có, nhưng chủ yếu là một kiểu lỗi có hệ thống: **box trùng** (15.5% box train, 22.6% box test). Đây là giới hạn của dữ liệu, phải ghi vào Limitations và phải xử lý (gộp box) trước khi đánh giá. Lỗi sai lớp ít (2/45 box trong mẫu, đều ở ảnh 12442). Không thấy thiếu nhãn hay box lệch.

**3. Lớp dễ nhầm?**
- **Giới hạn tốc độ ↔ Cấm còn lại ↔ Cấm rẽ:** cả ba đều là biển tròn nền trắng viền đỏ, chỉ khác nội dung ở giữa (số, mũi tên, hình xe). Khi biển nhỏ hơn 15 px thì gần như không phân biệt được. Chính người gán nhãn cũng nhầm (12442: "30" ghi thành Cấm rẽ, "10" ghi thành Cấm còn lại).
- **Cấm dừng và đỗ ↔ Hiệu lệnh:** đều có nền xanh. Khi chỉ còn vài pixel thì chỉ thấy một chấm xanh.
- Cấm ngược chiều (đỏ, vạch trắng) và Nguy hiểm (tam giác vàng) dễ nhận hơn nhờ màu và hình dạng.
- Vật dễ gây báo nhầm: logo hình tròn đỏ trắng (491), mặt sau biển (3154), biển chỉ dẫn xanh.

**4. Ảnh tiêu biểu** (đã lưu ở `reports/vis_baocao/`):
- **Biển xa:** `5070.jpg` + `5070_zoom1.jpg`. Cùng ảnh có biển 34 px rõ ràng và biển 6 px chỉ là chấm.
- **Biển cụm + nhãn sai:** `12442.jpg`, `12442_zoom2.jpg` (3 biển xếp dọc), `12442_zoom6.jpg` (3 biển tốc độ ngang nhau), `12442_nhan_sai.png`.
- **Cụm + box trùng:** `10183.jpg` + `10183_zoom0.jpg`, 6 box cho 2 biển.
- **Box trùng + nghiêng:** `11803.jpg` + `11803_zoom0.jpg`.
- **Biển bị che:** không có trong mẫu 20 ảnh. Cần xem thêm một mẫu khác (ví dụ `--seed 1`) để lấy ảnh minh họa.
