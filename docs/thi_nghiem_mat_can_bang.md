# Thí nghiệm mất cân bằng lớp (tuần 4)

## Câu hỏi

Cân bằng lại các lớp khi train có giúp các lớp hiếm (Cấm rẽ, Giới hạn tốc độ, Hiệu lệnh) không, đặc biệt là giảm nhầm lẫn giữa các biển tròn viền đỏ?

Lý do đặt câu hỏi này (theo phân tích lỗi tuần 3, `reports/errors/phan_tich_loi.md`):
- Cấm rẽ và Giới hạn tốc độ là hai lớp vừa ít mẫu vừa có AP thấp.
- Chúng hay bị nhầm thành "Cấm còn lại", một lớp nhiều mẫu hơn hẳn. Ở `yolov8s_1280`: Cấm rẽ có AP 75.0 (thấp nhất) và 14.3% biển bị gán sai lớp; Giới hạn tốc độ có 11.1% biển bị gán sai lớp.
- Lớp lớn nhất (Nguy hiểm, 1367 box train) nhiều gấp 4.4 lần lớp nhỏ nhất (Cấm rẽ, 310 box).

## Thiết kế: chỉ khác đúng một yếu tố so với cấu hình tốt nhất

Cấu hình tốt nhất do B gửi (#7) là `yolov8s_1280`: YOLOv8s, ảnh gốc, imgsz 1280, 50 epoch, batch 8. Kết quả: mAP@0.5 = 82.2, AP_small = 50.2. Môi trường của B: Kaggle T4, ultralytics 8.4.204.

| Tên | Dữ liệu train | Loss | Ghi chú |
|---|---|---|---|
| `yolov8s_1280` (của B) | `data/yolo/data.yaml` | mặc định | Đối chứng, **dùng lại kết quả của B** |
| `imb_rfs` | `data/yolo/data_rfs.yaml` (oversampling `thr=0.3`) | mặc định | Cân bằng bằng **dữ liệu** |
| `imb_clspw` | `data/yolo/data.yaml` | `--cls-pw 0.5` | Cân bằng bằng **loss** |

- **Dùng lại baseline của B, không train lại.** Đủ cả ba điều kiện:
  - cùng phiên bản (ghim `ultralytics==8.4.204` như B; đã kiểm tra bản này có `cls_pw`);
  - cùng script và tham số (`train_yolov8.py`, các tham số còn lại để mặc định như B);
  - cùng dữ liệu (`data/splits` đã gộp box trùng).
- **Lưu ý khi viết báo cáo:** baseline được train ở lần chạy khác. Train trên GPU không lặp lại được hoàn toàn, nên chênh lệch nhỏ hơn khoảng 1 điểm mAP chưa đủ để kết luận. Nếu còn GPU, train lại `yolov8s_1280` thêm một lần để biết kết quả tự dao động bao nhiêu.

### Cách 1: Oversampling (`scripts/oversample_yolo.py`)
- Dùng Repeat Factor Sampling (Gupta et al., LVIS, CVPR 2019): ảnh chứa lớp càng hiếm thì càng được lặp nhiều lần trong mỗi epoch.
  - `f_c` = tỉ lệ ảnh có lớp c.
  - `r_c = max(1, sqrt(thr / f_c))`.
  - Mỗi ảnh lặp theo `r_c` lớn nhất trong ảnh.
- Chỉ tạo một file danh sách ảnh có dòng lặp. Không sao chép ảnh, không sửa nhãn.
- Kết quả trên `data/yolo` với `thr = 0.3`:

| Lớp | % ảnh có lớp | Box trước | Box sau | Tăng |
|---|---|---|---|---|
| Cấm ngược chiều | 21.9% | 721 | 1007 | 1.40× |
| Cấm dừng và đỗ | 36.9% | 1328 | 1592 | 1.20× |
| Cấm rẽ | 9.4% | 310 | 548 | **1.77×** |
| Giới hạn tốc độ | 9.8% | 569 | 992 | **1.74×** |
| Cấm còn lại | 24.8% | 973 | 1291 | 1.33× |
| Nguy hiểm | 38.5% | 1367 | 1549 | 1.13× |
| Hiệu lệnh | 13.7% | 425 | 652 | **1.53×** |

- Lớp nhiều nhất / ít nhất: 4.41 → 2.91. 2933 ảnh thành 3619 dòng, nên mỗi epoch dài thêm 23%.
- **Giới hạn của cách này:** lặp ảnh thì lặp luôn các lớp phổ biến đứng chung ảnh. Vì vậy tăng `thr` lên 0.5 gần như không cân bằng thêm (2.91 → 2.79), chỉ làm epoch dài thêm 52%.

### Cách 2: Trọng số lớp trong loss (`--cls-pw`)
- Ultralytics ≥ 8.4.100 có sẵn tham số `cls_pw`. Mỗi lớp được trọng số `(1/số box)^cls_pw`, chuẩn hóa để trung bình bằng 1, rồi nhân vào loss phân loại (BCE).
- Đã thêm `--cls-pw` vào `scripts/train_yolov8.py`. Mặc định là 0, tức tắt, nên các lần chạy cũ không đổi.
- Với `cls_pw = 0.5`, trọng số các lớp là:

| Lớp | Trọng số |
|---|---|
| Cấm rẽ | 1.47 |
| Hiệu lệnh | 1.25 |
| Giới hạn tốc độ | 1.08 |
| Cấm ngược chiều | 0.96 |
| Cấm còn lại | 0.83 |
| Cấm dừng và đỗ | 0.71 |
| Nguy hiểm | 0.70 |

  Lớp nhiều nhất / ít nhất là 2.1, gần với mức cân bằng của cách 1.
- **Khác biệt quan trọng giữa hai cách:** `cls_pw` đếm theo **box**, oversampling đếm theo **ảnh**.
  - Giới hạn tốc độ có nhiều box (569) nhưng chỉ có trong ít ảnh (9.8%), vì biển tốc độ hay đứng thành cụm.
  - Nên `cls_pw` gần như không tăng trọng số cho lớp này (1.08), trong khi oversampling lặp nó 1.74 lần.
  - Nếu kết quả của lớp này khác nhau giữa hai cách, đây là lời giải thích.
- **Focal loss:** Ultralytics có lớp `FocalLoss` nhưng không gắn vào loss của detection và không có tham số để bật (`fl_gamma` cũ đã bị bỏ). Muốn dùng phải tự viết trainer riêng. Vì vậy chọn `cls_pw`.

## Chạy trên Kaggle (GPU T4)

Cách nhanh nhất là dùng notebook có sẵn `notebooks/imbalance_kaggle.ipynb`: import lên Kaggle, mỗi lần chạy đổi biến `RUN`. Các cell dưới đây là phiên bản rút gọn của cùng các bước đó.

Cell 1: cài đặt, ghim đúng phiên bản của B.
```bash
!git clone https://github.com/tmcng/deeplearning-project-sign-detection.git
%cd deeplearning-project-sign-detection
!pip install -q "ultralytics==8.4.204" wandb pycocotools
!pip show ultralytics | head -2
```

Cell 2: W&B, giống thí nghiệm rò rỉ.
```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

Cell 3: chuyển sang YOLO và tạo file oversampling.
```bash
IMAGES=/kaggle/input/za-traffic-2020/za_traffic_2020/traffic_train/images
!python scripts/coco_to_yolo.py --splits data/splits --images $IMAGES --out data/yolo --copy
!python scripts/oversample_yolo.py --yolo data/yolo --thr 0.3
```
Kiểm tra bảng in ra có khớp bảng ở trên không (2933 ảnh thành 3619 dòng).

Cell 4: train 2 lần. Tham số giống hệt lệnh của B, chỉ đổi `--data` hoặc thêm `--cls-pw`. Nên chạy thử `--epochs 1` trước.
```bash
!python scripts/train_yolov8.py --data data/yolo/data_rfs.yaml --model yolov8s.pt --imgsz 1280 --epochs 50 --batch 8 --name imb_rfs
!python scripts/train_yolov8.py --data data/yolo/data.yaml --model yolov8s.pt --imgsz 1280 --epochs 50 --batch 8 --cls-pw 0.5 --name imb_clspw
```
- Với `imb_clspw`, kiểm tra log có dòng `Class weights: [...]`. Nếu không có, nghĩa là `cls_pw` chưa được dùng.
- Mỗi lần có thể mất vài giờ trên T4, `imb_rfs` lâu hơn khoảng 23%. Kaggle giới hạn mỗi phiên 12 giờ, nên có thể chạy mỗi mô hình trong một phiên. Nhớ tải `runs/detect/<tên>/weights/best.pt` về sau mỗi phiên.

Cell 5: dự đoán trên val (suy luận thường ở 1280, không SAHI, giống B) và chấm điểm.
```bash
!python scripts/predict_yolo.py --weights runs/detect/imb_rfs/weights/best.pt   --imgsz 1280 --out preds/imb_rfs_val.json   --eval --name imb_rfs
!python scripts/predict_yolo.py --weights runs/detect/imb_clspw/weights/best.pt --imgsz 1280 --out preds/imb_clspw_val.json --eval --name imb_clspw
```

Cell 6: phân tích lỗi và so sánh. Phần này chạy được trên máy cá nhân, sau khi tải `preds/imb_*` và `reports/eval/imb_*.json` về.
```bash
python scripts/analyze_errors.py --gt data/splits/val.json --pred preds/imb_rfs_val.json   --name imb_rfs
python scripts/analyze_errors.py --gt data/splits/val.json --pred preds/imb_clspw_val.json --name imb_clspw
python scripts/compare_runs.py --base yolov8s_1280 --runs imb_rfs imb_clspw --out reports/imbalance/so_sanh.md
```
Baseline `yolov8s_1280` đã có sẵn `reports/eval/yolov8s_1280.json` và `reports/errors/yolov8s_1280/`.

## Phân tích khi có kết quả

Trả lời theo thứ tự:
1. **AP từng lớp:** ba lớp hiếm có tăng không, nhất là Cấm rẽ? Các lớp phổ biến có bị giảm không? Tăng lớp hiếm thường phải đổi lại bằng việc giảm lớp phổ biến.
2. **Tỉ lệ sai lớp** của Cấm rẽ (14.3% ở baseline) và Giới hạn tốc độ (11.1%) có giảm không? Ma trận nhầm lẫn còn nhầm sang "Cấm còn lại" nhiều như trước không? Nếu sai lớp giảm mà bỏ sót không giảm, thì cân bằng lớp chỉ sửa được lỗi nhận lớp, không giúp tìm biển.
3. **Báo nhầm nền** (74 box ở baseline) có tăng không? Tăng trọng số lớp hiếm có thể làm mô hình báo bừa lớp đó nhiều hơn.
4. **mAP chung:** có thể không đổi, vì lớp hiếm chiếm ít box.
5. **Nếu không cải thiện**, các lý do có thể:
   - Lỗi chính là **bỏ sót biển nhỏ**, không phải mất cân bằng lớp. Ở baseline, 74% biển dưới 8 px vẫn bị bỏ sót, trong khi chỉ 3.2% biển bị gán sai lớp.
   - Nhầm giữa các biển tròn viền đỏ là do **hình dạng giống nhau ở kích thước nhỏ**. Có thêm mẫu cũng không làm phần giữa biển rõ hơn.
   - Lặp ảnh thì lặp cả lớp phổ biến đi kèm.
   - Lặp ảnh nhiều dễ học thuộc các ảnh đó (overfit). Xem đường loss trên val trong W&B để kiểm tra.
6. **Độ tin cậy của kết quả:** mỗi cấu hình chỉ train 1 lần. Val chỉ có 98 biển Cấm rẽ, và baseline được train ở lần chạy khác. Vì vậy chênh lệch vài điểm AP của một lớp có thể chỉ là ngẫu nhiên.

## Kết quả

Train trên Kaggle T4, ultralytics 8.4.204. Cả hai lần đều chạy đủ 50 epoch, không dừng sớm. `args.yaml` xác nhận chỉ khác baseline đúng một yếu tố. Log train, biểu đồ và ma trận nhầm lẫn của Ultralytics nằm trong `reports/imbalance/<tên>/`.

| Tên | mAP@0.5 | mAP@0.5:0.95 | AP_small | AP50 Cấm rẽ | AP50 Giới hạn tốc độ | AP50 Hiệu lệnh | Sai lớp Cấm rẽ | Sai lớp Giới hạn tốc độ | Báo nhầm nền |
|---|---|---|---|---|---|---|---|---|---|
| yolov8s_1280 (baseline) | **82.2** | **56.6** | **50.2** | **75.0** | **77.7** | 78.6 | 14.3% | **11.1%** | 74 |
| imb_rfs (oversampling) | 81.0 | 55.8 | 49.6 | 71.4 | 73.1 | **80.7** | 19.4% | 14.4% | 77 |
| imb_clspw (cls_pw 0.5) | 81.1 | 55.7 | 49.7 | 74.9 | 71.6 | 78.7 | **10.2%** | 11.8% | **68** |

- Bảng đủ 7 lớp (AP, recall, sai lớp): `reports/imbalance/so_sanh.md`.
- Phân tích lỗi từng mô hình: `reports/errors/imb_*/`.

### Chênh lệch có ý nghĩa không? (`scripts/bootstrap_ap.py`)

Mỗi lần lấy mẫu lại ngẫu nhiên 696 ảnh val, có hoàn lại, rồi tính AP50 của cả ba mô hình trên cùng bộ ảnh đó. Lặp 1000 lần (paired bootstrap). Bảng dưới là chênh lệch AP50 so với baseline, kèm khoảng tin cậy 95%:

| Lớp | imb_rfs | imb_clspw |
|---|---|---|
| Cấm rẽ | −3.6 [−10.2, +2.7] | −0.1 [−5.4, +4.8] |
| Giới hạn tốc độ | −4.6 [−9.4, +0.2] | **−6.1 [−10.2, −1.6]** |
| Hiệu lệnh | +2.1 [−1.0, +5.9] | +0.1 [−3.6, +3.9] |
| Cấm còn lại | +0.9 [−1.8, +3.4] | −1.0 [−3.4, +1.7] |
| mAP@0.5 | −1.2 [−2.6, +0.4] | −1.1 [−2.5, +0.2] |

- **Chỉ có một thay đổi chắc chắn: `cls_pw` làm giảm AP của Giới hạn tốc độ.** Mọi thay đổi khác đều nằm trong khoảng dao động do chọn ảnh val.
- Khoảng tin cậy của Cấm rẽ rộng tới khoảng ±6 điểm, vì val chỉ có 98 biển lớp này. Muốn thấy được một cải thiện vài điểm thì cần tập val lớn hơn nhiều.
- Bootstrap chỉ đo dao động do chọn ảnh val. Nó chưa tính dao động giữa các lần train, vì mỗi cấu hình mới train 1 lần. Dao động thật còn lớn hơn.

### Kết luận

**Cả hai cách cân bằng lớp đều không cải thiện lớp hiếm, và mAP chung giảm khoảng 1 điểm (không có ý nghĩa thống kê).** Bốn lý do, đều có số liệu kiểm chứng:

1. **Lỗi chỉ bị chuyển chỗ, không mất đi.**
   - Tổng số biển bị nhầm lẫn giữa 3 lớp tròn viền đỏ không giảm: baseline 47, oversampling 56, `cls_pw` 50.
   - Ví dụ với `cls_pw`: Cấm rẽ có trọng số cao nhất (1.47) nên mô hình dự đoán Cấm rẽ nhiều hơn (79 → 90 box). Số biển Cấm rẽ được nhận đúng tăng thêm 3 (70 → 73), sai lớp giảm (14.3% → 10.2%). Nhưng số box dự đoán Cấm rẽ mà sai gần gấp đôi (9 → 17): biển Giới hạn tốc độ bị đoán thành Cấm rẽ tăng 3 → 6, biển Cấm còn lại bị đoán thành Cấm rẽ tăng 4 → 8.
   - Nên AP của Cấm rẽ đứng yên (75.0 → 74.9), còn Giới hạn tốc độ (trọng số chỉ 1.08) bị kéo sang Cấm rẽ và giảm 6.1 điểm.
   - **Trọng số chỉ đổi ranh giới quyết định giữa các lớp trông giống nhau. Nó không giúp mô hình phân biệt các lớp đó tốt hơn.**
2. **Oversampling lặp luôn chính lớp gây nhầm.**
   - 49% ảnh có Giới hạn tốc độ cũng có Cấm còn lại. Tỉ lệ chung chỉ là 25%, vì biển tốc độ hay đứng chung cột với biển cấm.
   - Vì vậy RFS lặp Cấm còn lại 1.33 lần, cùng với Giới hạn tốc độ (1.74 lần).
   - Kết quả: biển Giới hạn tốc độ bị đoán thành Cấm còn lại tăng 14 → 18, biển Cấm rẽ bị đoán thành Cấm còn lại tăng 8 → 15. Số box dự đoán "Cấm còn lại" mà sai tăng từ 37 lên 55.
3. **Mất cân bằng lớp không phải nguồn lỗi chính.**
   - Ở baseline, chỉ 3.2% biển bị gán sai lớp, trong khi 16.7% bị bỏ sót, chủ yếu là biển nhỏ: 74% biển dưới 8 px bị bỏ sót.
   - Cân bằng lớp không tác động tới bỏ sót. Recall theo kích thước gần như không đổi: nhóm 8–16 px lần lượt 74.6%, 71.1%, 73.5%.
   - Kể cả sửa hết lỗi nhầm lớp thì recall chung cũng chỉ tăng tối đa khoảng 3 điểm.
4. **Mức mất cân bằng vốn không lớn** (4.4 lần, theo số box). Các phương pháp như RFS được thiết kế cho dữ liệu đuôi dài, nơi lớp hiếm ít hơn hàng trăm lần.

Hai điểm phụ:
- **Không thấy học thuộc (overfit) do lặp ảnh.** Loss phân loại trên val của `imb_rfs` vẫn giảm tới cuối (0.81 ở epoch 8, 0.62 ở epoch 50), giống `imb_clspw`.
- **Vì sao `imb_clspw` báo nhầm nền ít hơn** (74 → 68): giảm trọng số các lớp phổ biến làm mô hình bớt tự tin hơn khi dự đoán chúng. Chênh lệch này nhỏ.

**Hướng tốt hơn cho lớp hiếm:** lỗi nằm ở chỗ ba lớp tròn viền đỏ trông giống nhau khi biển nhỏ, nên cần cho mô hình nhìn rõ phần giữa biển:
- tăng độ phân giải hoặc thêm đầu ra P2 cho biển nhỏ;
- dùng bước phân loại lại (crop biển rồi phân loại ở độ phân giải cao);
- sửa nhãn sai giữa các lớp này: tuần 1 đã thấy 2 box sai lớp đúng kiểu này trong 20 ảnh.

Cấu hình cuối cùng vẫn là `yolov8s_1280` không cân bằng lớp.
