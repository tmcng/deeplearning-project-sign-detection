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

Điền sau khi chạy. Bảng chi tiết sinh tự động ở `reports/imbalance/so_sanh.md`.

| Tên | mAP@0.5 | AP50 Cấm rẽ | AP50 Giới hạn tốc độ | AP50 Hiệu lệnh | Sai lớp Cấm rẽ | Sai lớp Giới hạn tốc độ |
|---|---|---|---|---|---|---|
| yolov8s_1280 (baseline) | 82.2 | 75.0 | 77.7 | 78.6 | 14.3% | 11.1% |
| imb_rfs | | | | | | |
| imb_clspw | | | | | | |
