# Thí nghiệm mất cân bằng lớp (tuần 4)

## Câu hỏi

Cân bằng lại các lớp khi train có giúp các lớp hiếm (Cấm rẽ, Giới hạn tốc độ, Hiệu lệnh) không, đặc biệt là giảm nhầm lẫn giữa các biển tròn viền đỏ?

Lý do đặt câu hỏi này (theo phân tích lỗi tuần 3, `reports/errors/phan_tich_loi.md`):
- Cấm rẽ và Giới hạn tốc độ là hai lớp vừa ít mẫu vừa có AP thấp.
- Chúng hay bị nhầm thành "Cấm còn lại", một lớp nhiều mẫu hơn hẳn: ở YOLO 640, 18–22% biển hai lớp này bị gán sai lớp.
- Lớp lớn nhất (Nguy hiểm, 1367 box train) nhiều gấp 4.4 lần lớp nhỏ nhất (Cấm rẽ, 310 box).

## Thiết kế: 3 lần train, chỉ khác đúng một yếu tố

| Tên | Dữ liệu train | Loss | Ý nghĩa |
|---|---|---|---|
| `imb_base` | `data.yaml` | mặc định | Đối chứng, train lại trong cùng môi trường |
| `imb_rfs` | `data_rfs.yaml` (oversampling `thr=0.3`) | mặc định | Cân bằng bằng **dữ liệu** |
| `imb_clspw` | `data.yaml` | `--cls-pw 0.5` | Cân bằng bằng **loss** |

- **Mô hình và cấu hình** lấy theo cấu hình tốt nhất B gửi (#7). Hiện tốt nhất là YOLOv8s train trên ô cắt 640 (`data/yolo_tiled`) + suy luận SAHI, đạt 83.9 mAP@0.5. Mọi tham số khác (epochs, imgsz, batch, seed, augmentation) giữ **giống hệt** giữa 3 lần.
- **Vì sao phải train lại baseline, không dùng lại số 83.9 của B:**
  - `cls_pw` cần Ultralytics ≥ 8.4.100, có thể khác bản B đã dùng.
  - Đổi phiên bản hoặc môi trường cũng làm kết quả lệch vài điểm.
  - Chỉ khi cùng môi trường thì chênh lệch mới đến từ cách cân bằng lớp.
- **Ghim phiên bản** `ultralytics==8.4.215`, là bản đã kiểm tra có `cls_pw`.

### Cách 1: Oversampling (`scripts/oversample_yolo.py`)
- Dùng Repeat Factor Sampling (Gupta et al., LVIS, CVPR 2019): ảnh chứa lớp càng hiếm thì càng được lặp nhiều lần trong mỗi epoch.
  - `f_c` = tỉ lệ ảnh có lớp c.
  - `r_c = max(1, sqrt(thr / f_c))`.
  - Mỗi ảnh lặp theo `r_c` lớn nhất trong ảnh.
- Chỉ tạo một file danh sách ảnh có dòng lặp. Không sao chép ảnh, không sửa nhãn.
- Thử trên tập ảnh gốc với `thr = 0.3`:

| Lớp | % ảnh có lớp | Box trước | Box sau | Tăng |
|---|---|---|---|---|
| Cấm ngược chiều | 21.9% | 721 | 1007 | 1.40× |
| Cấm dừng và đỗ | 36.9% | 1328 | 1592 | 1.20× |
| Cấm rẽ | 9.4% | 310 | 548 | **1.77×** |
| Giới hạn tốc độ | 9.8% | 569 | 992 | **1.74×** |
| Cấm còn lại | 24.8% | 973 | 1291 | 1.33× |
| Nguy hiểm | 38.5% | 1367 | 1549 | 1.13× |
| Hiệu lệnh | 13.7% | 425 | 652 | **1.53×** |

- Lớp nhiều nhất / ít nhất: 4.41 → 2.91. Mỗi epoch dài thêm 23%.
- **Giới hạn của cách này:** lặp ảnh thì lặp luôn các lớp phổ biến đứng chung ảnh. Vì vậy tăng `thr` lên 0.5 gần như không cân bằng thêm (2.91 → 2.79), chỉ làm epoch dài thêm 52%.
- Trên tập ô cắt, các con số sẽ hơi khác. Script tự in bảng khi chạy.

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

## Chạy trên Kaggle (GPU)

Cell 1: cài đặt, ghim phiên bản và ghi lại phiên bản đã dùng.
```bash
!git clone https://github.com/tmcng/deeplearning-project-sign-detection.git
%cd deeplearning-project-sign-detection
!pip install -q "ultralytics==8.4.215" wandb pycocotools
!pip show ultralytics | head -2
```

Cell 2: W&B, giống thí nghiệm rò rỉ.
```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

Cell 3: tạo tập ô cắt và file oversampling.
```bash
IMAGES=/kaggle/input/za-traffic-2020/za_traffic_2020/traffic_train/images
!python scripts/tile_yolo_dataset.py --splits data/splits --images $IMAGES --out data/yolo_tiled
!python scripts/oversample_yolo.py --yolo data/yolo_tiled --thr 0.3
```
Lưu lại bảng số box trước và sau mà lệnh thứ hai in ra, để đưa vào báo cáo.

Cell 4: train 3 lần.
- `COMMON` phải khớp đúng cấu hình B gửi (#7). Dòng dưới chỉ là mặc định của script.
- Nên chạy thử một lần `--epochs 1` trước.
- Mỗi lần có thể mất 1.5–2.5 giờ trên T4. Kaggle giới hạn mỗi phiên 12 giờ, nên có thể phải tách thành nhiều phiên, và nhớ tải kết quả về sau mỗi phiên.
```bash
COMMON="--model yolov8s.pt --imgsz 640 --epochs 50 --seed 0"
!python scripts/train_yolov8.py --data data/yolo_tiled/data.yaml     $COMMON --name imb_base
!python scripts/train_yolov8.py --data data/yolo_tiled/data_rfs.yaml $COMMON --name imb_rfs
!python scripts/train_yolov8.py --data data/yolo_tiled/data.yaml     $COMMON --cls-pw 0.5 --name imb_clspw
```
Với lần `imb_clspw`, kiểm tra log có dòng `Class weights: [...]`. Nếu không có, nghĩa là `cls_pw` chưa được dùng.

Cell 5: dự đoán trên val bằng SAHI (giống cấu hình tốt nhất) và chấm điểm.
```bash
for N in imb_base imb_rfs imb_clspw; do
  python scripts/predict_yolo.py --weights runs/detect/$N/weights/best.pt --imgsz 640 \
      --images data/yolo_tiled/images/val --class-map data/yolo_tiled/class_map.json \
      --sahi --slice-size 640 --overlap 0.2 --out preds/${N}_val.json --eval --name $N
done
```
Cell này viết theo cú pháp bash. Trong Kaggle, đặt `%%bash` ở dòng đầu cell.

Cell 6: phân tích lỗi và so sánh. Phần này chạy được cả trên máy cá nhân, sau khi tải `preds/imb_*` về.
```bash
for N in imb_base imb_rfs imb_clspw; do
  python scripts/analyze_errors.py --gt data/splits/val.json --pred preds/${N}_val.json --name $N
done
python scripts/compare_runs.py --base imb_base --runs imb_rfs imb_clspw --out reports/imbalance/so_sanh.md
```

## Phân tích khi có kết quả

Trả lời theo thứ tự:
1. **AP từng lớp:** ba lớp hiếm có tăng không, nhất là Cấm rẽ? Các lớp phổ biến có bị giảm không? Tăng lớp hiếm thường phải đổi lại bằng việc giảm lớp phổ biến.
2. **Tỉ lệ sai lớp** của Cấm rẽ và Giới hạn tốc độ có giảm không? Ma trận nhầm lẫn còn nhầm sang "Cấm còn lại" nhiều như trước không? Nếu sai lớp giảm mà bỏ sót không giảm, thì cân bằng lớp chỉ sửa được lỗi nhận lớp, không giúp tìm biển.
3. **Báo nhầm nền** có tăng không? Tăng trọng số lớp hiếm có thể làm mô hình báo bừa lớp đó nhiều hơn.
4. **mAP chung:** có thể không đổi, vì lớp hiếm chiếm ít box.
5. **Nếu không cải thiện**, các lý do có thể:
   - Lỗi chính của mô hình tốt nhất là **bỏ sót biển nhỏ**, không phải mất cân bằng lớp. 74% biển dưới 8 px vẫn bị bỏ sót.
   - Nhầm giữa các biển tròn viền đỏ là do **hình dạng giống nhau ở kích thước nhỏ**. Có thêm mẫu cũng không làm phần giữa biển rõ hơn.
   - Lặp ảnh thì lặp cả lớp phổ biến đi kèm.
   - Lặp ảnh nhiều dễ học thuộc các ảnh đó (overfit). Xem đường loss trên val trong W&B để kiểm tra.
6. **Độ tin cậy của kết quả:** mỗi cấu hình chỉ train 1 lần (1 seed). Val chỉ có 98 biển Cấm rẽ, nên chênh lệch vài điểm AP có thể chỉ là ngẫu nhiên. Nếu còn GPU, chạy thêm seed 1 cho `imb_base` để biết kết quả tự dao động bao nhiêu.

## Kết quả

Điền sau khi chạy. Bảng chi tiết sinh tự động ở `reports/imbalance/so_sanh.md`.

| Tên | mAP@0.5 | AP50 Cấm rẽ | AP50 Giới hạn tốc độ | AP50 Hiệu lệnh | Sai lớp Cấm rẽ | Sai lớp Giới hạn tốc độ |
|---|---|---|---|---|---|---|
| imb_base | | | | | | |
| imb_rfs | | | | | | |
| imb_clspw | | | | | | |
