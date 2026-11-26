# Hướng Dẫn Chạy Faster R-CNN (Train - Predict - Evaluate)

Tài liệu này tóm tắt ngắn gọn các cờ lệnh và quy trình thực thi trên cả **Kaggle (GPU)** và **Local (Máy cá nhân)**.

---

## 1. Giải thích nhanh các cờ (CLI Flags)

### Cho `faster_rcnn/train.py`
* `-c, --config`: Đường dẫn file cấu hình thí nghiệm (vd: `faster_rcnn/configs/experiment/exp01_baseline.yaml`).
* `--img-dir`: **[Mới]** Đường dẫn thư mục ảnh (override config, vd: `/kaggle/input/<tên-dataset>/images`).
* `--train-json`: **[Mới]** File COCO annotation train (override config, vd: `/kaggle/input/<tên-dataset>/train.json`).
* `--val-json`: **[Mới]** File COCO annotation val (override config, vd: `/kaggle/input/<tên-dataset>/val.json`).
* `--trainer`: Chọn profile phần cứng: `gpu_fp16` (Kaggle/Server T4), `gpu_fp32`, hoặc `debug_cpu` (chạy thử CPU local).
* `--epochs`: Số epoch huấn luyện (override file config).
* `--batch-size`: Kích thước batch (mặc định theo config, thường là 4).
* `--lr`: Tốc độ học learning rate (vd: `0.005`, `0.001`).
* `--no-wandb`: Tắt đồng bộ Weights & Biases (nếu không cần log).
* `--dry-run`: Chạy thử nhanh 1 batch để kiểm tra pipeline không bị lỗi.

### Cho `faster_rcnn/evaluate.py`
* `-c, --config`: File cấu hình thí nghiệm tương ứng.
* `--split`: Tập dữ liệu đánh giá: `val` (mặc định) hoặc `test`.
* `--img-dir`: **[Mới]** Đường dẫn thư mục ảnh (override config, vd: `/kaggle/input/<tên-dataset>/images`).
* `--gt`: **[Mới]** Đường dẫn file annotation COCO JSON ground-truth (override config, vd: `/kaggle/input/<tên-dataset>/val.json`).
* `--pred`: Đường dẫn file predictions JSON có sẵn (bỏ qua inference model, chấm điểm ngay lập tức bằng `pycocotools`).
* `--weights`: Đường dẫn checkpoint `.pth` hoặc `.zip` tùy chọn (mặc định tự tìm `best` $\rightarrow$ `last`).
* `--out-pred`: Đường dẫn lưu file predictions JSON (mặc định xuất thẳng vào `preds/`).
* `--name`: Tên thí nghiệm ghi vào `reports/eval/` và `results.md` (mặc định lấy theo config `_name`).
* `--conf, --conf-thresh`: Ngưỡng confidence score (mặc định `0.001` để pycocotools quét đủ đường cong PR; có thể chỉnh vd `--conf 0.0001` để bắt tối đa ứng viên hoặc `--conf 0.25` để lọc chặt).
* `--dry-run`: Chạy thử nhanh 1 batch inference sanity check.

---

## 2. Quy trình chạy trên Kaggle (GPU Server) khi Dữ liệu ở `/kaggle/input/`

Toàn bộ thao tác dưới đây được thực hiện trong **Kaggle Notebook** (chọn Accelerator là **GPU T4 x2** hoặc **GPU P100**).
Do thư mục ảnh và nhãn (COCO JSON) nằm trong thư mục dataset Kaggle Input (chế độ chỉ đọc: `/kaggle/input/<tên-dataset>/...`), bạn có thể chọn một trong các phương án sau:

### Phương án A: Truyền trực tiếp đường dẫn qua cờ CLI (Khuyên dùng — Tiện nhất, không cần sửa file)

Bạn không cần chạm vào bất kỳ file code hay file `.yaml` nào, chỉ cần truyền thẳng đường dẫn của Kaggle Input vào lệnh chạy:

#### 1. Đánh giá (Evaluate) tập Validation (`val`):
```bash
python faster_rcnn/evaluate.py \
  -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml \
  --split val \
  --img-dir /kaggle/input/<tên-dataset-ảnh>/images \
  --gt /kaggle/input/<tên-dataset-nhãn>/val.json
```

#### 2. Đánh giá (Evaluate) tập Test (`test`):
```bash
python faster_rcnn/evaluate.py \
  -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml \
  --split test \
  --img-dir /kaggle/input/<tên-dataset-ảnh>/images \
  --gt /kaggle/input/<tên-dataset-nhãn>/test.json
```

#### 3. Đánh giá khi dùng Weights có sẵn từ Kaggle Input:
Nếu bạn upload file trọng số checkpoint đã train trước đó lên Kaggle Input (dưới dạng file `.pth` hoặc `.zip`):
```bash
python faster_rcnn/evaluate.py \
  -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml \
  --split test \
  --img-dir /kaggle/input/<tên-dataset-ảnh>/images \
  --gt /kaggle/input/<tên-dataset-nhãn>/test.json \
  --weights /kaggle/input/<tên-dataset-weights>/fasterrcnn_fpn_aug_best.pth \
  --out-pred preds/predictions_test_exp03_fpn_aug.json
```

#### 4. Huấn luyện (Training) với Data & Label từ Kaggle Input:
```bash
python faster_rcnn/train.py \
  -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml \
  --img-dir /kaggle/input/<tên-dataset-ảnh>/images \
  --train-json /kaggle/input/<tên-dataset-nhãn>/train.json \
  --val-json /kaggle/input/<tên-dataset-nhãn>/val.json
```

---

### Phương án B: Tạo Symlink (Liên kết mềm) trong Notebook

Nếu không muốn gõ lại cờ `--img-dir` và `--gt` mỗi lần chạy lệnh, hãy tạo liên kết mềm trỏ từ Kaggle Input về thư mục mặc định của repo (`data/images` và `data/splits/`):

```bash
# Trong ô code Notebook Kaggle:
%cd /kaggle/working
!git clone https://github.com/tmcng/deeplearning-project-sign-detection.git
%cd deeplearning-project-sign-detection

# Cài thư viện
!pip install -q pycocotools wandb

# Tạo symlink cho thư mục ảnh
!mkdir -p data
!ln -sfn /kaggle/input/<tên-dataset-ảnh>/images data/images

# Tạo symlink cho các file nhãn JSON (nếu nằm trong Kaggle input)
!mkdir -p data/splits
!ln -sf /kaggle/input/<tên-dataset-nhãn>/train.json data/splits/train.json
!ln -sf /kaggle/input/<tên-dataset-nhãn>/val.json   data/splits/val.json
!ln -sf /kaggle/input/<tên-dataset-nhãn>/test.json  data/splits/test.json

# Kiểm tra symlink đã nhận đúng ảnh
!ls -lh data/images | head -n 5
```
Sau khi tạo symlink, bạn chỉ cần chạy lệnh ngắn gọn như bình thường:
```bash
# Đánh giá tập val
python faster_rcnn/evaluate.py -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml --split val

# Đánh giá tập test
python faster_rcnn/evaluate.py -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml --split test
```

---

### Phương án C: Cập nhật file cấu hình `faster_rcnn/configs/paths/server_gpu.yaml`

Nếu bạn muốn cấu hình cố định đường dẫn cho toàn bộ thí nghiệm trên GPU server, sửa trực tiếp file `faster_rcnn/configs/paths/server_gpu.yaml`:

```yaml
# faster_rcnn/configs/paths/server_gpu.yaml
project_root: null

data:
  img_dir:    "/kaggle/input/<tên-dataset-ảnh>/images"
  train_json: "/kaggle/input/<tên-dataset-nhãn>/train.json"   # hoặc "data/splits/train.json"
  val_json:   "/kaggle/input/<tên-dataset-nhãn>/val.json"     # hoặc "data/splits/val.json"
  test_json:  "/kaggle/input/<tên-dataset-nhãn>/test.json"    # hoặc "data/splits/test.json"

output:
  checkpoint_dir: "faster_rcnn/checkpoints"
  output_dir:     "preds"
  report_dir:     "reports/eval"

num_workers: 2
```

---

### Tải file kết quả từ Kaggle về Local

Kaggle chỉ lưu trữ dữ liệu sinh ra trong thư mục `/kaggle/working/`:
* File kết quả dự đoán: `preds/predictions_val_<tên_exp>.json` hoặc `preds/predictions_test_*.json`.
* Báo cáo đánh giá: `reports/eval/<tên_exp>.json`.
* Bảng so sánh tổng hợp: `reports/eval/results.md`.

**Cách tải về:**
Nén các thư mục kết quả lại và tải từ tab **Output** bên phải màn hình:
```bash
!zip -r faster_rcnn_results.zip preds/ reports/ faster_rcnn/checkpoints/
```

---

## 3. Quy trình chạy trên Local (Máy cá nhân)

> **Lưu ý về dữ liệu trên máy Local:**
> - Máy local có thể chỉ lưu trữ tập ảnh test rút gọn trong `data/images/` (871 ảnh). Tập train/val đầy đủ được lưu trữ trên GPU Server/Kaggle.
> - Do đó, trên local nên ưu tiên **Chế độ A** (chấm điểm từ predictions JSON có sẵn) hoặc chạy thử sanity check trên tập test (`--split test --dry-run`).

### Chế độ A: Chấm điểm nhanh từ file tải về (Không cần PyTorch/GPU)
Khi bạn đã tải file predictions từ Kaggle về thư mục `preds/`, chỉ cần chạy 1 dòng lệnh để tính điểm và cập nhật `results.md`:
```bash
# Chấm điểm EXP01
python faster_rcnn/evaluate.py --pred preds/predictions_val_exp01_baseline.json

# Chấm điểm EXP02
python faster_rcnn/evaluate.py --pred preds/predictions_val_exp02_small_anchor.json

# Chấm điểm EXP03
python faster_rcnn/evaluate.py --pred preds/predictions_val_exp03_fpn_aug.json
```

### Chế độ B: Chạy thử kiểm tra code trên CPU (Sanity check)
Nếu muốn test luồng inference model trên máy cá nhân không có GPU (dùng tập `test` có sẵn ảnh local):
```bash
# Test luồng evaluate 1 batch trên CPU với tập test
python faster_rcnn/evaluate.py -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml --split test --dry-run

# Test luồng train 1 batch trên CPU (yêu cầu ảnh train trong data/images)
python faster_rcnn/train.py -c faster_rcnn/configs/experiment/exp01_baseline.yaml --trainer debug_cpu --dry-run --no-wandb
```

### Chế độ C: Inference & Evaluate đầy đủ (Tập VAL hoặc TEST)

Hệ thống tự động phân biệt dữ liệu và file annotation thông qua cờ `--split` (toàn bộ ảnh để chung trong `data/images/`, loader sẽ mở đúng file JSON trong `data/splits/` để lọc đúng danh sách ảnh cần chạy):

| Cờ `--split` | File Annotation đọc vào | Hành vi xử lý | File dự đoán xuất ra |
| :---: | :--- | :--- | :--- |
| **`--split val`** *(Mặc định)* | `data/splits/val.json` | Tính `val/loss` $\rightarrow$ Inference $\rightarrow$ Chấm mAP | `preds/predictions_val_<tên_exp>.json` |
| **`--split test`** | `data/splits/test.json` | Bỏ qua tính loss $\rightarrow$ Inference $\rightarrow$ Chấm mAP | `preds/predictions_test_<tên_exp>.json` |

#### 1. Lệnh chạy cho tập Validation (`val`):
```powershell
# Chạy đánh giá tập val (tính val loss và mAP)
python faster_rcnn/evaluate.py -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml --split val
```

#### 2. Lệnh chạy cho tập Test (`test`):
```powershell
# Chạy đánh giá tập test (tự động bỏ qua tính loss, chỉ inference và chấm điểm mAP)
python faster_rcnn/evaluate.py -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml --split test

# Hoặc tùy biến chỉ định trực tiếp file weights và file predictions:
python faster_rcnn/evaluate.py `
  -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml `
  --split test `
  --weights faster_rcnn/checkpoints/fasterrcnn_fpn_aug_best.zip `
  --out-pred preds/predictions_test_exp03_fpn_aug.json
```

* **Kết quả sinh ra:**
  * File dự đoán: `preds/predictions_test_<tên_exp>.json` (hoặc `predictions_val_*.json`).
  * Báo cáo đánh giá JSON: `reports/eval/<tên_exp>.json`.
  * Tự động cập nhật bảng điểm so sánh vào: `reports/eval/results.md`.
