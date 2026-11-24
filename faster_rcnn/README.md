# Hướng Dẫn Chạy Faster R-CNN (Train - Predict - Evaluate)

Tài liệu này tóm tắt ngắn gọn các cờ lệnh và quy trình thực thi trên cả **Kaggle (GPU)** và **Local (Máy cá nhân)**.

---

## 1. Giải thích nhanh các cờ (CLI Flags)

### Cho `faster_rcnn/train.py`
* `-c, --config`: Đường dẫn file cấu hình thí nghiệm (vd: `faster_rcnn/configs/experiment/exp01_baseline.yaml`).
* `--trainer`: Chọn profile phần cứng: `gpu_fp16` (Kaggle/Server T4), `gpu_fp32`, hoặc `debug_cpu` (chạy thử CPU local).
* `--epochs`: Số epoch huấn luyện (override file config).
* `--batch-size`: Kích thước batch (mặc định theo config, thường là 4).
* `--lr`: Tốc độ học learning rate (vd: `0.005`, `0.001`).
* `--no-wandb`: Tắt đồng bộ Weights & Biases (nếu không cần log).
* `--dry-run`: Chạy thử nhanh 1 batch để kiểm tra pipeline không bị lỗi.

### Cho `faster_rcnn/evaluate.py`
* `-c, --config`: File cấu hình thí nghiệm tương ứng.
* `--split`: Tập dữ liệu đánh giá: `val` (mặc định) hoặc `test`.
* `--pred`: Đường dẫn file predictions JSON có sẵn (bỏ qua inference model, chấm điểm ngay lập tức bằng `pycocotools`).
* `--weights`: Đường dẫn checkpoint `.pth` tùy chọn (mặc định tự tìm `best` $\rightarrow$ `last`).
* `--out-pred`: Đường dẫn lưu file predictions JSON (mặc định xuất thẳng vào `preds/`).
* `--name`: Tên thí nghiệm ghi vào `reports/eval/` và `results.md` (mặc định lấy theo config).
* `--conf-thresh`: Ngưỡng confidence score (mặc định `0.001` để pycocotools quét đủ đường cong PR).

---

## 2. Quy trình chạy trên Kaggle (GPU Server)

Toàn bộ lệnh chạy từ thư mục gốc của dự án:

### Bước 1: Huấn luyện (Training)
Chọn thí nghiệm tương ứng (`exp01_baseline`, `exp02_fpn_small_anchor`, hoặc `exp03_fpn_aug`):

```bash
# EXP01: Baseline (Frozen backbone, anchor 32-512px)
python faster_rcnn/train.py -c faster_rcnn/configs/experiment/exp01_baseline.yaml

# EXP02: FPN + Anchor nhỏ (8-128px)
python faster_rcnn/train.py -c faster_rcnn/configs/experiment/exp02_fpn_small_anchor.yaml

# EXP03: FPN + Anchor nhỏ + Safe Augmentation
python faster_rcnn/train.py -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml
```
*(Thêm `--no-wandb` vào cuối nếu không muốn dùng WandB).*

### Bước 2: Inference & Chấm điểm (Predict + Evaluate)
Sau khi train xong, chạy lệnh đánh giá trên tập `val`:
```bash
# Sẽ tự động lấy weights best, xuất file preds/ và chấm mAP qua pycocotools
python faster_rcnn/evaluate.py -c faster_rcnn/configs/experiment/exp01_baseline.yaml --split val
python faster_rcnn/evaluate.py -c faster_rcnn/configs/experiment/exp02_fpn_small_anchor.yaml --split val
python faster_rcnn/evaluate.py -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml --split val
```
* Kết quả sinh ra:
  * File dự đoán: `preds/predictions_val_<tên_exp>.json`
  * Báo cáo JSON: `reports/eval/<tên_exp>.json`
  * Bảng so sánh: tự động nối thêm dòng vào `reports/eval/results.md`.

### Bước 3: Lấy file về Local
* **Cách 1 (Git - Khuyên dùng):** Commit các file trong `preds/` và `reports/eval/` rồi push lên nhánh `faster-rcnn`.
* **Cách 2 (Download):** Tải các file `preds/predictions_val_*.json` từ mục Output của Kaggle về đặt vào thư mục `preds/` trên máy local.

---

## 3. Quy trình chạy trên Local (Máy cá nhân)

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
Nếu muốn test luồng train/eval trên máy cá nhân không có GPU:
```bash
# Test luồng train 1 batch trên CPU
python faster_rcnn/train.py -c faster_rcnn/configs/experiment/exp01_baseline.yaml --trainer debug_cpu --dry-run --no-wandb

# Test luồng evaluate 1 batch
python faster_rcnn/evaluate.py -c faster_rcnn/configs/experiment/exp01_baseline.yaml --dry-run
```

### Chế độ C: Inference & Evaluate đầy đủ trên tập Test (Local / Server)
Khi bạn đã giải nén ảnh tập test vào `data/images/` và annotation tại `data/splits/test.json`:

```powershell
# Chạy inference và chấm điểm tập test cho thí nghiệm EXP03 (tự động nhận checkpoint .zip hoặc .pth)
python faster_rcnn/evaluate.py -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml --split test

# Hoặc chỉ định rõ file weights và đường dẫn file predictions xuất ra:
python faster_rcnn/evaluate.py `
  -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml `
  --split test `
  --weights faster_rcnn/checkpoints/fasterrcnn_fpn_aug_best.zip `
  --out-pred preds/predictions_test_exp03_fpn_aug.json
```
* **Kết quả sinh ra:**
  * File dự đoán test: `preds/predictions_test_exp03_fpn_aug.json`
  * Báo cáo đánh giá: `reports/eval/exp03_fpn_aug.json` và cập nhật vào `reports/eval/results.md`.
