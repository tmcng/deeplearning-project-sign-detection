# Hệ thống Cấu hình (Config System)

Thư mục này chứa toàn bộ siêu tham số theo kiến trúc **modular YAML** —
mỗi nhóm config được tách thành file riêng, dễ swap mà không cần sửa code.

## Cấu trúc

```
configs/
├── train.yaml                          # File điều phối chính
├── paths/
│   ├── local.yaml                      # Máy cá nhân (Windows / macOS)
│   └── server_gpu.yaml                 # Linux server / Kaggle / Colab
├── data/
│   ├── dataset_v1.yaml                 # 7 lớp biển báo, format gốc
│   └── dataset_v2_clean.yaml           # Profile dữ liệu; không tự thực hiện oversampling
├── augmentation/
│   ├── no_aug.yaml                     # Không aug (Baseline)
│   ├── basic.yaml                      # Safe Brightness/Contrast (không flip)
│   └── heavy.yaml                      # Khai báo cấu hình; dataset hiện chỉ dùng brightness/contrast
├── model/
│   ├── fasterrcnn_resnet50_fpn.yaml    # Kiến trúc v1 mặc định
│   └── fasterrcnn_resnet50_fpn_v2.yaml # Anchor nhỏ, trainable FPN
├── trainer/
│   ├── gpu_fp32.yaml                   # Training chuẩn FP32
│   ├── gpu_fp16.yaml                   # AMP FP16 (cần Tensor Core)
│   └── debug_cpu.yaml                  # Profile debug; thêm --device cpu để ép CPU
└── experiment/                         # Ablation study configs
    ├── exp01_baseline.yaml
    ├── exp02_fpn_small_anchor.yaml
    └── exp03_fpn_aug.yaml
```

## Cách Sử Dụng

> **Lưu ý:** Tất cả các lệnh dưới đây đều được thực thi từ **thư mục gốc của dự án** (`sign-detection/`).

### Chạy thí nghiệm cụ thể
```bash
python faster_rcnn/train.py --config faster_rcnn/configs/experiment/exp01_baseline.yaml
python -m faster_rcnn.train --config faster_rcnn/configs/experiment/exp02_fpn_small_anchor.yaml
python faster_rcnn/train.py --config faster_rcnn/configs/experiment/exp03_fpn_aug.yaml
```

### Debug nhanh (1 batch, CPU)
```bash
python -m faster_rcnn.train --config faster_rcnn/configs/train.yaml --trainer debug_cpu --device cpu --dry-run --no-wandb
```

### Override từng tham số qua CLI
```bash
python faster_rcnn/train.py --config faster_rcnn/configs/experiment/exp03_fpn_aug.yaml \
    --trainer gpu_fp16 \
    --batch-size 4 \
    --lr 0.001
```

## Quy tắc Thiết kế

CLI hiện tại là `faster_rcnn.train` và `faster_rcnn.predict`.
Inference chỉ xuất JSON; chạy `evaluation/evaluate.py` riêng để chấm.
Danh sách tham số và trách nhiệm file: [README Faster R-CNN](../README.md).

Các cờ CLI ghi đè YAML; `--set key=value` ghi đè sau cùng, ví dụ:

```powershell
python -m faster_rcnn.train --epochs 10 --no-wandb --set "trainer.optimizer.momentum=0.8"
python -m faster_rcnn.predict --split test --set "data.conf_threshold=0.001" --out-pred preds/test.json
```

Các file YAML là cấu hình, không tự triển khai tính năng. Code hiện dùng SGD,
MultiStepLR và brightness/contrast; các khai báo oversampling hay transforms nâng cao
không tự kích hoạt thuật toán tương ứng. Best checkpoint hiện chọn bằng validation loss mỗi epoch.

| Nguyên tắc | Giải thích |
|---|---|
| **1 biến đổi / 1 file** | Mỗi experiment chỉ ghi đè tham số khác với default |
| **Đường dẫn dữ liệu** | Đường dẫn tương đối được resolve từ gốc repo do loader xác định; trường YAML project_root hiện không ghi đè gốc này |
| **experiment/ = ablation log** | File exp phản ánh chính xác 1 thay đổi để so sánh công bằng |
| **trainer/ tách hardware** | Tách cấu hình phần cứng (FP16/FP32, num_workers) ra khỏi logic model |
