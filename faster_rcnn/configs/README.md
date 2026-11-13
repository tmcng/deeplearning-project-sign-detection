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
│   └── dataset_v2_clean.yaml           # Đã lọc box xấu, có oversampling
├── augmentation/
│   ├── no_aug.yaml                     # Không aug (Baseline)
│   ├── basic.yaml                      # Flip + Brightness/Contrast
│   └── heavy.yaml                      # Albumentations nâng cao
├── model/
│   ├── fasterrcnn_resnet50_fpn.yaml    # Kiến trúc v1 mặc định
│   └── fasterrcnn_resnet50_fpn_v2.yaml # Anchor nhỏ, trainable FPN
├── trainer/
│   ├── gpu_fp32.yaml                   # Training chuẩn FP32
│   ├── gpu_fp16.yaml                   # AMP FP16 (cần Tensor Core)
│   └── debug_cpu.yaml                  # Dry-run 1 batch, không W&B
└── experiment/                         # Ablation study configs
    ├── exp01_baseline.yaml
    ├── exp02_fpn_small_anchor.yaml
    └── exp03_fpn_aug.yaml
```

## Cách Sử Dụng

### Chạy thí nghiệm cụ thể
```bash
python train.py --config configs/experiment/exp01_baseline.yaml
python train.py --config configs/experiment/exp02_fpn_small_anchor.yaml
python train.py --config configs/experiment/exp03_fpn_aug.yaml
```

### Debug nhanh (1 batch, CPU)
```bash
python train.py --config configs/train.yaml --trainer debug_cpu
```

### Override từng tham số qua CLI
```bash
python train.py --config configs/experiment/exp03_fpn_aug.yaml \
    --trainer gpu_fp16 \
    --paths server_gpu
```

## Quy tắc Thiết kế

| Nguyên tắc | Giải thích |
|---|---|
| **1 biến đổi / 1 file** | Mỗi experiment chỉ ghi đè tham số khác với default |
| **Không hard-code path** | Tất cả path dùng `project_root` relative |
| **experiment/ = ablation log** | File exp phản ánh chính xác 1 thay đổi để so sánh công bằng |
| **trainer/ tách hardware** | Tách cấu hình phần cứng (FP16/FP32, num_workers) ra khỏi logic model |

