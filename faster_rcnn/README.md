# Faster R-CNN — lệnh chạy

Chạy từ thư mục gốc repository bằng PowerShell. Không đặt khoảng trắng sau dấu backtick cuối dòng.
Cần cài đủ torch/torchvision tương thích và các dependency trước khi chạy.

## Help

```powershell
python -m faster_rcnn.train --help
python -m faster_rcnn.predict --help
python evaluation/evaluate.py --help
```

## Đường dẫn ảnh

Sửa đường dẫn này thành thư mục có đủ ảnh train/val/test. Nếu mỗi tập nằm riêng, đặt ba biến khác nhau.

```powershell
$Data_dir = "data/images"
```

Các lệnh train bên dưới dùng GPU. Nếu chạy CPU, thay `--device cuda` bằng `--device cpu`, `--mixed-precision` bằng `--no-mixed-precision`, giảm `--batch-size 1` và `--num-workers 0`.

## EXP03 — FPN + small anchors + augmentation

### Train

```powershell
python -m faster_rcnn.train `
  -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml `
  --img-dir "$TrainImages" `
  --train-json data/splits/train.json `
  --val-json data/splits/val.json `
  --device cuda `
  --epochs 30 `
  --batch-size 4 `
  --num-workers 2 `
  --mixed-precision `
  --save-best `
  --checkpoint-dir faster_rcnn/checkpoints `
  --no-wandb
```

### Predict val

```powershell
python -m faster_rcnn.predict `
  -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml `
  --weights faster_rcnn/checkpoints/fasterrcnn_fpn_aug_best.pth `
  --img-dir "$ValImages" `
  --gt data/splits/val.json `
  --split val `
  --device cuda `
  --batch-size 4 `
  --num-workers 2 `
  --conf 0.001 `
  --no-dry-run `
  --out-pred preds/exp03_fpn_aug_val.json
```

### Chấm val

```powershell
python evaluation/evaluate.py `
  --gt data/splits/val.json `
  --pred preds/exp03_fpn_aug_val.json `
  --name exp03_fpn_aug_val `
  --out reports/eval
```

### Predict test

```powershell
python -m faster_rcnn.predict `
  -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml `
  --weights faster_rcnn/checkpoints/fasterrcnn_fpn_aug_best.pth `
  --img-dir "$TestImages" `
  --gt data/splits/test.json `
  --split test `
  --device cuda `
  --batch-size 4 `
  --num-workers 2 `
  --conf 0.001 `
  --no-dry-run `
  --out-pred preds/exp03_fpn_aug_test.json
```

### Chấm test

```powershell
python evaluation/evaluate.py `
  --gt data/splits/test.json `
  --pred preds/exp03_fpn_aug_test.json `
  --name exp03_fpn_aug_test `
  --out reports/eval
```

## Chạy EXP01 hoặc EXP02

Dùng lại các lệnh EXP03 ở trên, chỉ thay các tên sau trong cả bước train, predict và chấm điểm:

| Thành phần | EXP03 (mẫu) | EXP01 | EXP02 |
| --- | --- | --- | --- |
| Config trong `configs/experiment/` | `exp03_fpn_aug.yaml` | `exp01_baseline.yaml` | `exp02_fpn_small_anchor.yaml` |
| Weights trong `faster_rcnn/checkpoints/` | `fasterrcnn_fpn_aug_best.pth` | `fasterrcnn_baseline_best.pth` | `fasterrcnn_fpn_small_anchor_best.pth` |
| Tiền tố file predictions và `--name` | `exp03_fpn_aug` | `exp01_baseline` | `exp02_fpn_small_anchor` |

Giữ hậu tố `_val` / `_test`; đường dẫn `--pred` khi chấm phải khớp `--out-pred` khi predict.
Không cần đổi tay anchor hay augmentation: YAML đã cấu hình riêng cho từng exp.
EXP01 là baseline đóng băng backbone; EXP02 mở một phần backbone và dùng anchor nhỏ; EXP03 thêm augmentation vào EXP02.

## Thử một batch train trên CPU với dữ liệu test hiện có

Chỉ kiểm tra code, không lưu checkpoint và không dùng làm kết quả thí nghiệm.

```powershell
python -m faster_rcnn.train `
  -c faster_rcnn/configs/experiment/exp01_baseline.yaml `
  --img-dir data/images `
  --train-json data/splits/test.json `
  --val-json data/splits/test.json `
  --trainer debug_cpu `
  --device cpu `
  --batch-size 1 `
  --num-workers 0 `
  --dry-run `
  --no-wandb
```

## Thử đầy đủ một epoch trên CPU với dữ liệu test hiện có

Weights thử được lưu riêng; không dùng để đánh giá độ chính xác vì train và val đều lấy từ test.

```powershell
python -m faster_rcnn.train `
  -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml `
  --img-dir data/images `
  --train-json data/splits/test.json `
  --val-json data/splits/test.json `
  --device cpu `
  --epochs 1 `
  --batch-size 1 `
  --num-workers 0 `
  --no-mixed-precision `
  --save-best `
  --checkpoint-dir faster_rcnn/checkpoints/cpu_test `
  --no-wandb
```

## Thử inference một batch trên CPU

Dùng checkpoint exp03 đã có; thay đường dẫn weights nếu lưu ở nơi khác.

```powershell
python -m faster_rcnn.predict `
  -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml `
  --weights faster_rcnn/checkpoints/fasterrcnn_fpn_aug_best.pth `
  --img-dir data/images `
  --gt data/splits/test.json `
  --split test `
  --device cpu `
  --batch-size 1 `
  --num-workers 0 `
  --dry-run `
  --out-pred preds/exp03_test_smoke.json
```

## Kiểm tra code

```powershell
python -m unittest faster_rcnn.test_pipeline
python -m compileall -q faster_rcnn
```

Best checkpoint hiện được chọn theo validation loss thấp nhất. Chấm mAP là bước riêng bằng evaluator chung.
Dùng val để so sánh thí nghiệm; chỉ chạy các lệnh test sau khi đã chốt mô hình. Không chấm mAP toàn tập bằng file smoke.
