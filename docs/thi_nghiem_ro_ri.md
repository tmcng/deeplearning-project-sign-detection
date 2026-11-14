# Thí nghiệm rò rỉ dữ liệu: split ngẫu nhiên và split theo street_id

## Câu hỏi

Nếu chia ảnh ngẫu nhiên (không theo đường), val có còn đo đúng khả năng nhận biển trên đường mới không?

## Thiết kế

| | Split theo đường (`data/splits`) | Split ngẫu nhiên (`data/splits_random`) |
|---|---|---|
| Train | 2933 ảnh, 28 đường | 2933 ảnh, 33 đường |
| Val | 696 ảnh, 5 đường **không có trong train** | 696 ảnh, 32 đường **đều có trong train** |
| Test | 871 ảnh, 5 đường | **giống hệt** bên trái |

- Split ngẫu nhiên lấy chính train + val của split theo đường rồi chia lại theo từng ảnh. Hai mô hình có cùng số ảnh train và cùng test.
- Mức rò rỉ (từ `data/splits_random/split_stats.txt`): **96.1% ảnh val ngẫu nhiên có khung hình liền kề nằm trong train.**
- Hai lần train giữ nguyên mọi thứ (YOLOv8n, 50 epoch, imgsz 640, batch 16, seed 0), chỉ đổi dữ liệu.

Dự đoán:
- mAP trên val ngẫu nhiên sẽ cao hơn hẳn mAP trên val theo đường.
- Trên test (đường mới), hai mô hình sẽ gần như bằng nhau.
- Nếu đúng như vậy, split ngẫu nhiên đã đánh giá quá cao khả năng tổng quát hóa.

Test chỉ chấm ở tuần 5, cùng lúc với các mô hình cuối. Tuần 2 chỉ so hai số val.

## Chạy trên Kaggle (GPU T4 hoặc P100)

Cell 1: lấy code và cài thư viện. Nếu repo private, cần dùng token GitHub.
```bash
!git clone https://github.com/tmcng/deeplearning-project-sign-detection.git
%cd deeplearning-project-sign-detection
!pip install -q ultralytics wandb pycocotools
```

Cell 2: W&B. Thêm secret `WANDB_API_KEY` trong Add-ons > Secrets.
```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

Cell 3: chuyển hai split sang YOLO. Sửa `IMAGES` cho đúng thư mục ảnh của dataset trên Kaggle.
```bash
IMAGES=/kaggle/input/za-traffic-2020/za_traffic_2020/traffic_train/images
!python scripts/coco_to_yolo.py --splits data/splits        --images $IMAGES --out data/yolo        --copy
!python scripts/coco_to_yolo.py --splits data/splits_random --images $IMAGES --out data/yolo_random --copy
```
Kiểm tra số ảnh và số box in ra khớp với `split_stats.txt`.

Cell 4 và 5: train hai mô hình, mỗi mô hình khoảng 30 phút trên T4. Nên chạy thử `--epochs 1` trước để chắc mọi thứ chạy được.
```bash
!python scripts/train_yolo.py --yolo data/yolo        --splits data/splits        --name yolov8n_street --wandb --export-test
!python scripts/train_yolo.py --yolo data/yolo_random --splits data/splits_random --name yolov8n_random --wandb --export-test
```
`--export-test` chỉ xuất sẵn dự đoán test để tuần 5 khỏi phải train lại. Bây giờ chưa chấm file này.

Cell 6: chấm trên val của từng split.
```bash
!python scripts/evaluate.py --gt data/splits/val.json        --pred preds/yolov8n_street_val.json --name yolov8n_street --out reports/eval_ro_ri
!python scripts/evaluate.py --gt data/splits_random/val.json --pred preds/yolov8n_random_val.json --name yolov8n_random --out reports/eval_ro_ri
```

Cell 7: tải về `reports/eval_ro_ri/` và `preds/yolov8n_*` (Output > Download), rồi commit `reports/eval_ro_ri/`. File preds lớn, không cần commit.

## Kết quả

| Mô hình | mAP@0.5 val | mAP@0.5:0.95 val | mAP@0.5 test (tuần 5) |
|---|---|---|---|
| yolov8n_street | | | |
| yolov8n_random | | | |

Lưu ý khi viết vào Analysis:
- Hai tập val có phân bố lớp hơi khác nhau. "Cấm còn lại" chiếm 19% box val ngẫu nhiên và 24% box val theo đường. "Nguy hiểm" chiếm 22% và 17%.
- Val theo đường cũng có nhiều biển hơn trên mỗi ảnh (1636 so với 1419 box trên cùng 696 ảnh).
- Vì vậy chênh lệch trên val chưa hoàn toàn do rò rỉ. So trên cùng test ở tuần 5 mới là phép so sạch.
