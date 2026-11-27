# Phát hiện biển báo giao thông Việt Nam (Zalo AI Traffic Sign 2020)

Bài tập lớn môn **Học sâu**, Đề tài 1: so sánh và tối ưu mô hình **một giai đoạn (YOLOv8)** và **hai giai đoạn (Faster R-CNN)** cho bài toán phát hiện biển báo giao thông nhỏ trên ảnh camera hành trình Việt Nam (1622×626, 7 lớp).

## Kết quả cuối

Chấm bằng `pycocotools` trên tập test (871 ảnh, 5 tuyến đường mô hình chưa từng thấy). Tập test chỉ được chấm **một lần**, cho các mô hình cuối.

| Mô hình | mAP@0.5 val | mAP@0.5 test | mAP@0.5:0.95 test | AP_small test |
|---|---|---|---|---|
| YOLOv8s, cắt ô 640 + SAHI | **83.9** | **78.1** | **51.4** | **46.8** |
| YOLOv8s, imgsz 1280 | 82.2 | 76.8 | 50.8 | 45.1 |
| Faster R-CNN R50-FPN (exp03) | 75.2 | 69.3 | 41.8 | 35.9 |

- YOLOv8s 1280 hơn Faster R-CNN 7.5 điểm mAP@0.5. Khoảng tin cậy 95% bootstrap là [4.5, 10.2], tức có ý nghĩa thống kê.
- Cắt ô + SAHI hơn 1280 1.3 điểm, nhưng chênh lệch này không có ý nghĩa thống kê.
- Bảng đầy đủ: [reports/eval/results.md](reports/eval/results.md). Báo cáo cuối (LaTeX, template ACL): [report/main.tex](report/main.tex).

## Cấu trúc thư mục

```text
├── data/                    Nhãn COCO JSON (nhỏ, có trong repo)
│   ├── splits/              ★ train/val/test đã gộp box trùng, chia theo street_id. Mọi thí nghiệm dùng bộ này
│   ├── splits_raw/          Nhãn trước khi gộp box trùng (chỉ để đối chiếu)
│   └── leak/                Bộ chia riêng cho thí nghiệm rò rỉ dữ liệu
├── data_prep/               (A) Chuẩn bị dữ liệu: eda.py, dedup_boxes.py, make_split.py
├── evaluation/              (A) Đánh giá chung cho mọi mô hình
│   ├── evaluate.py          Chấm mAP bằng pycocotools (+ test_evaluate.py để tự kiểm tra)
│   ├── analyze_errors.py    Ma trận nhầm lẫn, recall theo kích thước và theo cụm biển
│   ├── bootstrap_ap.py      Kiểm định chênh lệch AP có ý nghĩa thống kê không
│   ├── compare_runs.py      So từng lớp giữa các thí nghiệm
│   ├── visualize.py         Vẽ nhãn và dự đoán, lọc biển nhỏ bị bỏ sót
│   ├── freeze_results.py    Chụp lại mã băm của kết quả (snapshot)
│   └── make_report_figures.py  Vẽ lại các hình của báo cáo vào report/figures/
├── yolo/                    (B) YOLOv8
│   ├── coco_to_yolo.py      Chuyển nhãn COCO sang YOLO (+ check_yolo_labels.py)
│   ├── train_yolov8.py      Train (640, 1280, P2, --cls-pw, W&B)
│   ├── predict_yolo.py      Xuất dự đoán ra COCO JSON (thường hoặc SAHI) rồi chấm
│   ├── tile_yolo_dataset.py Cắt ảnh thành ô 640×640 để train
│   └── notebooks/           Notebook chạy trọn gói trên Kaggle/Colab
├── faster_rcnn/             (C) Faster R-CNN ResNet50-FPN: train.py, evaluate.py, configs/ (xem README riêng)
├── streaming/               (C) Suy luận thời gian thực trên video, ONNX, tracker (xem README riêng)
├── experiments/             Thí nghiệm phân tích (A), mỗi thư mục có README mô tả thiết kế và kết quả
│   ├── leakage/             Rò rỉ dữ liệu: chia ngẫu nhiên và chia theo tuyến đường
│   └── imbalance/           Mất cân bằng lớp: oversampling (RFS) và trọng số lớp (cls_pw)
├── preds/                   File dự đoán <mô hình>_<val|test>.json (định dạng chung)
├── reports/                 Kết quả
│   ├── eval/                Điểm từng thí nghiệm + results.md (bảng tổng) + bootstrap_test.json
│   ├── errors/              Phân tích lỗi từng mô hình + phan_tich_loi.md (tổng hợp)
│   ├── eda/, eda_dedup/     Thống kê dữ liệu trước và sau khi gộp box trùng
│   ├── eval_leak/, imbalance/   Kết quả hai thí nghiệm phân tích
│   └── latency_fps/         Đo tốc độ YOLO và Faster R-CNN
├── report/                  Báo cáo cuối, tự chứa, tải thẳng lên Overleaf (compiler XeLaTeX)
│   ├── main.tex             Trang chính: tóm tắt + \input các mục
│   ├── sections/            intro, related, method, experiments, analysis, realtime, conclusion,
│   │                        limitations, contributions
│   ├── figures/             Hình (PDF), tạo bằng evaluation/make_report_figures.py
│   └── custom.bib, acl.sty, acl_natbib.bst
└── tests/                   Kiểm thử cho streaming
```

Không có trong repo (tải hoặc tạo lại): ảnh gốc `za_traffic_2020/`, dữ liệu YOLO `data/yolo*/`, trọng số `*.pt`.

## Tái tạo kết quả

Mọi lệnh chạy từ thư mục gốc của repo.

### 1. Cài đặt và tải dữ liệu
```bash
pip install -r requirements.txt
```
Tải bộ **Zalo AI 2020 Traffic Sign Detection** (Kaggle dataset `za-traffic-2020`), giải nén sao cho ảnh nằm ở `za_traffic_2020/traffic_train/images/`.

### 2. Gộp box trùng và chia tập
Bước này **không cần chạy lại**, vì kết quả đã có sẵn trong `data/splits/`. Các lệnh dưới đây chỉ để tái tạo.
```bash
# 17,5% box là box lặp trên cùng một biển -> gộp lại
python data_prep/dedup_boxes.py --ann za_traffic_2020/traffic_train/train_traffic_sign_dataset.json --out data/clean/train_dedup.json
# Chia 70/15/15 theo tuyến đường (street_id), phân tầng theo lớp hiếm nhất, seed 42
python data_prep/make_split.py --ann data/clean/train_dedup.json --out data/splits --group-key street_id
# Thống kê dữ liệu
python data_prep/eda.py --ann data/clean/train_dedup.json --out reports/eda_dedup
```

### 3. YOLOv8 (chạy trên GPU, ví dụ Kaggle T4; ultralytics 8.4.204)
Chạy trọn gói bằng notebook [yolo/notebooks/yolo_kaggle_colab.ipynb](yolo/notebooks/yolo_kaggle_colab.ipynb), hoặc bằng dòng lệnh:
```bash
python yolo/coco_to_yolo.py --splits data/splits --images za_traffic_2020/traffic_train/images --out data/yolo
python yolo/train_yolov8.py --data data/yolo/data.yaml --model yolov8s.pt --imgsz 1280 --epochs 50 --batch 8 --name yolov8s_1280
python yolo/predict_yolo.py --weights runs/detect/yolov8s_1280/weights/best.pt --imgsz 1280 \
    --gt data/splits/val.json --images data/yolo/images/val --out preds/yolov8s_1280_val.json --eval --name yolov8s_1280

# Cắt ô 640 + SAHI
python yolo/tile_yolo_dataset.py --splits data/splits --images za_traffic_2020/traffic_train/images --out data/yolo_tiled
python yolo/train_yolov8.py --data data/yolo_tiled/data.yaml --model yolov8s.pt --imgsz 640 --epochs 40 --batch 16 --name yolov8s_tiled_640
python yolo/predict_yolo.py --weights runs/detect/yolov8s_tiled_640/weights/best.pt --imgsz 640 \
    --images data/yolo_tiled/images/val --sahi --slice-size 640 --overlap 0.2 \
    --out preds/yolov8s_tiled_640_sahi_val.json --eval --name yolov8s_tiled_640_sahi
```

### 4. Faster R-CNN
```bash
python faster_rcnn/train.py -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml
python faster_rcnn/evaluate.py -c faster_rcnn/configs/experiment/exp03_fpn_aug.yaml --split val
```
Cách chạy trên Kaggle và ý nghĩa các tham số: [faster_rcnn/README.md](faster_rcnn/README.md).

### 5. Đánh giá và phân tích (CPU là đủ)
Mọi mô hình xuất dự đoán theo cùng một định dạng: `[{"image_id", "category_id", "bbox": [x, y, w, h], "score"}]`, tọa độ pixel trên ảnh gốc, `category_id` là id gốc COCO, xuất ở `conf=0.001`.
```bash
python evaluation/evaluate.py       --gt data/splits/val.json --pred preds/<tên>_val.json --name <tên>
python evaluation/analyze_errors.py --gt data/splits/val.json --pred preds/<tên>_val.json --name <tên>
python evaluation/bootstrap_ap.py   --gt data/splits/val.json --base <mô hình gốc> --runs <mô hình khác>
python evaluation/visualize.py      --gt data/splits/val.json --images za_traffic_2020/traffic_train/images \
    --pred preds/<tên>_val.json --only-missed-small --n 20 --out reports/missed_<tên>
```
Tập test (`data/splits/test.json`) chỉ dùng cho các mô hình cuối, mỗi mô hình chấm một lần. Đổi `_val` thành `_test` trong các lệnh trên.

### 6. Thí nghiệm phân tích và demo
- Rò rỉ dữ liệu: [experiments/leakage/README.md](experiments/leakage/README.md)
- Mất cân bằng lớp: [experiments/imbalance/README.md](experiments/imbalance/README.md)
- Suy luận thời gian thực trên video: [streaming/README.md](streaming/README.md), chạy bằng `python -m streaming.demo ...`

### 7. Báo cáo cuối
```bash
# Vẽ lại các hình (cần đã chạy analyze_errors.py trên test cho 3 mô hình cuối)
python evaluation/make_report_figures.py --images za_traffic_2020/traffic_train/images
```
Biên dịch trên Overleaf:
1. Nén thư mục `report/` thành file zip, rồi chọn **New Project → Upload Project**.
2. Vào **Menu → Compiler**, chọn **XeLaTeX** (vì báo cáo viết tiếng Việt và dùng `fontspec`).
3. Trang chính là `main.tex`.

## Phân công

| Thành viên | Phụ trách chính | Phần mở rộng |
| :--- | :--- | :--- |
| **A – Dữ liệu & Đánh giá** | Gộp box trùng, EDA, chia tập theo `street_id`, script đánh giá chung, phân tích lỗi | Rò rỉ dữ liệu, mất cân bằng lớp |
| **B – YOLOv8** | Pipeline YOLOv8 (640/1280, P2), W&B, quản lý repo | Cắt ô + SAHI |
| **C – Faster R-CNN & So sánh** | Faster R-CNN ResNet50-FPN, anchor nhỏ, so sánh tốc độ và độ chính xác | Suy luận thời gian thực trên video, demo |
