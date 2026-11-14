# Phát hiện Biển báo Giao thông Việt Nam (Zalo AI Traffic Sign 2020)

Kho mã nguồn bài tập lớn môn **Học sâu (Deep Learning)** – Đề tài 1: So sánh và tối ưu hóa mô hình **One-stage (YOLOv8)** và **Two-stage (Faster R-CNN)** cho bài toán phát hiện biển báo giao thông nhỏ trên ảnh camera hành trình Việt Nam.

## Phân công thành viên

| Thành viên | Phụ trách chính | Phần mở rộng |
| :--- | :--- | :--- |
| **A – Dữ liệu & Đánh giá** | Làm sạch dữ liệu (gộp box trùng), EDA, chia tập theo `street_id`, script đánh giá `pycocotools`, phân tích lỗi | Xử lý mất cân bằng lớp |
| **B – YOLOv8** | Pipeline YOLOv8 (`imgsz=640/1280`, kiến trúc `P2`), tích hợp Weights & Biases, quản lý repo | Tiling ảnh độ phân giải cao + suy luận cắt mảnh (SAHI) |
| **C – Faster R-CNN & So sánh** | Faster R-CNN ResNet50-FPN, tinh chỉnh anchor nhỏ, báo cáo ACL, so sánh tốc độ/độ chính xác | Suy luận thời gian thực (real-time) trên video & demo |

---

## Cấu trúc thư mục

```text
├── data/
│   ├── splits/                  # Nhãn COCO JSON đã làm sạch (train.json, val.json, test.json)
│   └── splits_raw/              # Nhãn gốc trước khi gộp 17.5% box trùng (chỉ để đối chiếu)
├── docs/
│   ├── notes_du_lieu.md         # Ghi chú phân tích ảnh, lỗi nhãn và biển đứng cụm
│   ├── proposal_du_lieu.tex     # Mục Dữ liệu (LaTeX) cho Proposal
│   ├── tom_tat_FPN.md           # Tóm tắt bài báo Feature Pyramid Networks (CVPR 2017)
│   └── tom_tat_YOLO.md          # Tóm tắt bài báo YOLO & kiến trúc YOLOv8
├── notebooks/
│   └── yolo_kaggle_colab.ipynb  # Notebook chạy trọn gói YOLOv8 (640, 1280, P2, Tiling, SAHI) trên Kaggle/Colab
├── reports/
│   ├── eda/                     # Thống kê trên nhãn gốc
│   └── eda_dedup/               # Thống kê trên nhãn đã gộp box trùng
└── scripts/
    ├── eda.py                   # Thống kê phân phối lớp và kích thước bounding box
    ├── dedup_boxes.py           # Gộp các bounding box bị gán nhãn lặp trên cùng một biển báo
    ├── make_split.py            # Chia train/val/test theo street_id và phân tầng theo lớp
    ├── coco_to_yolo.py          # Chuyển đổi nhãn COCO JSON sang định dạng Ultralytics YOLO
    ├── check_yolo_labels.py     # Vẽ kiểm tra nhãn YOLO (.txt) trên ảnh gốc
    ├── train_yolov8.py            # Huấn luyện YOLOv8 (hỗ trợ imgsz=640/1280, YOLOv8-P2, W&B)
    ├── predict_yolo.py          # Xuất dự đoán YOLO ra COCO JSON (hỗ trợ chuẩn & SAHI) và chấm điểm
    ├── tile_yolo_dataset.py     # Cắt mảnh (Tiling) ảnh 1622x626 cho tập huấn luyện YOLO
    ├── evaluate.py              # Script chấm điểm chung bằng pycocotools (mAP@0.5, mAP@0.5:0.95, AP_small)
    ├── test_evaluate.py         # Kiểm thử tự động cho evaluate.py
    └── visualize.py             # Vẽ đối chiếu nhãn thật và dự đoán, lọc biển nhỏ bị bỏ sót
```

---

## Hướng dẫn cài đặt và chạy nhanh

### 1. Cài đặt môi trường & tải dữ liệu
```bash
pip install -r requirements.txt
```
Tải bộ dữ liệu **Zalo AI 2020 Traffic Sign Detection** (Kaggle dataset `za-traffic-2020`) và giải nén vào thư mục gốc sao cho ảnh nằm tại `za_traffic_2020/traffic_train/images/`.

### 2. Chuyển đổi dữ liệu sang định dạng YOLO (Thành viên B)
```bash
python scripts/coco_to_yolo.py --splits data/splits --images za_traffic_2020/traffic_train/images --out data/yolo
python scripts/check_yolo_labels.py --yolo data/yolo --split train --n 5
```

### 3. Huấn luyện và xuất dự đoán YOLOv8
Có thể chạy trực tiếp trên **Kaggle / Google Colab** bằng notebook [`notebooks/yolo_kaggle_colab.ipynb`](notebooks/yolo_kaggle_colab.ipynb) hoặc chạy qua dòng lệnh:

```bash
# Baseline YOLOv8s ở imgsz=640
python scripts/train_yolov8.py --data data/yolo/data.yaml --model yolov8s.pt --imgsz 640 --name yolov8s_640
python scripts/predict_yolo.py --weights runs/detect/yolov8s_640/weights/best.pt \
    --imgsz 640 --out preds/yolov8s_640_val.json --eval --name yolov8s_640

# Phương pháp chính: YOLOv8s ở imgsz=1280 (hoặc thêm tầng P2 với --model yolov8s-p2.yaml --pretrained yolov8s.pt)
python scripts/train_yolov8.py --data data/yolo/data.yaml --model yolov8s.pt --imgsz 1280 --batch 8 --name yolov8s_1280
python scripts/predict_yolo.py --weights runs/detect/yolov8s_1280/weights/best.pt \
    --imgsz 1280 --out preds/yolov8s_1280_val.json --eval --name yolov8s_1280

# Phần mở rộng (Tuần 4): Suy luận cắt mảnh SAHI & Tiling
python scripts/predict_yolo.py --weights runs/detect/yolov8s_640/weights/best.pt \
    --imgsz 640 --sahi --slice-size 640 --overlap 0.2 \
    --out preds/yolov8s_640_sahi_val.json --eval --name yolov8s_640_sahi
```

### 4. Đánh giá chung cho mọi mô hình (`pycocotools`)
Mọi file dự đoán trong `preds/<ten_mo_hinh>_val.json` (từ cả YOLOv8 và Faster R-CNN) đều chấm bằng cùng một lệnh:
```bash
python scripts/evaluate.py --gt data/splits/val.json --pred preds/<file_du_doan>.json --name <ten_thi_nghiem>
```
Bảng tổng hợp tự động được cập nhật tại `reports/eval/results.md`.
