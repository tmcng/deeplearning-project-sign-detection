# Module Real-Time Video Streaming & Telemetry Benchmark

Hệ sinh thái suy luận thời gian thực (Real-time Video Inference), bám đuôi đối tượng (IoU Tracking) và đo đạc hiệu năng (Auditable Telemetry) cho bài toán **Nhận diện biển báo giao thông trên Camera hành trình (DashCam)**.

Được thiết kế và tái cấu trúc theo nguyên lý **Clean Architecture** và các mẫu thiết kế chuẩn công nghiệp (**Strategy Pattern**, **Factory Pattern**, **Pipeline Pattern**, **Decoupled Telemetry & Visualization**).

---

## 📑 Mục Lục
1. [Kiến trúc hệ sinh thái (Architecture)](#1-kiến-trúc-hệ-sinh-thái)
2. [Bảng kết quả tối ưu hóa hiệu năng (Optimization Ladder)](#2-bảng-kết-quả-tối-ưu-hóa-hiệu-năng)
3. [Hướng dẫn chạy trên Local (Laptop GPU RTX 3050 / Windows)](#3-hướng-dẫn-chạy-trên-local-laptop-rtx-3050)
4. [Hướng dẫn chạy trên Kaggle Notebook (GPU T4 / P100)](#4-hướng-dẫn-chạy-trên-kaggle-notebook)
5. [Xuất & Chạy với ONNX Runtime (DirectML / TensorRT)](#5-xuất--chạy-với-onnx-runtime)
6. [Tối ưu hóa suy luận nâng cao với TensorRT](#6-tối-ưu-hóa-suy-luận-nâng-cao-với-tensorrt)
7. [Mô phỏng luồng Camera RTSP qua MediaMTX (Thời gian thực)](#7-mô-phỏng-luồng-camera-rtsp-qua-mediamtx)
8. [Bảng tra cứu tham số dòng lệnh CLI](#8-bảng-tra-cứu-tham-số-dòng-lệnh-cli)

---

## 1. Kiến trúc hệ sinh thái

```
streaming/
├── __init__.py           # Package exports & clean public APIs
├── config.py             # Type-safe Dataclasses (PipelineConfig, Detection, FrameMetrics)
├── tracker.py            # Greedy Class-aware IoU Tracker (Decoupled tracking logic)
├── visualizer.py         # StreamVisualizer (HUD telemetry, Bounding Box, Tracking ID, Khung ROI)
├── metrics.py            # MetricsCollector (Thu thập latency, tính toán mean/p50/p95, xuất CSV/JSON)
├── pipeline.py           # StreamingInferencePipeline điều phối toàn bộ chu trình xử lý
├── export_onnx.py        # Tiện ích CLI chuyển đổi mô hình Faster R-CNN và YOLO sang ONNX
├── backends/             # Strategy Pattern cho các engine suy luận
│   ├── base.py           # Interface trừu tượng BaseDetector
│   ├── factory.py        # DetectorFactory tự động định tuyến backend theo file & tham số
│   ├── pytorch_frcnn.py  # Faster R-CNN PyTorch (hỗ trợ CUDA, FP16 AMP, torch.compile)
│   ├── yolo_backend.py   # YOLOv8 Backend (hỗ trợ .pt, .onnx, TensorRT .engine)
│   └── onnx_backend.py   # ONNX Runtime (hỗ trợ TensorRT, CUDA, DirectML cho Windows RTX 3050)
├── inference.py          # Production CLI Entrypoint (python -m streaming.inference)
├── __main__.py           # Package-level executable (python -m streaming)
├── demo.py               # Backward-compatibility wrapper (python -m streaming.demo)
└── README.md             # Tài liệu đặc tả kỹ thuật
```

---

## 2. Bảng kết quả tối ưu hóa hiệu năng

Toàn bộ các phép đo được thực hiện trên cùng 300 khung hình video dashcam chuẩn (`data/DashCam/video_da_cat.mp4` hoặc `data/DashCam/traffic.mp4`, độ phân giải 1280×720), tự động loại trừ 10 khung hình khởi động (warmup):

| Cấp độ | Môi trường / Thiết bị | Chiến lược tối ưu hóa | Infer FPS | Pipeline FPS | Độ trễ (Latency/frame) | Tăng tốc (Speedup) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | **CPU** *(Intel 4 cores)* | Baseline (FP32, Full-frame 1280×720) | `0.21 FPS` | `0.21 FPS` | ~4.912 ms | **1.0x** *(Gốc)* |
| **1** | **GPU T4** *(Kaggle Cloud)* | Chuyển dịch lên GPU (FP32, Full-frame) | `4.93 FPS` | `4.74 FPS` | ~202.8 ms | **23.5x** |
| **2** | **GPU T4** *(Kaggle Cloud)* | **FP16 AMP + Cắt nắp ca-pô** (`hood-cut`) | `10.01 FPS` | `9.20 FPS` | ~99.9 ms | **47.6x** *(x2.0 lần)* |
| **3** | **GPU T4** *(Kaggle Cloud)* | **FP16 AMP + Quét tập trung lề phải** (`right`) | **`13.11 FPS`** | **`11.86 FPS`** | **~76.3 ms** | **62.4x** *(x2.7 lần)* |

### 💡 Luận điểm khoa học chính:
1. **Domain Knowledge (Kiến thức miền):** Biển báo tại Việt Nam không bao giờ nằm ở nắp ca-pô xe hay sát gầm xe (chiếm 25% phía dưới khung hình). Việc cắt bỏ vùng dư thừa giúp giảm 50% số proposals của RPN và giảm tải tính toán cho RoIAlign.
2. **Bảo toàn góc nhìn toàn cảnh:** Dù mô hình chỉ inference trên vùng ROI, hệ thống tự động bù tọa độ offset để vẽ bounding box lên toàn bộ video góc rộng 1280×720.
3. **Khai thác phần cứng:** Kích hoạt nhân Tensor Cores thông qua FP16 AMP giúp tăng tốc ma trận mà không làm suy giảm độ chính xác mAP.

---

## 3. Hướng dẫn chạy trên Local (Laptop RTX 3050)

### Bước 1: Cài đặt thư viện
```powershell
pip install -r requirements.txt
# Đảm bảo PyTorch đã nhận GPU NVIDIA RTX 3050:
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Bước 2: Chạy Demo trực tiếp (Mở cửa sổ UI màn hình)
Bỏ cờ `--headless` để hiển thị cửa sổ trực quan:

```powershell
python -m streaming.demo `
  --model faster-rcnn `
  --weights faster_rcnn/checkpoints/fasterrcnn_fpn_aug_best.zip `
  --config faster_rcnn/configs/experiment/exp03_fpn_aug.yaml `
  --source data/DashCam/video_da_cat.mp4 `
  --device cuda:0 `
  --fp16 `
  --roi-preset right `
  --track `
  --save-video `
  --output-dir demo_laptop_rtx3050
```

* **Thao tác:** Nhấn phím **`q`** trên cửa sổ video để dừng demo bất kỳ lúc nào.
* **Kết quả:** Video được lưu tại `demo_laptop_rtx3050/demo.mp4`, thống kê độ trễ lưu tại `summary.json` và `timings.csv`.

---

## 4. Hướng dẫn chạy trên Kaggle Notebook

### Bước 1: Clone repo và pull nhánh `faster-rcnn`
Trong cell đầu tiên của notebook Kaggle (bật GPU T4 hoặc P100):
```bash
!git clone https://github.com/tmcng/deeplearning-project-sign-detection.git
%cd deeplearning-project-sign-detection
!git checkout faster-rcnn
!git pull origin faster-rcnn
```

### Bước 2: Chuẩn hóa weights từ Kaggle Input
Do cơ chế auto-unzip của Kaggle Datasets với file `.zip`, bạn dùng cell Python sau để chuẩn hóa weights vào `faster_rcnn/checkpoints/`:

```python
import os, zipfile, torch
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
DST_DIR = Path("faster_rcnn/checkpoints")
DST_DIR.mkdir(parents=True, exist_ok=True)

for name in ["fasterrcnn_baseline_best1", "fasterrcnn_fpn_small_anchor_best", "fasterrcnn_fpn_aug_best"]:
    target = [p for p in KAGGLE_INPUT.rglob("version") if name in p.as_posix()]
    if target:
        real_dir = target[0].parent
        out_zip = DST_DIR / f"{name}.zip"
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as z:
            for root, dirs, files in os.walk(real_dir):
                for f in files:
                    fp = Path(root) / f
                    z.write(fp, "archive/" + fp.relative_to(real_dir).as_posix())
        print(f"✓ Đã chuẩn hóa: {out_zip.name}")
```

### Bước 3: Chạy Benchmark & Xuất Video

#### A. Chạy với Faster R-CNN:
```bash
!python -m streaming.demo \
  --model faster-rcnn \
  --weights faster_rcnn/checkpoints/fasterrcnn_fpn_aug_best.zip \
  --config faster_rcnn/configs/experiment/exp03_fpn_aug.yaml \
  --source /kaggle/input/datasets/hpwj123/video1/traffic.mp4 \
  --device cuda:0 \
  --fp16 \
  --roi-preset right \
  --headless \
  --save-video \
  --track \
  --output-dir run_kaggle_gpu_opt
```

#### B. Chạy với YOLO (Checkpoint PyTorch `.pt`):
```bash
!python -m streaming.demo \
  --model yolo \
  --weights faster_rcnn/checkpoints/best.pt \
  --source /kaggle/input/datasets/hpwj123/video1/traffic.mp4 \
  --device cuda:0 \
  --fp16 \
  --headless \
  --save-video \
  --track \
  --output-dir run_kaggle_yolo_opt
```

### Bước 4: Xem trực tiếp video trong Notebook
```python
from IPython.display import Video
Video("run_kaggle_gpu_opt/demo.mp4", embed=True, width=700)
```

---

## 5. Xuất & Chạy với ONNX Runtime

Mô hình sau khi xuất sang ONNX đã được "nướng cứng" (baked-in) toàn bộ cấu trúc mạng và tham số, do đó khi suy luận **không cần mang theo file config YAML** và có thể chạy độc lập tốc độ cao trên ONNX Runtime.

### Bước 1: Xuất checkpoint sang định dạng ONNX

Thư mục lưu mô hình ONNX quy chuẩn: `faster_rcnn/checkpoints/to_onnx/`

#### A. Xuất mô hình Faster R-CNN:
```powershell
python -m streaming.export_onnx `
  --config faster_rcnn/configs/experiment/exp03_fpn_aug.yaml `
  --weights faster_rcnn/checkpoints/fasterrcnn_fpn_aug_best.zip `
  --output faster_rcnn/checkpoints/to_onnx/fasterrcnn_fpn_aug.onnx
```

#### B. Xuất mô hình YOLO:
Bạn có thể xuất mô hình YOLO (`.pt`) sang ONNX theo một trong các cách sau:

* **Cách 1: Sử dụng tiện ích đồng bộ của module `streaming`:**
  ```powershell
  python -m streaming.export_onnx `
    --model yolo `
    --weights faster_rcnn/checkpoints/best.pt `
    --output faster_rcnn/checkpoints/to_onnx/yolo.onnx `
    --opset 14 `
    --imgsz 640
  ```

* **Cách 2: Sử dụng trực tiếp CLI của Ultralytics:**
  ```powershell
  yolo export model=faster_rcnn/checkpoints/best.pt format=onnx opset=14 imgsz=640 dynamic=True
  ```

* **Cách 3: Chạy script Python ngắn:**
  ```python
  from ultralytics import YOLO
  model = YOLO("faster_rcnn/checkpoints/best.pt")
  model.export(format="onnx", opset=14, dynamic=True, imgsz=640)
  ```

> 💡 **Giải thích tham số xuất ONNX:**
> * `--opset 14`: Phiên bản tập toán tử ONNX chuẩn. Mức 14 hỗ trợ đầy đủ toán tử Detection/NMS hiện đại và đạt độ tương thích cao nhất trên ONNX Runtime, DirectML (Windows RTX 3050) và TensorRT.
> * `--imgsz 640`: Kích thước cạnh ảnh đầu vào ($640 \times 640$ pixels). Đây là độ phân giải chuẩn của YOLO, cân bằng tối ưu giữa tốc độ thời gian thực (FPS cao, độ trễ thấp) và độ chính xác mAP.

### Bước 2: Chạy suy luận qua ONNX Runtime
Backend ONNX Runtime tự động ưu tiên: `TensorrtExecutionProvider` ➔ `CUDAExecutionProvider` ➔ `DmlExecutionProvider` (DirectML cho Windows RTX 3050):

* **Chạy với Faster R-CNN ONNX:**
  ```powershell
  python -m streaming.demo `
    --model onnx `
    --weights faster_rcnn/checkpoints/to_onnx/fasterrcnn_fpn_aug.onnx `
    --source data/DashCam/video_da_cat.mp4 `
    --device cuda:0 `
    --track `
    --output-dir demo_onnx_faster_rcnn
  ```

* **Chạy với YOLO ONNX:**
  ```powershell
  python -m streaming.demo `
    --model yolo `
    --weights faster_rcnn/checkpoints/to_onnx/yolo.onnx `
    --source data/DashCam/video_da_cat.mp4 `
    --device cuda:0 `
    --track `
    --output-dir demo_onnx_yolo
  ```

---

## 6. Tối ưu hóa suy luận nâng cao với TensorRT

Đối với phần cứng GPU NVIDIA có nhân Tensor Cores, TensorRT là giải pháp tăng tốc phần cứng đạt hiệu năng cao nhất hiện nay.

### Chạy suy luận YOLO với TensorRT Engine (`.engine`):
```powershell
# Chạy trực tiếp qua streaming demo:
python -m streaming.demo `
  --model yolo `
  --weights faster_rcnn/checkpoints/best.engine `
  --source data/DashCam/video_da_cat.mp4 `
  --device cuda:0 `
  --output-dir demo_yolo_trt
```
*(Cách xuất file `.engine` từ checkpoint YOLO: `yolo export model=faster_rcnn/checkpoints/best.pt format=engine device=0`)*

### 💡 Tại sao TensorRT Engine (.engine) không áp dụng trực tiếp được cho Faster R-CNN?
* **Kiến trúc Two-Stage phức tạp và luồng điều khiển động (Data-dependent dynamic control flow):** Faster R-CNN chứa tầng RPN (Region Proposal Network) sinh ra số lượng hộp ứng viên biến thiên liên tục tùy vào độ phức tạp của từng khung hình. Sau đó là các phép toán hình học động (`RoIAlign` / `RoIPooling`) và thuật toán lọc trùng lặp đa lớp `batched_nms` viết bằng các custom C++ kernel của `torchvision`.
* **Giới hạn của TensorRT Engine:** TensorRT được thiết kế để tối ưu hóa đồ thị tĩnh (Static Graph Optimization: gộp kernel, cấp phát tĩnh trước vùng nhớ VRAM). TensorRT thuần không thể biên dịch trực tiếp các custom C++ ops của `torchvision` nếu không tự viết riêng các C++ TensorRT Plugin phức tạp (như `EfficientNMS_TRT`, `ROIAlign_TRT`).
* **Ưu thế của YOLO:** YOLO là kiến trúc Single-Stage dạng đồ thị tensor thuần nhất (End-to-End Tensor). Dữ liệu chảy thẳng qua các tầng tích chập từ ảnh đầu vào $640 \times 640 \times 3$ đến ma trận đầu ra cố định $[1, 84, 8400]$, không phụ thuộc vào dynamic proposals hay custom C++ kernels, do đó Ultralytics có thể tự động build sang TensorRT `.engine` hoàn toàn tự động chỉ với 1 lệnh.
* *(Giải pháp cho Faster R-CNN trên Windows GPU NVIDIA là xuất sang **ONNX** và chạy qua **ONNX Runtime (DirectML / CUDA)** như đã hướng dẫn ở Phần 5).*

---

## 7. Mô phỏng luồng Camera RTSP qua MediaMTX (Thời gian thực)

Dành cho kịch bản kiểm tra khả năng truyền dẫn mạng thời gian thực qua giao thức RTSP:

1. **Terminal 1 - Khởi động RTSP Server:**
   ```powershell
   D:\Downloads\mediamtx_v1.21.3_windows_amd64\mediamtx.exe
   ```

2. **Terminal 2 - Phát lặp video qua RTSP:**
   ```powershell
   ffmpeg -re -stream_loop -1 -i data/DashCam/video_da_cat.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/live/traffic
   ```

3. **Terminal 3 - Chạy hệ thống suy luận thời gian thực với mô hình ONNX:**
   * **Với Faster R-CNN (Chạy qua ONNX Runtime - DirectML / CUDA):**
     ```powershell
     python -m streaming.demo `
       --model onnx `
       --weights faster_rcnn/checkpoints/to_onnx/fasterrcnn_fpn_aug.onnx `
       --source rtsp://127.0.0.1:8554/live/traffic `
       --device cuda:0 `
       --track `
       --output-dir demo_rtsp_live_faster_rcnn
     ```
   * **Với YOLO (Chạy qua YOLO ONNX Engine):**
     ```powershell
     python -m streaming.demo `
       --model yolo `
       --weights faster_rcnn/checkpoints/to_onnx/yolo.onnx `
       --source rtsp://127.0.0.1:8554/live/traffic `
       --device cuda:0 `
       --track `
       --output-dir demo_rtsp_live_yolo
     ```

   > 💡 **Ghi chú:** Khi chạy với RTSP stream thực tế, sử dụng định dạng **ONNX** là phương án tối ưu nhất vì mô hình đã được tối ưu hóa đồ thị tính toán, không cần nạp PyTorch config cồng kềnh, giúp giảm thiểu tối đa độ trễ (latency) và hiện tượng drop frame qua đường truyền mạng. (Nếu muốn kiểm thử với checkpoint PyTorch gốc, bạn có thể truyền `--weights faster_rcnn/checkpoints/fasterrcnn_fpn_aug_best.zip` kèm `--config`).

---

## 8. Bảng tra cứu tham số dòng lệnh CLI

| Tham số | Mặc định | Ý nghĩa & Hướng dẫn |
| :--- | :---: | :--- |
| `--model` | *(bắt buộc)* | Loại kiến trúc: `faster-rcnn`, `yolo`, hoặc `onnx` |
| `--weights` | *(bắt buộc)* | Đường dẫn file trọng số: `.zip`, `.pth`, `.pt`, `.onnx`, `.engine` |
| `--config` | `None` | File cấu hình YAML bắt buộc đối với Faster R-CNN |
| `--source` | `rtsp://...` | Nguồn video: file `.mp4` cục bộ hoặc URL luồng `rtsp://` |
| `--device` | `cpu` | Thiết bị chạy: `cpu` hoặc `cuda:0` |
| `--fp16` | `False` | Bật bán chính xác Half-Precision (FP16 AMP) kích hoạt Tensor Cores |
| `--compile` | `False` | Bật JIT Graph Compilation của PyTorch 2.x (`torch.compile`) |
| `--roi-preset` | `none` | Chọn vùng cắt tối ưu: `right` (quét lề phải) hoặc `hood-cut` (bỏ ca-pô) |
| `--roi` | `None` | Tùy chỉnh vùng cắt tự do: `x1 y1 x2 y2` (dải giá trị `0.0` đến `1.0`) |
| `--track` | `False` | Bật thuật toán gán ID bám đuôi đối tượng (IoU Tracking) |
| `--save-video` | `False` | Ghi luồng kết quả ra file `demo.mp4` trong thư mục đầu ra |
| `--headless` | `False` | Tắt cửa sổ giao diện OpenCV (bắt buộc khi chạy trên Colab/Kaggle) |
| `--conf` | `0.3` | Ngưỡng tin cậy confidence threshold để lọc bounding box |
| `--max-frames` | `300` | Số khung hình tối đa xử lý |
| `--warmup` | `10` | Số khung hình khởi động loại trừ khỏi thống kê đo đạc |
| `--output-dir` | *(bắt buộc)* | Thư mục lưu kết quả (bắt buộc là thư mục mới hoặc rỗng để bảo toàn log) |
