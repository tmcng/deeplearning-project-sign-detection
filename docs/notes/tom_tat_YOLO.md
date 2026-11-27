# Tóm tắt bài YOLO & YOLOv8 – trình bày cho nhóm trong 15 phút

**Bài báo gốc:** Redmon, Divvala, Girshick, Farhadi. *You Only Look Once: Unified, Real-Time Object Detection*. CVPR 2016.  
**Tài liệu kiến trúc hiện đại:** Jocher, Chaurasia, Qiu. *Ultralytics YOLOv8* (2023) & Wang et al. *YOLOv7 / CSPNet / PANet*.

Đọc bài gốc để hiểu tư tưởng **One-stage Detector**, sau đó nắm rõ thay đổi ở **YOLOv8** (phiên bản nhóm sử dụng trong bài tập lớn) để trả lời vấn đáp.

---

## 1. Tư tưởng cốt lõi: One-stage vs. Two-stage

Các mô hình hai giai đoạn (Two-stage) như **Faster R-CNN** tách bài toán thành 2 bước tuần tự:
1. **RPN (Region Proposal Network):** tìm các vùng nghi ngờ có vật thể (foreground vs. background).
2. **RoI Head:** cắt đặc trưng ở từng vùng đề xuất để phân loại lớp và tinh chỉnh tọa độ box.

**YOLO (You Only Look Once)** đặt lại toàn bộ bài toán phát hiện vật thể thành **một bài toán hồi quy duy nhất (single regression problem)**:
- Chia ảnh đầu vào thành lưới ô vuông $S \times S$ (hoặc ở nhiều tỉ lệ stride khác nhau).
- Nếu tâm của một vật thể rơi vào ô lưới nào, **ô lưới đó chịu trách nhiệm** dự đoán tọa độ bounding box và phân phối xác suất các lớp của vật thể đó.
- Chỉ đi qua mạng CNN **đúng 1 lần (one-stage)** → tốc độ suy luận rất nhanh (real-time), nhìn được toàn bộ ngữ cảnh ảnh nên ít báo nhầm nền (background false positives) hơn so với nhìn từng vùng cục bộ.

**Đánh đổi kinh điển:** YOLO đời đầu gặp khó khăn khi **nhiều vật thể nhỏ đứng sát nhau** trong cùng một ô lưới. Các phiên bản sau (YOLOv3 → YOLOv8) khắc phục bằng dự đoán đa tỉ lệ (Multi-scale FPN/PAN).

---

## 2. Kiến trúc YOLOv8 (phiên bản dùng trong đề tài)

YOLOv8 gồm 3 phần chính: **Backbone – Neck – Decoupled Head**.

| Thành phần | Cấu trúc chính | Vai trò |
| :--- | :--- | :--- |
| **Backbone** | CSPDarknet + khối **C2f** + **SPPF** | Trích xuất đặc trưng ở các mức stride 4 (C2), 8 (C3), 16 (C4), 32 (C5). Khối C2f nối tắt nhiều nhánh gradient giúp mạng nhẹ nhưng học đặc trưng tốt hơn C3 cũ. |
| **Neck** | **PAN-FPN** (Path Aggregation Network) | Kết hợp đường **Top-down (FPN)** truyền ngữ nghĩa mạnh từ tầng sâu xuống tầng nông, và đường **Bottom-up thứ hai (PAN)** truyền thông tin vị trí chính xác từ tầng nông ngược lên tầng sâu. |
| **Head** | **Decoupled Head** + **Anchor-free** | Tách riêng nhánh phân loại (Classification – BCE Loss) và nhánh hồi quy tọa độ (Regression – CIoU + DFL Loss). Mặc định xuất ở 3 tầng **P3 (stride 8), P4 (stride 16), P5 (stride 32)**. |

### Ba điểm khác biệt lớn nhất của YOLOv8 cần nhớ khi vấn đáp:
1. **Anchor-free (không dùng anchor cố định):** Thay vì đặt trước các hộp anchor với kích thước/tỉ lệ cố định rồi học độ lệch như YOLOv3/v5 hay Faster R-CNN, mỗi điểm (grid cell) trên feature map trực tiếp dự đoán khoảng cách từ tâm ô tới 4 cạnh `(trái, trên, phải, dưới)` của bounding box.
2. **Decoupled Head (đầu dự đoán tách rời):** Bài toán "vật này thuộc lớp gì" cần đặc trưng bất biến không gian, còn "viền vật nằm ở pixel nào" cần đặc trưng nhạy vị trí. Tách 2 nhánh conv riêng cho classification và regression giúp hội tụ nhanh và chính xác hơn.
3. **Hàm Loss và Gán nhãn động (Task-Aligned Assigner – TAL):**
   - Chọn mẫu dương (positive samples) động dựa trên điểm kết hợp giữa xác suất phân loại $s$ và độ chồng lấp $\text{IoU}$: $t = s^\alpha \cdot \text{IoU}^\beta$.
   - **Classification loss:** Binary Cross-Entropy (BCE) cho từng lớp.
   - **Box regression loss:** kết hợp **CIoU Loss** (tính cả diện tích chồng lấp, khoảng cách tâm và tỉ lệ khung hình) + **DFL (Distribution Focal Loss)** giúp dự đoán biên box mượt khi viền biển báo bị mờ.

---

## 3. Liên hệ trực tiếp với dữ liệu Zalo AI 2020 của nhóm

Từ số liệu EDA đã làm sạch của A (4.500 ảnh 1622×626, 9.077 box, 7 lớp):

1. **Vì sao YOLOv8 mặc định ở `imgsz=640` bỏ sót nhiều biển báo nhỏ?**
   - Trên ảnh gốc 1622×626, **77,6% box có diện tích dưới $32^2$ px** (trung vị cạnh chỉ **16,5 px**).
   - Khi resize cạnh dài từ 1622 về `imgsz=640` (tỉ lệ thu nhỏ $\approx 2,53$ lần), **68,9% biển báo chỉ còn dưới 10 px** (biển trung vị còn $\sim 6,5$ px).
   - Đầu ra nhỏ nhất của YOLOv8 chuẩn là tầng **P3 có stride = 8**. Một biển 6,5 px khi chia cho stride 8 thì kích thước trên feature map P3 **chưa tới $1 \times 1$ ô** → thông tin chi tiết bên trong biển (số giới hạn tốc độ, mũi tên cấm rẽ) gần như biến mất.

2. **Ba hướng xử lý của B trong kế hoạch thực nghiệm:**
   - **Tăng độ phân giải đầu vào (`imgsz=1280`):** Giữ ảnh gần với kích thước gốc 1622×626 hơn, giúp biển báo trên feature map P3 to gấp đôi theo chiều cạnh (gấp 4 lần diện tích).
   - **Kiến trúc `yolov8s-p2.yaml` (thêm tầng P2, stride = 4):** Kéo đường FPN/PAN xuống tận tầng C2 để tạo đầu dự đoán P2 có độ phân giải $320 \times 320$ (ở `imgsz=1280`), chuyên bắt các biển báo rất xa (6–15 px).
   - **Tiling khi huấn luyện + SAHI khi suy luận (Tuần 4):** Cắt ảnh gốc thành các ô (tile) chồng lấp nhau. Một biển 16 px trên ảnh 1622×626 khi nằm trong tile 640×640 rồi đưa vào mạng sẽ giữ nguyên kích thước 16 px thay vì bị bóp nhỏ về 6 px.

3. **Lưu ý cấu hình huấn luyện và hậu xử lý (NMS) trên dữ liệu này:**
   - **Giữ lật ngang (`fliplr=0.5`):** Lớp "Cấm rẽ" trong bộ dữ liệu đã gộp chung cấm rẽ trái và cấm rẽ phải nên lật ngang không làm sai nhãn.
   - **NMS theo từng lớp (`agnostic_nms=False`, mặc định) và giữ ngưỡng IoU NMS khoảng `0.6–0.7`:** Có 27% ảnh train và 38% ảnh val/test chứa biển đứng cụm sát nhau (hay gặp nhất là *Cấm ngược chiều + Hiệu lệnh* hoặc 3 biển *Giới hạn tốc độ* trên giá long môn). Giữ NMS theo từng lớp giúp hai biển khác lớp đứng sát nhau không xóa nhầm nhau.
   - **Xuất dự đoán ở `conf=0.001` và ánh xạ lại `class_map.json`:** YOLO đánh số lớp `0..6`, trong khi COCO JSON dùng `category_id` `1..7`. Khi xuất file JSON gửi A chấm bằng `pycocotools`, phải đổi ngược qua `class_map.json` và dùng ngưỡng `conf=0.001` để tính đầy đủ đường cong Precision–Recall.

---

## 4. Câu hỏi vấn đáp có thể gặp

1. **YOLOv8 là anchor-based hay anchor-free? Nó dự đoán box như thế nào?**  
   → Anchor-free. Tại mỗi tâm ô lưới $(c_x, c_y)$ trên feature map (P3, P4, P5), mạng dự đoán 4 khoảng cách $(l, t, r, b)$ từ tâm ô tới 4 cạnh trái, trên, phải, dưới của box, sau đó nhân với stride của tầng tương ứng để ra tọa độ pixel.
2. **Vì sao Neck của YOLOv8 dùng PAN-FPN thay vì chỉ dùng FPN như Faster R-CNN?**  
   → FPN chỉ có nhánh top-down (đưa ngữ nghĩa từ sâu xuống nông). PAN thêm một nhánh bottom-up nữa sau FPN để đưa thông tin định vị sắc nét (cạnh, góc ở tầng nông) ngược lên các tầng sâu, giúp định vị biên box chính xác hơn.
3. **Tại sao khi xuất dự đoán để tính mAP lại để `conf=0.001` mà khi chạy demo lại để `conf=0.25` hoặc `0.3`?**  
   → Chỉ số mAP tích phân diện tích dưới đường cong Precision–Recall trên mọi mức Recall từ 0 đến 1. Nếu cắt ở `0.25`, phần đuôi Recall sẽ bị cụt làm mAP thấp giả tạo. Ngược lại, khi hiển thị demo cho người xem thì cần lọc bỏ các box độ tin cậy thấp để tránh rối mắt.
4. **Nếu bật `agnostic_nms=True` trên bộ dữ liệu này thì chuyện gì xảy ra?**  
   → NMS sẽ xét chồng lấp bất kể lớp dự đoán. Với các biển đứng cụm khác lớp (như *Cấm ngược chiều* gắn sát *Hiệu lệnh*), hoặc một biển mà mô hình phân vân giữa 2 lớp tròn viền đỏ, NMS có thể triệt tiêu box đúng nếu box lớp kia có điểm cao hơn một chút.
5. **So sánh nhanh tốc độ và độ chính xác giữa YOLOv8 và Faster R-CNN trên biển báo nhỏ?**  
   → YOLOv8 (one-stage) nhanh hơn nhiều lần, phù hợp chạy real-time; tuy nhiên ở cấu hình mặc định `imgsz=640`, stride 8 của P3 làm mất nhiều biển nhỏ. Khi tăng lên `imgsz=1280`, thêm đầu P2 hoặc dùng SAHI, YOLOv8 cải thiện mạnh `AP_small` nhưng đánh đổi bằng thời gian suy luận tăng lên.

---

## BibTeX

```bibtex
@inproceedings{redmon2016you,
  title     = {You Only Look Once: Unified, Real-Time Object Detection},
  author    = {Redmon, Joseph and Divvala, Santosh and Girshick, Ross and Farhadi, Ali},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {779--788},
  year      = {2016}
}

@software{jocher2023yolo,
  author    = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  title     = {Ultralytics YOLOv8},
  version   = {8.0.0},
  year      = {2023},
  url       = {https://github.com/ultralytics/ultralytics}
}
```
