# Tóm tắt bài FPN – trình bày cho nhóm trong 15 phút

**Bài báo:** Lin, Dollár, Girshick, He, Hariharan, Belongie. *Feature Pyramid Networks for Object Detection*. CVPR 2017.

Đọc bài gốc trước, dùng file này để ôn và trình bày. Tập trung vào Hình 1, Hình 3 và mục 3–4 của bài.

---

## 1. Vấn đề: vì sao CNN phát hiện vật nhỏ kém

CNN như ResNet giảm dần kích thước feature map qua các tầng:

| Tầng | Stride (ảnh bị chia) | Độ phân giải | Ngữ nghĩa |
| --- | --- | --- | --- |
| C2 | 4 | cao | yếu (cạnh, góc) |
| C3 | 8 | | |
| C4 | 16 | | |
| C5 | 32 | thấp | mạnh (biết "đây là biển báo") |

Detector cũ chỉ dùng tầng cuối (C5 hoặc C4). Một biển 16 px ở stride 32 chỉ chiếm **nửa ô** feature map, nên thông tin gần như biến mất. Còn các tầng nông giữ được chi tiết thì lại "chưa hiểu" đó là vật gì.

Cách cũ để khắc phục là **image pyramid**: chạy mô hình trên nhiều kích thước ảnh. Cách này chính xác nhưng chậm gấp nhiều lần.

## 2. Ý tưởng FPN

Lấy ngữ nghĩa mạnh từ tầng sâu "đổ ngược" xuống các tầng nông có độ phân giải cao. Chi phí thêm rất nhỏ.

Ba thành phần (vẽ lại Hình 3 khi trình bày):

1. **Bottom-up:** đường đi bình thường của ResNet, lấy C2–C5.
2. **Top-down:** bắt đầu từ C5, **phóng to ×2** (nearest neighbor) để về kích thước tầng dưới.
3. **Lateral connection:** C_i qua conv 1×1 để về 256 kênh, rồi **cộng từng phần tử** với bản phóng to từ trên xuống. Sau đó conv 3×3 để giảm nhiễu răng cưa do phóng to.

Kết quả: P2–P5, **tầng nào cũng 256 kênh và đều có ngữ nghĩa mạnh**, nhưng giữ độ phân giải riêng. (Thêm P6 cho RPN bằng cách lấy mẫu thưa P5.)

## 3. Dùng FPN trong Faster R-CNN

- **RPN:** mỗi tầng chỉ dùng anchor một kích thước, tầng nông cho vật nhỏ, tầng sâu cho vật lớn: P2 → 32², P3 → 64², P4 → 128², P5 → 256², P6 → 512²; mỗi kích thước có 3 tỉ lệ khung (1:2, 1:1, 2:1).
- **RoI head:** mỗi đề xuất được gán về tầng phù hợp với kích thước của nó:

  k = ⌊k0 + log₂(√(wh) / 224)⌋, với k0 = 4

  Box càng nhỏ thì k càng nhỏ, tức lấy đặc trưng từ tầng độ phân giải cao hơn.

Kết quả chính của bài: FPN cải thiện rõ trên COCO so với chỉ dùng một tầng, và **cải thiện nhiều nhất với vật thể nhỏ**.

## 4. Liên quan trực tiếp đến nhóm mình (phần quan trọng nhất khi trình bày)

**Với C (Faster R-CNN):** `fasterrcnn_resnet50_fpn` của torchvision dùng đúng các anchor trên, nhỏ nhất là **32 px**. Biển báo của mình có trung vị cạnh **16 px**, nhỏ hơn cả anchor nhỏ nhất. Đây là lý do việc tuần 3 của C là **giảm kích thước anchor** (ví dụ bắt đầu từ 8 hoặc 16) và/hoặc tăng kích thước ảnh đầu vào.

**Với B (YOLOv8):** neck của YOLOv8 cũng là một dạng pyramid (kiểu PAN, có thêm đường bottom-up lần hai), nhưng đầu ra chỉ dùng **P3–P5 (stride 8, 16, 32), không có P2**. Ở `imgsz=640`, biển trung vị còn khoảng 6 px, tức **chưa tới 1 ô** ở stride 8. Hai hướng cho B:
- tăng `imgsz=1280` (biển to gấp đôi trên feature map);
- dùng cấu hình `yolov8-p2.yaml` của Ultralytics, thêm đầu ra ở P2 (stride 4) dành riêng cho vật nhỏ.

**Với cả nhóm:** tiling (phần mở rộng của B) là một cách làm "image pyramid" thủ công: cắt ảnh để biển trông to hơn so với khung hình.

## 5. Câu hỏi vấn đáp có thể gặp

1. Vì sao không dùng luôn C2 để phát hiện vật nhỏ? → C2 có độ phân giải cao nhưng ngữ nghĩa yếu; FPN bổ sung ngữ nghĩa từ trên xuống.
2. Vì sao dùng phép cộng chứ không nối kênh (concat)? → Giữ số kênh cố định 256 ở mọi tầng, để dùng chung một head cho tất cả các tầng.
3. Conv 1×1 ở lateral connection để làm gì? → Đưa số kênh của C_i (256–2048) về 256 cho khớp với nhánh top-down.
4. FPN khác image pyramid ở đâu? → Image pyramid chạy lại cả mạng trên nhiều kích thước ảnh (chậm); FPN chỉ chạy mạng một lần và tạo pyramid trên feature map (nhanh).
5. Một biển 20×20 px sẽ được RoI head lấy đặc trưng ở tầng nào? → k = ⌊4 + log₂(20/224)⌋ = ⌊4 − 3,49⌋ = 0, bị chặn về tầng thấp nhất là P2.

## BibTeX

```bibtex
@inproceedings{lin2017feature,
  title     = {Feature Pyramid Networks for Object Detection},
  author    = {Lin, Tsung-Yi and Doll{\'a}r, Piotr and Girshick, Ross and He, Kaiming and Hariharan, Bharath and Belongie, Serge},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {2117--2125},
  year      = {2017}
}
```
