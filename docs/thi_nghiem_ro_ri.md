# Thí nghiệm rò rỉ dữ liệu: chia ngẫu nhiên theo ảnh và chia theo street_id

## Câu hỏi

Nếu chia ảnh ngẫu nhiên (không theo đường), tập val có còn đo đúng khả năng nhận biển trên đường mới không?

Ảnh trong bộ dữ liệu là các khung hình liên tiếp quay trên cùng một con đường. Chia ngẫu nhiên thì hai khung hình gần như giống hệt nhau có thể rơi vào cả train lẫn val. Khi đó mô hình chỉ cần nhớ chứ không cần tổng quát hóa.

## Thiết kế (`scripts/make_leak_splits.py`, dữ liệu trong `data/leak/`)

- **Kho ảnh** là `data/splits/train.json` (2933 ảnh, 28 đường). Kho này được chia lại theo hai cách, cùng tỉ lệ:
  - `random`: chia ngẫu nhiên từng ảnh;
  - `street`: chia theo `street_id`, cả con đường nằm trọn ở train' hoặc val'.
- **Tập kiểm tra chung** là `data/splits/val.json`: 696 ảnh, 5 đường mà cả hai mô hình chưa từng thấy. Tập này được ghi thành `test.json` trong mỗi thư mục.
- **Không dùng `data/splits/test.json`.** Tập test thật vẫn để dành, chỉ chấm một lần ở tuần 5.

| | `street` | `random` |
|---|---|---|
| Train' | 1994 ảnh, 23 đường | 1994 ảnh, 28 đường |
| Val' | 939 ảnh, 5 đường **không có trong train'** | 939 ảnh, 28 đường **đều có trong train'** |
| Ảnh val' có khung hình liền kề trong train' | 0 / 939 | **827 / 939 (88.1%)** |
| Tập kiểm tra chung | 696 ảnh, 5 đường mới | giống hệt bên trái |

- **Train:** `notebooks/kaggle_leakage.ipynb`, Kaggle GPU.
  - Hai mô hình dùng cùng cấu hình: YOLOv8n, imgsz 1280, 50 epoch (patience 15), seed 0, `deterministic=True`. Chỉ khác cách chia dữ liệu.
  - Xuất dự đoán bằng `scripts/export_yolo_preds.py` (`conf=0.001`), chấm bằng `scripts/evaluate.py`.
- **Kết quả:**
  - Điểm số: `reports/eval_leak/`.
  - Dự đoán: `preds/leak_*.json`.
  - Số liệu chia tập: `data/leak/leak_stats.txt`.

## Kết quả

| Cách chia | mAP@0.5 val' | mAP@0.5 đường mới | Chênh lệch | mAP@0.5:0.95 val' | mAP@0.5:0.95 đường mới |
|---|---|---|---|---|---|
| `random` | **78.5** | 72.8 | **+5.7** | 50.7 | 48.5 |
| `street` | 62.9 | 69.9 | −7.0 | 39.9 | 47.0 |

## Kết luận

- **Chia ngẫu nhiên làm val đánh giá cao hơn thực tế.**
  - Trên val', mô hình `random` hơn mô hình `street` tới 15.6 điểm mAP@0.5.
  - Trên cùng tập đường mới, nó chỉ hơn 2.9 điểm.
  - Vậy phần lớn khoảng cách trên val' là ảo, do 88% ảnh val' có khung hình gần như giống hệt trong train'.
- **Nếu tin val' ngẫu nhiên, ta đánh giá cao hơn 5.7 điểm** so với khả năng thật trên đường mới (78.5 so với 72.8).
- **Phần hơn 2.9 điểm còn lại trên đường mới** có thể vì mô hình `random` được train trên cả 28 đường, còn mô hình `street` chỉ thấy 23 đường. Thấy nhiều đường hơn thì dữ liệu đa dạng hơn.
- **Val' của `street` thấp hơn đường mới (62.9 so với 69.9).** Nghĩa là 5 đường trong val' khó hơn 5 đường trong tập kiểm tra chung. Mỗi tập chỉ có 5 đường, nên kết quả phụ thuộc nhiều vào việc đường nào rơi vào tập nào. Điều này nên ghi vào Limitations.
- **Lưu ý khi so hai cột val'.** Hai tập val' có phân bố lớp khác nhau: val' của `street` có 780/1636 box là Nguy hiểm, val' của `random` chỉ có 414/1805. Vì vậy so trên cùng tập đường mới mới là phép so sạch.
- **Hệ quả cho cả dự án:** chia theo `street_id` là bắt buộc. Mọi kết quả chính của nhóm đều dùng `data/splits` (chia theo đường).
