# Báo cáo bổ sung: Faster R-CNN, đo tốc độ và demo

Phạm vi thực hiện: Faster R-CNN; phần chạy YOLO do thành viên phụ trách YOLO bàn giao. Không chặn bàn giao Faster R-CNN vì chưa có checkpoint YOLO. Trạng thái: bản thảo dựa trên kết quả validation đã có; chưa đóng băng kết quả hiệu năng cuối cùng. Số liệu độ chính xác dưới đây lấy từ JSON trong `reports/eval`, không phải kết quả chạy lại trong phiên này. FPS phải được bổ sung từ trace chạy thật với đúng checkpoint và cấu hình.

## Abstract

Nghiên cứu khảo sát phát hiện bảy nhóm biển báo giao thông Việt Nam từ ảnh camera hành trình, tập trung vào khó khăn do vật thể nhỏ. Hai họ mô hình YOLOv8 và Faster R-CNN được so sánh bằng các chỉ số COCO trên tập validation của dự án. Với Faster R-CNN, chuỗi thí nghiệm thay đổi cấu hình fine-tuning, anchor, kích thước ảnh và tăng cường độ sáng/độ tương phản cải thiện mAP@0.5:0.95 từ 41,6% lên 48,6% và AP_small từ 33,5% lên 41,3%. Cấu hình YOLOv8s huấn luyện trên ảnh cắt và suy luận SAHI đạt mAP@0.5:0.95 57,9% trong các báo cáo hiện có. Một pipeline demo đọc video hoặc RTSP được dùng để quan sát phát hiện trên chuỗi ảnh và thu trace thời gian. Chưa có đủ phép đo tốc độ đã kiểm chứng để kết luận mô hình đáp ứng thời gian thực; kết luận độ chính xác hiện giới hạn ở tập validation.

## Introduction

Biển báo trong ảnh camera hành trình thường chỉ chiếm một phần nhỏ khung hình, trong khi nền đường phố và điều kiện chiếu sáng thay đổi. Việc phát hiện biển báo vì vậy cần đồng thời định vị chính xác vật thể nhỏ và giữ chi phí suy luận phù hợp với tốc độ video. Dự án nghiên cứu đầu ra gồm hộp bao, lớp và độ tin cậy cho bảy nhóm biển báo; chưa đánh giá hệ thống ra quyết định điều khiển phương tiện.

Mục tiêu thực nghiệm gồm so sánh độ chính xác của YOLOv8 và Faster R-CNN, khảo sát chuỗi cải tiến Faster R-CNN, và đo tốc độ xử lý trên cùng nguồn video và phần cứng. Đóng góp thực nghiệm là bộ kết quả validation, phân tích sự thay đổi giữa các cấu hình, cùng quy trình demo có lưu dự đoán và thời gian từng khung hình. RTSP phát từ video lặp lại là môi trường mô phỏng truyền luồng; tốc độ suy luận và độ trễ truyền luồng cần được báo cáo riêng.

Nguồn nội bộ: mô tả bài toán trong `docs/proposal.tex`; số liệu trong `reports/eval/*.json`; cấu hình trong `faster_rcnn/configs/experiment`. Các số liệu này không thay thế đánh giá trên tập test độc lập.

## Ablation Faster R-CNN

Đơn vị chỉ số trong bảng là phần trăm; chênh lệch là điểm phần trăm. AP_small là AP COCO trên nhóm vật thể nhỏ, không phải AP50_small.

| Cấu hình | mAP@0.5 | mAP@0.5:0.95 | AP_small | AP_medium | AP_large |
| --- | ---: | ---: | ---: | ---: | ---: |
| exp01_baseline | 64,8 | 41,6 | 33,5 | 68,1 | 50,4 |
| exp02_fpn_small_anchor | 72,5 | 47,7 | 40,3 | 71,2 | 51,9 |
| exp03_fpn_aug | 75,2 | 48,6 | 41,3 | 71,8 | 53,4 |

| Yếu tố theo YAML hiện tại | exp01 | exp02 | exp03 |
| --- | --- | --- | --- |
| Kiến trúc | ResNet50 + FPN | ResNet50 + FPN | ResNet50 + FPN |
| trainable_backbone_layers | 0 | 3 | 3 |
| Anchor sizes | 32, 64, 128, 256, 512 | 8, 16, 32, 64, 128 | 8, 16, 32, 64, 128 |
| Anchor aspect ratios | 0,5; 1; 2 | 0,8; 1; 1,25 | 0,8; 1; 1,25 |
| min_size / max_size | 800 / 1333 | 800 / 1800 | 800 / 1800 |
| Augmentation | Không | Không | Brightness/contrast; không lật ngang |

Từ exp01 sang exp02, mAP@0.5:0.95 tăng 6,17 điểm và AP_small tăng 6,85 điểm, tính từ JSON chưa làm tròn. Đây là thay đổi **một nhóm yếu tố đồng thời**, chưa tách được tác động riêng của anchor, aspect ratio, kích thước ảnh và fine-tuning backbone. Cả ba cấu hình đều dùng FPN, nên không diễn giải đây là thí nghiệm “thêm FPN”. `trainable_backbone_layers=0` cũng không có nghĩa toàn bộ detector bị đóng băng.

Từ exp02 sang exp03, khác biệt khai báo chính là augmentation độ sáng/độ tương phản; mAP@0.5:0.95 tăng 0,84 điểm, AP_small tăng 0,98 điểm và mAP@0.5 tăng 2,75 điểm. Toàn chuỗi exp01→exp03 tăng 7,01 điểm mAP@0.5:0.95 và 7,83 điểm AP_small. Chưa có kết quả nhiều seed hoặc khoảng tin cậy cho chuỗi này; không khẳng định ý nghĩa thống kê hay quan hệ nhân quả chỉ từ một lượt mỗi cấu hình. YAML hiện tại cần đối chiếu với metadata checkpoint/log huấn luyện trước khi xem là cấu hình đã thực thi.

Nếu cần ablation từng yếu tố, cần huấn luyện bổ sung với cùng split, seed, lịch học và tiêu chí chọn checkpoint, mỗi lượt chỉ đổi một yếu tố. Chưa có các lượt đó trong bảng trên; không suy diễn hoặc điền số liệu dự kiến.

## Giao thức đo FPS và đóng băng

1. Ghi SHA256 checkpoint, cấu hình, video; phiên bản Python/PyTorch/TorchVision/Ultralytics, GPU/CPU, precision và Git revision. Dùng cùng video, đoạn bắt đầu, số khung hình, batch=1 và điều kiện lưu/hiển thị cho hai mô hình.
2. Chạy warmup riêng; bỏ thời gian warmup và tải mô hình khỏi phép đo. Đồng bộ CUDA trước/sau đoạn đo khi dùng GPU. Lưu thời gian từng khung hình để tính FPS tổng số frame chia tổng thời gian, median và p95 latency; không lấy trung bình FPS từng frame.
3. Phân biệt tốc độ hàm detector (tiền xử lý, suy luận, hậu xử lý theo định nghĩa CLI) với tốc độ pipeline gồm đọc, vẽ, tracking và ghi. Benchmark file cục bộ để đo throughput; demo RTSP để kiểm tra vận hành. RTSP có thể bị giới hạn bởi nhịp phát và buffer; FPS cao không chứng minh độ trễ đầu-cuối thấp.
4. Lặp ít nhất ba lần cho mỗi cấu hình dự định đưa vào kết luận; ghi số frame thực tế và mọi lỗi đọc/khởi tạo. Demo vài frame chỉ là smoke test. Không ghép FPS YOLO chuẩn 1280 với AP của `yolov8s_tiled_640_sahi`; SAHI phải đo đúng pipeline cắt và hợp nhất nếu dùng AP đó.
5. Kiểm tra video đầu ra đọc được, snapshot có nhãn/hộp, log dự đoán hợp lệ, timestamp hợp lý. Tracking IoU chỉ phục vụ demo; chưa có đánh giá MOTA/IDF1 và không khẳng định giữ ID qua che khuất.
6. Tạo manifest bằng `evaluation/freeze_results.py` để chốt nguồn số liệu hiện có. Manifest là bản chụp nguồn, không tự chứng nhận thí nghiệm hoặc đóng băng kết luận cuối. Có thể bàn giao bản chụp độ chính xác Faster R-CNN kèm smoke test có ghi rõ giới hạn. Đóng băng hiệu năng Faster R-CNN sau benchmark đủ dài, bằng chứng demo và đối chiếu checkpoint. Phần so sánh tốc độ hai mô hình được bổ sung khi thành viên YOLO bàn giao trace cùng giao thức; đây không phải điều kiện chặn phần việc Faster R-CNN.

### Smoke test CPU đã chạy (chưa phải benchmark cuối)

Lượt exp03 xử lý 12 frame đầu của video 1280×720, loại 2 frame warmup, còn 10 frame đo; CPU 4 threads, FP32, batch=1, bật tracking và ghi video, chạy headless. Môi trường ghi nhận Python 3.12.9, PyTorch 2.15.0+cpu, TorchVision 0.29.2+cpu và OpenCV 5.0.0. Nguồn là file cục bộ, chưa xác minh vận hành RTSP bằng lượt này.

| Cấu hình | Frames đo | FPS detector | FPS pipeline | p95 detector (ms) | Bằng chứng |
| --- | ---: | ---: | ---: | ---: | --- |
| exp01_baseline | 10 | 0,230790 | 0,230283 | 4661,38 | `artifacts/verification/demo_baseline_smoke/summary.json` |
| exp02_fpn_small_anchor | 10 | 0,216179 | 0,215761 | 4859,90 | `artifacts/verification/demo_anchors_smoke/summary.json` |
| exp03_fpn_aug | 10 | 0,208878 | 0,208446 | 4912,72 | `artifacts/verification/demo_aug_smoke/summary.json` |
| exp03_fpn_aug (RTSP TCP) | 4 | 0,213628 | 0,213194 | 5058,52 | `artifacts/verification/demo_rtsp_smoke/summary.json` |

`infer_ms` bao gồm preprocessing, inference, postprocessing và chuyển kết quả về CPU; `pipeline_ms` tính từ đọc frame qua vẽ, tracking, encode và log detection, loại thời gian ghi CSV. Trace từng frame nằm cạnh summary. Đây là các lượt ngắn để xác minh chạy thật, không phải tốc độ ổn định hay kết quả triển khai GPU. Tốc độ CPU quan sát được (~0.21 - 0.23 FPS) thấp hơn nhiều so với nguồn 30 FPS; chưa đáp ứng xử lý đủ frame theo thời gian thực trên phần cứng CPU hiện tại.

Đã kiểm tra vận hành truyền luồng RTSP thực tế qua MediaMTX và FFmpeg lặp `data/DashCam/traffic.mp4`. Pipeline kết nối qua TCP transport và hoàn thành suy luận 5 frame (1 warmup, 4 frame đo) với detector FPS đạt 0,2136 và pipeline FPS đạt 0,2132 trên CPU. Log server, publisher, lệnh thực thi và trace lưu tại `artifacts/verification/rtsp_transport` và `artifacts/verification/demo_rtsp_smoke`.

Mỗi video smoke xuất chỉ chứa 12 frame ở 30 FPS, tương đương **0,4 giây**, không phải phim demo hoàn chỉnh. Metadata baseline và exp03 ghi revision `aabe845` với worktree dirty; exp02 ghi `9b804c2`. SHA256 implementation được ghi tại thời điểm chạy; không gán các lượt cũ cho commit mới hơn. Kết quả tốc độ cuối cần lượt đo dài và lặp lại. Phần YOLO được thành viên phụ trách bổ sung sau, không chặn bàn giao này.

## Kịch bản demo video

Kịch bản sau là đề xuất, chưa phải bản quay đã hoàn thành. Bản quay đề xuất dài 60–90 giây, dùng cùng đoạn `data/DashCam/traffic.mp4` để người xem đối chiếu. Lệnh chạy và quy trình kiểm chứng xem `streaming/README.md`; xem tùy chọn bằng `python -m streaming.demo --help`.

| Thời lượng | Nội dung | Bằng chứng lưu |
| --- | --- | --- |
| 0–10 giây | Giới thiệu bài toán, tên mô hình, cấu hình và thiết bị | Màn hình thông tin lượt chạy |
| 10–35 giây | Faster R-CNN: khung hình, lớp, confidence, hộp và ID nếu bật tracking | Video có annotation, snapshot, JSONL |
| 35–60 giây | Tùy chọn: thành viên YOLO bổ sung cùng đoạn; nếu chưa có, trình bày thêm Faster R-CNN | Video có annotation, snapshot, JSONL |
| 60–75 giây | Hiện bảng độ chính xác validation và tốc độ thực đo | Bảng có liên kết trace |
| 75–90 giây | Nêu ví dụ bỏ sót/nhầm lớp quan sát được và hạn chế | Khung hình lỗi thực tế; không dựng lỗi giả |

Khởi động MediaMTX bằng đường dẫn đã có trên máy. Từ thư mục chứa video, phát luồng bằng `ffmpeg -re -stream_loop -1 -i traffic.mp4 -c copy -f rtsp rtsp://localhost:8554/live/traffic`. Sau đó chạy detector với nguồn RTSP. Với benchmark định lượng, dùng nguồn file để hai mô hình nhận cùng chuỗi frame; không so sánh hai lần bắt RTSP tại thời điểm khác nhau. Nếu phát trực tiếp và inference chậm hơn nguồn, phải nêu rõ chiến lược bỏ frame/buffering và đo độ trễ riêng.

## Conclusion

Các kết quả validation cho thấy cấu hình Faster R-CNN exp03 đạt mAP@0.5:0.95 48,6% và AP_small 41,3%, cao hơn baseline lần lượt 7,01 và 7,83 điểm phần trăm. Mức tăng lớn nhất trong chuỗi gắn với nhóm thay đổi fine-tuning, anchor và giới hạn kích thước ảnh; augmentation mang lại mức tăng bổ sung nhỏ hơn. Trong các báo cáo YOLO hiện có, cấu hình tiled + SAHI đạt mAP@0.5:0.95 57,9%. Việc lựa chọn cấu hình triển khai cần bổ sung tốc độ thực đo trên cùng phần cứng và pipeline tương ứng. Kết quả hiện tại hỗ trợ kết luận về độ chính xác validation, chưa đủ để tuyên bố hệ thống đáp ứng yêu cầu thời gian thực.

## Limitations

- Kết quả chính dùng validation; chưa có kết quả test độc lập cho toàn bộ cấu hình được so sánh ở đây. Chọn cấu hình trên validation có thể tạo thiên lệch lạc quan.
- Chuỗi Faster R-CNN chưa tách độc lập mọi yếu tố và chưa đánh giá nhiều seed. Cấu hình YAML trong repository không đủ chứng minh mọi override khi huấn luyện; cần metadata checkpoint/log gốc.
- Độ chính xác ảnh tĩnh không đo tính ổn định theo thời gian. Video demo chưa có nhãn chuẩn nên chỉ cho bằng chứng vận hành, không cho AP trên video.
- Nguồn RTSP localhost phát video lặp lại chưa đại diện mạng thực, camera thực, mất gói hay dịch chuyển miền dữ liệu. Không suy ra độ trễ capture-to-display từ thời gian xử lý Python.
- Throughput phụ thuộc thiết bị, kích thước ảnh, precision và SAHI. Benchmark ngắn hoặc CPU smoke test không thay thế kết quả GPU triển khai.
- Tracking IoU đơn giản có thể đổi ID khi che khuất, ra/vào khung hình hoặc detector bỏ sót; chưa có ground truth tracking để đánh giá.
