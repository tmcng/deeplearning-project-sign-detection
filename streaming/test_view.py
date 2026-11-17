import cv2
import time

# Đường dẫn camera ảo của bạn
RTSP_URL = "rtsp://localhost:8554/live/traffic"

print("Đang kết nối luồng RTSP... Bấm 'q' để thoát.")
cap = cv2.VideoCapture(RTSP_URL)

# Tối ưu cấu hình bộ đệm để xem thời gian thực không trễ
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Không thể kết nối! Kiểm tra xem FFmpeg và MediaMTX đã bật chưa.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Đang chờ khung hình hoặc luồng bị reset...")
        time.sleep(0.5)
        continue

    # Hiển thị video lên màn hình máy tính
    cv2.imshow("Luong RTSP mo phong Realtime qua OpenCV", frame)

    # Bấm 'q' để tắt cửa sổ
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
