import cv2
import mediapipe as mp
import math
import time
import csv
import os
from collections import deque

# --- 1. CẤU HÌNH HỆ THỐNG ---
REAL_HAND_WIDTH_CM = 7.0  # Chiều rộng lòng bàn tay thực tế (cm)
CALIB_FILE = "calibration_data.txt"
DATA_FOLDER = "collected_data"

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)


# --- 2. CÁC HÀM TIỆN ÍCH ---
def save_calibration(focal_len):
    with open(CALIB_FILE, "w") as f:
        f.write(str(focal_len))


def load_calibration():
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, "r") as f:
            try:
                return float(f.read())
            except:
                return None
    return None


# --- 3. KHỞI TẠO CÔNG CỤ ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Biến trạng thái
FOCAL_LENGTH = load_calibration()
is_calibrated = True if FOCAL_LENGTH else False
history_dist = deque(maxlen=10)  # Bộ lọc làm mượt khoảng cách
history_gap = deque(maxlen=5)  # Bộ lọc làm mượt khoảng cách ngón tay

recording = False
data_log = []
label = 0  # 0: Healthy (H), 1: Parkinson (P)
start_time = 0

# Khởi tạo biến pixel mặc định để tránh lỗi "Undefined"
px_w = 0
dist_px = 0
calib_w25 = None

cap = cv2.VideoCapture(0)

print("--- HỆ THỐNG ĐÃ SẴN SÀNG ---")
print("HƯỚNG DẪN PHÍM BẤM:")
print("[1]: Lưu mốc 25cm | [2]: Lưu mốc 40cm")
print("[H]: Nhãn HEALTHY | [P]: Nhãn PARKINSON")
print("[S]: Bắt đầu/Dừng ghi CSV | [R]: Reset hiệu chuẩn | [Q]: Thoát")

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)  # Lật ảnh như gương
    h, w, c = img.shape

    # Chuyển màu sang RGB để MediaPipe xử lý
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Lấy tọa độ các điểm quan trọng
            p5 = hand_lms.landmark[5]  # Gốc ngón trỏ
            p17 = hand_lms.landmark[17]  # Gốc ngón út
            p8 = hand_lms.landmark[8]  # Đầu ngón trỏ
            p4 = hand_lms.landmark[4]  # Đầu ngón cái

            # TÍNH KHOẢNG CÁCH PIXEL (Dùng math.hypot để fix lỗi Math Domain Error)
            px_w = math.hypot(p5.x * w - p17.x * w, p5.y * h - p17.y * h)
            dist_px = math.hypot(p8.x * w - p4.x * w, p8.y * h - p4.y * h)

            # Vẽ xương tay lên màn hình
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            if is_calibrated and px_w > 0:
                # 1. Tính khoảng cách từ Cam đến Tay (Z)
                raw_cam_dist = (REAL_HAND_WIDTH_CM * FOCAL_LENGTH) / px_w
                history_dist.append(raw_cam_dist)
                avg_dist = sum(history_dist) / len(history_dist)

                # 2. Quy đổi Pixel ngón tay sang mm thực tế
                raw_gap_mm = (dist_px * avg_dist * 10) / FOCAL_LENGTH
                history_gap.append(raw_gap_mm)
                avg_gap = sum(history_gap) / len(history_gap)

                # --- VẼ GIAO DIỆN THEO ẢNH MẪU ---
                # Hộp nền xanh lá đậm
                cv2.rectangle(img, (20, 20), (460, 160), (0, 100, 0), -1)

                # Hiển thị Cam Dist (Xanh neon)
                cv2.putText(img, f"Cam Dist: {avg_dist:.1f} cm", (40, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                # Hiển thị Gap (Đỏ rực)
                cv2.putText(img, f"Gap: {avg_gap:.1f} mm", (40, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)

                # Lưu dữ liệu vào log nếu đang quay
                if recording:
                    current_ts = time.time() - start_time
                    data_log.append([round(current_ts, 3), round(avg_dist, 2), round(avg_gap, 2), label])

            elif not is_calibrated:
                # Giao diện thông báo đang trong chế độ hiệu chuẩn
                cv2.rectangle(img, (20, 20), (600, 100), (0, 0, 150), -1)
                status_msg = "DOI MOC 25CM (Phim 1)" if calib_w25 is None else "DOI MOC 40CM (Phim 2)"
                cv2.putText(img, status_msg, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Hiển thị nhãn (Label) và trạng thái Ghi (REC)
    label_color = (0, 255, 255) if label == 1 else (255, 255, 0)
    cv2.putText(img, f"Label: {'PARKINSON' if label == 1 else 'HEALTHY'}",
                (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)

    if recording:
        cv2.circle(img, (w - 40, 40), 20, (0, 0, 255), -1)
        cv2.putText(img, "REC", (w - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("PARKINSON HUST - DATA COLLECTOR", img)

    # --- XỬ LÝ PHÍM BẤM ---
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Đổi phím nhãn sang H và P để không trùng với 1 và 2
    elif key == ord('h'):
        label = 0
        print("Đã chọn nhãn: HEALTHY")
    elif key == ord('p'):
        label = 1
        print("Đã chọn nhãn: PARKINSON")

    elif key == ord('r'):
        is_calibrated = False
        calib_w25 = None
        if os.path.exists(CALIB_FILE): os.remove(CALIB_FILE)
        print("Đã xóa hiệu chuẩn. Hãy làm lại mốc 1 và 2.")

    # HIỆU CHUẨN MỐC 1
    elif key == ord('1') and not is_calibrated:
        if px_w > 0:
            calib_w25 = px_w
            print(f"Lưu mốc 1 (25cm): {px_w:.2f} px. Bây giờ hãy bấm phím '2' tại mốc 40cm.")
        else:
            print("Lỗi: Không thấy bàn tay để lưu mốc 1!")

    # HIỆU CHUẨN MỐC 2
    elif key == ord('2') and not is_calibrated:
        if calib_w25 is not None and px_w > 0:
            calib_w40 = px_w
            # Tính toán tiêu cự Focal Length
            f1 = (25 * calib_w25) / REAL_HAND_WIDTH_CM
            f2 = (40 * calib_w40) / REAL_HAND_WIDTH_CM
            FOCAL_LENGTH = (f1 + f2) / 2
            save_calibration(FOCAL_LENGTH)
            is_calibrated = True
            print(f"HIỆU CHUẨN THÀNH CÔNG! Focal Length: {FOCAL_LENGTH:.2f}")
        else:
            print("Lỗi: Bạn chưa lưu mốc 1 hoặc không thấy bàn tay!")

    # GHI DỮ LIỆU CSV
    elif key == ord('s'):
        if not recording:
            if not is_calibrated:
                print("Lỗi: Phải hiệu chuẩn xong mới được ghi dữ liệu!")
                continue
            recording = True
            data_log = []
            start_time = time.time()
            print("--- ĐANG GHI DỮ LIỆU... ---")
        else:
            recording = False
            if len(data_log) > 0:
                file_name = f"{DATA_FOLDER}/record_{int(time.time())}_L{label}.csv"
                with open(file_name, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Timestamp', 'Cam_Dist_cm', 'Finger_Gap_mm', 'Label'])
                    writer.writerows(data_log)
                print(f"--- ĐÃ LƯU FILE: {file_name} ---")
            else:
                print("Không có dữ liệu để lưu.")

cap.release()
cv2.destroyAllWindows()
    