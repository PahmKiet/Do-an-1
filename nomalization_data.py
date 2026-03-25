import cv2
import mediapipe as mp
import math
import collections
import os

# --- THÔNG SỐ CỦA BẢO ---
REAL_HAND_WIDTH_CM = 7.0
CALIB_FILE = "calibration_data.txt"  # Tên file lưu mốc hiệu chuẩn


# Hàm lưu cấu hình vào file
def save_calibration(focal_len):
    with open(CALIB_FILE, "w") as f:
        f.write(str(focal_len))


# Hàm đọc cấu hình từ file
def load_calibration():
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, "r") as f:
            try:
                return float(f.read())
            except:
                return None
    return None


# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Biến trạng thái và bộ lọc làm mượt
calib_w25 = None
calib_w40 = None
history_dist = collections.deque(maxlen=10)

# KIỂM TRA DỮ LIỆU CŨ KHI KHỞI ĐỘNG
FOCAL_LENGTH = load_calibration()
is_calibrated = True if FOCAL_LENGTH else False

cap = cv2.VideoCapture(0)

if not is_calibrated:
    print("--- CHẾ ĐỘ HIỆU CHUẨN (25cm & 40cm) ---")
    print("1. Đặt tay cách Cam đúng 25cm -> Nhấn phím '1'")
    print("2. Đặt tay cách Cam đúng 40cm -> Nhấn phím '2'")
else:
    print(f"--- ĐÃ TẢI CẤU HÌNH: Focal Length = {FOCAL_LENGTH:.2f} ---")
    print("Nhấn 'r' nếu muốn hiệu chuẩn lại từ đầu.")

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # 1. Lấy pixel chiều rộng bàn tay (Điểm 5 đến 17)
            p5 = hand_lms.landmark[5]
            p17 = hand_lms.landmark[17]
            px_w = math.sqrt((p5.x * w - p17.x * w) ** 2 + (p5.y * h - p17.y * h) ** 2)

            # 2. Lấy pixel khoảng cách ngón trỏ (8) và cái (4)
            p8 = hand_lms.landmark[8]
            p4 = hand_lms.landmark[4]
            dist_px = math.sqrt((p8.x * w - p4.x * w) ** 2 + (p8.y * h - p4.y * h) ** 2)

            # --- VẼ LANDMARKS ---
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            # --- TÍNH TOÁN ---
            if is_calibrated:
                # Tính khoảng cách từ Cam đến Tay (cm)
                raw_cam_dist = (REAL_HAND_WIDTH_CM * FOCAL_LENGTH) / px_w
                history_dist.append(raw_cam_dist)
                current_dist_cam = sum(history_dist) / len(history_dist)

                # Quy đổi Pixel sang mm thực tế
                finger_gap_mm = (dist_px * current_dist_cam * 10) / FOCAL_LENGTH

                # Hiển thị thông số
                cv2.rectangle(img, (10, 10), (450, 130), (0, 100, 0), -1)
                cv2.putText(img, f"Cam Dist: {current_dist_cam:.1f} cm", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f"Gap: {finger_gap_mm:.1f} mm", (20, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                # Giao diện chờ Hiệu chuẩn
                cv2.rectangle(img, (10, 10), (450, 100), (0, 0, 150), -1)
                status = "DOI MOC 25CM (1)" if not calib_w25 else "DOI MOC 40CM (2)"
                cv2.putText(img, status, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("PARKINSON HUST - MEASUREMENT", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):  # Reset để hiệu chuẩn lại
        is_calibrated = False
        calib_w25 = calib_w40 = None
        if os.path.exists(CALIB_FILE):
            os.remove(CALIB_FILE)
        print("Đã xóa cấu hình cũ. Hãy hiệu chuẩn lại mốc 25-40cm.")
    elif key == ord('1') and not is_calibrated:
        calib_w25 = px_w
        print(f"Lưu mốc 25cm: {int(px_w)} px")
    elif key == ord('2') and not is_calibrated:
        if calib_w25:
            calib_w40 = px_w
            # Tính và lưu ngay lập tức
            f1 = (25 * calib_w25) / REAL_HAND_WIDTH_CM
            f2 = (40 * calib_w40) / REAL_HAND_WIDTH_CM
            FOCAL_LENGTH = (f1 + f2) / 2
            save_calibration(FOCAL_LENGTH)
            is_calibrated = True
            print(f"HIEU CHUAN XONG! Da luu file: {FOCAL_LENGTH:.2f}")
        else:
            print("Ban phai bam mốc 25cm (phím 1) trước!")

cap.release()
cv2.destroyAllWindows()
