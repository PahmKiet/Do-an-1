import cv2
import mediapipe as mp
import math
import collections
import os

# --- THÔNG SỐ CỦA BẢO ---
REAL_HAND_WIDTH_CM = 7.0  # Chiều rộng thực tế lòng bàn tay (cm)
CALIB_FILE = "calibration_data.txt"


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


# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Biến trạng thái
history_dist = collections.deque(maxlen=10)
history_gap = collections.deque(maxlen=5)  # Lọc nhiễu cho Gap ngón tay

FOCAL_LENGTH = load_calibration()
is_calibrated = True if FOCAL_LENGTH else False

cap = cv2.VideoCapture(0)

print("--- HỆ THỐNG ĐO KHOẢNG CÁCH NGÓN TAY ---")
if is_calibrated:
    print(f"Cấu hình sẵn sàng: Focal = {FOCAL_LENGTH:.2f}")

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Lấy tọa độ điểm 5 và 17 (Chiều rộng lòng bàn tay)
            p5 = hand_lms.landmark[5]
            p17 = hand_lms.landmark[17]
            # Tính pixel chiều rộng bàn tay (Dùng công thức hypot cho chuẩn)
            px_w = math.hypot(p5.x * w - p17.x * w, p5.y * h - p17.y * h)

            # Lấy tọa độ đầu ngón cái (4) và trỏ (8)
            p4 = hand_lms.landmark[4]
            p8 = hand_lms.landmark[8]
            dist_px = math.hypot(p8.x * w - p4.x * w, p8.y * h - p4.y * h)

            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            if is_calibrated and px_w > 0:
                # 1. Tính khoảng cách từ Cam đến Tay (Z)
                current_dist_cam = (REAL_HAND_WIDTH_CM * FOCAL_LENGTH) / px_w
                history_dist.append(current_dist_cam)
                avg_dist = sum(history_dist) / len(history_dist)

                # 2. Quy đổi Pixel sang mm dùng tỷ lệ trực tiếp (Ratio Method)
                # 1 pixel = (70mm / Chiều rộng lòng bàn tay pixel)
                ratio_mm_px = (REAL_HAND_WIDTH_CM * 10) / px_w
                raw_gap_mm = dist_px * ratio_mm_px

                # Trừ sai số hình học (Offset) để chạm nhau về ~0
                # Nếu vẫn thấy dư, Bảo chỉnh số 15 này lên 18 hoặc 20 nhé
                gap_mm_final = max(0, raw_gap_mm - 15)

                history_gap.append(gap_mm_final)
                avg_gap = sum(history_gap) / len(history_gap)

                # HIỂN THỊ GIAO DIỆN
                cv2.rectangle(img, (10, 10), (450, 150), (0, 64, 0), -1)
                cv2.putText(img, f"Cam Dist: {avg_dist:.1f} cm", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f"Gap: {avg_gap:.1f} mm", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            else:
                cv2.rectangle(img, (10, 10), (500, 100), (0, 0, 128), -1)
                txt = "DOI MOC 25CM (1)" if 'calib_w25' not in locals() or not calib_w25 else "DOI MOC 40CM (2)"
                cv2.putText(img, txt, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("HUST PARKINSON MEASURE", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

    if key == ord('r'):
        is_calibrated = False
        if os.path.exists(CALIB_FILE): os.remove(CALIB_FILE)
        print("Reset thành công!")

    # Logic hiệu chuẩn nếu chưa có file
    if not is_calibrated:
        if key == ord('1'):
            calib_w25 = px_w
            print(f"Lưu mốc 25cm: {px_w:.1f} px")
        elif key == ord('2'):
            if 'calib_w25' in locals() and calib_w25:
                calib_w40 = px_w
                f1 = (25 * calib_w25) / REAL_HAND_WIDTH_CM
                f2 = (40 * calib_w40) / REAL_HAND_WIDTH_CM
                FOCAL_LENGTH = (f1 + f2) / 2
                save_calibration(FOCAL_LENGTH)
                is_calibrated = True
                print(f"XONG! Focal Length: {FOCAL_LENGTH:.2f}")

cap.release()
cv2.destroyAllWindows()