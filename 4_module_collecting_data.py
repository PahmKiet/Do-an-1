import cv2
import mediapipe as mp
import math
import time
import csv
import os
from collections import deque

# --- 1. CẤU HÌNH HỆ THỐNG ---
REAL_HAND_WIDTH_CM = 7.0  # Chiều rộng thực tế lòng bàn tay (cm)
CALIB_FILE = "calibration_data.txt"
DATA_FOLDER = "collected_data"
OFFSET_MM = 18.0  # Khoảng cách bù để khi chạm tay Gap = 0

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# --- 2. CÁC HÀM TIỆN ÍCH ---
def save_calibration(focal_len):
    with open(CALIB_FILE, "w") as f:
        f.write(str(focal_len))

def load_calibration():
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, "r") as f:
            try: return float(f.read())
            except: return None
    return None

# --- 3. KHỞI TẠO CÔNG CỤ ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

FOCAL_LENGTH = load_calibration()
is_calibrated = True if FOCAL_LENGTH else False
history_dist = deque(maxlen=10)
history_gap = deque(maxlen=7)

recording = False
data_log = []
label = 0  # 0: Healthy, 1: Parkinson
start_time = 0
px_w, dist_px, calib_w25 = 0, 0, None

cap = cv2.VideoCapture(0)

print("--- HỆ THỐNG SẴN SÀNG ---")

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Lấy tọa độ điểm mốc
            p5, p17 = hand_lms.landmark[5], hand_lms.landmark[17]
            p8, p4 = hand_lms.landmark[8], hand_lms.landmark[4]

            # Tính Pixel
            px_w = math.hypot(p5.x * w - p17.x * w, p5.y * h - p17.y * h)
            dist_px = math.hypot(p8.x * w - p4.x * w, p8.y * h - p4.y * h)

            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            if is_calibrated and px_w > 0:
                # 1. Tính Khoảng cách Cam (Z)
                current_dist = (REAL_HAND_WIDTH_CM * FOCAL_LENGTH) / px_w
                history_dist.append(current_dist)
                avg_dist = sum(history_dist) / len(history_dist)

                # 2. Tính Gap mm và Khử sai số 18mm
                raw_gap = (dist_px * avg_dist * 10) / FOCAL_LENGTH
                gap_final = max(0, raw_gap - OFFSET_MM)
                history_gap.append(gap_final)
                avg_gap = sum(history_gap) / len(history_gap)

                # HIỂN THỊ THÔNG SỐ (Góc trái trên)
                cv2.rectangle(img, (15, 15), (380, 145), (0, 80, 0), -1)
                cv2.putText(img, f"Dist: {avg_dist:.1f} cm", (30, 65), 2, 1.0, (0, 255, 0), 2)
                cv2.putText(img, f"Gap: {avg_gap:.1f} mm", (30, 120), 2, 1.2, (0, 0, 255), 3)

                if recording:
                    data_log.append([round(time.time() - start_time, 3), round(avg_dist, 2), round(avg_gap, 2), label])

            elif not is_calibrated:
                cv2.rectangle(img, (15, 15), (550, 80), (0, 0, 150), -1)
                msg = "DOI MOC 25CM (Phim 1)" if calib_w25 is None else "DOI MOC 40CM (Phim 2)"
                cv2.putText(img, msg, (30, 55), 2, 0.9, (255, 255, 255), 2)

    # --- BẢNG HƯỚNG DẪN (Góc phải dưới) ---
    overlay = img.copy()
    cv2.rectangle(overlay, (w-280, h-200), (w-15, h-15), (40, 40, 40), -1)
    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    instructions = [("H: HEALTHY", (255, 255, 0)), ("P: PARKINSON", (255, 0, 255)),
                    ("S: START/STOP", (0, 255, 0)), ("R: RESET", (0, 0, 255)), ("Q: QUIT", (255, 255, 255))]
    for i, (text, color) in enumerate(instructions):
        cv2.putText(img, text, (w-265, h-165 + i*32), 1, 1.2, color, 2)

    # Trạng thái Ghi & Nhãn hiện tại
    mode_txt = "PARKINSON" if label == 1 else "HEALTHY"
    cv2.putText(img, f"MODE: {mode_txt}", (20, h-25), 2, 0.9, (255, 255, 0), 2)
    if recording:
        cv2.circle(img, (w-40, 40), 15, (0, 0, 255), -1)
        cv2.putText(img, "REC", (w-110, 50), 2, 0.8, (0, 0, 255), 2)

    cv2.imshow("HUST PARKINSON - DATA COLLECTOR", img)

    # --- XỬ LÝ PHÍM ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('h'): label = 0; print("Chon: HEALTHY")
    elif key == ord('p'): label = 1; print("Chon: PARKINSON")
    elif key == ord('r'):
        is_calibrated = False; calib_w25 = None
        if os.path.exists(CALIB_FILE): os.remove(CALIB_FILE)
        print("Reset!")
    elif key == ord('1') and not is_calibrated:
        calib_w25 = px_w; print("Luu 25cm")
    elif key == ord('2') and not is_calibrated:
        if calib_w25:
            f1, f2 = (25 * calib_w25)/7.0, (40 * px_w)/7.0
            FOCAL_LENGTH = (f1 + f2)/2
            save_calibration(FOCAL_LENGTH); is_calibrated = True; print("Xong!")
    elif key == ord('s'):
        if not recording:
            if not is_calibrated: continue
            recording, data_log, start_time = True, [], time.time()
            print("Recording...")
        else:
            recording = False
            if data_log:
                fname = f"{DATA_FOLDER}/record_{int(time.time())}_L{label}.csv"
                with open(fname, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Time', 'Dist_cm', 'Gap_mm', 'Label'])
                    writer.writerows(data_log)
                print(f"Saved: {fname}")

cap.release()
cv2.destroyAllWindows()