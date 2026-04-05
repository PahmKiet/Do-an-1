import cv2
import mediapipe as mp
import math
import os
import pandas as pd

# Cấu hình hệ thống
REAL_HAND_WIDTH_CM = 7.0
CALIB_FILE = "calibration_data.txt"
DATA_FOLDER = "collected_data"
OFFSET_MM = 18.0

if not os.path.exists(DATA_FOLDER): os.makedirs(DATA_FOLDER)


def run_collection(source, label, output_filename):
    focal_length = 500.0
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, "r") as f:
            try:
                focal_length = float(f.read())
            except:
                pass

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    is_video_file = isinstance(source, str)

    raw_data = []
    recording = False  # Trạng thái ghi
    frame_rec_count = 0

    print("--- HD: Nhấn 'r' để BẮT ĐẦU/DỪNG ghi | Nhấn 'q' để THOÁT ---")

    while cap.isOpened():
        success, img = cap.read()
        if not success: break

        # Lật cam nếu dùng webcam cho thuận tay
        if not is_video_file: img = cv2.flip(img, 1)

        h, w, _ = img.shape
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

                # Tính Gap mm
                p5, p17 = hand_lms.landmark[5], hand_lms.landmark[17]
                p8, p4 = hand_lms.landmark[8], hand_lms.landmark[4]
                px_w = math.hypot(p5.x * w - p17.x * w, p5.y * h - p17.y * h)
                dist_px = math.hypot(p8.x * w - p4.x * w, p8.y * h - p4.y * h)

                if px_w > 0:
                    dist_cm = (REAL_HAND_WIDTH_CM * focal_length) / px_w
                    gap_mm = max(0, (dist_px * dist_cm * 10) / focal_length - OFFSET_MM)

                    # Hiện mm lên màn hình
                    cv2.putText(img, f"Gap: {round(gap_mm, 1)}mm", (50, 50), 2, 0.8, (0, 255, 0), 2)

                    # CHỈ GHI KHI ĐANG BẬT RECORDING
                    if recording:
                        timestamp = frame_rec_count / fps
                        raw_data.append([round(timestamp, 3), round(dist_cm, 2), round(gap_mm, 2), label])
                        frame_rec_count += 1

        # GIAO DIỆN TRẠNG THÁI
        if recording:
            cv2.circle(img, (30, 30), 10, (0, 0, 255), -1)  # Đèn đỏ nháy
            cv2.putText(img, f"RECORDING: {len(raw_data)} frames", (50, 30), 2, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(img, "READY - Press 'r' to Record", (w - 350, 30), 2, 0.6, (0, 255, 255), 2)

        cv2.imshow("HUST PARKINSON COLLECTOR", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'):
            recording = not recording
            if not recording and len(raw_data) > 0:  # Dừng ghi thì thoát để xử lý tiếp
                print(f"✅ Đã ghi xong {len(raw_data)} khung hình.")
                break

    cap.release()
    cv2.destroyAllWindows()

    if raw_data:
        df = pd.DataFrame(raw_data, columns=['Time', 'Dist_cm', 'Gap_mm', 'Label'])
        df.to_csv(os.path.join(DATA_FOLDER, output_filename), index=False)
        return True
    return False