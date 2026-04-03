import cv2
import mediapipe as mp
import math
import os
import pandas as pd

# Cấu hình giữ nguyên từ đồ án của Bảo
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
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    is_video_file = isinstance(source, str)
    raw_data, frame_count, recording = [], 0, is_video_file

    while cap.isOpened():
        success, img = cap.read()
        if not success: break
        if not is_video_file: img = cv2.flip(img, 1)  # Lật cam nếu dùng webcam

        h, w, _ = img.shape
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
                p5, p17 = hand_lms.landmark[5], hand_lms.landmark[17]
                p8, p4 = hand_lms.landmark[8], hand_lms.landmark[4]
                px_w = math.hypot(p5.x * w - p17.x * w, p5.y * h - p17.y * h)
                dist_px = math.hypot(p8.x * w - p4.x * w, p8.y * h - p4.y * h)

                if px_w > 0:
                    dist_cm = (REAL_HAND_WIDTH_CM * focal_length) / px_w
                    gap_mm = max(0, (dist_px * dist_cm * 10) / focal_length - OFFSET_MM)
                    cv2.putText(img, f"Gap: {round(gap_mm, 1)}mm", (50, 50), 2, 0.8, (0, 255, 0), 2)
                    if recording:
                        raw_data.append([round(frame_count / fps, 3), round(dist_cm, 2), round(gap_mm, 2), label])

        cv2.putText(img, "REC" if recording else "WAITING", (w - 150, 50), 2, 0.7,
                    (0, 0, 255) if recording else (0, 255, 255), 2)
        cv2.imshow("HUST PARKINSON SYSTEM", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('s') and not is_video_file:
            recording = not recording
            if not recording: break
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    if raw_data:
        pd.DataFrame(raw_data, columns=['Time', 'Dist_cm', 'Gap_mm', 'Label']).to_csv(
            os.path.join(DATA_FOLDER, output_filename), index=False)
        return True
    return False