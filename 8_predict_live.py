import cv2
import mediapipe as mp
import numpy as np
import pickle
import math
from scipy.signal import butter, filtfilt, medfilt, find_peaks
from scipy.fft import fft, fftfreq

# 1. Cấu hình MediaPipe và Hằng số chuẩn hóa
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
REAL_HAND_WIDTH_CM = 7.0
A_COEFF, B_COEFF = -9.72e-4, 0.86
OFFSET_MM = 18.0

# 2. Nạp "Bộ não" AI
model = pickle.load(open('parkinson_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


def get_features(gap_data, fs=30):
    # Lọc nhiễu
    gap_median = medfilt(gap_data, kernel_size=5)
    b, a = butter(5, 6 / (0.5 * fs), btype='low')
    gap_clean = filtfilt(b, a, gap_median)

    # Tính Hz bằng FFT
    N = len(gap_clean)
    yf = np.abs(fft(gap_clean - np.mean(gap_clean)))
    xf = fftfreq(N, 1 / fs)
    idx = np.where((xf > 0.5) & (xf < 10))
    hz_fft = abs(xf[idx][np.argmax(yf[idx])]) if len(idx[0]) > 0 else 0

    # Tính Hz bằng Peaks và Amplitude
    peaks, _ = find_peaks(gap_clean, height=15, distance=int(fs / 4), prominence=10)
    hz_peaks = len(peaks) / (len(gap_clean) / fs)
    amp = np.mean(gap_clean[peaks]) if len(peaks) > 0 else 0

    return [hz_peaks, hz_fft, amp]


def run_live():
    cap = cv2.VideoCapture(0)
    gap_buffer = []
    prediction_text = "COLLECTING DATA..."
    color = (255, 255, 0)

    print("--- Đang chạy dự đoán Real-time. Nhấn 'q' để thoát ---")

    while cap.isOpened():
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                p5, p17 = hand_lms.landmark[5], hand_lms.landmark[17]
                p8, p4 = hand_lms.landmark[8], hand_lms.landmark[4]
                px_w = math.hypot(p5.x * w - p17.x * w, p5.y * h - p17.y * h)
                dist_px = math.hypot(p8.x * w - p4.x * w, p8.y * h - p4.y * h)

                if px_w > 0:
                    r_coeff = A_COEFF * px_w + B_COEFF
                    gap_mm = max(0, dist_px * r_coeff - OFFSET_MM)
                    gap_buffer.append(gap_mm)

        # Khi đủ 150 khung hình (khoảng 5 giây với 30fps)
        if len(gap_buffer) >= 150:
            feats = get_features(gap_buffer)
            # Chuẩn hóa và Dự đoán
            feats_scaled = scaler.transform([feats])
            prob = model.predict_proba(feats_scaled)[0]
            label = model.predict(feats_scaled)[0]

            if label == 1:
                prediction_text = f"WARNING: PARKINSON ({round(prob[1] * 100)}%)"
                color = (0, 0, 255)
            else:
                prediction_text = f"HEALTHY ({round(prob[0] * 100)}%)"
                color = (0, 255, 0)

            gap_buffer = []  # Reset buffer để lấy đợt tiếp theo

        cv2.putText(img, prediction_text, (50, 50), 2, 1, color, 2)
        cv2.imshow("PARKINSON REAL-TIME DIAGNOSIS", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live()