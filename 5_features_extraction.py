import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
SOURCE_FOLDER = "collected_data"
RESULT_FOLDER = "features_extraction"
OUTPUT_CSV = os.path.join(RESULT_FOLDER, "final_features.csv")

# Tự động tạo thư mục kết quả nếu chưa có
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)
    print(f"--- Đã tạo thư mục: {RESULT_FOLDER} ---")


def process_file(file_path):
    df = pd.read_csv(file_path)
    time = df['Time'].values
    gap = df['Gap_mm'].values
    label = df['Label'].iloc[0]
    filename = os.path.basename(file_path)

    # 2. Thuật toán tìm Đỉnh (Peaks)
    peaks, _ = find_peaks(gap, height=15, distance=8)

    if len(peaks) < 3:
        return None

    # 3. Tính toán Đặc trưng
    duration = time[-1] - time[0]
    freq = len(peaks) / duration
    amps = gap[peaks]
    mean_amp = np.mean(amps)

    # Tính Decay (Độ suy giảm)
    first_3 = np.mean(amps[:3])
    last_3 = np.mean(amps[-3:])
    decay = (first_3 - last_3) / first_3 if first_3 > 0 else 0

    # Tính Variability (Biến thiên nhịp)
    intervals = np.diff(time[peaks])
    rhythm_var = np.std(intervals) / np.mean(intervals) if len(intervals) > 0 else 0

    # 4. VẼ BIỂU ĐỒ VÀ LƯU ẢNH (Để Bảo đưa vào Slide)
    plt.figure(figsize=(10, 4))
    plt.plot(time, gap, label='Gap (mm)', color='#1f77b4')
    plt.plot(time[peaks], gap[peaks], "x", color='red', label='Cú nhấp tay')
    plt.title(f"Phân tích: {filename} (Label: {label})")
    plt.ylabel("Biên độ (mm)")
    plt.xlabel("Thời gian (s)")
    plt.legend()

    # Lưu ảnh vào thư mục features_extraction
    plot_name = filename.replace(".csv", ".png")
    plt.savefig(os.path.join(RESULT_FOLDER, plot_name))
    plt.close()  # Đóng để giải phóng bộ nhớ

    return {
        'File_Name': filename,
        'Hz': round(freq, 2),
        'Amplitude_mm': round(mean_amp, 2),
        'Decay_Rate': round(decay, 3),
        'Rhythm_Var': round(rhythm_var, 3),
        'Label': label
    }


# --- CHƯƠNG TRÌNH CHÍNH ---
all_results = []
files = [f for f in os.listdir(SOURCE_FOLDER) if f.endswith('.csv')]

if not files:
    print(f"Lỗi: Không tìm thấy file CSV nào trong '{SOURCE_FOLDER}'")
else:
    for f in files:
        res = process_file(os.path.join(SOURCE_FOLDER, f))
        if res:
            all_results.append(res)
            print(f"Successfully processed: {f}")

    # Xuất file CSV tổng hợp
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n--- XỬ LÝ HOÀN TẤT ---")
        print(f"1. File tổng hợp: {OUTPUT_CSV}")
        print(f"2. Ảnh biểu đồ đã lưu vào thư mục: {RESULT_FOLDER}")