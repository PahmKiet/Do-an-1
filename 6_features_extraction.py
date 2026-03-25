import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# --- CẤU HÌNH ---
RAW_FOLDER = "collected_data"  # Để lấy dữ liệu thô vẽ so sánh
SOURCE_FOLDER = "filtered_data"  # Để lấy dữ liệu sạch trích đặc trưng
RESULT_FOLDER = "features_extraction"
OUTPUT_CSV = os.path.join(RESULT_FOLDER, "final_features.csv")

if not os.path.exists(RESULT_FOLDER): os.makedirs(RESULT_FOLDER)


def get_dominant_freq(gap_data, fs=30):
    N = len(gap_data)
    yf = np.abs(fft(gap_data - np.mean(gap_data)))
    xf = fftfreq(N, 1 / fs)
    idx = np.where((xf > 0.5) & (xf < 10))
    return abs(xf[idx][np.argmax(yf[idx])]) if len(idx[0]) > 0 else 0


def process_file(filename):
    # 1. Đọc dữ liệu thô và dữ liệu sạch
    df_raw = pd.read_csv(os.path.join(RAW_FOLDER, filename))
    # File sạch có tên là filtered_... (theo logic file signal_processing)
    df_clean = pd.read_csv(os.path.join(SOURCE_FOLDER, f"filtered_{filename}"))

    time = df_clean['Time'].values
    gap_raw = df_raw['Gap_mm'].values
    gap_clean = df_clean['Gap_mm'].values
    label = df_clean['Label'].iloc[0]

    # 2. Tìm Đỉnh và FFT trên dữ liệu sạch
    peaks, _ = find_peaks(gap_clean, height=15, distance=8)
    if len(peaks) < 3: return None

    hz_fft = get_dominant_freq(gap_clean)
    hz_peaks = len(peaks) / (time[-1] - time[0])

    # 3. VẼ BIỂU ĐỒ SO SÁNH "TRƯỚC & SAU"
    plt.figure(figsize=(11, 5))
    # Vẽ đường thô (màu xám, nhạt)
    plt.plot(time, gap_raw, label='Dữ liệu thô (Raw)', color='gray', alpha=0.3, linestyle='--')
    # Vẽ đường sạch (màu xanh lá, đậm)
    plt.plot(time, gap_clean, label='Dữ liệu đã lọc (Butterworth)', color='#2ca02c', linewidth=2)
    # Đánh dấu đỉnh
    plt.plot(time[peaks], gap_clean[peaks], "ro", markersize=7, label='Đỉnh nhấp tay')

    plt.title(f"So sánh & Trích xuất: {filename} | FFT: {hz_fft:.2f}Hz")
    plt.ylabel("Khoảng cách (mm)")
    plt.xlabel("Thời gian (s)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)

    # Lưu ảnh duy nhất vào features_extraction
    plt.savefig(os.path.join(RESULT_FOLDER, filename.replace(".csv", "_COMPARISON.png")))
    plt.close()

    return {
        'File_Name': filename,
        'Hz_Peaks': round(hz_peaks, 2),
        'Hz_FFT': round(hz_fft, 2),
        'Amplitude_mm': round(np.mean(gap_clean[peaks]), 2),
        'Label': label
    }


# --- CHẠY ---
all_results = []
files = [f for f in os.listdir(RAW_FOLDER) if f.endswith('.csv')]

for f in files:
    if os.path.exists(os.path.join(SOURCE_FOLDER, f"filtered_{f}")):
        res = process_file(f)
        if res:
            all_results.append(res)
            print(f"Done: {f}")

pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)
print(f"\n--- Xong! Kiểm tra ảnh so sánh trong {RESULT_FOLDER} ---")