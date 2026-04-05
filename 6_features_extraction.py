import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq


# --- HÀM BỔ TRỢ (PHẢI CÓ ĐỂ FIX LỖI NAMEERROR) ---
def get_dominant_freq(gap_data, fs=30):
    if len(gap_data) == 0: return 0
    N = len(gap_data)
    yf = np.abs(fft(gap_data - np.mean(gap_data)))
    xf = fftfreq(N, 1 / fs)
    idx = np.where((xf > 0.5) & (xf < 10))  # Chỉ lấy dải tần Parkinson
    return abs(xf[idx][np.argmax(yf[idx])]) if len(idx[0]) > 0 else 0


def run_extraction(filename):
    # Đọc dữ liệu
    raw_p = os.path.join("collected_data", filename)
    clean_p = os.path.join("filtered_data", filename)

    if not os.path.exists(clean_p): return None

    df_raw = pd.read_csv(raw_p)
    df_clean = pd.read_csv(clean_p)
    time, gap = df_clean['Time'].values, df_clean['Gap_mm'].values

    # Chốt chặn dữ liệu
    if len(time) < 10 or (time[-1] - time[0]) <= 0: return None
    fs = 1 / (time[1] - time[0])

    # Tìm đỉnh (Ràng buộc theo Slide báo cáo)
    peaks, _ = find_peaks(gap, height=15, distance=int(fs / 4), prominence=10)

    if len(peaks) < 3:
        print(f"⚠️ Video {filename} không đủ 3 đỉnh nhấp tay rõ ràng.")
        return None

    # Tính toán (An toàn nhờ hàm get_dominant_freq phía trên)
    hz_fft = get_dominant_freq(gap, fs=fs)
    hz_peaks = len(peaks) / (time[-1] - time[0])
    amp_mean = np.mean(gap[peaks])

    # Vẽ Comparison Plot (Giữ nguyên giao diện Bảo thích)
    plt.figure(figsize=(11, 5))
    plt.plot(df_raw['Time'], df_raw['Gap_mm'], color='gray', alpha=0.3, label='Raw')
    plt.plot(time, gap, color='green', label='Filtered')
    plt.plot(time[peaks], gap[peaks], "ro", label='Peaks')
    plt.title(f"Analysis: {filename} | FFT: {hz_fft:.2f}Hz")
    plt.savefig(os.path.join("features_extraction", filename.replace(".csv", "_COMPARISON.png")))
    plt.close()

    feature = {'File_Name': filename, 'Hz_Peaks': round(hz_peaks, 2), 'Hz_FFT': round(hz_fft, 2),
               'Amplitude_mm': round(amp_mean, 2), 'Label': df_clean['Label'].iloc[0]}

    out_path = "features_extraction/final_features.csv"
    pd.DataFrame([feature]).to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
    return feature