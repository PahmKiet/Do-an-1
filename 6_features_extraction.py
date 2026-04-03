import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq


# Hàm định nghĩa Tần số trội (BẮT BUỘC PHẢI CÓ)
def get_dominant_freq(gap_data, fs=30):
    N = len(gap_data)
    yf = np.abs(fft(gap_data - np.mean(gap_data)))
    xf = fftfreq(N, 1 / fs)
    idx = np.where((xf > 0.5) & (xf < 10))
    return abs(xf[idx][np.argmax(yf[idx])]) if len(idx[0]) > 0 else 0


def run_extraction(filename):
    df_raw = pd.read_csv(os.path.join("collected_data", filename))
    df_clean = pd.read_csv(os.path.join("filtered_data", filename))
    time, gap = df_clean['Time'].values, df_clean['Gap_mm'].values

    # Chốt chặn 1: Tránh chia cho 0 nếu video quá ngắn
    duration = time[-1] - time[0]
    if duration <= 0: return None

    fs = 1 / (time[1] - time[0]) if len(time) > 1 else 30
    peaks, _ = find_peaks(gap, height=15, distance=int(fs / 4), prominence=10)

    # Chốt chặn 2: Tránh Mean of empty slice (Nếu không tìm thấy đỉnh)
    if len(peaks) < 3:
        print(f"⚠️ Video {filename} không đủ đỉnh nhấp tay rõ ràng.")
        return None

    hz_fft = get_dominant_freq(gap, fs=fs)
    hz_peaks = len(peaks) / duration
    amp_mean = np.mean(gap[peaks])

    # Vẽ Comparison Plot (Giữ nguyên giao diện của Bảo)
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