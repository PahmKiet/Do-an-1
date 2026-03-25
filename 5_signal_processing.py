import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt

# --- CẤU HÌNH ĐƯỜNG DẪN ---
RAW_FOLDER = "collected_data"
FILTERED_FOLDER = "filtered_data"

# Tự động tạo thư mục filtered_data nếu chưa có
for folder in [RAW_FOLDER, FILTERED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"--- Đã tạo thư mục: {folder} ---")


def butter_lowpass_filter(data, cutoff=6, fs=30, order=5):
    """Hàm lọc thông thấp Butterworth để làm mịn tín hiệu."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def process_all_raw_files():
    """
    Quét collected_data, lọc mượt, lưu file sạch vào filtered_data
    và vẽ ảnh sóng thô vào collected_data.
    """
    files = [f for f in os.listdir(RAW_FOLDER) if f.endswith('.csv')]

    if not files:
        print(f"Lỗi: Không tìm thấy dữ liệu thô (.csv) trong '{RAW_FOLDER}'")
        return

    print(f"--- ĐANG LỌC DỮ LIỆU & VẼ ẢNH RAW TỪ {RAW_FOLDER} ---")

    for filename in files:
        # 1. Đọc file thô
        raw_path = os.path.join(RAW_FOLDER, filename)
        df = pd.read_csv(raw_path)

        # --- BƯỚC MỚI: VẼ ẢNH SÓNG THÔ (DÙNG ĐỂ SO SÁNH) ---
        plt.figure(figsize=(10, 4))
        plt.plot(df['Time'].values, df['Gap_mm'].values, color='gray', alpha=0.6, label='Gap Raw (Nhiễu)')
        plt.title(f"Sóng Thô (Raw Signal): {filename} (Label: {df['Label'].iloc[0]})")
        plt.ylabel("Biên độ (mm)")
        plt.xlabel("Thời gian (s)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Lưu ảnh vào cùng thư mục collected_data
        raw_plot_name = filename.replace(".csv", "_RAW_PLOT.png")
        plt.savefig(os.path.join(RAW_FOLDER, raw_plot_name))
        plt.close()  # Đóng để giải phóng bộ nhớ

        # 2. Thực hiện chuỗi lọc kép (Median -> Butterworth)
        gap_median = medfilt(df['Gap_mm'].values, kernel_size=5)
        gap_filtered = butter_lowpass_filter(gap_median)

        # 3. Cập nhật lại dữ liệu và lưu sang filtered_data
        df_filtered = df.copy()
        df_filtered['Gap_mm'] = np.round(gap_filtered, 2)

        output_path = os.path.join(FILTERED_FOLDER, f"filtered_{filename}")
        df_filtered.to_csv(output_path, index=False)
        print(f"Xong: {filename} -> filtered_{filename} & {raw_plot_name}")

    print(f"--- HOÀN TẤT! Đã có file sạch trong '{FILTERED_FOLDER}' và ảnh Raw trong '{RAW_FOLDER}' ---")


if __name__ == "__main__":
    process_all_raw_files()