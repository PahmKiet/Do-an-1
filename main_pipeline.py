import os
import importlib

# Nạp module bằng tên file (vì tên file bắt đầu bằng số)
m4 = importlib.import_module("4_module_collecting_data")
m5 = importlib.import_module("5_signal_processing")
m6 = importlib.import_module("6_features_extraction")


def main():
    print("\n--- HỆ THỐNG XỬ LÝ DỮ LIỆU TỰ ĐỘNG ---")
    print("1. Dùng Camera | 2. Dùng Video")
    mode = input("Chọn: ")
    source = 0 if mode == '1' else input("Kéo file video vào đây: ").strip().replace('"', '')

    if mode == '2' and not os.path.exists(source): return

    label = input("Nhãn (0: Khỏe, 1: Parkinson): ")
    v_name = "cam_session" if mode == '1' else os.path.basename(source).split('.')[0]
    fname = f"record_{v_name}_L{label}.csv"

    if m4.run_collection(source, label, fname):
        m5.run_processing(fname)
        res = m6.run_extraction(fname)
        if res: print(f"Xong: {res['Hz_FFT']}Hz | {res['Amplitude_mm']}mm")
    else:
        print("Lỗi dữ liệu.")


if __name__ == "__main__":
    while True:
        main()
        if input("\nTiếp tục? (y/n): ").lower() != 'y': break