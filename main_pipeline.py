import os
import importlib

# Nạp các file 4, 5, 6
m4 = importlib.import_module("4_module_collecting_data")
m5 = importlib.import_module("5_signal_processing")
m6 = importlib.import_module("6_features_extraction")

def main():
    print("\n🎬 --- HỆ THỐNG PHÂN TÍCH PARKINSON HUST ---")
    print("1. Dùng Camera | 2. Dùng Video")
    mode = input("👉 Chọn chế độ: ").strip()

    if mode == '1':
        source = 0
        v_name = "cam_live"
    else:
        path = input("📂 Nhập đường dẫn hoặc kéo Video vào đây: ").strip().replace('"', '').replace("'", "")
        if not os.path.exists(path):
            print("❌ Lỗi: Không thấy file!"); return
        source = path
        v_name = os.path.basename(path).split('.')[0]

    label = input("🏷️ Gán nhãn (0: Khỏe, 1: Parkinson): ").strip()
    fname = f"record_{v_name}_L{label}.csv"

    # Chạy quy trình
    print("\n--- BƯỚC 1: QUAY VIDEO (Nhấn 'r' để REC) ---")
    if m4.run_collection(source, label, fname):
        print("--- BƯỚC 2: LỌC TÍN HIỆU ---")
        m5.run_processing(fname)
        print("--- BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG ---")
        res = m6.run_extraction(fname)
        if res:
            print(f"\n✅ THÀNH CÔNG: {res['Hz_FFT']} Hz | {res['Amplitude_mm']} mm")
    else:
        print("❌ Lỗi: Không có dữ liệu được ghi.")

if __name__ == "__main__":
    while True:
        main()
        if input("\nLàm tiếp mẫu khác? (y/n): ").lower() != 'y': break