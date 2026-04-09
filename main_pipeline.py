# import os
# import importlib
#
# # Nạp các file 4, 5, 6
# m4 = importlib.import_module("4_module_collecting_data")
# m5 = importlib.import_module("5_signal_processing")
# m6 = importlib.import_module("6_features_extraction")
#
# def main():
#     print("\n🎬 --- HỆ THỐNG PHÂN TÍCH PARKINSON HUST ---")
#     print("1. Dùng Camera | 2. Dùng Video")
#     mode = input("👉 Chọn chế độ: ").strip()
#
#     if mode == '1':
#         source = 0
#         v_name = "cam_live"
#     else:
#         path = input("📂 Nhập đường dẫn hoặc kéo Video vào đây: ").strip().replace('"', '').replace("'", "")
#         if not os.path.exists(path):
#             print("❌ Lỗi: Không thấy file!"); return
#         source = path
#         v_name = os.path.basename(path).split('.')[0]
#
#     label = input("🏷️ Gán nhãn (0: Khỏe, 1: Parkinson): ").strip()
#     fname = f"record_{v_name}_L{label}.csv"
#
#     # Chạy quy trình
#     print("\n--- BƯỚC 1: QUAY VIDEO (Nhấn 'r' để REC) ---")
#     if m4.run_collection(source, label, fname):
#         print("--- BƯỚC 2: LỌC TÍN HIỆU ---")
#         m5.run_processing(fname)
#         print("--- BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG ---")
#         res = m6.run_extraction(fname)
#         if res:
#             print(f"\n✅ THÀNH CÔNG: {res['Hz_FFT']} Hz | {res['Amplitude_mm']} mm")
#     else:
#         print("❌ Lỗi: Không có dữ liệu được ghi.")
#
# if __name__ == "__main__":
#     while True:
#         main()
#         if input("\nLàm tiếp mẫu khác? (y/n): ").lower() != 'y': break
import os
import importlib
import datetime  # Thêm thư viện thời gian

# Nạp các module
m4 = importlib.import_module("4_module_collecting_data")
m5 = importlib.import_module("5_signal_processing")
m6 = importlib.import_module("6_features_extraction")


def main():
    print("\n" + "=" * 40)
    print("🚀 HỆ THỐNG GHI NHẬN DỮ LIỆU KHÔNG GHI ĐÈ")
    print("=" * 40)
    print("1. Dùng Camera | 2. Dùng Video")
    mode = input("👉 Chọn chế độ (1/2): ").strip()

    if mode == '1':
        source = 0
        v_name = "cam"
    else:
        path_input = input("📂 Kéo file video vào đây: ").strip()
        source = path_input.strip('& ').strip("'").strip('"').strip()
        if not os.path.exists(source):
            print("❌ Không thấy file!");
            return
        v_name = os.path.basename(source).split('.')[0]

    label = input("🏷️ Nhãn (0: Khỏe, 1: Parkinson): ").strip()

    # --- KHU VỰC TẠO TÊN FILE DUY NHẤT ---
    # Lấy thời gian hiện tại: NămThángNgày_GiờPhútGiây
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Tên file sẽ có dạng: record_cam_L1_20240321_143005.csv
    fname = f"record_{v_name}_L{label}_{now}.csv"

    print(f"\n📂 Tên bản ghi mới: {fname}")

    # Chạy dây chuyền 3 bước
    if m4.run_collection(source, label, fname):
        print("--- Đang lọc tín hiệu... ---")
        m5.run_processing(fname)

        print("--- Đang trích xuất đặc trưng & vẽ biểu đồ... ---")
        res = m6.run_extraction(fname)

        if res:
            print(f"\n✅ ĐÃ LƯU BẢN GHI: {fname}")
            print(f"📊 Chỉ số: {res['Hz_FFT']}Hz | {res['Amplitude_mm']}mm")
    else:
        print("❌ Lỗi dữ liệu.")


if __name__ == "__main__":
    while True:
        main()
        if input("\nTiếp tục bản ghi mới? (y/n): ").lower() != 'y': break