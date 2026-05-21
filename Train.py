import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# ==========================================
# 1. ĐƯỜNG DẪN THƯ MỤC CHỨA CẢ FILE EXCEL VÀ PNG
# ==========================================
folder_path = r"D:\Parkinson_HUST\Test_Hello_World\collected_data_2"

# --- BỘ LỌC THÔNG MINH ---
file_list = glob.glob(os.path.join(folder_path, "*.csv")) + glob.glob(os.path.join(folder_path, "*.xlsx"))

print(f"-> Hệ thống nhận diện được: {len(file_list)} file dữ liệu số (Excel/CSV).")
print(f"-> Các file ảnh .png trong thư mục đã được tự động bỏ qua để tránh lỗi.")

if len(file_list) == 0:
    raise ValueError("❌ Không tìm thấy file Excel hoặc CSV nào trong thư mục! Anh kiểm tra lại đường dẫn nha.")

# ==========================================
# 2. HÀM TRÍCH XUẤT ĐẶC TRƯNG CỬA SỔ (WINDOWING)
# ==========================================
WINDOW_SIZE = 60
STEP_SIZE = 15


def extract_window_features(data, window_size, step_size):
    X_features, y_labels = [], []
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[i: i + window_size]

        # Tính toán các đặc trưng thống kê biên độ run
        features = [
            window["Dist_cm"].mean(),
            window["Dist_cm"].std(),
            window["Dist_cm"].max(),
            window["Dist_cm"].min(),
            window["Gap_mm"].mean(),
            window["Gap_mm"].std(),
            window["Gap_mm"].max(),
            window["Gap_mm"].min()
        ]
        label = window["Label"].mode()[0]
        X_features.append(features)
        y_labels.append(label)

    feature_names = ["dist_mean", "dist_std", "dist_max", "dist_min", "gap_mean", "gap_std", "gap_max", "gap_min"]
    return pd.DataFrame(X_features, columns=feature_names), np.array(y_labels)


# ==========================================
# 3. CHIA FILE THEO TỶ LỆ TRAIN/TEST (CHỐNG CAO ẢO)
# ==========================================
# Thay vì trộn cửa sổ, ta chia riêng danh sách file để Train và Test độc lập
train_files, test_files = train_test_split(file_list, test_size=0.25, random_state=42)

print(f"-> Chia file thành công: Dùng {len(train_files)} file để Train, {len(test_files)} file để Test.")


# Hàm phụ để gộp dữ liệu từ một danh sách file cụ thể
def load_and_combine(files_to_load):
    X_list, y_list = [], []
    for file in files_to_load:
        if file.endswith('.csv'):
            df_file = pd.read_csv(file)
        else:
            df_file = pd.read_excel(file)

        required_cols = ["Dist_cm", "Gap_mm", "Label"]
        if not all(col in df_file.columns for col in required_cols):
            continue

        X_file, y_file = extract_window_features(df_file, WINDOW_SIZE, STEP_SIZE)
        if len(X_file) > 0:
            X_list.append(X_file)
            y_list.append(y_file)

    if len(X_list) == 0:
        return pd.DataFrame(), np.array([])

    return pd.concat(X_list, ignore_index=True), np.concatenate(y_list)


print("\n⏳ Đang tự động trích xuất dữ liệu từ các file...")
X_train, y_train = load_and_combine(train_files)
X_test, y_test = load_and_combine(test_files)

print(f"\n🎉 Gom dữ liệu thành công!")
print(
    f"-> Số mẫu tập TRAIN: {X_train.shape[0]} | Phân bố nhãn Train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(
    f"-> Số mẫu tập TEST : {X_test.shape[0]} | Phân bố nhãn Test : {dict(zip(*np.unique(y_test, return_counts=True)))}")

# ==========================================
# 4. CẤU HÌNH XGBOOST GIẢM OVERFITTING
# ==========================================
print("\n🤖 Đang huấn luyện mô hình XGBoost...")
# Kìm hãm sức mạnh của cây (max_depth=3) và giảm tốc độ học để chống học vẹt nhiễu dữ liệu
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# ==========================================
# 5. ĐÁNH GIÁ VÀ LƯU MÔ HÌNH
# ==========================================
y_pred = model.predict(X_test)
print("\n================ KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ================")
print(f"Độ chính xác (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred, zero_division=0))

model.save_model("model_tong_hop_tot_nhat.json")
print("\n🎉 Đã lưu mô hình thành công thành file: 'model_tong_hop_tot_nhat.json'")